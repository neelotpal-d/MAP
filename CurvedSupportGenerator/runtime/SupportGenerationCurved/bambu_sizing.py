"""Regular Bambu tree radius/auto-brim calculations, adapted to saved routes.

Ported from Bambu Lab BambuStudio (AGPL-3.0), commit
79df7ae309be398ed30c9f397c13c9911c2edc57. Original source and license are
in upstream_bambu/. This is not Bambu's planner or collision-grid quantizer.
"""
import numpy as np
import pyvista as pv

UPSTREAM_COMMIT = "79df7ae309be398ed30c9f397c13c9911c2edc57"


def branch_radius(distance, diameter=5.0, angle=5.0, interface=True):
    """TreeSupport::calc_branch_radius before collision-grid ceilRadius.

    The regular-tree implementation multiplies by the angle in radians,
    not tan(angle); the organic-tree implementation is a different policy.
    """
    if not np.isfinite([diameter, angle]).all() or diameter <= 0 or not 0 <= angle <= 15:
        raise ValueError("Require positive branch diameter and angle in [0,15].")
    distance = np.asarray(distance, dtype=float)
    base = diameter / 2
    radius = np.where(distance > base, base + (distance - base) * np.deg2rad(angle), distance)
    radius = np.clip(radius, 0.4, 10.0)
    return np.maximum(radius, base) if interface else radius


def route_radii(points, lines, contact_heights, direction, diameter=5., angle=5.):
    """Adapt parent-height accumulation to the preserved route graph.

    Shared nodes use the greatest vertical distance to a contributing contact.
    Contact points and the separate terminal/roof construction are preserved.
    """
    top = np.full(len(points), -np.inf)
    for line, height in zip(lines, contact_heights):
        np.maximum.at(top, line, height)
    distance = np.maximum(0, top - np.asarray(points) @ direction)
    return branch_radius(distance, diameter, angle), distance


def polygon_moments(polygon):
    """Port Brim.cpp's signed-ring moments and centroid-axis correction, mm^4."""
    from shapely.geometry.polygon import orient
    polygon = orient(polygon, sign=1.)
    center = np.array([polygon.centroid.x, polygon.centroid.y])
    # Translating to the centroid implements the same parallel-axis correction
    # while avoiding cancellation for large workspace coordinates.
    moment = np.zeros(2)
    for ring in [polygon.exterior, *polygon.interiors]:
        p = np.asarray(ring.coords)[:-1] - center
        q = np.roll(p, -1, axis=0)
        cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        moment += np.array([np.sum((p[:, 1]**2 + p[:, 1]*q[:, 1] + q[:, 1]**2)*cross),
                            np.sum((p[:, 0]**2 + p[:, 0]*q[:, 0] + q[:, 0]**2)*cross)]) / 12
    return moment


def auto_expansion(polygon, speed=80., height=0.):
    """Port ExPolygon::map_moment_to_expansion, with all lengths in mm."""
    if not np.isfinite([speed, height]).all() or speed < 0 or height < 0:
        raise ValueError("Require finite nonnegative speed and contact height.")
    if height <= 0 or speed <= 0:
        return 0.
    ixx, iyy = polygon_moments(polygon)
    if min(ixx, iyy) <= 0:
        raise ValueError("Platform footprint has zero second moment.")
    xmin, ymin, xmax, ymax = polygon.bounds
    height_to_area = max(height / ixx * (ymax-ymin), height / iyy * (xmax-xmin)) * height / 1920
    return float(np.clip(height_to_area * speed, 1., 10.))


class FlangeField:
    """Union of tree with a first-layer offset polygon, cleared against model."""
    def __init__(self, geometry, polygon, value):
        self.geometry, self.polygon, self.value = geometry, polygon, value

    def evaluate(self, points):
        import shapely
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        distances = self.geometry.distance(points)
        xy = points[:, [0, 2]]
        lateral = shapely.distance(shapely.points(xy), self.polygon.boundary)
        lateral = np.where(shapely.contains_xy(self.polygon, xy[:, 0], xy[:, 1]), -lateral, lateral)
        flange = np.maximum(lateral, self.geometry.side_gap - distances)
        flange = np.maximum(flange, np.abs(points[:, 1] - self.value) - 1e-6)
        # Accepted flange points already certify union membership. Only points
        # rejected by the flange need an original-tree query (including contact
        # allowances near the model). This avoids evaluating every branch over
        # the entire expanded platform footprint.
        outside = flange > 0
        if np.any(outside):
            flange[outside] = np.minimum(self.geometry.evaluate(points[outside], distances[outside]), flange[outside])
        return self.geometry.constrain_bounds(points,flange) if getattr(self.geometry,"maximum_projection",None) is not None else flange


def expand_platform_layer(surface, geometry, value, roots, speed=80.):
    """Offset actual first-layer polygons; then apply existing clearance checks.

    Bambu's width equation is retained. Shapely replaces its integer polygon
    operations; the project's continuous model distance replaces its planar
    collision map. The flange is layer geometry, not the coarse solid preview.
    """
    import shapely
    from shapely.geometry import Point
    from .tree_geometry import CachedLayerField, refined_clip_layer
    if not surface.n_cells:
        return surface, CachedLayerField(geometry), dict(component_count=0, widths_mm=[]), 0
    triangles = np.asarray(surface.points)[surface.regular_faces][:, :, [0, 2]]
    # Match libslic3r's 1e-6 mm integer-coordinate resolution. Float contour
    # seams must not become tiny independent islands with a spurious 10 mm
    # automatic brim. Clearance is rechecked after the polygon operation.
    footprint = shapely.union_all(shapely.polygons(triangles), grid_size=1e-6)
    polygons = [footprint] if footprint.geom_type == "Polygon" else list(footprint.geoms)
    heights_by_component = {}
    for root in roots:
        candidates = [i for i, polygon in enumerate(polygons)
                      if polygon.distance(Point(root[:2])) <= root[3]+1e-6]
        if candidates:
            # TreeSupport.cpp::avoid_object_remove_extra_small_parts retains
            # the largest collision-clipped piece of each node. Apply that
            # rule to the actual slice components associated with this root;
            # numerical slivers must not seed independent automatic flanges.
            chosen = max(candidates, key=lambda i: polygons[i].area)
            heights_by_component.setdefault(chosen, []).append(root[2])
    expanded, details = [], []
    for number, polygon in enumerate(polygons):
        if polygon.area <= 1e-10:
            continue
        heights = heights_by_component.get(number, [])
        if not heights:
            # No platform-reaching route contributes to this island.
            expanded.append(polygon)
            continue
        height = max(0., max(heights) - (value - geometry.plate))
        width = auto_expansion(polygon, speed, height)
        expanded.append(polygon.buffer(width, quad_segs=16))
        details.append(dict(width_mm=width, contact_height_mm=height, initial_area_mm2=polygon.area))
    polygon = shapely.union_all(expanded).simplify(1e-6, preserve_topology=True)
    # Constrained triangulation keeps holes and expanded polygon boundaries.
    faces = np.array([np.asarray(triangle.exterior.coords)[:3]
        for triangle in shapely.get_parts(shapely.constrained_delaunay_triangles(polygon))])
    xy = faces.reshape(-1, 2)
    points = np.column_stack((xy[:, 0], np.full(len(xy), value), xy[:, 1]))
    cells = np.column_stack((np.full(len(faces), 3), np.arange(len(points)).reshape(-1, 3)))
    candidate = pv.PolyData(points, cells.ravel()).clean()
    candidate["__world_coordinates"] = np.asarray(candidate.points, dtype=float).copy()
    candidate = candidate.subdivide_adaptive(max_edge_len=1., max_n_passes=12, max_n_tris=500000)
    candidate.points = np.asarray(candidate["__world_coordinates"], dtype=float).copy()
    del candidate.point_data["__world_coordinates"]
    evaluator = CachedLayerField(FlangeField(geometry, polygon, value))
    trimmed, removed = refined_clip_layer(candidate, evaluator)
    return trimmed, evaluator, dict(component_count=len(details), components=details,
        widths_mm=[d["width_mm"] for d in details], initial_area_mm2=float(surface.area),
        expanded_area_mm2=float(trimmed.area)), removed

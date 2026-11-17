"""Local model-offset roof, independent of the print scalar field."""
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from .tree_geometry import smooth_maximum, TaperedTree, sample_mesh_distance


def contact_patch(tree, terminal_segment=3939, radius=3, minimum_support_height=0.):
    if not 0 <= terminal_segment < len(tree.segments) or not tree.terminal[terminal_segment]:
        raise ValueError("Roof seed must identify a terminal segment.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Roof patch radius must be positive.")
    normal = tree.tip_normals[terminal_segment]
    center = tree.b[terminal_segment] - tree.contact_gap * normal
    mesh = tree.mesh.triangulate()
    centers = mesh.cell_centers().points
    normals = np.asarray(mesh.cell_normals)
    allowed = (np.linalg.norm(centers - center, axis=1) <= radius)
    allowed &= normals @ tree.direction < -0.2
    allowed &= normals @ normal > 0.25
    from .support_region import height_mask
    if minimum_support_height>0:
        allowed &= height_mask(centers,tree.direction,tree.plate,minimum_support_height)
    seed = int(mesh.find_closest_cell(center))
    if not allowed[seed]:
        raise ValueError("Roof seed is not inside a suitably oriented overhang patch.")
    # Connected faces through shared edges; nearby unrelated features excluded.
    faces = np.asarray(mesh.regular_faces)
    edges = {}
    for cell in np.flatnonzero(allowed):
        for first, second in zip(faces[cell], np.roll(faces[cell], -1)):
            edges.setdefault(tuple(sorted((int(first), int(second)))), []).append(int(cell))
    adjacency = {int(cell): set() for cell in np.flatnonzero(allowed)}
    for cells in edges.values():
        for cell in cells:
            adjacency[cell].update(other for other in cells if other != cell)
    chosen, pending = set(), [seed]
    while pending:
        cell = pending.pop()
        if cell in chosen:
            continue
        chosen.add(cell)
        pending.extend(adjacency[cell] - chosen)
    selected = faces[sorted(chosen)]
    patch = pv.PolyData(np.asarray(mesh.points, dtype=float),
                        np.column_stack((np.full(len(selected), 3), selected)).ravel()).clean()
    return patch, {"terminal_segment": terminal_segment, "patch_radius": radius,
                   "patch_faces": patch.n_cells, "patch_area_mm2": float(patch.area),
                   "model_face_ids": sorted(chosen),
                   "contact_center": center.tolist()}


class OffsetRoof:
    def __init__(self, patch, gap=0.2, thickness=0.6, rounding=0.1, side_mesh=None, side_gap=0.4,direction=(0,1,0)):
        if not np.all(np.isfinite([gap, thickness, rounding])) or gap < 0 or thickness <= 0 or rounding < 0 or rounding >= thickness:
            raise ValueError("Require nonnegative gap/rounding and roof thickness > rounding.")
        self.patch, self.gap, self.thickness, self.rounding = patch, gap, thickness, rounding
        self.function = vtkImplicitPolyDataDistance()
        self.function.SetInput(patch)
        self.sampler = vtkSampleImplicitFunctionFilter()
        self.sampler.SetImplicitFunction(self.function)
        self.sampler.SetComputeGradients(False)
        self.input_mesh = pv.PolyData()
        self.lateral_clearance = None
        self.side_gap = side_gap
        if side_mesh is not None and side_mesh.n_cells:
            from .side_clearance import LateralClearance
            self.lateral_clearance=LateralClearance(side_mesh,direction,side_gap)
        bounds = np.asarray(patch.bounds).reshape(3, 2)
        margin = gap + thickness
        self.bounds = bounds + np.array([-margin, margin])

    def evaluate(self, points, model_distance):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        distance = np.asarray(model_distance, dtype=float)
        result = np.full(len(points), np.inf)
        eligible = np.all((points >= self.bounds[:, 0]) & (points <= self.bounds[:, 1]), axis=1)
        if np.any(eligible):
            patch_distance = sample_mesh_distance(self.function, self.sampler, points[eligible], self.input_mesh).astype(float)
            # Finite open patch supplies the lateral extent: its unsigned
            # distance rounds the outside perimeter. Signed distance restricts
            # the roof to the outward side. The full model enforces clearance.
            lower = self.gap - distance[eligible]
            outer = np.abs(patch_distance) - (self.gap + self.thickness)
            roof = smooth_maximum(lower, outer, self.rounding)
            roof = smooth_maximum(roof, -patch_distance, self.rounding)
            if self.lateral_clearance is not None:
                other_distance = self.lateral_clearance.distance(points[eligible])
                roof = smooth_maximum(roof, self.side_gap - other_distance, self.rounding)
            result[eligible] = roof
        return result


class RoofTree:
    """Exact union of already-cleared tree and local offset roof."""
    def __init__(self, tree, roof):
        self.tree, self.roof = tree, roof
        self.roof_sample_bounds = roof.bounds

    def __getattr__(self, name):
        return getattr(self.tree, name)

    def evaluate(self, points, distances=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        distances = self.tree.distance(points) if distances is None else np.asarray(distances)
        return self.tree.constrain_bounds(points,np.minimum(self.tree.evaluate(points, distances), self.roof.evaluate(points, distances)))


def attach_roof(tree, terminal=3939, radius=3, thickness=0.6, rounding=0.1,minimum_support_height=0.):
    patch, report = contact_patch(tree, terminal, radius,minimum_support_height)
    faces = np.asarray(tree.mesh.regular_faces)
    keep = np.ones(len(faces), dtype=bool)
    keep[report["model_face_ids"]] = False
    other = pv.PolyData(np.asarray(tree.mesh.points, dtype=float),
                        np.column_stack((np.full(keep.sum(), 3), faces[keep])).ravel()).clean()
    roof = OffsetRoof(patch, tree.contact_gap, thickness, rounding, other, tree.side_gap,tree.direction)
    return RoofTree(tree, roof), report


def direct_layer(geometry, value, spacing=1, local_spacing=0.1):
    """Analytic-y candidate plane independent of the sampled solid preview."""
    if not np.all(np.isfinite([spacing, local_spacing])) or min(spacing, local_spacing) <= 0:
        raise ValueError("Direct-layer spacings must be finite and positive.")
    bounds = geometry.roof.bounds
    padding = float(max(geometry.ra.max(), geometry.rb.max())) + spacing
    lower = np.minimum(geometry.a.min(axis=0), geometry.b.min(axis=0)) - padding
    upper = np.maximum(geometry.a.max(axis=0), geometry.b.max(axis=0)) + padding
    lower = np.minimum(lower, bounds[:, 0] - spacing)
    upper = np.maximum(upper, bounds[:, 1] + spacing)
    coordinates = []
    for axis in (0, 2):
        regular = np.arange(lower[axis], upper[axis] + spacing, spacing)
        fine = np.arange(bounds[axis, 0] - spacing, bounds[axis, 1] + spacing, local_spacing) if bounds[1, 0] <= value <= bounds[1, 1] else np.empty(0)
        coordinates.append(np.unique(np.concatenate((regular, fine))))
    x, z = coordinates
    if len(x) * len(z) > 500_000:
        raise ValueError("Direct-layer candidate exceeds 500,000 points.")
    xx, zz = np.meshgrid(x, z, indexing="ij")
    points = np.column_stack((xx.ravel(), np.full(xx.size, value), zz.ravel()))
    ids = np.arange(len(points)).reshape(len(x), len(z))
    # Conservative primitive boxes retain every possible inside point while
    # avoiding subdivision/evaluation of the entire empty model bounding box.
    extent = (np.maximum(geometry.ra, geometry.rb) + geometry.blend_extent)[:, None]
    boxes = np.stack((np.minimum(geometry.a, geometry.b) - extent,
                      np.maximum(geometry.a, geometry.b) + extent), axis=-1)
    boxes = np.concatenate((boxes, bounds[None]), axis=0)
    active = layer_box_mask(x, z, boxes, value)
    a, b, c, d = ids[:-1, :-1][active], ids[1:, :-1][active], ids[1:, 1:][active], ids[:-1, 1:][active]
    faces = np.concatenate((np.column_stack((np.full(len(a), 3), a, b, c)),
                            np.column_stack((np.full(len(a), 3), a, c, d))))
    return pv.PolyData(points, faces.ravel()).clean(polys_to_lines=False, lines_to_points=False)


def layer_box_mask(x, z, boxes, value):
    """Union of grid rectangles, with batched bounds and a 2D prefix sum."""
    boxes = np.asarray(boxes)
    boxes = boxes[(boxes[:, 1, 0] <= value) & (value <= boxes[:, 1, 1])]
    lo_x = np.clip(np.searchsorted(x, boxes[:, 0, 0], side="left") - 1, 0, len(x) - 1)
    hi_x = np.clip(np.searchsorted(x, boxes[:, 0, 1], side="right"), 0, len(x) - 1)
    lo_z = np.clip(np.searchsorted(z, boxes[:, 2, 0], side="left") - 1, 0, len(z) - 1)
    hi_z = np.clip(np.searchsorted(z, boxes[:, 2, 1], side="right"), 0, len(z) - 1)
    counts = np.zeros((len(x), len(z)), dtype=np.int64)
    np.add.at(counts, (lo_x, lo_z), 1)
    np.add.at(counts, (hi_x, lo_z), -1)
    np.add.at(counts, (lo_x, hi_z), -1)
    np.add.at(counts, (hi_x, hi_z), 1)
    return counts.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0


def geometry_for_layer(geometry, value):
    """Keep only primitives capable of contributing on this analytic-y plane."""
    tree = geometry.tree
    margin = np.maximum(tree.ra, tree.rb)
    if tree.geometry_mode == "smooth":
        margin += max(tree.branch_blend, tree.clearance_blend, tree.rim_blend)
    keep = (np.minimum(tree.a[:, 1], tree.b[:, 1]) - margin <= value)
    keep &= np.maximum(tree.a[:, 1], tree.b[:, 1]) + margin >= value
    ids = np.flatnonzero(keep)
    if not len(ids):
        return geometry
    if tree.geometry_mode == "hard":
        # Hard union is idempotent; shared route segments need evaluating once.
        ids = tree.unique_primitive_ids[keep[tree.unique_primitive_ids]]
    sliced = TaperedTree(tree.segments[ids], tree.mesh, tree.direction, tree.plate,
                         tree.side_gap, tree.contact_gap, tree.roof_clearance_mode,
                         tree.tip_normals[ids], tree.terminal_cap, tree.geometry_mode,
                         tree.branch_blend, tree.clearance_blend, tree.rim_blend, distance_source=tree)
    return RoofTree(sliced, geometry.roof)

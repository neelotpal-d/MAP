"""Experimental, local support for one plate-visible overhang.

Project author: Neelotpal Dutta
"""
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.distance import pdist

from .contact_coverage import surface_patches
from .field_grid import GridField
from .platform_router.acceleration import NativeDistance
from .platform_router.context import save
from .platform_router.engine import load_support_field
from .tree_geometry import TaperedTree, configure_geometry_threads

MODE = 'separated-posts'
TIP_RADIUS = .5
NECK_LENGTH = 1.2
PITCH = 1.5
BODY_RADIUS = .7
TAPER_LENGTH = 1.5
PATCH_RADIUS = 3.98


def _roof_y(triangles, xz):
    """First nonvertical triangle intersection when looking up from the plate."""
    if not len(triangles):
        return np.full(len(xz), np.inf)
    origin = triangles[:, 0]
    edges = triangles[:, 1:] - origin[:, None]
    basis = edges[:, :, [0, 2]].transpose(0, 2, 1)
    valid = abs(np.linalg.det(basis)) > 1e-12
    if not valid.any():
        return np.full(len(xz), np.inf)
    uv = np.einsum('fij,pfj->pfi', np.linalg.inv(basis[valid]), xz[:, None]-origin[None, valid][:, :, [0, 2]])
    inside = (uv >= -1e-9).all(axis=2) & (uv.sum(axis=2) <= 1+1e-9)
    y = origin[valid, 1] + np.einsum('pfi,fi->pf', uv, edges[valid, :, 1])
    return np.where(inside, y, np.inf).min(axis=1)


def _footprint(mesh, native, triangles, angle, min_y):
    import shapely

    centers = triangles.mean(axis=1)
    normals = np.asarray(mesh.cell_normals)
    _, gradient = native.evaluate_scalars_and_gradients(centers)
    length = np.linalg.norm(gradient, axis=1)
    dot = np.einsum('ij,ij->i', normals, gradient) / np.maximum(length, 1e-15)
    candidates = np.flatnonzero((length > 1e-8) & (dot < -np.cos(np.deg2rad(angle))) &
                                (normals[:, 1] <= -.85) & (centers[:, 1] > min_y))
    visible = []
    for i in candidates:
        x, y, z = centers[i]
        hit, _ = mesh.ray_trace((x, 0., z), (x, y+.03, z), first_point=True)
        if len(hit) and abs(hit[1]-y) <= .01:
            visible.append(i)
    if visible:
        visible = np.asarray(visible)
        labels, count = surface_patches(mesh.regular_faces[visible])
        patches = []
        for label in range(count):
            patch = triangles[visible[labels == label]]
            region = shapely.union_all([shapely.Polygon(t[:, [0, 2]]) for t in patch])
            if region.area >= 15.:
                diameter = shapely.maximum_inscribed_circle(region, tolerance=.02)
                patches.append((diameter.length, region, np.asarray(diameter.coords[0])))
        if patches:
            return max(patches, key=lambda p: p[0])
    raise ValueError('Separated posts need a broad downward overhang with direct visibility from the plate.')


def _place(mesh, field, triangles, region, center, gap, min_y):
    import shapely

    offsets = PITCH*np.array([(i+.5*(j % 2), np.sqrt(3.)/2*j)
                             for j in range(-3, 4) for i in range(-3, 4)])
    xz = center + offsets[np.linalg.norm(offsets, axis=1) <= PATCH_RADIUS]
    xz = xz[shapely.contains_xy(region.buffer(-BODY_RADIUS), *xz.T)]
    projected = triangles[:, :, [0, 2]]
    lower, upper = projected.min(axis=1), projected.max(axis=1)
    distance = NativeDistance(mesh)
    grid = np.linspace(-BODY_RADIUS, BODY_RADIUS, 29)
    disk = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    disk = disk[np.linalg.norm(disk, axis=1) <= BODY_RADIUS+1e-10]
    accepted = []
    for xy in xz:
        local = ((lower <= xy+BODY_RADIUS) & (upper >= xy-BODY_RADIUS)).all(axis=1)
        nearby = triangles[local]
        y = _roof_y(nearby, xy[None])[0]
        if not np.isfinite(y) or y <= min_y:
            continue
        lo, hi = y-(2*gap+.1), y
        if distance([[xy[0], lo, xy[1]]])[0] < gap:
            continue
        for _ in range(24):
            mid = (lo+hi)/2
            if distance([[xy[0], mid, xy[1]]])[0] >= gap+2e-6:
                lo = mid
            else:
                hi = mid
        if lo <= NECK_LENGTH+TAPER_LENGTH+min_y or np.any(_roof_y(nearby, xy+disk) < lo):
            continue
        # The preserved scalar must increase along the plate-to-tip axis.
        knots = field.origin[1]+field.spacing*np.arange(field.shape[1])
        ys = np.r_[0., knots[(knots > 0) & (knots < lo)], lo]
        points = np.column_stack((np.full(len(ys), xy[0]), ys, np.full(len(ys), xy[1])))
        if np.all(np.diff(field.evaluate_scalars(points)) > 0):
            accepted.append(points[-1])
    if len(accepted) < 3:
        raise ValueError('Fewer than three clear plate-to-overhang posts fit this patch and support field.')
    return np.asarray(accepted), len(xz)


def route(pipeline, output, *, clearance=.4, overhang_angle=60., minimum_support_height=.6, maximum_contacts=5000):
    from SupportGenerationVoxel.model_field import TetrahedralVertexField, load_closed_mesh
    from .compatible_export import model_volume

    inputs = pipeline['input']
    source = inputs.get('support_field_source')
    if inputs.get('field_mode') != 'tetrahedral' or not source:
        raise ValueError('Separated posts require a tetrahedral model with its matching saved support field.')
    if not np.isfinite([clearance, overhang_angle, minimum_support_height]).all() or (
            clearance <= 0 or not 0 < overhang_angle < 90 or minimum_support_height < 0):
        raise ValueError('Invalid separated-post clearance, overhang angle or minimum height.')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a fresh separated-post output: {output}')
    configure_geometry_threads(1)
    scalar = inputs.get('tetrahedral_scalar', 'scalar')
    native = TetrahedralVertexField(inputs['tetrahedral_volume'], scalar)
    mesh = load_closed_mesh(inputs['mesh'])
    model_volume(inputs['mesh'], mesh, inputs['tetrahedral_volume'], _loaded_volume=native.volume)
    if mesh.bounds[2] < -1e-6:
        raise ValueError('Separated posts require a zero-height plate and a model above it.')
    data, values, _ = load_support_field(source, inputs['tetrahedral_volume'], scalar)
    field = GridField(data, values)
    triangles = np.asarray(mesh.points, float)[mesh.regular_faces]
    _, region, center = _footprint(mesh, native, triangles, overhang_angle, minimum_support_height+clearance)
    tips, considered = _place(mesh, field, triangles, region, center, clearance, minimum_support_height)
    count = len(tips)
    if count > maximum_contacts:
        raise ValueError(f'{count} separated posts exceed maximum_contacts={maximum_contacts}.')
    neck, body, bed = (tips.copy() for _ in range(3))
    neck[:, 1] -= NECK_LENGTH
    body[:, 1] -= NECK_LENGTH+TAPER_LENGTH
    bed[:, 1] = 0.
    nodes = np.vstack((tips, neck, body, bed))
    i = np.arange(3*count)
    graph = pv.PolyData(nodes, lines=np.column_stack((np.full(len(i), 2), i, i+count)).ravel())
    graph.point_data['node_kind'] = np.repeat([1, 0, 0, 2], count)
    graph.point_data['is_root'] = np.asarray(graph['node_kind'] == 2, dtype=np.uint8)
    report = dict(routing_mode=MODE, experimental_overhang_only=True, clearance_mm=clearance,
                  support_field_source=source, seed_count=count, rooted_contacts=count, unrooted_contacts=0,
                  platform_roots=count, stalled_fronts=0, considered_posts=considered,
                  center_xz_mm=center.tolist(), tip_radius_mm=TIP_RADIUS, neck_length_mm=NECK_LENGTH,
                  minimum_spacing_mm=float(pdist(tips[:, [0, 2]]).min()), body_radius_mm=BODY_RADIUS,
                  taper_length_mm=TAPER_LENGTH, wall_width_mm=.5,
                  limitations=['One selected overhang only. Plate visibility and scalar ordering are sampled.'])
    output.mkdir(parents=True, exist_ok=False)
    graph.save(output/'support_centerlines.vtp')
    pv.PolyData(tips).save(output/'contacts.vtp')
    save(output/'module_report.json', report)
    save(output/'summary.json', dict(mesh=str(Path(inputs['mesh']).resolve()), input=inputs,
         support_field_source=source, arguments={'plate': 0.}, skeleton_experiment=report))
    print(f'Separated-post trial: {count} posts on one overhang at {center.tolist()}.', flush=True)


class SeparatedPostTree(TaperedTree):
    """Vertical cylinders and a smooth taper, each clipped to its axial interval."""

    def primitive(self, points, ids, t):
        delta = points - self.a[ids]
        depth = self.b[ids, 1]-points[:, 1]
        blend = np.clip((depth-NECK_LENGTH)/TAPER_LENGTH, 0., 1.)
        radius = TIP_RADIUS + (BODY_RADIUS-TIP_RADIUS)*(blend*blend*(3.-2.*blend))
        radius[self.platform_base[ids]] = BODY_RADIUS
        radial = np.linalg.norm(delta[:, [0, 2]], axis=1) - radius
        return np.maximum.reduce((radial, -delta[:, 1], points[:, 1]-self.b[ids, 1]))


def shape_for(graph, report, mesh, gap, base_height):
    count = report['seed_count']
    points = np.asarray(graph.points)
    if points.shape != (4*count, 3) or not np.isfinite(base_height) or not 0 < base_height < points[2*count:3*count, 1].min():
        raise ValueError('Invalid separated-post geometry or planar base height.')
    bed = points[-count:]
    rows = np.column_stack((bed, points[:count], np.full(count, BODY_RADIUS),
                            np.full(count, TIP_RADIUS), np.ones(count)))
    bases = np.column_stack((bed, bed+[0., base_height, 0.], np.full((count, 2), BODY_RADIUS), np.full(count, 3)))
    return SeparatedPostTree(np.vstack((rows, bases)), mesh, (0, 1, 0), 0., gap, gap,
                             terminal_cap='flat_disc', prune_pairs=True, distance_backend='certified')

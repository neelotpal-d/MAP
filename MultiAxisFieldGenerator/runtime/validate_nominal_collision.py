"""Dense part-point inclusion checks against analytical, nominal tool volumes.

No per-pose tool mesh, SDF gate, tool-surface sampling, top-k, or training occurs.
All default part samples become both obstacle points and tool-pose roots. A
point q collides at p when q is inside the nominal cone/cylinders and f(q)<f(p).
This checks point coverage, not continuous triangle/solid intersections.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from fields import load_field

HERE = Path(__file__).resolve().parent
COMPONENTS = ("cone", "inner_cylinder", "outer_cylinder")


def validate_tool(tool):
    if tool.get("units") != "mm":
        raise ValueError("Tool dimensions must be in mm")
    for name in COMPONENTS:
        shape = tool[name]
        if not np.isfinite([shape["z_min"], shape["z_max"]]).all() or shape["z_min"] < 0 or shape["z_max"] <= shape["z_min"]:
            raise ValueError(f"Invalid axial interval for {name}")
        radius = shape["radius_max"] if name == "cone" else shape["radius"]
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"Invalid radius for {name}")
    if not 0 < tool["cone"]["half_angle_degrees"] < 90:
        raise ValueError("Cone half-angle must be between 0 and 90 degrees")


def inclusion_masks(axial, radial_squared, tool, tolerance_mm=0.):
    """Radial interior with optional side/end-cap numerical tolerance.

    Cone tolerance is measured perpendicular to its sloped side. Axial/radial
    tests use the same tolerance at the tip and uppermost cap. Internal cap
    planes remain inclusive to avoid gaps at joins. This conservatively counts
    touching at the outer-cylinder shoulder; it is not a penetration-depth test.
    """
    masks = []
    max_height = max(tool[name]["z_max"] for name in COMPONENTS)
    for name in COMPONENTS:
        shape = tool[name]
        # Do not open gaps at internal joins by eroding each component cap.
        lower = shape["z_min"] + (tolerance_mm if shape["z_min"] == 0 else 0.)
        upper = shape["z_max"] - (tolerance_mm if shape["z_max"] == max_height else 0.)
        within_height = (axial >= lower) & (axial <= upper)
        radius = shape.get("radius", shape.get("radius_max")) - tolerance_mm
        mask = within_height & (radial_squared < max(radius, 0.)**2)
        if name == "cone":
            angle = math.radians(shape["half_angle_degrees"])
            limit = (axial*math.sin(angle)-tolerance_mm)/math.cos(angle)
            mask = mask & (limit > 0) & (radial_squared < limit.clamp_min(0).square())
        masks.append(mask)
    return masks


@torch.no_grad()
def check_pairs(roots, root_values, directions, obstacles, obstacle_values, tool, *,
                root_batch=64, point_batch=8192, scalar_tolerance=0., geometry_tolerance_mm=1e-4,
                contact_radius_mm=0., min_gradient_norm=1e-8, progress=None):
    """Vectorized CPU/CUDA checks with memory O(root_batch*point_batch).

    Scalar-sort both clouds so each root block only visits obstacle points
    below its largest field value. Return union counts, overlapping component
    counts and one worst-field-drop witness per root. Output order matches the
    input roots even though checking order is scalar-sorted.
    """
    if root_batch < 1 or point_batch < 1:
        raise ValueError("Batch sizes must be positive")
    if min(scalar_tolerance, geometry_tolerance_mm, contact_radius_mm, min_gradient_norm) < 0:
        raise ValueError("Tolerances cannot be negative")
    if not np.isfinite([scalar_tolerance, geometry_tolerance_mm, contact_radius_mm, min_gradient_norm]).all():
        raise ValueError("Tolerances must be finite")
    validate_tool(tool)
    if geometry_tolerance_mm >= min(tool[name].get('radius', tool[name].get('radius_max')) for name in COMPONENTS):
        raise ValueError("Geometry tolerance must be smaller than every tool radius")
    nr, nq = len(roots), len(obstacles)
    if nr == 0 or nq == 0:
        raise ValueError("Both point clouds must be nonempty")
    for array in [roots, root_values, directions, obstacles, obstacle_values]:
        if not torch.isfinite(array).all():
            raise ValueError("Point, direction and scalar tensors must be finite")
    if roots.shape != (nr, 3) or directions.shape != (nr, 3) or obstacles.shape != (nq, 3) or root_values.shape != (nr,) or obstacle_values.shape != (nq,):
        raise ValueError("Expected Nx3 points/directions and one scalar per point")
    norms = directions.norm(dim=1)
    valid = norms > min_gradient_norm
    unit = directions/norms.clamp_min(max(min_gradient_norm, 1e-15))[:, None]
    q_order = obstacle_values.argsort()
    sorted_values, q = obstacle_values[q_order], obstacles[q_order]
    p_order = root_values.argsort()
    counts = torch.zeros(nr, dtype=torch.long, device=roots.device)
    component_counts = torch.zeros(nr, 3, dtype=torch.long, device=roots.device)
    worst_drop = torch.zeros(nr, dtype=roots.dtype, device=roots.device)
    witness = torch.full((nr,), -1, dtype=torch.long, device=roots.device)
    evaluated_pairs = 0
    total_possible = nr*nq
    for start in range(0, nr, root_batch):
        ids = p_order[start:start+root_batch]
        p, n, fp = roots[ids], unit[ids], root_values[ids]
        prefix = int(torch.searchsorted(sorted_values, fp.max()-scalar_tolerance, right=False))
        local_count = torch.zeros(len(ids), dtype=torch.long, device=roots.device)
        local_components = torch.zeros(len(ids), 3, dtype=torch.long, device=roots.device)
        local_drop = torch.zeros(len(ids), dtype=roots.dtype, device=roots.device)
        local_witness = torch.full((len(ids),), -1, dtype=torch.long, device=roots.device)
        for qstart in range(0, prefix, point_batch):
            block = q[qstart:min(prefix, qstart+point_batch)]
            dx = block[None, :, 0]-p[:, None, 0]
            dy = block[None, :, 1]-p[:, None, 1]
            dz = block[None, :, 2]-p[:, None, 2]
            axial = dx*n[:, None, 0]+dy*n[:, None, 1]+dz*n[:, None, 2]
            distance_squared = dx.square()+dy.square()+dz.square()
            radial_squared = (distance_squared-axial.square()).clamp_min(0)
            drop = fp[:, None]-sorted_values[None, qstart:qstart+len(block)]
            eligible = (drop > scalar_tolerance) & valid[ids, None] & (distance_squared > contact_radius_mm**2)
            union = torch.zeros_like(eligible)
            for component, mask in enumerate(inclusion_masks(axial, radial_squared, tool, geometry_tolerance_mm)):
                hits = eligible & mask
                local_components[:, component] += hits.sum(1)
                union |= hits
            local_count += union.sum(1)
            tile_drop, tile_index = torch.where(union, drop, 0.).max(1)
            better = tile_drop > local_drop
            local_witness[better] = q_order[qstart+tile_index[better]]
            local_drop = torch.maximum(local_drop, tile_drop)
            evaluated_pairs += len(ids)*len(block)
        counts[ids], component_counts[ids] = local_count, local_components
        worst_drop[ids], witness[ids] = local_drop, local_witness
        if progress:
            progress(min(start+root_batch, nr), nr)
    return dict(counts=counts, component_counts=component_counts, max_field_drop=worst_drop,
                witness_indices=witness, valid_roots=valid, evaluated_pairs=evaluated_pairs,
                total_possible_pairs=total_possible)


def sample_part(mesh_path, volume_path, volume_random, surface_random, seed):
    """All tet vertices/centroids and surface face centroids, plus random fill.

    Volume random points are volume-weighted tetrahedron samples, exactly inside
    the supplied tetrahedral mesh. Surface random points are area-weighted.
    No neural SDF is used for part membership. No deterministic seed is dropped.
    """
    import igl
    import pyvista as pv
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    volume = pv.read(str(volume_path))
    cells = volume.cells_dict
    if set(cells) != {int(pv.CellType.TETRA)}:
        raise ValueError("Expected a pure linear tetrahedral volume mesh")
    tetra = np.asarray(volume.points)[cells[int(pv.CellType.TETRA)]]
    edge = tetra[:, 1:]-tetra[:, :1]
    volumes = np.abs(np.linalg.det(edge))/6
    valid = volumes > 1e-12
    tetra, volumes = tetra[valid], volumes[valid]
    if len(tetra) == 0:
        raise ValueError("Volume mesh has no nondegenerate tetrahedra")
    for actual, expected in [(volume.points.min(0), vertices.min(0)), (volume.points.max(0), vertices.max(0))]:
        if not np.allclose(actual, expected, atol=1e-3, rtol=1e-5):
            raise ValueError("Surface/volume bounds differ; check that meshes share coordinates")
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
    triangles, area = triangles[area > 1e-12], area[area > 1e-12]
    rng = np.random.default_rng(seed)
    points = [np.asarray(volume.points), tetra.mean(1), vertices, triangles.mean(1)]
    sizes = dict(tet_vertices=len(volume.points), tet_centroids=len(tetra),
                 surface_vertices=len(vertices), surface_centroids=len(triangles),
                 volume_random=volume_random, surface_random=surface_random)
    if volume_random:
        ids = rng.choice(len(tetra), volume_random, p=volumes/volumes.sum())
        bary = rng.exponential(size=(volume_random, 4))
        bary /= bary.sum(1, keepdims=True)
        points.append((tetra[ids]*bary[:, :, None]).sum(1))
    if surface_random:
        ids = rng.choice(len(triangles), surface_random, p=area/area.sum())
        uv = rng.random((surface_random, 2))
        root = np.sqrt(uv[:, :1])
        bary = np.column_stack((1-root[:, 0], root[:, 0]*(1-uv[:, 1]), root[:, 0]*uv[:, 1]))
        points.append((triangles[ids]*bary[:, :, None]).sum(1))
    # Exact coordinate deduplication only; keep float64 geometry for optional
    # double-precision inclusion checks.
    cloud = np.unique(np.concatenate(points), axis=0)
    return cloud, dict(groups_before_dedup=sizes, unique_points=len(cloud),
                       zero_volume_tetrahedra_discarded=int((~valid).sum()),
                       max_tet_edge_mm=float(max(np.linalg.norm(tetra[:, a]-tetra[:, b], axis=1).max()
                                                 for a in range(4) for b in range(a))),
                       deterministic_coverage="all supplied tet vertices, tet centroids, surface vertices and face centroids")


def field_samples(field, points_mm, center, scale, root_ids, device, chunk_size):
    """Scalar-only obstacle queries; first input derivatives only at pose roots."""
    from collision import scalar_values
    with torch.no_grad():
        values = scalar_values(field, torch.tensor((points_mm-center)/scale, dtype=torch.float32, device=device),
                               chunk_size).cpu().numpy()
    gradients = []
    for batch in np.array_split(root_ids, max(1, math.ceil(len(root_ids)/chunk_size))):
        x = torch.tensor((points_mm[batch]-center)/scale, dtype=torch.float32, device=device, requires_grad=True)
        out = field(x, compute_grads=False)["scalars"]
        gradients.append(torch.autograd.grad(out.sum(), x, create_graph=False)[0].detach().cpu().numpy())
    return values, np.concatenate(gradients)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Default: run/latest.pt")
    p.add_argument("--tool-config", type=Path, default=HERE.parent/'tools'/'abb_nominal.json')
    p.add_argument("--mesh", type=Path)
    p.add_argument("--volume-mesh", type=Path)
    p.add_argument("--output", type=Path, help="Default: run/nominal_validation; refuses to overwrite a report")
    p.add_argument("--volume-samples", type=int, default=25000, help="Additional random interior samples")
    p.add_argument("--surface-samples", type=int, default=25000, help="Additional random surface samples")
    p.add_argument("--max-roots", type=int, help="Optional randomized root subset; default checks EVERY part sample as a pose")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root-batch", type=int, default=64)
    p.add_argument("--point-batch", type=int, default=8192)
    p.add_argument("--field-batch", type=int, default=4096)
    p.add_argument("--geometry-dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--geometry-tolerance-mm", type=float, default=1e-4)
    p.add_argument("--contact-radius-mm", type=float, default=0., help="Optional excluded ball at the nozzle; zero excludes only identical points")
    p.add_argument("--scalar-relative-tolerance", type=float, default=1e-6)
    p.add_argument("--scalar-absolute-tolerance", type=float, default=1e-8)
    p.add_argument("--initial", action="store_true", help="Validate the run's initializer instead of latest.pt")
    return p


def main():
    args = parser().parse_args()
    if min(args.volume_samples, args.surface_samples, args.geometry_tolerance_mm, args.contact_radius_mm,
           args.scalar_relative_tolerance, args.scalar_absolute_tolerance) < 0:
        raise ValueError("Sample counts and tolerances cannot be negative")
    if min(args.root_batch, args.point_batch, args.field_batch) < 1 or (args.max_roots is not None and args.max_roots < 1):
        raise ValueError("Batch sizes and any root cap must be positive")
    if args.initial and args.checkpoint:
        raise ValueError("Choose --initial or --checkpoint, not both")
    args.run = args.run.resolve()
    config = json.loads((args.run/"config.json").read_text())
    tool = json.loads(args.tool_config.read_text())
    validate_tool(tool)
    output = args.output or args.run/("nominal_validation_initial" if args.initial else "nominal_validation")
    if (output/"report.json").exists():
        raise FileExistsError(f"Report exists: {output}; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    from visualize_fertility import local_input
    mesh = args.mesh or local_input(config["mesh"], ["fertility.obj"], "fertility.obj")
    volume = args.volume_mesh or Path(config.get('volume_mesh') or mesh.with_suffix('.ele'))
    checkpoint = args.checkpoint or args.run/"latest.pt"
    if args.initial:
        checkpoint = local_input(config["start_checkpoint"],
                                ["fertility_initializer.pt", "parametersTest_batched_fertility_10_128_7_7.pt"],
                                "fertility_initializer.pt")
    device = torch.device(args.device)
    torch.set_num_threads(int(os.environ.get('MULTIAXIS_NATIVE_THREADS', 0)) or min(8, os.cpu_count() or 1))
    started = time.perf_counter()
    print("Sampling supplied tetrahedral volume and surface meshes...", flush=True)
    if volume.exists():
        points, sampling = sample_part(mesh, volume, args.volume_samples, args.surface_samples, args.seed)
    else:
        from root_sampling import SurfaceRootDomain
        import igl
        vertices, faces = igl.read_triangle_mesh(str(mesh))
        center = np.asarray(config['normalization']['center_mm'])
        scale = config['normalization']['scale_mm']
        domain = SurfaceRootDomain(mesh, center, scale, 'cpu')
        random = domain.sample(2*args.volume_samples, torch.Generator().manual_seed(args.seed))[:args.volume_samples].numpy()*scale+center
        rng = np.random.default_rng(args.seed+2)
        ids = rng.choice(len(domain.triangles), args.surface_samples, p=domain.probabilities)
        uv = rng.random((len(ids), 2)); a = np.sqrt(uv[:, 0])
        bary = np.column_stack((1-a, a*(1-uv[:, 1]), a*uv[:, 1]))
        surface = (domain.triangles[ids]*bary[:, :, None]).sum(1)
        points = np.unique(np.concatenate((vertices, domain.triangles.mean(1), random, surface)), axis=0)
        sampling = dict(backend='surface_rejection', unique_points=len(points), seed=args.seed,
                        volume_random=args.volume_samples, surface_random=args.surface_samples,
                        deterministic_coverage='surface vertices and triangle centroids; no tetrahedral seeds')
    root_ids = np.arange(len(points))
    if args.max_roots is not None and args.max_roots < len(root_ids):
        root_ids = np.random.default_rng(args.seed+1).choice(root_ids, args.max_roots, replace=False)
    field, state = load_field(checkpoint, device, **config.get('field_spec', {}))
    field.requires_grad_(False).eval()
    center = np.array(config["normalization"]["center_mm"])
    scale = config["normalization"]["scale_mm"]
    if "normalization" in state:
        normal = state["normalization"]
        if not np.allclose(normal["center_mm"], center) or not np.isclose(normal["scale_mm"], scale):
            raise ValueError("Checkpoint normalization differs from the run")
    print(f"Field queries: {len(points):,} part points; normals at {len(root_ids):,} tool roots...", flush=True)
    values, gradients = field_samples(field, points, center, scale, root_ids, device, args.field_batch)
    scalar_epsilon = max(args.scalar_absolute_tolerance, args.scalar_relative_tolerance*float(np.ptp(values)))
    dtype = getattr(torch, args.geometry_dtype)
    q = torch.tensor(points-center, dtype=dtype, device=device)
    p = q[torch.tensor(root_ids, device=device)]
    f = torch.tensor(values, dtype=dtype, device=device)
    n = torch.tensor(gradients, dtype=dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    pair_start = time.perf_counter()
    last_percent = [-1]

    def progress(done, total):
        percent = int(20*done/total)
        if percent != last_percent[0]:
            print(f"Analytical checks: {done:,}/{total:,} roots ({100*done/total:.1f}%)", flush=True)
            last_percent[0] = percent

    result = check_pairs(p, f[torch.tensor(root_ids, device=device)], n, q, f, tool,
                         root_batch=args.root_batch, point_batch=args.point_batch,
                         scalar_tolerance=scalar_epsilon, geometry_tolerance_mm=args.geometry_tolerance_mm,
                         contact_radius_mm=args.contact_radius_mm, progress=progress)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pair_seconds = time.perf_counter()-pair_start
    arrays = {key: value.cpu().numpy() for key, value in result.items() if torch.is_tensor(value)}
    bad = arrays["counts"] > 0
    valid = arrays["valid_roots"]
    report = dict(tool=tool, checkpoint=str(checkpoint.resolve()),
                  seed=args.seed, mesh=str(mesh.resolve()), volume_mesh=str(volume.resolve()),
                  normalization=config['normalization'],
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  sampling=sampling, roots_checked=len(root_ids), all_sampled_points_used_as_roots=len(root_ids) == len(points),
                  valid_roots=int(valid.sum()), invalid_gradient_roots=int((~valid).sum()),
                  colliding_roots=int(bad.sum()), collision_root_fraction=float(bad.sum()/max(1, valid.sum())),
                  colliding_pairs=int(arrays["counts"].sum()),
                  component_pair_counts=dict(zip(COMPONENTS, arrays["component_counts"].sum(0).tolist())),
                  max_field_drop=float(arrays["max_field_drop"].max()), scalar_tolerance=scalar_epsilon,
                  geometry_tolerance_mm=args.geometry_tolerance_mm, contact_radius_mm=args.contact_radius_mm,
                  evaluated_pairs=result["evaluated_pairs"], total_possible_pairs=result["total_possible_pairs"],
                  device=str(device), geometry_dtype=args.geometry_dtype,
                  root_batch=args.root_batch, point_batch=args.point_batch,
                  analytical_seconds=pair_seconds, total_seconds=time.perf_counter()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  limitations="Point-sampled validation of an assumed nominal enclosure, not a continuous intersection or fabrication certificate; invalid-gradient roots are unverified.")
    np.savez_compressed(output/"points.npz", points_mm=points, scalars=values)
    np.savez_compressed(output/"roots.npz", root_point_indices=root_ids, gradients=gradients, **arrays)
    np.savez_compressed(output/"witnesses.npz", roots_mm=points[root_ids[bad]],
                        obstacles_mm=points[arrays["witness_indices"][bad]],
                        field_drops=arrays["max_field_drop"][bad])
    (output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(event='nominal_collision_result', report=str((output/'report.json').resolve()))), flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report, sampled clouds, per-root counts and collision witnesses: {output}", flush=True)


if __name__ == "__main__":
    main()

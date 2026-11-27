"""Run independent envelope construction and experimental field transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtimeEnv import configure_openmp_runtime
configure_openmp_runtime()

import numpy as np
import pyvista as pv
from SupportGenerationCurved.config import load_config, argument_defaults, resolve_config_path
from SupportGenerationCurved.envelope import build_envelope
from SupportGenerationCurved.transfer import blended_taylor, reconstruct, validate_samples
from SupportGenerationCurved.trajectories import trace_to_plate, overhang_mask
from SupportGenerationCurved.layers import export_layers, extend_schedule
from SupportGenerationCurved.output import prepare_output
from SupportGenerationCurved.boundary import sample_boundary
from SupportGenerationCurved.support_region import height_mask,resolve_minimum_height


class AnalyticCurvedField:
    def __init__(self, curvature):
        self.curvature = curvature

    def evaluate_scalars_and_gradients(self, points):
        points = np.asarray(points, dtype=float)
        gradients = np.zeros_like(points)
        gradients[:, 1] = 1
        gradients[:, (0, 2)] = 2 * self.curvature * points[:, (0, 2)]
        return points[:, 1] + self.curvature * (points[:, 0] ** 2 + points[:, 2] ** 2), gradients


def random_surface_samples(mesh, count, seed):
    triangles = np.asarray(mesh.points, dtype=float)[np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(triangles), size=count, p=area / area.sum())
    u, v = rng.random((2, count))
    root = np.sqrt(u)
    barycentric = np.column_stack((1 - root, root * (1 - v), root * v))
    return np.einsum("ni,nij->nj", barycentric, triangles[ids]), cross[ids] / area[ids, None]


def run(args):
    # Windows spawned layer workers import this module. They only need geometry,
    # so keep the neural model stack out of worker startup and memory.
    from SupportGenerationVoxel.config import load_config as load_legacy_config
    from SupportGenerationVoxel.model_field import AnalyticYField, TetrahedralVertexField, TrustedModelField, load_closed_mesh
    from SupportGenerationVoxel.contacts import derive_layer_values, load_layer_values
    if args.clean_output:
        target = args.output.resolve()
        for source in (args.samples, args.layer_values, args.model_config, args.mesh, args.config, args.tree_source,args.model_volume,args.model_layers_source,args.skeleton_source,args.tetrahedral_volume):
            if source and (source.resolve() == target or target in source.resolve().parents):
                raise ValueError(f"Input must be outside the directory being cleaned: {source}")
    output = prepare_output(args.output, clean=args.clean_output)
    started = time.perf_counter()
    timings = {}
    config = load_config(args.config) if args.config else (load_legacy_config(args.model_config) if args.model_config else None)
    mesh_path = args.mesh or (Path(config["input"]["mesh"]) if config else ROOT / "inputShapes/bunny.obj")
    mesh = load_closed_mesh(mesh_path)
    if mesh.n_open_edges:
        raise ValueError("Envelope subtraction needs a closed, manifold model surface.")
    mode = args.field_mode or (config["input"].get("field_mode", "checkpoint") if config else "analytic_y")
    if mode == "tetrahedral" and args.support_field_mode != "relaxed":
        # A vertex FEM field only exists inside the model.  The legacy native
        # blend asks the source field at exterior support points, so the
        # reconstructed envelope field is the valid support scalar here.
        args.support_field_mode = "relaxed"
    timings["config_and_mesh_load"] = time.perf_counter() - started
    stage = time.perf_counter()
    if args.samples:
        with np.load(args.samples) as data:
            source_points, source_values, source_gradients = validate_samples(data["points"], data["scalars"], data["gradients"])
        evaluator = None
        mode = "sample_fallback"
    else:
        if mode == "analytic_y":
            evaluator = AnalyticYField()
        elif mode == "analytic_curved":
            evaluator = AnalyticCurvedField(args.curvature)
        elif mode == "checkpoint" and config:
            normal_path = config["input"].get("normalization_mesh") or mesh_path
            evaluator = TrustedModelField(config["input"]["checkpoint"], load_closed_mesh(normal_path).points, config["network"])
        elif mode == "tetrahedral":
            volume_path = args.tetrahedral_volume or (config["input"].get("tetrahedral_volume") if config else None)
            if not volume_path:
                raise ValueError("Tetrahedral field mode requires input.tetrahedral_volume.")
            scalar_name = args.tetrahedral_scalar or (config["input"].get("tetrahedral_scalar", "scalar") if config else "scalar")
            evaluator = TetrahedralVertexField(volume_path, scalar_name)
        else:
            raise ValueError("Checkpoint mode requires --model-config with checkpoint and network settings.")
        source_points, normals = random_surface_samples(mesh, args.sample_count, args.seed)
        if args.interior_depth > 0:
            rng = np.random.default_rng(args.seed + 1)
            interior = source_points - rng.uniform(0, args.interior_depth, (len(source_points), 1)) * normals
            distances = pv.PolyData(interior).compute_implicit_distance(mesh)["implicit_distance"]
            source_points = np.vstack((source_points, interior[np.asarray(distances) < 0]))
        source_values, source_gradients = evaluator.evaluate_scalars_and_gradients(source_points)
        source_points, source_values, source_gradients = validate_samples(source_points, source_values, source_gradients)
    timings["field_load_and_sampling"] = time.perf_counter() - stage
    stage = time.perf_counter()
    np.savez_compressed(output / "trusted_samples.npz", points=source_points, scalars=source_values, gradients=source_gradients)
    timings["sample_export"] = time.perf_counter() - stage
    print(f"Trusted samples: {len(source_points):,}; source={mode}", flush=True)

    direction = args.build_direction or (config.get("domain", {}).get("build_direction", [0, 1, 0]) if config else [0, 1, 0])
    stage = time.perf_counter()
    plate = args.plate if args.plate is not None else (config.get("domain", {}).get("build_plate_projection") if config else None)
    direction = np.asarray(direction, dtype=float)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-12:
        raise ValueError("Build direction must be finite and nonzero.")
    direction /= np.linalg.norm(direction)
    plate = float(np.min(np.asarray(mesh.points) @ direction)) if plate is None else float(plate)
    args.minimum_support_height=resolve_minimum_height(args.minimum_support_height,mesh.points,direction,plate)
    trajectory_points = None
    trajectory_report = None
    if args.envelope_mode == "gradient":
        surface_points, surface_normals = random_surface_samples(mesh, args.sample_count, args.seed)
        if evaluator is not None:
            _, surface_gradients = evaluator.evaluate_scalars_and_gradients(surface_points)
        else:
            _, surface_gradients = blended_taylor(surface_points, source_points, source_values, source_gradients)
        selected = overhang_mask(surface_normals, surface_gradients, args.support_threshold_angle) if args.trajectory_region == "overhang" else np.ones(len(surface_points), dtype=bool)
        selected &= height_mask(surface_points,direction,plate,args.minimum_support_height)
        if not np.any(selected):
            raise ValueError("No sampled overhangs found. Use --trajectory-region all or --envelope-mode convex.")
        step = args.trajectory_step or min(args.spacing, args.padding * 0.5)
        if step > args.padding:
            raise ValueError("Trajectory step must not exceed padding; adjacent sample balls need overlap.")
        timings["guide_sample_preparation"] = time.perf_counter() - stage
        np.savez_compressed(output / "trajectory_inputs.npz", points=surface_points[selected], gradients=surface_gradients[selected], direction=direction, plate=plate, step=step, turn_degrees=args.trajectory_turn)
        stage = time.perf_counter()
        trajectories, trajectory_report = trace_to_plate(surface_points[selected], surface_gradients[selected], direction, plate, step, args.trajectory_turn)
        trajectory_report.update({"region": args.trajectory_region, "surface_samples": len(surface_points),
                                  "selected_samples": int(np.count_nonzero(selected)),
                                  "support_threshold_angle": args.support_threshold_angle,
                                  "minimum_support_height_mm": args.minimum_support_height})
        trajectories.save(output / "envelope_trajectories.vtp")
        trajectory_points = np.asarray(trajectories.points)
        timings["trajectory_trace_and_export"] = time.perf_counter() - stage
        print(f"Gradient guides: {trajectory_report['trajectory_count']:,} trajectories to plate", flush=True)
    stage = time.perf_counter()
    if args.skeleton_source:
        from SupportGenerationCurved.skeleton_geometry import load_skeleton
        skeleton,_=load_skeleton(args.skeleton_source,allow_stalled=args.allow_stalled_skeleton,skip_unrooted=args.skip_unrooted_contacts)
        folder=args.skeleton_source if args.skeleton_source.is_dir() else args.skeleton_source.parent
        source_summary=json.loads((folder/'summary.json').read_text())
        if Path(source_summary['mesh']).resolve()!=Path(mesh_path).resolve():
            raise ValueError('Saved skeleton belongs to a different model mesh.')
        trajectory_points=np.vstack((trajectory_points,skeleton.points)) if trajectory_points is not None else skeleton.points
    elif args.tree_source and mode in ('checkpoint','tetrahedral'):
        tree_points=pv.read(args.tree_source/'tree.vtp').points
        bridge_path=args.tree_source/'model_bridges.json'
        if bridge_path.exists():
            bridges=json.loads(bridge_path.read_text())['bridges']
            if bridges:
                bridge_points=[np.linspace(r['lower'],r['upper'],max(2,int(np.ceil(r['length_mm']/args.spacing))+1)) for r in bridges]
                tree_points=np.vstack((tree_points,*bridge_points))
        trajectory_points=np.vstack((trajectory_points,tree_points)) if trajectory_points is not None else tree_points
    overlap=np.sqrt(3.)*args.spacing if args.support_field_mode=='relaxed' else 0.
    domain, envelope_report = build_envelope(mesh, args.spacing, args.padding, direction, plate, args.maximum_points, trajectory_points,args.geometry_threads,
        model_overlap=overlap)
    envelope_report["trajectories"] = trajectory_report
    timings["envelope"] = time.perf_counter() - stage
    print(f"Envelope: {domain.n_points:,} points, {domain.n_cells:,} tetrahedra ({timings['envelope']:.2f}s)", flush=True)
    stage = time.perf_counter()
    anchors = np.asarray(domain["model_anchor"], dtype=bool)
    anchor_points = np.asarray(domain.points, dtype=float)[anchors]
    boundary_mode = "projected_taylor" if mode == "tetrahedral" and args.boundary_mode == "auto" else args.boundary_mode
    anchor_values, gradients, boundary_error, boundary_report = sample_boundary(
        mesh, anchor_points, evaluator, (source_points, source_values, source_gradients), boundary_mode)
    print(f"Boundary sampling: {boundary_report['mode']}; max model distance {boundary_error.max():.4g} mm", flush=True)
    timings["anchor_preparation"] = time.perf_counter() - stage
    stage = time.perf_counter()
    domain.save(output / "envelope_mesh.vtu")
    np.savez_compressed(output / "transfer_inputs.npz", anchor_scalars=anchor_values, anchor_gradients=gradients)
    timings["module_fixture_export"] = time.perf_counter() - stage
    stage = time.perf_counter()
    baseline, _ = blended_taylor(domain.points, source_points, source_values, source_gradients)
    scalar, vector, actual_gradient, transfer_report = reconstruct(domain, anchor_values, gradients, baseline,
        upward_length=args.field_upward_length,build_direction=direction,platform_upward=args.field_platform_upward,
        upward_ramp=args.field_upward_ramp)
    timings["transfer"] = time.perf_counter() - stage
    print(f"Field transfer finished ({timings['transfer']:.2f}s)", flush=True)
    stage = time.perf_counter()
    domain.point_data["scalar"] = scalar
    domain.point_data["taylor_baseline"] = baseline
    domain.point_data["extended_gradient"] = vector
    domain.cell_data["reconstructed_gradient"] = actual_gradient
    domain.point_data["scalar_minus_baseline"] = scalar - baseline
    domain.save(output / "envelope_field.vtu")
    domain.extract_surface(algorithm="dataset_surface").save(output / "envelope_boundary.vtp")
    timings["volume_export"] = time.perf_counter() - stage

    stage = time.perf_counter()
    schedule_path = args.layer_values or (config["input"].get("layer_values") if config else None)
    schedule = load_layer_values(schedule_path) if schedule_path else derive_layer_values(source_values, source_gradients, args.layer_spacing, 0, 10000, 256)
    schedule_extension = None
    if not schedule_path and args.tree_source and mode in ('checkpoint', 'tetrahedral'):
        schedule, schedule_extension = extend_schedule(schedule, [float(np.min(scalar)), float(np.max(scalar))])
        print(f"Shared schedule extended for support coverage: +{schedule_extension['prepended_levels']} lower, "
              f"+{schedule_extension['appended_levels']} upper levels; {len(schedule)} total.", flush=True)
    np.savetxt(output / "main_scalar_values.txt", schedule, fmt="%.17g")
    lower, upper = max(0, args.minimum_layer), min(len(schedule) - 1, args.maximum_layer if args.maximum_layer is not None else len(schedule) - 1)
    if lower > upper:
        raise ValueError("Requested layer range does not intersect the scalar schedule.")
    indices = np.arange(lower, upper + 1) if args.all_layers else np.unique(np.linspace(lower, upper, min(args.preview_layers, max(0, upper - lower + 1))).astype(int))
    timings["schedule"] = time.perf_counter() - stage
    prepared_compatible_model=None
    if args.export_compatible_layers and not args.model_layers_source:
        # Prepare the model before the support preview/filter lifetime. The
        # previous late VTU re-read intermittently failed in native VTK after
        # preview generation. A FEM source already loaded by the evaluator can
        # be copied directly; do not mutate that evaluator's locator dataset.
        from SupportGenerationCurved.compatible_export import model_volume
        stage=time.perf_counter()
        volume_path=args.model_volume or args.tetrahedral_volume
        loaded=(evaluator.volume if mode=='tetrahedral' and volume_path is not None
                and Path(volume_path).resolve()==evaluator.path else None)
        prepared_compatible_model=model_volume(mesh_path,mesh,volume_path,_loaded_volume=loaded)
        timings['compatible_model_preparation']=time.perf_counter()-stage
    tree_report = None
    if args.tree_source:
        if mode not in ('analytic_y','analytic_curved','checkpoint','tetrahedral'):
            raise ValueError('Tree slicing requires a callable model field.')
        from SupportGenerationCurved.tree_geometry import export_geometry
        stage = time.perf_counter()
        tree_report = export_geometry(args.tree_source, mesh, schedule, indices, output,
            args.tree_spacing, args.side_clearance, args.contact_gap, args.tip_radius, args.roof_clearance_mode, args.terminal_cap,
            args.geometry_mode, args.branch_blend, args.clearance_blend, args.rim_blend,
            args.roof_mode, args.roof_terminal, args.roof_radius, args.roof_thickness, args.roof_rounding, args.roof_layer_spacing,
            args.layer_trim, args.geometry_threads, args.layer_workers, args.layer_compression,
            args.branch_sizing, args.branch_diameter, args.branch_diameter_angle, args.platform_flange, args.support_speed,
            args.minimum_support_height,plate,args.support_ceiling,args.tip_alignment,args.tip_length,args.contact_layers_below,
            evaluator if mode!='analytic_y' else None,mode,domain if mode in ('checkpoint','tetrahedral') else None,layer_batch_size=args.layer_batch_size,
            model_distance_backend=args.model_distance_backend,geometry_query_batch_size=args.geometry_query_batch_size,trim_pair_pruning=args.trim_pair_pruning,
            model_tree_mode=args.model_tree_mode,skeleton_source=args.skeleton_source,allow_stalled_skeleton=args.allow_stalled_skeleton,
            support_field_mode=args.support_field_mode,model_root_foot_radius=args.model_root_foot_radius,
            model_root_foot_alignment=args.model_root_foot_alignment,model_root_foot_shape=args.model_root_foot_shape,
            platform_base_radius_factor=args.platform_base_radius_factor,platform_base_height=args.platform_base_height,
            build_direction=direction, vtk_threads_per_worker=args.vtk_threads_per_worker, maximum_trunk_diameter=args.maximum_trunk_diameter,skip_unrooted_contacts=args.skip_unrooted_contacts,planar_platform_bases=args.planar_platform_bases,platform_base_layer_height=args.platform_base_layer_height,
            layer_seed_spacing=args.layer_seed_spacing,adaptive_layer_tolerance=args.adaptive_layer_tolerance,
            trim_boundary_tolerance_mm=args.trim_boundary_tolerance_mm,
            maximum_platform_trunk_diameter=args.maximum_platform_trunk_diameter,
            include_support_gradients=args.include_support_gradients,
            support_layer_spacing=args.support_layer_spacing if args.support_layer_spacing is not None else min(args.layer_spacing,args.maximum_support_layer_height),
            maximum_support_layer_height=args.maximum_support_layer_height,
            support_gradient_probe_spacing=args.support_gradient_probe_spacing,slicing_mode=args.slicing_mode,
            model_layer_spacing=args.layer_spacing,_planning_model_volume=prepared_compatible_model[0] if prepared_compatible_model else None)
        timings["tree_geometry_and_layers"] = time.perf_counter() - stage
        if tree_report.get('online_slicing'):
            schedule=np.loadtxt(output/'main_scalar_values.txt')
            indices=np.asarray(tree_report['online_slicing']['selected_indices'],dtype=int)
    stage = time.perf_counter()
    manifest = (export_layers(domain, schedule, indices, output / "envelope_layers", analytic_y=mode == "analytic_y", contour_backend=args.contour_backend, layer_compression=args.layer_compression)
                if args.export_envelope_layers else [])
    timings["layer_export"] = time.perf_counter() - stage
    compatible_report=None
    if args.export_compatible_layers:
        from SupportGenerationCurved.compatible_export import export_compatible
        stage=time.perf_counter()
        support_records=json.loads((output/'tree_layer_manifest.json').read_text()) if tree_report else []
        compatible_report=export_compatible(mesh_path,mesh,schedule,indices,support_records,output,
            evaluator=evaluator,field_mode=mode,volume_path=args.model_volume or args.tetrahedral_volume,compression=args.layer_compression,
            model_layers_source=args.model_layers_source,model_query_backend=args.model_query_backend,model_query_threads=args.geometry_threads,
            _prepared_model=prepared_compatible_model)
        timings['compatible_model_support_export']=time.perf_counter()-stage
    stage = time.perf_counter()
    validation = {"finite": bool(np.all(np.isfinite(scalar))),
                  "anchor_scalar_error_max": transfer_report["anchor_scalar_error_max"],
                  "cavity_boundary_distance_max": float(boundary_error.max()),
                  "cavity_boundary_distance_rms": float(np.sqrt(np.mean(boundary_error ** 2))),
                  "affine_regression_passed": None}
    if mode in ("analytic_y", "analytic_curved"):
        exact, _ = evaluator.evaluate_scalars_and_gradients(domain.points)
        validation["analytic_scalar_error_max"] = float(np.max(np.abs(scalar - exact)))
        validation["analytic_scalar_error_rms"] = float(np.sqrt(np.mean((scalar - exact) ** 2)))
        validation["taylor_analytic_error_max"] = float(np.max(np.abs(baseline - exact)))
        if mode == "analytic_y":
            validation["affine_regression_passed"] = validation["analytic_scalar_error_max"] < 1e-6
            if not validation["affine_regression_passed"]:
                raise RuntimeError(f"Affine field regression failed: {validation}")
    timings["validation"] = time.perf_counter() - stage
    (output / "layer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"status": "experimental_field_transfer", "mesh": str(Path(mesh_path).resolve()),
              "field_mode": mode, "envelope": envelope_report, "transfer": transfer_report,
              "boundary_sampling": boundary_report,
              "tree_geometry": tree_report,
              "compatible_export":compatible_report,
              "validation": validation, "schedule_source": ('online_current_layers' if tree_report and tree_report.get('online_slicing') else 'provided' if schedule_path else 'derived_approximation'),
              "schedule_extension": schedule_extension,
              "exported_layer_count": len(manifest), "envelope_layer_export_enabled":args.export_envelope_layers,
              "layer_export_mode": "all" if args.all_layers else "preview",
              "timings_seconds": timings, "total_seconds": time.perf_counter() - started,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "limitations": ["Independent cavity mesh approximates the original boundary.",
                  "Natural outer/plate boundary conditions do not guarantee printable layer directions or spacing.",
                  "Envelope layers are computational-domain surfaces, not final support geometry.",
                  "Optional tree geometry reuses a preserved skeleton; new planning and toolpaths are not implemented."]}
    if mode=='analytic_curved' and tree_report:
        report['layer_scalar_sources']={'model':'shared global analytic callable',
            'support':'shared global analytic callable','envelope':'reconstructed transfer field (diagnostic only)'}
        report['limitations']+=['Nonplanar model/support test uses a shared global analytic field, not the reconstructed envelope field.',
            'Saved contact selection and tree routing are retained; support completeness for the new field is not certified.']
    elif mode in ('checkpoint','tetrahedral') and tree_report:
        report['layer_scalar_sources']={'model':'Native neural field; model-grid contours projected to neural isovalues.',
            'support':tree_report.get('transferred_support_field'), 'envelope':'reconstructed exterior FEM field'}
        if mode == 'tetrahedral':
            report['layer_scalar_sources']['model'] = 'Piecewise-linear scalar interpolated from model tetrahedron vertices.'
        report['limitations']+=['Model-rooted supports have a lower and upper removable contact gap.',
            'No-room gaps are audited in the source model_bridges.json; complete contact coverage is not certified.',
            'Layer spacing, scalar monotonicity along every branch and machine collision safety remain downstream checks.']
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Stage timings (seconds): " + json.dumps(timings), flush=True)
    print(f"Pipeline execution: {report['total_seconds']:.2f}s; envelope layers: {len(manifest)}; "
          f"model/support OBJ: {compatible_report['model_count'] if compatible_report else 0}/{compatible_report['support_count'] if compatible_report else 0}", flush=True)
    print(json.dumps(validation, indent=2), flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Dedicated SupportGenerationCurved JSON; CLI flags override its input/settings defaults.")
    parser.add_argument("--clean-output", action="store_true", help="Clear generated artifacts in this output directory before rerunning.")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument('--model-volume',type=Path,help='Matching TetGen .ele/.node pair or tetrahedral VTU; otherwise auto-detect beside model surface.')
    parser.add_argument('--model-layers-source',type=Path,help='Existing model OBJ layers with the identical scalar schedule; combine components per main index.')
    parser.add_argument('--export-compatible-layers',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument("--model-config", type=Path, help="Reuse the voxel pipeline's checkpoint/network input settings only.")
    parser.add_argument("--field-mode", choices=("analytic_y", "analytic_curved", "checkpoint", "tetrahedral"))
    parser.add_argument("--tetrahedral-volume", type=Path, help="All-tetra VTU containing the vertex scalar field (input.tetrahedral_volume in JSON).")
    parser.add_argument("--tetrahedral-scalar", help="Point-data scalar name in --tetrahedral-volume (input.tetrahedral_scalar in JSON).")
    parser.add_argument("--samples", type=Path, help="Fallback NPZ with points, scalars, gradients in world units.")
    parser.add_argument("--boundary-mode", choices=("auto", "direct", "projected_taylor", "sample_taylor"), default="auto", help="auto evaluates cavity nodes directly for callable fields; projected_taylor retains the original boundary transfer as backup.")
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--padding", type=float, default=6.0)
    parser.add_argument("--envelope-mode", choices=("gradient", "convex"), default="gradient")
    parser.add_argument("--trajectory-region", choices=("overhang", "all"), default="overhang")
    parser.add_argument("--support-threshold-angle", type=float, default=45.0)
    parser.add_argument("--minimum-support-height",type=float,help="Evaluate model support regions at/above this height in mm above plate; default is the model's lowest point. Trunks still reach the plate.")
    parser.add_argument("--trajectory-step", type=float)
    parser.add_argument("--trajectory-turn", type=float, default=2.25, help="Degrees turned toward downward build direction per step.")
    parser.add_argument("--build-direction", type=float, nargs=3)
    parser.add_argument('--field-upward-length',type=float,default=0.,help='Vector regularization length in mm; 0 preserves legacy diffusion, smaller positive values pull more strongly toward build-up.')
    parser.add_argument('--field-upward-ramp',type=float,default=0.,help='Gradually enable upward attraction over this distance from boundary anchors, in mm; 0 keeps uniform attraction.')
    parser.add_argument('--support-field-mode',choices=('native_blend','relaxed'),default='native_blend',help='relaxed: slice support directly with reconstructed scalar; native_blend retains the original near-model neural blend.')
    parser.add_argument('--field-platform-upward',action=argparse.BooleanOptionalAction,default=False,help='Fix non-model platform vector nodes to build-up; model boundary anchors take priority.')
    parser.add_argument("--plate", type=float)
    parser.add_argument("--maximum-points", type=int, default=500000)
    parser.add_argument("--sample-count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--interior-depth", type=float, default=0.0)
    parser.add_argument("--curvature", type=float, default=0.002)
    parser.add_argument("--layer-values", type=Path)
    parser.add_argument("--layer-spacing", type=float, default=0.2)
    parser.add_argument('--slicing-mode',choices=('online','sampled'),default='sampled',
        help='online: extract current planning layers and use their mean gradient norms for one shared schedule; sampled: precomputed support sampling.')
    parser.add_argument('--include-support-gradients',action=argparse.BooleanOptionalAction,default=True,
        help='Consider actual support gradients. Online mode shares new model/support levels; sampled mode preserves model levels.')
    parser.add_argument('--support-layer-spacing',type=float,default=None,
        help='Nominal support spacing in mm; default uses model spacing up to the support maximum.')
    parser.add_argument('--maximum-support-layer-height',type=float,default=.5,
        help='Maximum sampled gradient-based support spacing estimate in mm; actual normal/backing checks remain downstream.')
    parser.add_argument('--support-gradient-probe-spacing',type=float,default=2.,
        help='Axial spacing in mm for branch gradient probes; all branches are considered.')
    parser.add_argument("--contour-backend", choices=("auto", "legacy", "linear_grid"), default="auto")
    parser.add_argument("--layer-compression", choices=("zlib", "lz4", "none"), default="zlib")
    parser.add_argument("--preview-layers", type=int, default=5)
    parser.add_argument("--all-layers", action="store_true", help="Export every scheduled layer in the selected range, rather than a preview subset.")
    parser.add_argument('--export-envelope-layers',action=argparse.BooleanOptionalAction,default=True,
                        help='Export diagnostic envelope slices in addition to final model/support slices.')
    parser.add_argument("--minimum-layer", type=int, default=1)
    parser.add_argument("--maximum-layer", type=int)
    parser.add_argument("--tree-source", type=Path, help="Optional preserved skeleton directory for the analytic-y tapered-tip experiment.")
    parser.add_argument('--skeleton-source',type=Path,help='Use an existing fully rooted progressive skeleton directory/VTP for sweep geometry and envelope coverage.')
    parser.add_argument('--planar-platform-bases',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--platform-base-layer-height',type=float,default=.2)
    parser.add_argument('--skip-unrooted-contacts',action='store_true',help='Omit unrooted contacts and their unfinished branches; slice only rooted support components.')
    parser.add_argument('--allow-stalled-skeleton',action='store_true',help='Keep existing stalled branches and continue layer extraction; do not remove them or label their ends as valid contacts.')
    parser.add_argument('--model-root-foot-radius',type=float,default=0.,help='Optional maximum model-root foot radius in mm; 0 keeps saved contact radii. Requires --skeleton-source.')
    parser.add_argument('--model-root-foot-alignment',type=float,default=.95,help='Minimum model normal/gradient alignment across a proposed wider foot.')
    parser.add_argument('--model-root-foot-shape',choices=('conformal','planar'),default='conformal',help='Validate wider feet against the actual curved surface; planar retains the original flat-disk screen.')
    parser.add_argument('--platform-base-radius-factor',type=float,default=0.,help='For imported skeleton platform roots, add a plate-clipped horizontal base disc at this multiple of local trunk radius; 0 disables it.')
    parser.add_argument('--platform-base-height',type=float,default=.6,help='Vertical height in mm of the optional platform base disc primitive.')
    parser.add_argument("--tree-spacing", type=float, default=1.0, help="Support solid preview grid spacing; does not set layer extraction resolution.")
    parser.add_argument('--layer-seed-spacing',type=float,default=1.,help='Independent fallback layer seed spacing; relaxed FEM layers use the transfer mesh directly.')
    parser.add_argument('--adaptive-layer-tolerance',type=float,default=.05,help='Local layer-outline detail target in mm; exact clearance checks remain active.')
    parser.add_argument('--trim-boundary-tolerance-mm',type=float,default=0.,help='Retain verified whole-face boundary deviations within this physical mm budget with warnings; zero is strict.')
    parser.add_argument("--side-clearance", type=float, default=0.4)
    parser.add_argument("--contact-gap", type=float, default=0.2)
    parser.add_argument('--fast-clearance','--fast-mode',dest='fast_clearance',action='store_true',
                        help='Use a uniform 0.3 mm model clearance, overriding side/contact gaps and skipping separate side-wall queries.')
    parser.add_argument("--tip-radius", type=float, default=0.6)
    parser.add_argument("--roof-clearance-mode", choices=("plateau", "legacy_blend"), default="plateau")
    parser.add_argument("--terminal-cap", choices=("flat_disc", "rounded"), default="flat_disc")
    parser.add_argument('--model-tree-mode',choices=('struts','merged'),default='struts',help='Experimental local merging of model-rooted struts; requires uniform gaps and rounded caps.')
    parser.add_argument("--support-ceiling",choices=("none","contact_max"),default="none",help="contact_max clips analytic-Y support to the highest retained contact plane minus contact gap.")
    parser.add_argument("--tip-alignment",choices=("legacy","gradient"),default="legacy")
    parser.add_argument("--contact-layers-below",type=int,default=3,help="Additional preceding schedule layers to mark as contact (metadata only; default: 3).")
    parser.add_argument("--tip-length",type=float,default=3.,help="Minimum length of the reserved gradient-aligned contact approach in mm.")
    parser.add_argument("--geometry-mode", choices=("hard", "smooth"), default="hard")
    parser.add_argument("--branch-blend", type=float, default=0.3)
    parser.add_argument("--clearance-blend", type=float, default=0.1)
    parser.add_argument("--rim-blend", type=float, default=0.1)
    parser.add_argument("--roof-mode", choices=("none", "offset_patch"), default="none")
    parser.add_argument("--roof-terminal", type=int, default=3939)
    parser.add_argument("--roof-radius", type=float, default=3)
    parser.add_argument("--roof-thickness", type=float, default=0.6)
    parser.add_argument("--roof-rounding", type=float, default=0.1)
    parser.add_argument("--roof-layer-spacing", type=float, default=0.1)
    parser.add_argument("--layer-trim", choices=("checked", "surface_scalar", "surface_scalar_bisect"), default="checked")
    parser.add_argument("--geometry-threads", type=int, default=8)
    parser.add_argument('--vtk-threads-per-worker',type=int,default=None,help='Explicit VTK threads per layer process; omitted retains legacy budget splitting.')
    parser.add_argument("--layer-workers", type=int, default=0, help="Layer processes; 0 selects four for >=32 layers, otherwise one.")
    parser.add_argument('--layer-batch-size',type=int,default=1,help='Number of support layers trimmed together per worker; 1 keeps the independent-layer path.')
    parser.add_argument('--model-distance-backend',choices=('exact','certified'),default='exact',help='Model-distance queries for support geometry/field; certified bands keep near-boundary queries exact. Native model slicing remains exact.')
    parser.add_argument('--model-query-backend',choices=('vtk','libigl','open3d'),default='vtk',help='Closest-point queries for native model boundary projection; grid signed distances remain VTK.')
    parser.add_argument('--geometry-query-batch-size',type=int,default=2000,help='Points per vectorized support-geometry query; larger batches use more temporary memory.')
    parser.add_argument('--trim-pair-pruning',action=argparse.BooleanOptionalAction,default=False,help='Reject branch/point pairs outside conservative primitive bounds during layer trimming.')
    parser.add_argument("--branch-sizing", choices=("preserved", "bambu"), default="preserved")
    parser.add_argument("--branch-diameter", type=float, default=5.)
    parser.add_argument('--maximum-trunk-diameter',type=float,default=None,help='Separate cap for load-based trunk thickening on imported skeletons.')
    parser.add_argument('--maximum-platform-trunk-diameter',type=float,default=None,help='Independent diameter cap on platform-rooted branches; omitted inherits the general trunk cap. Contact tips remain separate.')
    parser.add_argument("--branch-diameter-angle", type=float, default=5.)
    parser.add_argument("--platform-flange", choices=("none", "bambu_auto"), default="none")
    parser.add_argument("--support-speed", type=float, default=80.)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", type=Path)
    selected, _ = selector.parse_known_args(argv)
    if selected.config:
        try:
            parser.set_defaults(**argument_defaults(load_config(selected.config)))
        except (ValueError, OSError) as error:
            parser.error(str(error))
    args = parser.parse_args(argv)
    if args.maximum_trunk_diameter is not None:
        if not np.isfinite(args.maximum_trunk_diameter) or args.maximum_trunk_diameter < args.branch_diameter:
            parser.error('Maximum trunk diameter must be finite and at least the minimum branch diameter.')
        if not args.skeleton_source:
            parser.error('--maximum-trunk-diameter requires --skeleton-source.')
    if args.maximum_platform_trunk_diameter is not None:
        if not np.isfinite(args.maximum_platform_trunk_diameter) or args.maximum_platform_trunk_diameter < args.branch_diameter:
            parser.error('Platform trunk diameter cap must be finite and at least the minimum branch diameter.')
        if not args.skeleton_source:
            parser.error('--maximum-platform-trunk-diameter requires --skeleton-source.')
    if args.vtk_threads_per_worker is not None and args.vtk_threads_per_worker < 1:
        parser.error('VTK threads per worker must be positive.')
    if not np.isfinite(args.model_root_foot_radius) or args.model_root_foot_radius<0 or not 0<args.model_root_foot_alignment<1:
        parser.error('Require a nonnegative finite foot radius and alignment in (0,1).')
    if args.model_root_foot_radius>0 and not args.skeleton_source:
        parser.error('--model-root-foot-radius requires --skeleton-source.')
    if (not np.isfinite([args.platform_base_radius_factor,args.platform_base_height]).all()
            or args.platform_base_radius_factor<0 or args.platform_base_height<=0):
        parser.error('Platform base radius factor must be nonnegative and height must be positive.')
    if not np.isfinite(args.platform_base_layer_height) or args.platform_base_layer_height<=0:
        parser.error('Platform base layer height must be finite and positive.')
    if not np.isfinite([args.field_upward_length,args.field_upward_ramp]).all() or min(args.field_upward_length,args.field_upward_ramp)<0:
        parser.error('Field upward length and ramp must be finite and nonnegative.')
    if args.fast_clearance:
        args.side_clearance=args.contact_gap=0.3
    if not np.isfinite(args.tip_length) or args.tip_length<=0:
        parser.error('Tip length must be finite and positive.')
    if isinstance(args.contact_layers_below,bool) or not isinstance(args.contact_layers_below,int) or args.contact_layers_below < 0:
        parser.error("Contact layers below must be a nonnegative integer.")
    if args.geometry_threads < 1:
        parser.error("Geometry thread count must be positive.")
    if args.layer_workers < 0:
        parser.error("Layer worker count must be nonnegative.")
    if args.layer_batch_size<1:
        parser.error('Layer batch size must be positive.')
    if args.geometry_query_batch_size<1:
        parser.error('Geometry query batch size must be positive.')
    if args.config and args.model_config:
        parser.error("Use --config or legacy --model-config, not both.")
    # Input paths are repository-relative, independent of the launch folder.
    if args.config:
        args.config = resolve_config_path(args.config)
    if args.skeleton_source and not args.skeleton_source.is_absolute():
        candidate=Path.cwd()/args.skeleton_source
        if candidate.exists():args.skeleton_source=candidate.resolve()
    for name in ("model_config", "mesh", "samples", "layer_values", "tree_source", "model_volume", "model_layers_source", "skeleton_source", "tetrahedral_volume"):
        path = getattr(args, name)
        if path and not path.is_absolute():
            setattr(args, name, (ROOT / path).resolve())
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices and value is not None and value not in action.choices:
            parser.error(f"Invalid configured {action.dest}: {value}")
    if args.minimum_support_height is not None and (not np.isfinite(args.minimum_support_height) or args.minimum_support_height<0):
        parser.error("Minimum support height must be finite and nonnegative.")
    if args.slicing_mode=='online' and args.model_layers_source:
        parser.error('Online shared-level slicing requires regenerating model layers; existing model-layer reuse needs sampled mode.')
    numeric = (args.spacing, args.padding, args.interior_depth, args.curvature, args.layer_spacing,
               args.layer_seed_spacing,args.adaptive_layer_tolerance)
    if not np.all(np.isfinite(numeric)) or args.spacing <= 0 or args.padding <= 0 or args.layer_spacing <= 0 or args.interior_depth < 0 or args.sample_count < 2 or args.preview_layers < 1 or args.maximum_points < 8:
        parser.error("Require finite parameters, positive spacing/padding/counts, and nonnegative interior depth.")
    if min(args.layer_seed_spacing,args.adaptive_layer_tolerance)<=0:
        parser.error('Layer seed spacing and adaptive layer tolerance must be positive.')
    support_spacing=min(args.layer_spacing,args.maximum_support_layer_height) if args.support_layer_spacing is None else args.support_layer_spacing
    if args.include_support_gradients and (not np.isfinite([support_spacing,args.maximum_support_layer_height,args.support_gradient_probe_spacing]).all()
            or min(support_spacing,args.support_gradient_probe_spacing)<=0 or args.maximum_support_layer_height<support_spacing):
        parser.error('Require positive support target/probe spacing and maximum support height >= target spacing.')
    if not 0 < args.support_threshold_angle <= 90 or not 0 < args.trajectory_turn <= 90 or (args.trajectory_step is not None and (not np.isfinite(args.trajectory_step) or args.trajectory_step <= 0)):
        parser.error("Require threshold/turn angles in (0,90] and a finite positive trajectory step.")
    if not np.all(np.isfinite([args.tree_spacing, args.side_clearance, args.contact_gap, args.tip_radius])) or args.tree_spacing <= 0 or args.tip_radius <= 0 or not 0 <= args.contact_gap <= args.side_clearance:
        parser.error("Require positive geometry spacing/radius and 0 <= contact gap <= side clearance.")
    return args


if __name__ == "__main__":
    run(parse_args())

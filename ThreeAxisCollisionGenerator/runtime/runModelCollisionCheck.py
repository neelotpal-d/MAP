"""Run ``scalarFieldCollisionCheck.py`` from a saved pipeline model directory.

The pipeline already records the mesh, scalar-network frequencies, output
checkpoint, and (for newer runs) the network depth.  This helper resolves
those values so collision checks do not need a long manually assembled CLI.
Tool geometry can be kept in a small reusable JSON file and overridden on the
command line when needed.
"""

import argparse
import json
import subprocess
import sys
import torch
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "model_runs"
COLLISION_SCRIPT = REPOSITORY_ROOT / "scalarFieldCollisionCheck.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a pipeline model and run scalarFieldCollisionCheck.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model-name", help="Pipeline model directory name under --run-root.")
    selection.add_argument("--run-dir", type=Path, help="Pipeline model directory containing pipeline_manifest.json.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Override the checkpoint resolved from the manifest.")
    parser.add_argument(
        "--tool-config",
        type=Path,
        default=None,
        help="Optional JSON with tool bounds/orientation and any scalarFieldCollisionCheck arguments.",
    )
    parser.add_argument(
        "--write-tool-config-template",
        action="store_true",
        help="Write collision_tool_config.sample.json in the model directory, then exit.",
    )

    contour = parser.add_mutually_exclusive_group()
    contour.add_argument("--use-voxels", action="store_true", help="Extract visualization contours from a voxel volume.")
    contour.add_argument("--use-tet", action="store_true", help="Extract visualization contours from the model TetGen volume.")
    contour.add_argument("--use-surface", action="store_true", help="Use the surface mesh as the contour source.")
    parser.add_argument("--voxel-spacing", type=float, default=1.0)
    parser.add_argument("--voxel-padding", type=float, default=1.0)

    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--grid-spacing", type=float, default=None, help="World-space spacing of interior collision roots.")
    sampling.add_argument(
        "--point-density",
        type=float,
        default=None,
        help="Requested interior roots per world-unit cubed; converted to grid spacing density^(-1/3).",
    )
    parser.add_argument("--max-sampled-points", type=int, default=400000)
    parser.add_argument("--padding", type=float, default=1.0)
    parser.add_argument("--inside-threshold", type=float, default=0.0)
    parser.add_argument("--inside-method", choices=("vtk", "igl"), default="vtk")
    parser.add_argument("--candidate-oversampling", type=float, default=4.0)
    parser.add_argument("--scalar-batch-size", type=int, default=50000)
    parser.add_argument("--origin-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--build-direction", type=float, nargs=3, default=None, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--in-plane-u", type=float, nargs=3, default=None, metavar=("UX", "UY", "UZ"))
    for axis in ("x", "y", "z"):
        parser.add_argument(f"--tool-{axis}-min", type=float, default=None)
        parser.add_argument(f"--tool-{axis}-max", type=float, default=None)

    parser.add_argument("--debug-viz", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-all-colliding-origins", action="store_true")
    parser.add_argument("--viz-colliding-origin-rank", type=int, default=0)
    parser.add_argument("--viz-contour-count", type=int, default=20)
    parser.add_argument("--viz-contour-step-world", type=float, default=0.2)
    parser.add_argument("--viz-cross-section", action="store_true")
    parser.add_argument("--viz-cross-section-axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--viz-cross-section-position-world", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write and print the resolved config without starting the checker.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint(manifest: dict, model_dir: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {override}")
        return override.resolve()

    configured = Path(manifest.get("paths", {}).get("output_checkpoint", ""))
    if configured.is_file():
        return configured.resolve()

    candidates = sorted((model_dir / "checkpoints").glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No checkpoint found in the manifest or model checkpoints directory.")
    return candidates[-1].resolve()


def load_training_record(checkpoint: Path) -> dict:
    record = checkpoint.with_name(f"{checkpoint.stem}_run_config.json")
    if not record.is_file():
        return {}
    return read_json(record).get("arguments", {})


def tool_config_template() -> dict:
    return {
        "build_direction": [0.0, 1.0, 0.0],
        "in_plane_u": [1.0, 0.0, 0.0],
        "tool_x_min": -28.5,
        "tool_x_max": 28.5,
        "tool_y_min": 4.2,
        "tool_y_max": 100.0,
        "tool_z_min": -60.0,
        "tool_z_max": 15.6,
    }


def set_if_given(config: dict, name: str, value) -> None:
    if value is not None:
        config[name] = value


def build_config(args: argparse.Namespace, model_dir: Path, manifest: dict, checkpoint: Path) -> dict:
    training = load_training_record(checkpoint)
    config = read_json(args.tool_config) if args.tool_config is not None else {}
    if args.tool_config is None and training.get('tool_box_config'):
        from boxToolSampling import nominal_box
        config.update(nominal_box(read_json(Path(training['tool_box_config']))))
    if args.tool_config is not None and not args.tool_config.is_file():
        raise FileNotFoundError(f"Tool config does not exist: {args.tool_config}")

    if all(k in config for k in ('cone', 'inner_cylinder', 'outer_cylinder')):
        from axisymmetricTool import validate_nominal
        validate_nominal(config)
        config = dict(config, axisymmetric_tool={k: config[k] for k in ('units', 'cone', 'inner_cylinder', 'outer_cylinder')})

    mesh_path = Path(manifest["mesh_path"]).resolve()
    node_path = manifest.get("node_path")
    ele_path = manifest.get("ele_path")
    if args.use_tet:
        contour_source = "tet"
    elif args.use_surface:
        contour_source = "surface"
    else:
        # Voxel is the robust default for runs without an available volume mesh.
        contour_source = "voxel"

    if args.point_density is not None:
        if args.point_density <= 0.0:
            raise ValueError("--point-density must be positive.")
        grid_spacing = args.point_density ** (-1.0 / 3.0)
    else:
        grid_spacing = args.grid_spacing if args.grid_spacing is not None else 0.5

    config.update({
        "mesh_path": str(mesh_path),
        "normalization_model_path": str(mesh_path),
        "checkpoint_path": str(checkpoint),
        "field_type": "neural",
        "contour_source": contour_source,
        "voxel_spacing": args.voxel_spacing,
        "voxel_padding": args.voxel_padding,
        "volume_node_path": node_path,
        "volume_ele_path": ele_path,
        "w0": training.get("w0", manifest.get("w0", 1.0)),
        "w0_initial": training.get("w0_initial", manifest.get("w0_initial", 1.0)),
        "num_layers": training.get("num_layers", 5),
        "dim_hidden": training.get("dim_hidden", 128),
        "device": args.device,
        "padding": args.padding,
        "grid_spacing": grid_spacing,
        "inside_threshold": args.inside_threshold,
        "inside_method": args.inside_method,
        "candidate_oversampling": args.candidate_oversampling,
        "max_sampled_points": args.max_sampled_points,
        "scalar_batch_size": args.scalar_batch_size,
        "origin_batch_size": args.origin_batch_size,
        "debug_viz": args.debug_viz,
        "viz": args.viz,
        "viz_all_colliding_origins": args.viz_all_colliding_origins,
        "viz_colliding_origin_rank": args.viz_colliding_origin_rank,
        "viz_contour_count": args.viz_contour_count,
        "viz_contour_step_world": args.viz_contour_step_world,
        "viz_cross_section": args.viz_cross_section,
        "viz_cross_section_axis": args.viz_cross_section_axis,
        "viz_cross_section_position_world": args.viz_cross_section_position_world,
    })

    # A nominal shape config may contain the original frame. The trained
    # checkpoint supplies the learned frame unless a CLI override is explicit.
    import torch
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    learned = checkpoint_data.get('learned_tool_frame')
    if learned is not None:
        config['build_direction'] = learned['n']
        config['in_plane_u'] = learned['u']
        print('Using learned global tool frame from the selected checkpoint.')
    set_if_given(config, "build_direction", args.build_direction)
    set_if_given(config, "in_plane_u", args.in_plane_u)
    for axis in ("x", "y", "z"):
        set_if_given(config, f"tool_{axis}_min", getattr(args, f"tool_{axis}_min"))
        set_if_given(config, f"tool_{axis}_max", getattr(args, f"tool_{axis}_max"))
    config.setdefault("build_direction", training.get("up_direction", manifest.get("up_direction", [0.0, 1.0, 0.0])))
    config.setdefault("in_plane_u", training.get("in_plane_u", [1.0, 0.0, 0.0]))
    return config


def main() -> None:
    args = parse_args()
    model_dir = args.run_dir.resolve() if args.run_dir else (args.run_root / args.model_name).resolve()
    manifest_path = model_dir / "pipeline_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No pipeline manifest found: {manifest_path}")
    manifest = read_json(manifest_path)

    if args.write_tool_config_template:
        template_path = model_dir / "collision_tool_config.sample.json"
        template_path.write_text(json.dumps(tool_config_template(), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote tool configuration template: {template_path}")
        return

    checkpoint = resolve_checkpoint(manifest, model_dir, args.checkpoint_path)
    config = build_config(args, model_dir, manifest, checkpoint)
    output_dir = model_dir / "collision_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = output_dir / f"collision_check_{timestamp}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    print(f"Model run: {model_dir.name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Contour source: {config['contour_source']}; collision-root spacing: {config['grid_spacing']:.6g} world units")
    print(f"Resolved config: {config_path}")
    if args.dry_run:
        return
    subprocess.run([sys.executable, str(COLLISION_SCRIPT), "--config-path", str(config_path)], check=True)


if __name__ == "__main__":
    main()

"""Run the portable fixed-axis surface/support/strength field optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from .job_options import resolve_frequencies


RUNTIME = Path(__file__).resolve().parent / "runtime"
PIPELINE = RUNTIME / "runBandedSurfaceQualityPipeline.py"
CHECKER = RUNTIME / "runModelCollisionCheck.py"

# Options passed without a value when true.  All other supported options use
# their JSON key with underscores converted to the runner's hyphenated flag.
BOOLEAN_OPTIONS = {
    "disable_surface_quality_loss", "enable_support_free_loss", "disable_surface_scalar_loss",
    "affine_y_residual", "no_fuse_regular_losses", "debug_loss_breakdown",
    "interactive_loss_controls", "balanced_norm_batches", "enable_input_curvature",
    "input_curvature_boundary_coverage",
    "enable_tangent_twist", "boundary_vertex_normal_smoothing",
    "exclude_surface_transition_samples", "skip_automatic_selection_window",
    "fast_selection_preview", "base_direction_parallel_only",
    "disable_scalar_gradient_division", "detach_scalar_gradient_denominator",
    "skip_constraint_sample_preview", "disable_collision", "skip_surface_selection",
    "select_surface_interactively", "reuse_saved_preserved_regions",
    "reuse_generated_constraint_samples", "no_train", "skip_sdf_verification",
    "skip_sdf", "skip_initializer_verification", "retrain_initializer", "yes",
    "initializer_lightweight_preview", "no_sdf_fuse_batches",
    "disable_norm_banded_control", "show_norm_control_batches",
    "show_local_edge_pairs", "initial_gradient_diagnostics", "create_missing",
    "retrain_existing", "strength_all_signs", "show_final_preview", "normalize_collision_by_gradient", "disable_orientation", "optimize_yaw",
}
PATH_OPTIONS = {"mesh_path", "node_path", "ele_path", "start_checkpoint", "run_root", "strength_file", "tool_plane_config", "tool_box_config", "stress_orientation_record"}
REQUIRED = {"model_name", "mesh_path", "w0"}


def _resolve(value, base: Path) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def _flatten_job(raw: dict, job_path: Path, output_override: Path | None) -> dict:
    allowed = {"output", "pipeline", "strength", "workflow"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown job keys: {sorted(unknown)}")
    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("Job requires a 'pipeline' object.")
    sections = {"model", "sdf", "initializer", "training", "surface_quality", "support_free", "collision", "execution", "orientation"}
    unknown = set(pipeline) - sections
    if unknown:
        raise ValueError(f"Unknown pipeline sections: {sorted(unknown)}")
    values: dict = {}
    for name, section in pipeline.items():
        if not isinstance(section, dict):
            raise ValueError(f"pipeline.{name} must be an object.")
        duplicate = set(values) & set(section)
        if duplicate:
            raise ValueError(f"Duplicate option(s) across pipeline sections: {sorted(duplicate)}")
        values.update(section)
    if values.get('disable_orientation') is False and not values.get('optimize_yaw'):
        raise ValueError('This package uses a fixed tool axis; use an orientation package to learn a pose')
    values['disable_orientation'] = not bool(values.get('optimize_yaw'))
    output = output_override or raw.get("output")
    if output is None:
        raise ValueError("Job requires 'output', or pass --output.")
    values["run_root"] = _resolve(output, job_path.parent)
    for name in PATH_OPTIONS & set(values):
        if values[name] is not None and name != "run_root":
            values[name] = _resolve(values[name], job_path.parent)
    missing = REQUIRED - set(values)
    if missing:
        raise ValueError(f"Missing required model options: {sorted(missing)}")
    strength = raw.get("strength", {})
    if not isinstance(strength, dict):
        raise ValueError("strength must be an object when supplied.")
    workflow = raw.get('workflow', 'strength' if strength.get('enabled') else 'surface-quality').replace('_', '-')
    if workflow not in {'strength', 'surface-quality', 'support-free', 'combined'}:
        raise ValueError('workflow must be strength, support-free, surface-quality or combined')
    if 'objective' in values and values['objective'] != workflow:
        raise ValueError('pipeline objective conflicts with workflow')
    values['objective'] = workflow
    supported = {'enabled', 'file', 'weight', 'neighbors', 'max_distance_mm', 'point_columns',
                 'direction_columns', 'signed_column', 'tensile_only', 'min_tensile_stress', 'seed',
                 'sampling', 'samples', 'batch_size', 'fps_filter_order'}
    if set(strength)-supported:
        raise ValueError('Unknown strength settings')
    if workflow in {'strength', 'combined'}:
        if not strength.get('enabled') or not strength.get('file'):
            raise ValueError('Strength and combined workflows require strength.enabled and strength.file')
        for key, value in strength.items():
            if key not in {'enabled', 'tensile_only'}:
                values['strength_'+key] = value
        values['strength_all_signs'] = not strength.get('tensile_only', True)
        values['strength_file'] = _resolve(values['strength_file'], job_path.parent)
        if not Path(values['strength_file']).is_file():
            raise FileNotFoundError(values['strength_file'])
    elif strength.get('enabled'):
        raise ValueError('Choose workflow strength or combined to enable the supplied strength objective')
    if values.get('disable_collision'):
        values['skip_sdf'] = True
    return resolve_frequencies(values)


def _command(values: dict) -> list[str]:
    from decimal import Decimal
    def argument(value):
        # argparse otherwise treats negative scientific notation as an option.
        # Decimal expands the shortest float representation without rounding.
        return format(Decimal(str(value)), 'f') if isinstance(value, float) else str(value)
    command = [sys.executable, str(PIPELINE)]
    for key, value in values.items():
        if key == "balanced_norm_batches":
            command.append("--balanced-norm-batches" if value else "--no-balanced-norm-batches")
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOLEAN_OPTIONS:
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        command.append(flag)
        if isinstance(value, (list, tuple)):
            command.extend(argument(item) for item in value)
        else:
            command.append(argument(value))
    return command


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    if any(item == '--run-dir' or item.startswith('--run-dir=') for item in argv):
        parser.set_defaults(job=None)
    else:
        parser.add_argument("job", type=Path, nargs='?', help="JSON job; relative paths are resolved from this file.")
    parser.add_argument('--run-dir', type=Path, help='Inspect a saved run without its original job')
    parser.add_argument("--output", type=Path, help="Override the job's output/run-root directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the job and print the exact pipeline command.")
    parser.add_argument('--view', action='store_true', help='Compare initial/final gradient-height layers')
    parser.add_argument("--collision-check", action="store_true", help="Run the bundled saved-model collision checker instead of training.")
    parser.add_argument("--tool-config", type=Path, help="Collision-checker tool configuration JSON.")
    args, passthrough = parser.parse_known_args(argv)
    if not PIPELINE.is_file() or not CHECKER.is_file():
        raise FileNotFoundError("Bundled runtime is incomplete.")
    if args.view and args.collision_check:
        parser.error('Choose either view or collision-check')
    if args.run_dir:
        if args.job or args.output or not (args.view or args.collision_check):
            parser.error('--run-dir requires view/check without a job or output override')
        model_dir = args.run_dir.resolve()
        values = {}
        command = []
    else:
        if args.job is None:
            parser.error('Supply a job or --run-dir with --view/--collision-check')
        job_path = args.job.resolve()
        raw = json.loads(job_path.read_text(encoding="utf-8"))
        values = _flatten_job(raw, job_path, args.output)
        model_dir = Path(values['run_root']) / str(values['model_name'])
        command = _command(values)
    if args.collision_check:
        command = [sys.executable, str(CHECKER), "--run-dir", str(model_dir)]
        if args.tool_config:
            command.extend(("--tool-config", _resolve(args.tool_config, job_path.parent if args.job else Path.cwd())))
    if args.view:
        command = [sys.executable, str(RUNTIME/'viewFieldComparison.py'), '--run-dir', str(model_dir)]
    command.extend(passthrough)
    print(json.dumps({"resolved_options": values, "command": command}, indent=2))
    if args.dry_run:
        return 0
    subprocess.run(command, check=True, cwd=None if args.view else str(RUNTIME))
    return 0

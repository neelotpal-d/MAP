"""Reuse output directories while deleting only recognized generated artifacts."""
from pathlib import Path
import re


RUN_FILES = {
    "model_input.json",
    "skipped_contacts.json",
    "skeleton_field_audit.json",
    "stall_diagnostic.json",
    "support_centerlines.vtp", "model_tree_plan.json",
    "resolved_config.json", "contacts.csv", "routing_paths.json", "routing_failures.json", "support_field.npz", "contact_regions.npz",
    "routing_domain.npz",
    "routing_diagnostic.png", "patch_diagnostic.json",
    "model_bridges.json",
    "native_model_volume.vtu", "swan_pipeline_report.json",
    "support_scalar_grid.vtu",
    "tree.vtp", "contacts.vtp",
    "summary.json", "layer_manifest.json", "module_report.json",
    "model_layer_manifest.json", "support_layer_manifest.json", "compatible_export.json",
    "trusted_samples.npz", "trajectory_inputs.npz", "transfer_inputs.npz",
    "envelope_mesh.vtu", "envelope_field.vtu", "envelope_boundary.vtp",
    "envelope_trajectories.vtp", "trajectories.vtp", "main_scalar_values.txt",
    "preview.png", "envelope_preview.png",
    "model_preview.png", "support_preview.png",
    "model_field.vtp", "model_field_report.json", "model_field_preview.png", "combined_field_preview.png",
    "support_solid.vtp", "tree_geometry_report.json", "tree_layer_manifest.json",
    "contact_coarse.vtp", "contact_fine.vtp", "contact_comparison.png", "contact_diagnostic.json",
    "layer_membership.npz",
    "side_gap_diagnostic.json",
    "roof_patch.vtp", "roof_comparison.png", "roof_comparison.json", "roof_solid.vtp",
    "roof_layers.png",
}


def _remove_files(directory, names):
    for name in names:
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"Refusing to clean a linked artifact: {path}")
        if path.is_file():
            path.unlink()


def _clean_run(directory):
    _remove_files(directory, RUN_FILES)
    for name in ("envelope_layers", "layers", "tree_layers", "model_layers", "support_layers"):
        layers = directory / name
        if layers.is_symlink():
            raise ValueError(f"Refusing to clean a linked layer directory: {layers}")
        if layers.is_dir():
            for path in layers.iterdir():
                if path.suffix.lower() not in (('.vtp','.obj') if name in ('model_layers','support_layers') else ('.vtp',)):
                    continue
                if re.fullmatch(r"\d+_\d+" if name in ('model_layers','support_layers') else r"\d+", path.stem):
                    if path.is_symlink():
                        raise ValueError(f"Refusing to clean a linked layer: {path}")
                    path.unlink()
            if not any(layers.iterdir()):
                layers.rmdir()


def prepare_output(output, clean=False, benchmark=False):
    raw = Path(output).absolute()
    if raw.is_symlink() or any(parent.is_symlink() for parent in raw.parents):
        raise ValueError("Output cleanup does not support linked directory paths.")
    output = raw.resolve()
    workspace = Path(__file__).resolve().parents[1]
    if output in (Path(output.anchor), workspace, Path(__file__).resolve().parent):
        raise ValueError(f"Choose a dedicated output subdirectory: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        if not clean:
            raise FileExistsError(f"Output is nonempty; use --clean-output to reuse it: {output}")
        # Resolve each final target before deletion; never recursively remove
        # the output directory or its unknown contents.
        if benchmark:
            for child in output.iterdir():
                if re.fullmatch(r"run_\d{3,}", child.name) and child.is_dir():
                    if child.is_symlink() or child.resolve().parent != output:
                        raise ValueError(f"Run directory escapes output: {child}")
                    _clean_run(child)
                    if any(child.iterdir()):
                        remaining=', '.join(sorted(path.name for path in child.iterdir()))
                        raise ValueError(f"Run contains unrecognized files; preserved them: {child} ({remaining})")
                    child.rmdir()
            _remove_files(output, {"benchmark.json", "swan_pipeline_report.json", "tip_radius_comparison.json", "tip_radius_comparison.png", "tip_radius_layers.png"})
            for path in output.glob("run_*.log"):
                if re.fullmatch(r"run_\d{3,}\.log", path.name):
                    _remove_files(output, {path.name})
        else:
            _clean_run(output)
    output.mkdir(parents=True, exist_ok=True)
    return output

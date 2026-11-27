"""Configuration for the currently implemented curved-support stages only."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "trim_boundary_tolerance_mm",
    "correct_layer_scalar",
    "slicing_mode",
    "include_support_gradients", "support_layer_spacing", "maximum_support_layer_height", "support_gradient_probe_spacing",
    "layer_seed_spacing", "adaptive_layer_tolerance",
    "vtk_threads_per_worker", "maximum_trunk_diameter", "maximum_platform_trunk_diameter",
    "model_root_foot_radius", "model_root_foot_alignment", "model_root_foot_shape", "platform_base_radius_factor", "platform_base_height",
    "field_upward_length", "field_platform_upward", "field_upward_ramp", "support_field_mode",
    "allow_stalled_skeleton", "skip_unrooted_contacts", "planar_platform_bases", "platform_base_layer_height",
    "boundary_mode", "spacing", "padding", "envelope_mode", "trajectory_region", "export_compatible_layers", "export_envelope_layers",
    "support_threshold_angle", "minimum_support_height", "trajectory_step", "trajectory_turn", "build_direction",
    "plate", "maximum_points", "sample_count", "seed", "interior_depth", "curvature",
    "layer_spacing", "preview_layers", "all_layers", "minimum_layer", "maximum_layer", "contour_backend", "layer_compression",
    "tree_spacing", "side_clearance", "contact_gap", "fast_clearance", "model_tree_mode", "tip_radius", "roof_clearance_mode", "terminal_cap", "support_ceiling",
    "geometry_mode", "branch_blend", "clearance_blend", "rim_blend", "tip_alignment", "tip_length", "contact_layers_below",
    "roof_mode", "roof_terminal", "roof_radius", "roof_thickness", "roof_rounding", "roof_layer_spacing",
    "layer_trim", "geometry_threads", "layer_workers", "layer_batch_size", "model_distance_backend", "model_query_backend", "geometry_query_batch_size", "trim_pair_pruning",
    "branch_sizing", "branch_diameter", "branch_diameter_angle", "platform_flange", "support_speed",
}
INPUT_KEYS = {"mesh", "checkpoint", "normalization_mesh", "layer_values", "field_mode", "samples", "tree_source", "model_volume", "model_layers_source", "skeleton_source", "tetrahedral_volume", "tetrahedral_scalar"}
NETWORK_DEFAULTS = {"device": "auto", "dim_hidden": 128, "num_layers": 5,
                    "w0": 50.0, "w0_initial": 50.0, "batch_size": 32768}


def resolve_config_path(path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path, ROOT / path, Path(__file__).resolve().parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Configuration not found. Checked " + ", ".join(str(p) for p in candidates))


def resolve_config_arguments(arguments):
    """Freeze caller-relative config paths before a subprocess changes cwd."""
    result = list(arguments)
    for index, value in enumerate(result):
        if value == "--config" and index + 1 < len(result):
            result[index + 1] = str(resolve_config_path(result[index + 1]))
        elif value.startswith("--config="):
            result[index] = "--config=" + str(resolve_config_path(value.split("=", 1)[1]))
    return result


def load_config(path):
    path = resolve_config_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(config) - {"input", "network", "settings"}
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    inputs = dict(config.get("input", {}))
    settings = dict(config.get("settings", {}))
    network = dict(NETWORK_DEFAULTS)
    overrides = config.get("network", {})
    if set(inputs) - INPUT_KEYS or set(settings) - SETTINGS or set(overrides) - set(network):
        raise ValueError("Unknown input, settings, or network key in curved configuration.")
    network.update(overrides)
    for key in ("mesh", "checkpoint", "normalization_mesh", "layer_values", "samples", "tree_source", "model_volume", "model_layers_source", "skeleton_source", "tetrahedral_volume"):
        if inputs.get(key):
            candidate = Path(inputs[key])
            inputs[key] = str(candidate if candidate.is_absolute() else (ROOT / candidate).resolve())
    return {"input": inputs, "network": network, "settings": settings,
            "_config_path": str(path.resolve())}


def argument_defaults(config):
    defaults = dict(config["settings"])
    for key in ("mesh", "layer_values", "samples", "tree_source", "model_volume", "model_layers_source", "skeleton_source", "tetrahedral_volume"):
        if config["input"].get(key):
            defaults[key] = Path(config["input"][key])
    if config["input"].get("tetrahedral_scalar"):
        defaults["tetrahedral_scalar"] = str(config["input"]["tetrahedral_scalar"])
    if config["input"].get("field_mode"):
        defaults["field_mode"] = config["input"]["field_mode"]
    return defaults

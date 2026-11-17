from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_CONFIG: dict[str, Any] = {
    "paths_relative_to": "repository",
    "input": {
        "field_mode": "checkpoint",
        "model_manifest": None,
        "mesh": None,
        "checkpoint": None,
        "normalization_mesh": None,
        "layer_values": None,
    },
    "network": {
        "device": "auto",
        "dim_hidden": 128,
        "num_layers": 5,
        "w0": 1.0,
        "w0_initial": 1.0,
        "batch_size": 32768,
    },
    "layers": {
        "world_spacing": 0.2,
        "epsilon": 0.0,
        "maximum_count": 1000,
        "surface_probe_count": 256,
    },
    "contacts": {
        "overhang_angle_degrees": 135.0,
        "support_threshold_angle_degrees": 45.0,
        "base_exclusion": 0.8,
        "minimum_layer": 2,
        "tip_spacing": 3.0,
        "minimum_bucket_area": 0.0,
        "maximum_seeds": 300,
        "surface_offset": 0.25,
        "coverage_phase_tolerance": None,
    },
    "domain": {
        "voxel_spacing": 1.0,
        "padding": 8.0,
        "build_direction": [0.0, 1.0, 0.0],
        "build_plate_projection": None,
        "model_clearance": 0.25,
        "clearance_voxel_margin": None,
        "seed_snap_distance": 3.0,
        "maximum_voxels": 8000000,
        "cache_domain": True,
    },
    "routing": {
        "branch_radius": 1.2,
        "maximum_branch_radius": 4.0,
        "radius_power": 2.5,
        "minimum_attach_phase_drop": 1.0,
        "maximum_upward_step": 0.0,
        "maximum_branch_angle_degrees": 45.0,
        "lateral_penalty": 0.35,
        "clearance_penalty": 0.2,
        "maximum_expansions_per_seed": 750000,
        "prevent_corner_cutting": True,
        "minimum_layer_spacing": 0.08,
        "maximum_layer_spacing": 1.2,
        "spacing_policy": "reject",
        "fail_on_unrouted": True,
    },
    "field": {
        "extension_mode": "surface_transfer",
        "screen_weight": 0.35,
        "smooth_weight": 1.0,
        "cg_relative_tolerance": 1e-8,
        "cg_max_iterations": 4000,
        "direct_fallback": True,
    },
    "interface": {
        "enabled": False,
        "mode": "point_collar",
        "contact_gap": 0.25,
        "collar_radius": 0.6,
        "roof_penetration": 1.0,
        "roof_outward_extension": None,
        "roof_coverage_scale": 1.5,
        "roof_geometry": "solid_patch",
        "roof_transition_top_radius": None,
        "roof_transition_maximum_angle_degrees": 45.0,
        "clearance_voxel_margin": None,
        "virtual_transfer_shell": False,
    },
    "base_foot": {
        "enabled": False,
        "bottom_radius": 4.5,
        "height": 3.5,
        "maximum_flare_angle_degrees": 45.0,
    },
    "validation": {
        # This is a geometric, layer-order coverage diagnostic for the
        # conformal roof.  It is intentionally opt-in while the roof is still
        # a prototype: the result is reported separately from hard collision
        # checks, rather than silently declaring a support printable.
        "roof_layer_coverage": {
            "enabled": False,
            "footprint_radius": None,
            "maximum_uncovered_fraction": 0.05,
        },
    },
    "export": {
        "output_directory": "./support_generation_voxel_output",
        "minimum_layer": 1,
        "maximum_layer": None,
        "minimum_component_area": 0.0,
        "save_debug_volume": True,
        "clean_output": False,
        "overwrite_known_outputs": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_manifest(config: dict[str, Any], base: Path) -> None:
    input_cfg = config["input"]
    manifest_path = _resolve_path(input_cfg.get("model_manifest"), base)
    if manifest_path is None:
        return
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model manifest does not exist: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("paths", {})
    input_cfg["model_manifest"] = str(manifest_path)
    if input_cfg.get("mesh") is None:
        input_cfg["mesh"] = manifest.get("mesh_path") or manifest.get("original_mesh_path")
    if input_cfg.get("checkpoint") is None:
        input_cfg["checkpoint"] = paths.get("output_checkpoint")
    if input_cfg.get("normalization_mesh") is None:
        input_cfg["normalization_mesh"] = input_cfg.get("mesh")

    network_cfg = config["network"]
    if "w0" in manifest:
        network_cfg["w0"] = float(manifest["w0"])
    if "w0_initial" in manifest:
        network_cfg["w0_initial"] = float(manifest["w0_initial"])
    if "num_layers" in manifest:
        network_cfg["num_layers"] = int(manifest["num_layers"])
    if "up_direction" in manifest:
        config["domain"]["build_direction"] = manifest["up_direction"]


def _validate(config: dict[str, Any]) -> None:
    input_cfg = config["input"]
    mode = input_cfg.get("field_mode", "checkpoint")
    if mode not in {"checkpoint", "analytic_y"}:
        raise ValueError("input.field_mode must be 'checkpoint' or 'analytic_y'.")
    required_paths = ("mesh", "checkpoint") if mode == "checkpoint" else ("mesh",)
    for name in required_paths:
        path = input_cfg.get(name)
        if path is None:
            requirement = "directly or through input.model_manifest" if name == "checkpoint" else "for every field mode"
            raise ValueError(f"input.{name} is required ({requirement}).")
        if not Path(path).is_file():
            raise FileNotFoundError(f"input.{name} does not exist: {path}")
    if input_cfg.get("normalization_mesh") and not Path(input_cfg["normalization_mesh"]).is_file():
        raise FileNotFoundError(f"input.normalization_mesh does not exist: {input_cfg['normalization_mesh']}")
    if input_cfg.get("layer_values") and not Path(input_cfg["layer_values"]).is_file():
        raise FileNotFoundError(f"input.layer_values does not exist: {input_cfg['layer_values']}")

    if float(config["domain"]["voxel_spacing"]) <= 0.0:
        raise ValueError("domain.voxel_spacing must be positive.")
    margin = config["domain"].get("clearance_voxel_margin")
    if margin is not None and float(margin) < 0.0:
        raise ValueError("domain.clearance_voxel_margin must be non-negative or null.")
    if float(config["contacts"]["tip_spacing"]) <= 0.0:
        raise ValueError("contacts.tip_spacing must be positive.")
    coverage_phase_tolerance = config["contacts"].get("coverage_phase_tolerance")
    if coverage_phase_tolerance is not None and float(coverage_phase_tolerance) <= 0.0:
        raise ValueError("contacts.coverage_phase_tolerance must be positive or null.")
    if not 90.0 <= float(config["contacts"]["overhang_angle_degrees"]) < 180.0:
        raise ValueError("contacts.overhang_angle_degrees must lie in [90, 180).")
    threshold_angle = float(config["contacts"]["support_threshold_angle_degrees"])
    if not 0.0 < threshold_angle <= 90.0:
        raise ValueError("contacts.support_threshold_angle_degrees must lie in (0, 90].")
    if config["routing"]["spacing_policy"] not in {"warn", "reject"}:
        raise ValueError("routing.spacing_policy must be 'warn' or 'reject'.")
    angle = float(config["routing"]["maximum_branch_angle_degrees"])
    if not 0.0 < angle < 90.0:
        raise ValueError("routing.maximum_branch_angle_degrees must lie in (0, 90).")
    if float(config["routing"]["branch_radius"]) <= 0.0:
        raise ValueError("routing.branch_radius must be positive.")
    if float(config["routing"]["maximum_branch_radius"]) < float(config["routing"]["branch_radius"]):
        raise ValueError("routing.maximum_branch_radius must be at least routing.branch_radius.")
    extension_mode = config["field"].get("extension_mode", "auto")
    if extension_mode not in {"surface_transfer", "skeleton_screened", "direct_field"}:
        raise ValueError(
            "field.extension_mode must be 'surface_transfer', 'skeleton_screened', or 'direct_field'."
        )
    if float(config["field"]["screen_weight"]) <= 0.0:
        raise ValueError("field.screen_weight must be positive so the voxel solve is well posed.")
    interface = config["interface"]
    if interface.get("mode", "point_collar") not in {"point_collar", "sdf_trimmed_roof"}:
        raise ValueError("interface.mode must be 'point_collar' or 'sdf_trimmed_roof'.")
    if float(interface["contact_gap"]) < 0.0:
        raise ValueError("interface.contact_gap must be non-negative.")
    if float(interface["collar_radius"]) <= 0.0:
        raise ValueError("interface.collar_radius must be positive.")
    if float(interface.get("roof_penetration", 0.0)) < 0.0:
        raise ValueError("interface.roof_penetration must be non-negative.")
    if interface.get("roof_outward_extension") is not None and float(interface["roof_outward_extension"]) <= 0.0:
        raise ValueError("interface.roof_outward_extension must be positive or null.")
    if float(interface.get("roof_coverage_scale", 0.0)) <= 0.0:
        raise ValueError("interface.roof_coverage_scale must be positive.")
    if interface.get("roof_geometry", "solid_patch") not in {"solid_patch", "tapered_heads"}:
        raise ValueError("interface.roof_geometry must be 'solid_patch' or 'tapered_heads'.")
    transition_radius = interface.get("roof_transition_top_radius")
    if transition_radius is not None and float(transition_radius) < float(interface["collar_radius"]):
        raise ValueError("interface.roof_transition_top_radius must be at least interface.collar_radius or null.")
    transition_angle = float(interface.get("roof_transition_maximum_angle_degrees", 45.0))
    if not 0.0 < transition_angle < 90.0:
        raise ValueError("interface.roof_transition_maximum_angle_degrees must lie in (0, 90).")
    if not isinstance(interface.get("virtual_transfer_shell", False), bool):
        raise ValueError("interface.virtual_transfer_shell must be boolean.")
    interface_margin = interface.get("clearance_voxel_margin")
    if interface_margin is not None and float(interface_margin) < 0.0:
        raise ValueError("interface.clearance_voxel_margin must be non-negative or null.")
    roof_coverage = config["validation"].get("roof_layer_coverage", {})
    if not isinstance(roof_coverage.get("enabled", False), bool):
        raise ValueError("validation.roof_layer_coverage.enabled must be boolean.")
    footprint_radius = roof_coverage.get("footprint_radius")
    if footprint_radius is not None and float(footprint_radius) <= 0.0:
        raise ValueError("validation.roof_layer_coverage.footprint_radius must be positive or null.")
    maximum_uncovered_fraction = float(roof_coverage.get("maximum_uncovered_fraction", 0.05))
    if not 0.0 <= maximum_uncovered_fraction <= 1.0:
        raise ValueError("validation.roof_layer_coverage.maximum_uncovered_fraction must lie in [0, 1].")
    base_foot = config["base_foot"]
    if float(base_foot["bottom_radius"]) <= 0.0 or float(base_foot["height"]) <= 0.0:
        raise ValueError("base_foot.bottom_radius and base_foot.height must be positive.")
    flare_angle = float(base_foot["maximum_flare_angle_degrees"])
    if not 0.0 < flare_angle < 90.0:
        raise ValueError("base_foot.maximum_flare_angle_degrees must lie in (0, 90).")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, merge, resolve, and validate one support-generation config."""

    config_path = Path(path).resolve()
    user = json.loads(config_path.read_text(encoding="utf-8"))
    config = _deep_merge(DEFAULT_CONFIG, user)
    relative_mode = config.get("paths_relative_to", "repository")
    if relative_mode == "repository":
        base = REPOSITORY_ROOT
    elif relative_mode == "config":
        base = config_path.parent
    else:
        raise ValueError("paths_relative_to must be 'repository' or 'config'.")

    _load_manifest(config, base)
    for name in ("mesh", "checkpoint", "normalization_mesh", "layer_values"):
        resolved = _resolve_path(config["input"].get(name), base)
        config["input"][name] = None if resolved is None else str(resolved)
    output = _resolve_path(config["export"]["output_directory"], base)
    assert output is not None
    config["export"]["output_directory"] = str(output)
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(REPOSITORY_ROOT)
    _validate(config)
    return config

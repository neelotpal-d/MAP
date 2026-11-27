"""
Config-driven end-to-end surface pipeline up to height-augmented toolpaths.

Stages:
- optional isotropic remeshing
- optional support-only boundary expansion
- contour/toolpath generation
- selectable 3-axis or first-pass 5-axis height computation

Machine G-code export intentionally remains a separate stage.
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
import gc
import json
import multiprocessing as mp
import shutil
from pathlib import Path
import re
import time

import numpy as np

from adaptive_path_decimation import apply_adaptive_point_decimation
from batch_compute_heights import collect_layer_files
from batch_compute_heights import compute_heights_with_effective_cap
from batch_compute_heights import infer_model_lift_from_metadata
from batch_compute_heights import load_existing_metadata
from batch_compute_heights import previous_main_indices
from batch_compute_heights import write_height_metadata
from batch_compute_projected_nearest_heights import run_nearest_surface_height_stage
from batch_compute_heights_5axis import _compute_layer_heights_5axis_job
from batch_compute_heights_5axis import write_height_metadata as write_height_metadata_5axis
from batch_expand_boundary_support_meshes import expand_boundary_vertices
from batch_expand_boundary_support_meshes import iter_obj_files as iter_expand_obj_files
from batch_expand_boundary_support_meshes import save_obj_mesh
from batch_generate_paths import _build_batch_metadata
from batch_generate_paths import iter_layer_meshes
from batch_generate_paths import process_layer
from batch_generate_hybrid_paths import _build_batch_metadata as _build_hybrid_batch_metadata
from batch_generate_hybrid_paths import process_hybrid_layer
from batch_meshlab_isotropic_remesh import iter_obj_files as iter_remesh_obj_files
from batch_meshlab_isotropic_remesh import remesh_file
from detect_top_layers_by_coverage import classify_layers
from extrusion_height import compute_point_heights_3axis
from extrusion_height import load_path_rows
from extrusion_height import load_reference_mesh_3axis
from extrusion_height import save_path_with_heights
from heat_method_core import HeatMethodParams
from mesh_distance_workflow import load_obj_mesh


LAYER_OBJ_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.obj$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the remesh -> path -> height pipeline from a JSON config file.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="JSON config describing folders and processing parameters.",
    )
    return parser


def load_config(config_path: Path) -> tuple[dict[str, object], Path]:
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Top-level config must be a JSON object.")
    return config, config_path.parent.resolve()


def resolve_path(base_dir: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Expected a non-empty path string in the config.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Expected a non-empty path string in the config.")
    path = Path(normalized)
    return (base_dir / path).resolve() if not path.is_absolute() else path


def require_section(config: dict[str, object], section_name: str) -> dict[str, object]:
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be a JSON object.")
    return section


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    resolved = directory.resolve()
    if (resolved / ".git").exists():
        raise ValueError(f"Refusing to clear git/workspace directory: {resolved}")
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def iter_layer_obj_files(input_dir: Path) -> list[Path]:
    meshes = [path for path in input_dir.iterdir() if path.is_file() and LAYER_OBJ_PATTERN.match(path.name)]
    return sorted(
        meshes,
        key=lambda path: (
            int(LAYER_OBJ_PATTERN.match(path.name).group("main")),
            int(LAYER_OBJ_PATTERN.match(path.name).group("sub")),
        ),
    )


def mesh_surface_area(mesh_path: Path) -> float:
    """Return the true 3D triangle area of one layer mesh."""
    vertices, faces = load_obj_mesh(str(mesh_path))
    triangles = vertices[faces]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    return float(0.5 * double_areas.sum())


def cleanup_stage_workers(*, stage_name: str, pause_seconds: float = 0.25) -> None:
    gc.collect()
    children = mp.active_children()
    if not children:
        print(f"[cleanup] No lingering worker processes after {stage_name}")
        return
    print(f"[cleanup] Terminating {len(children)} lingering worker process(es) after {stage_name}")
    for child in children:
        try:
            child.terminate()
        except Exception:
            pass
    deadline = time.perf_counter() + max(0.0, float(pause_seconds))
    for child in children:
        remaining = max(0.0, deadline - time.perf_counter())
        try:
            child.join(timeout=remaining)
        except Exception:
            pass
    survivors = [child for child in children if child.is_alive()]
    for child in survivors:
        try:
            child.kill()
            child.join(timeout=0.1)
        except Exception:
            pass
    gc.collect()
    print(f"[cleanup] Worker cleanup complete after {stage_name}")


def build_multi_material_input_stage(
    *,
    base_dir: Path,
    materials_cfg: list[object],
    output_dir: Path,
    clean_output: bool,
) -> tuple[Path, list[dict[str, object]]]:
    if not materials_cfg:
        raise ValueError("materials must be a non-empty list when provided.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")

    material_entries: list[dict[str, object]] = []
    main_sub_offsets: dict[int, int] = {}

    for material_index, material_obj in enumerate(materials_cfg):
        if not isinstance(material_obj, dict):
            raise ValueError("Each entry in materials must be a JSON object.")
        material_name = str(material_obj.get("name", f"material_{material_index}"))
        input_key = "input_surfaces" if "input_surfaces" in material_obj else "input_layers_dir"
        if input_key not in material_obj:
            raise ValueError("Each materials entry must define 'input_surfaces' or 'input_layers_dir'.")
        material_dir = resolve_path(base_dir, material_obj[input_key])
        layer_files = iter_layer_obj_files(material_dir)
        if not layer_files:
            raise FileNotFoundError(f"No layer meshes named like <main>_<sub>.obj were found in {material_dir}")

        grouped_by_main: dict[int, list[Path]] = {}
        for layer_file in layer_files:
            match = LAYER_OBJ_PATTERN.match(layer_file.name)
            assert match is not None
            main_value = int(match.group("main"))
            grouped_by_main.setdefault(main_value, []).append(layer_file)

        for main_value in sorted(grouped_by_main):
            current_offset = main_sub_offsets.get(main_value, 0)
            for local_sub_index, layer_file in enumerate(grouped_by_main[main_value]):
                match = LAYER_OBJ_PATTERN.match(layer_file.name)
                assert match is not None
                original_sub = int(match.group("sub"))
                combined_sub = current_offset + local_sub_index
                combined_name = f"{main_value}_{combined_sub}.obj"
                shutil.copy2(layer_file, output_dir / combined_name)
                material_entries.append(
                    {
                        "layer_main": main_value,
                        "layer_sub": combined_sub,
                        "material_name": material_name,
                        "material_index": material_index,
                        "source_dir": str(material_dir),
                        "source_mesh": layer_file.name,
                        "original_main": main_value,
                        "original_sub": original_sub,
                        "combined_mesh": combined_name,
                    }
                )
            main_sub_offsets[main_value] = current_offset + len(grouped_by_main[main_value])

    manifest_path = output_dir / "layer_material_manifest.json"
    manifest_payload = {
        "multi_material": True,
        "merge_rule": "per-main sub indices are reassigned by configured material-folder order",
        "materials": material_entries,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    print(f"Built combined multi-material input set: {output_dir}")
    print(f"Saved material manifest: {manifest_path.name}")
    return output_dir, material_entries


def _material_lookup(material_entries: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(entry["layer_main"]), int(entry["layer_sub"])): entry
        for entry in material_entries
    }


def _layer_key_from_path(path: Path) -> tuple[int, int]:
    main_str, sub_str = path.stem.split("_", maxsplit=1)
    return int(main_str), int(sub_str)


def _material_entry_for_layer(
    material_entries: list[dict[str, object]] | None,
    layer_path: Path,
) -> dict[str, object] | None:
    if not material_entries:
        return None
    return _material_lookup(material_entries).get(_layer_key_from_path(layer_path))


def _material_layer_rank(
    material_entries: list[dict[str, object]] | None,
    layer_path: Path,
) -> int | None:
    material_entry = _material_entry_for_layer(material_entries, layer_path)
    if material_entry is None:
        return None
    material_index = int(material_entry["material_index"])
    material_layers = [
        entry
        for entry in material_entries or []
        if int(entry["material_index"]) == material_index
    ]
    material_layers.sort(
        key=lambda entry: (
            int(entry.get("original_main", entry["layer_main"])),
            int(entry.get("original_sub", entry["layer_sub"])),
            int(entry["layer_main"]),
            int(entry["layer_sub"]),
        ),
    )
    target_key = (int(material_entry["layer_main"]), int(material_entry["layer_sub"]))
    for rank, entry in enumerate(material_layers):
        if (int(entry["layer_main"]), int(entry["layer_sub"])) == target_key:
            return rank
    return None


def _main_layer_ranks_from_paths(paths: list[Path]) -> dict[int, int]:
    main_indices = sorted({int(path.stem.split("_", maxsplit=1)[0]) for path in paths})
    return {main: rank for rank, main in enumerate(main_indices)}


def _main_layer_ranks_from_keys(keys: list[object]) -> dict[int, int]:
    main_indices = sorted({int(getattr(key, "main")) for key in keys})
    return {main: rank for rank, main in enumerate(main_indices)}


def _hybrid_material_overrides(
    hybrid_cfg: dict[str, object],
) -> dict[int, dict[str, object]]:
    raw_overrides = hybrid_cfg.get("material_overrides", [])
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, list):
        raise ValueError("hybrid_contours.material_overrides must be a list when provided.")

    overrides: dict[int, dict[str, object]] = {}
    for raw_override in raw_overrides:
        if not isinstance(raw_override, dict):
            raise ValueError("Each hybrid_contours.material_overrides entry must be a JSON object.")
        if "material_index" not in raw_override:
            raise ValueError("Each hybrid material override must define material_index.")
        material_index = int(raw_override["material_index"])
        entry: dict[str, object] = {}
        if raw_override.get("boundary_count") is not None:
            entry["boundary_count"] = float(raw_override["boundary_count"])
        if raw_override.get("boundary_count_factor") is not None:
            entry["boundary_count_factor"] = float(raw_override["boundary_count_factor"])
        if raw_override.get("parallel_spacing") is not None:
            entry["parallel_spacing"] = float(raw_override["parallel_spacing"])
        if raw_override.get("parallel_spacing_factor") is not None:
            entry["parallel_spacing_factor"] = float(raw_override["parallel_spacing_factor"])
        if raw_override.get("first_layer_count") is not None:
            entry["first_layer_count"] = float(raw_override["first_layer_count"])
        if raw_override.get("disable_parallel_after_first_layer_count") is not None:
            entry["disable_parallel_after_first_layer_count"] = int(raw_override["disable_parallel_after_first_layer_count"])
        if raw_override.get("first_parallel_spacing") is not None:
            entry["first_parallel_spacing"] = float(raw_override["first_parallel_spacing"])
        if raw_override.get("first_parallel_spacing_factor") is not None:
            entry["first_parallel_spacing_factor"] = float(raw_override["first_parallel_spacing_factor"])
        if raw_override.get("resample_spacing_factor") is not None:
            entry["resample_spacing_factor"] = float(raw_override["resample_spacing_factor"])
        if raw_override.get("parallel_resample_spacing") is not None:
            entry["parallel_resample_spacing"] = float(raw_override["parallel_resample_spacing"])
        if raw_override.get("parallel_resample_spacing_factor") is not None:
            entry["parallel_resample_spacing_factor"] = float(raw_override["parallel_resample_spacing_factor"])
        if raw_override.get("boundary_start_offset_count") is not None:
            entry["boundary_start_offset_count"] = int(raw_override["boundary_start_offset_count"])
        if raw_override.get("parallel_direction_mode") is not None:
            direction_mode = str(raw_override["parallel_direction_mode"]).strip().lower()
            if direction_mode not in {"connect_nearest", "one_direction"}:
                raise ValueError("hybrid_contours.material_overrides.parallel_direction_mode must be 'connect_nearest' or 'one_direction'.")
            entry["parallel_direction_mode"] = direction_mode
        overrides[material_index] = entry
    return overrides


def _load_hybrid_top_layer_override(
    *,
    hybrid_cfg: dict[str, object],
    base_dir: Path,
    generated_report_path: Path | None = None,
) -> tuple[set[str], dict[str, object]]:
    raw = hybrid_cfg.get("top_layer_overrides")
    if raw is None:
        if generated_report_path is None:
            return set(), {}
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("hybrid_contours.top_layer_overrides must be a JSON object when provided.")
    report_value = raw.get("classification_json")
    if report_value is None and generated_report_path is not None:
        report_path = generated_report_path
    elif report_value is None:
        return set(), {}
    else:
        report_path = resolve_path(base_dir, report_value)
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report_data, dict):
        raise ValueError("Top-layer classification JSON must be a JSON object.")
    raw_layers = report_data.get("layers", [])
    if not isinstance(raw_layers, list):
        raise ValueError("Top-layer classification JSON must contain a 'layers' list.")
    selected: set[str] = set()
    for item in raw_layers:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("top_like", False)):
            continue
        if "layer_main" not in item or "layer_sub" not in item:
            continue
        selected.add(f"{int(item['layer_main'])}_{int(item['layer_sub'])}")
    parsed: dict[str, object] = {
        "classification_json": str(report_path),
        "selected_layer_count": int(len(selected)),
        "width_check_enabled": bool(raw.get("width_check_enabled", False)),
        "width_check_max_factor": float(raw.get("width_check_max_factor", 2.0)),
    }
    if raw.get("boundary_count") is not None:
        parsed["boundary_count"] = int(raw["boundary_count"])
    if raw.get("boundary_count_factor") is not None:
        parsed["boundary_count_factor"] = float(raw["boundary_count_factor"])
    if raw.get("parallel_spacing") is not None:
        parsed["parallel_spacing"] = float(raw["parallel_spacing"])
    if raw.get("parallel_spacing_factor") is not None:
        parsed["parallel_spacing_factor"] = float(raw["parallel_spacing_factor"])
    if raw.get("parallel_resample_spacing") is not None:
        parsed["parallel_resample_spacing"] = float(raw["parallel_resample_spacing"])
    if raw.get("parallel_resample_spacing_factor") is not None:
        parsed["parallel_resample_spacing_factor"] = float(raw["parallel_resample_spacing_factor"])
    if raw.get("resample_spacing") is not None:
        parsed["resample_spacing"] = float(raw["resample_spacing"])
    if raw.get("resample_spacing_factor") is not None:
        parsed["resample_spacing_factor"] = float(raw["resample_spacing_factor"])
    return selected, parsed


def _run_hybrid_top_layer_classifier(
    *,
    mesh_dir: Path,
    hybrid_cfg: dict[str, object],
    base_dir: Path,
    output_dir: Path,
) -> Path | None:
    raw = hybrid_cfg.get("top_layer_classifier")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("hybrid_contours.top_layer_classifier must be a JSON object when provided.")
    if not bool(raw.get("enabled", False)):
        return None

    output_json_value = raw.get("output_json", str(output_dir / "top_layers_report.json"))
    output_json = resolve_path(base_dir, output_json_value)
    global_boundary_mesh_value = raw.get("global_boundary_mesh")
    global_boundary_mesh = (
        resolve_path(base_dir, global_boundary_mesh_value)
        if global_boundary_mesh_value is not None
        else None
    )

    mesh_files = collect_layer_files(mesh_dir, "obj")
    if not mesh_files:
        raise FileNotFoundError(f"No layer OBJ files were found in {mesh_dir}")

    print(f"[top-layer] Running classifier for {len(mesh_files)} layers")
    results = classify_layers(
        mesh_files,
        axis=str(raw.get("axis", "y")),
        lookahead_main_count=int(raw.get("lookahead_main_count", 3)),
        min_uncovered_fraction=float(raw.get("min_uncovered_fraction", 0.3)),
        k_nearest_vertices=int(raw.get("k_nearest_vertices", 12)),
        sample_mode=str(raw.get("sample_mode", "face_centroids")),
        tolerance=float(raw.get("tolerance", 1e-8)),
        layer_workers=int(raw.get("layer_workers", 1)),
        tree_query_workers=int(raw.get("tree_query_workers", 1)),
        boundary_precheck_enabled=bool(raw.get("boundary_precheck_enabled", False)),
        boundary_precheck_distance=float(raw.get("boundary_precheck_distance", 1.5)),
        boundary_precheck_min_fraction=float(raw.get("boundary_precheck_min_fraction", 0.30)),
        global_boundary_mesh=global_boundary_mesh,
        global_boundary_distance=float(raw.get("global_boundary_distance", 1.5)),
        global_boundary_min_fraction=float(raw.get("global_boundary_min_fraction", 0.30)),
        normalize_global_boundary_base_to_zero=bool(
            raw.get("normalize_global_boundary_base_to_zero", raw.get("normalize_base_to_zero", False)),
        ),
    )
    payload = {
        "mesh_dir": str(mesh_dir),
        "axis": str(raw.get("axis", "y")),
        "lookahead_main_count": int(raw.get("lookahead_main_count", 3)),
        "min_uncovered_fraction": float(raw.get("min_uncovered_fraction", 0.3)),
        "k_nearest_vertices": int(raw.get("k_nearest_vertices", 12)),
        "sample_mode": str(raw.get("sample_mode", "face_centroids")),
        "tolerance": float(raw.get("tolerance", 1e-8)),
        "layer_workers": int(raw.get("layer_workers", 1)),
        "tree_query_workers": int(raw.get("tree_query_workers", 1)),
        "boundary_precheck_enabled": bool(raw.get("boundary_precheck_enabled", False)),
        "boundary_precheck_distance": float(raw.get("boundary_precheck_distance", 1.5)),
        "boundary_precheck_min_fraction": float(raw.get("boundary_precheck_min_fraction", 0.30)),
        "global_boundary_mesh": str(global_boundary_mesh) if global_boundary_mesh is not None else None,
        "global_boundary_distance": float(raw.get("global_boundary_distance", 1.5)),
        "global_boundary_min_fraction": float(raw.get("global_boundary_min_fraction", 0.30)),
        "normalize_global_boundary_base_to_zero": bool(
            raw.get("normalize_global_boundary_base_to_zero", raw.get("normalize_base_to_zero", False)),
        ),
        "layers": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    top_like_count = sum(1 for item in results if bool(item.get("top_like", False)))
    print(f"[top-layer] Saved classification: {output_json} (top-like {top_like_count}/{len(results)})")
    return output_json


def _apply_hybrid_top_layer_override(
    *,
    layer_mesh: Path,
    boundary_count: int,
    spacing: float,
    resample_spacing: float | None,
    effective_boundary_count: int,
    effective_parallel_spacing: float | None,
    effective_resample_spacing: float | None,
    effective_parallel_resample_spacing: float | None,
    top_layer_stems: set[str],
    top_layer_override: dict[str, object],
) -> tuple[int, float | None, float | None, float | None, bool, float, dict[str, object] | None]:
    if not top_layer_override or layer_mesh.stem not in top_layer_stems:
        return (
            effective_boundary_count,
            effective_parallel_spacing,
            effective_resample_spacing,
            effective_parallel_resample_spacing,
            False,
            2.0,
            None,
        )
    if "boundary_count" in top_layer_override:
        effective_boundary_count = max(1, int(top_layer_override["boundary_count"]))
    elif "boundary_count_factor" in top_layer_override:
        effective_boundary_count = max(1, int(round(boundary_count * float(top_layer_override["boundary_count_factor"]))))
    if "parallel_spacing" in top_layer_override:
        effective_parallel_spacing = float(top_layer_override["parallel_spacing"])
    elif "parallel_spacing_factor" in top_layer_override:
        effective_parallel_spacing = float(spacing) * float(top_layer_override["parallel_spacing_factor"])
    if "parallel_resample_spacing" in top_layer_override:
        effective_parallel_resample_spacing = float(top_layer_override["parallel_resample_spacing"])
    elif "parallel_resample_spacing_factor" in top_layer_override:
        effective_parallel_resample_spacing = float(spacing) * float(top_layer_override["parallel_resample_spacing_factor"])
    if "resample_spacing" in top_layer_override:
        effective_resample_spacing = float(top_layer_override["resample_spacing"])
    elif "resample_spacing_factor" in top_layer_override:
        effective_resample_spacing = float(spacing) * float(top_layer_override["resample_spacing_factor"])
    width_check_enabled = bool(top_layer_override.get("width_check_enabled", False))
    width_check_max_factor = float(top_layer_override.get("width_check_max_factor", 2.0))
    applied = {
        "top_layer_override_applied": True,
        "top_layer_classification_json": str(top_layer_override["classification_json"]),
        "top_layer_effective_boundary_count": int(effective_boundary_count),
        "top_layer_width_check_enabled": width_check_enabled,
        "top_layer_width_check_max_factor": width_check_max_factor,
        "top_layer_effective_parallel_spacing": None if effective_parallel_spacing is None else float(effective_parallel_spacing),
        "top_layer_effective_resample_spacing": None if effective_resample_spacing is None else float(effective_resample_spacing),
        "top_layer_effective_parallel_resample_spacing": None if effective_parallel_resample_spacing is None else float(effective_parallel_resample_spacing),
    }
    return (
        effective_boundary_count,
        effective_parallel_spacing,
        effective_resample_spacing,
        effective_parallel_resample_spacing,
        width_check_enabled,
        width_check_max_factor,
        applied,
    )


def _effective_hybrid_layer_settings(
    *,
    layer_mesh: Path,
    material_entries: list[dict[str, object]] | None,
    material_overrides: dict[int, dict[str, object]],
    boundary_count: int,
    resample_spacing: float | None,
    spacing: float,
) -> tuple[int, float | None, float | None, float | None, int, str, bool, dict[str, object]]:
    material_entry = _material_entry_for_layer(material_entries, layer_mesh)
    material_layer_rank = _material_layer_rank(material_entries, layer_mesh)
    material_index = None if material_entry is None else int(material_entry["material_index"])
    effective_boundary_count = int(boundary_count)
    effective_parallel_spacing: float | None = None
    effective_resample_spacing = resample_spacing
    effective_parallel_resample_spacing = resample_spacing
    effective_boundary_start_offset_count = 0
    effective_parallel_direction_mode = "connect_nearest"
    effective_disable_parallel = False
    applied_override: dict[str, object] = {}

    if material_index is not None and material_index in material_overrides:
        override = material_overrides[material_index]
        if "boundary_count" in override:
            effective_boundary_count = int(round(override["boundary_count"]))
        elif "boundary_count_factor" in override:
            effective_boundary_count = int(round(boundary_count * override["boundary_count_factor"]))
        effective_boundary_count = max(1, effective_boundary_count)

        if "parallel_spacing" in override:
            effective_parallel_spacing = float(override["parallel_spacing"])
        elif "parallel_spacing_factor" in override:
            effective_parallel_spacing = float(spacing) * float(override["parallel_spacing_factor"])
        first_layer_count = int(override.get("first_layer_count", 0))
        first_layer_override_applied = (
            material_layer_rank is not None
            and first_layer_count > 0
            and material_layer_rank < first_layer_count
        )
        if first_layer_override_applied:
            if "first_parallel_spacing" in override:
                effective_parallel_spacing = float(override["first_parallel_spacing"])
            elif "first_parallel_spacing_factor" in override:
                effective_parallel_spacing = float(spacing) * float(override["first_parallel_spacing_factor"])

        if "parallel_resample_spacing" in override:
            effective_parallel_resample_spacing = float(override["parallel_resample_spacing"])
        elif "parallel_resample_spacing_factor" in override:
            effective_parallel_resample_spacing = float(spacing) * float(override["parallel_resample_spacing_factor"])
        if "resample_spacing_factor" in override:
            effective_resample_spacing = float(spacing) * float(override["resample_spacing_factor"])
        if "boundary_start_offset_count" in override:
            effective_boundary_start_offset_count = max(0, int(override["boundary_start_offset_count"]))
        if effective_boundary_count <= effective_boundary_start_offset_count:
            effective_boundary_count = effective_boundary_start_offset_count + 1
        if "parallel_direction_mode" in override:
            effective_parallel_direction_mode = str(override["parallel_direction_mode"])
        disable_parallel_after_first_layer_count = int(override.get("disable_parallel_after_first_layer_count", -1))
        if (
            material_layer_rank is not None
            and disable_parallel_after_first_layer_count >= 0
            and material_layer_rank >= disable_parallel_after_first_layer_count
        ):
            effective_disable_parallel = True

        applied_override = {
            "material_index": material_index,
            "material_name": material_entry.get("material_name"),
            "material_layer_rank": material_layer_rank,
            "base_boundary_count": int(boundary_count),
            "effective_boundary_count": int(effective_boundary_count),
            "base_parallel_spacing": float(spacing),
            "effective_parallel_spacing": None if effective_parallel_spacing is None else float(effective_parallel_spacing),
            "first_layer_count": first_layer_count,
            "first_layer_override_applied": first_layer_override_applied,
            "base_resample_spacing": None if resample_spacing is None else float(resample_spacing),
            "effective_resample_spacing": None if effective_resample_spacing is None else float(effective_resample_spacing),
            "effective_parallel_resample_spacing": None if effective_parallel_resample_spacing is None else float(effective_parallel_resample_spacing),
            "effective_boundary_start_offset_count": int(effective_boundary_start_offset_count),
            "effective_parallel_direction_mode": effective_parallel_direction_mode,
            "disable_parallel_after_first_layer_count": disable_parallel_after_first_layer_count if disable_parallel_after_first_layer_count >= 0 else None,
            "effective_disable_parallel": bool(effective_disable_parallel),
            "override": override,
        }

    return (
        effective_boundary_count,
        effective_parallel_spacing,
        effective_resample_spacing,
        effective_parallel_resample_spacing,
        effective_boundary_start_offset_count,
        effective_parallel_direction_mode,
        effective_disable_parallel,
        applied_override,
    )


def attach_material_metadata(
    *,
    target_dir: Path,
    metadata_filename: str,
    material_entries: list[dict[str, object]],
) -> None:
    metadata_path = target_dir / metadata_filename
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        return

    manifest_filename = "layer_material_manifest.json"
    (target_dir / manifest_filename).write_text(
        json.dumps(
            {
                "multi_material": True,
                "materials": material_entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lookup = _material_lookup(material_entries)
    metadata["multi_material"] = True
    metadata["layer_material_manifest_file"] = manifest_filename

    layers = metadata.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            key = (int(layer.get("layer_main", -1)), int(layer.get("layer_sub", -1)))
            material_entry = lookup.get(key)
            if material_entry is None:
                continue
            layer["material_name"] = material_entry["material_name"]
            layer["material_index"] = material_entry["material_index"]
            layer["source_material_dir"] = material_entry["source_dir"]
            layer["original_source_mesh"] = material_entry["source_mesh"]
            layer["original_main"] = material_entry["original_main"]
            layer["original_sub"] = material_entry["original_sub"]

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_remesh_stage(
    *,
    input_dir: Path,
    output_dir: Path,
    remesh_cfg: dict[str, object],
    clean_output: bool,
) -> Path:
    enabled = bool(remesh_cfg.get("enabled", True))
    min_input_surface_area = float(remesh_cfg.get("min_input_surface_area", 0.0))
    if min_input_surface_area < 0.0:
        raise ValueError("remesh.min_input_surface_area must be non-negative.")
    if not enabled:
        if min_input_surface_area > 0.0:
            print(
                "Remesh disabled; ignoring remesh.min_input_surface_area because input layers must remain intact.",
            )
        print(f"Remesh disabled; using input surfaces directly: {input_dir}")
        return input_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    obj_files = iter_remesh_obj_files(input_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {input_dir}")

    if min_input_surface_area > 0.0:
        retained_files: list[Path] = []
        for obj_path in obj_files:
            area = mesh_surface_area(obj_path)
            if area < min_input_surface_area:
                print(
                    f"Skipped small-area layer: {obj_path.name} "
                    f"(area={area:.6g} < {min_input_surface_area:.6g})",
                )
            else:
                retained_files.append(obj_path)
        print(
            f"Input-area filter retained {len(retained_files)} / {len(obj_files)} OBJ files "
            f"(minimum={min_input_surface_area:.6g}).",
        )
        obj_files = retained_files
        if not obj_files:
            raise ValueError("remesh.min_input_surface_area excluded every OBJ file.")

    suffix = str(remesh_cfg.get("suffix", ""))
    iterations = int(remesh_cfg.get("iterations", 20))
    adaptive = bool(remesh_cfg.get("adaptive", False))
    selectedonly = bool(remesh_cfg.get("selectedonly", False))
    targetlen_pct = float(remesh_cfg.get("targetlen_pct", 0.5))
    targetlen_abs_raw = remesh_cfg.get("targetlen_abs")
    targetlen_abs = None if targetlen_abs_raw is None else float(targetlen_abs_raw)
    featuredeg = float(remesh_cfg.get("featuredeg", 30.0))
    checksurfdist = bool(remesh_cfg.get("check_surfdist", False))
    max_surfdist_pct = float(remesh_cfg.get("max_surfdist_pct", 1.4142))
    splitflag = bool(remesh_cfg.get("splitflag", True))
    collapseflag = bool(remesh_cfg.get("collapseflag", True))
    swapflag = bool(remesh_cfg.get("swapflag", True))
    smoothflag = bool(remesh_cfg.get("smoothflag", True))
    reprojectflag = bool(remesh_cfg.get("reprojectflag", True))
    save_vertex_normal = bool(remesh_cfg.get("save_vertex_normal", False))
    post_smooth_iterations = int(remesh_cfg.get("post_smooth_iterations", 0))
    post_smooth_alpha = float(remesh_cfg.get("post_smooth_alpha", 0.2))
    report_timing = bool(remesh_cfg.get("report_timing", False))
    file_workers = int(remesh_cfg.get("file_workers", 1))
    if file_workers <= 0:
        raise ValueError("remesh.file_workers must be positive.")
    if post_smooth_iterations < 0:
        raise ValueError("remesh.post_smooth_iterations must be non-negative.")

    print(f"Remeshing {len(obj_files)} OBJ files into {output_dir}")
    print(f"Remesh file workers: {file_workers}")
    if targetlen_abs is None:
        print(f"Remesh target length mode: percentage ({targetlen_pct})")
    else:
        print(f"Remesh target length mode: absolute ({targetlen_abs})")
    if post_smooth_iterations > 0:
        print(
            f"Remesh post-smoothing enabled: iterations={post_smooth_iterations}, "
            f"alpha={post_smooth_alpha}",
        )
    if report_timing:
        print("Remesh timing report enabled (load, preparation, native filter, and save per layer).")
    if file_workers == 1:
        for obj_path in obj_files:
            output_path = output_dir / f"{obj_path.stem}{suffix}.obj"
            v0, f0, v1, f1 = remesh_file(
                obj_path,
                output_path,
                iterations=iterations,
                adaptive=adaptive,
                selectedonly=selectedonly,
                targetlen_pct=targetlen_pct,
                targetlen_abs=targetlen_abs,
                featuredeg=featuredeg,
                checksurfdist=checksurfdist,
                max_surfdist_pct=max_surfdist_pct,
                splitflag=splitflag,
                collapseflag=collapseflag,
                swapflag=swapflag,
                smoothflag=smoothflag,
                reprojectflag=reprojectflag,
                save_vertex_normal=save_vertex_normal,
                post_smooth_iterations=post_smooth_iterations,
                post_smooth_alpha=post_smooth_alpha,
                report_timing=report_timing,
            )
            print(f"Remeshed: {obj_path.name}: V {v0} -> {v1}, F {f0} -> {f1}")
    else:
        from batch_meshlab_isotropic_remesh import _remesh_job
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=file_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    _remesh_job,
                    obj_path=obj_path,
                    output_path=output_dir / f"{obj_path.stem}{suffix}.obj",
                    iterations=iterations,
                    adaptive=adaptive,
                    selectedonly=selectedonly,
                    targetlen_pct=targetlen_pct,
                    targetlen_abs=targetlen_abs,
                    featuredeg=featuredeg,
                    checksurfdist=checksurfdist,
                    max_surfdist_pct=max_surfdist_pct,
                    splitflag=splitflag,
                    collapseflag=collapseflag,
                    swapflag=swapflag,
                    smoothflag=smoothflag,
                    reprojectflag=reprojectflag,
                    save_vertex_normal=save_vertex_normal,
                    post_smooth_iterations=post_smooth_iterations,
                    post_smooth_alpha=post_smooth_alpha,
                    report_timing=report_timing,
                )
                for obj_path in obj_files
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda item: item[0].name)
        for obj_path, v0, f0, v1, f1 in results:
            print(f"Remeshed: {obj_path.name}: V {v0} -> {v1}, F {f0} -> {f1}")

    return output_dir


def run_support_expansion_stage(
    *,
    input_dir: Path,
    output_dir: Path,
    expand_cfg: dict[str, object],
    clean_output: bool,
) -> Path:
    enabled = bool(expand_cfg.get("enabled", False))
    if not enabled:
        print(f"Support-mesh expansion disabled; using contour meshes for heights: {input_dir}")
        return input_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    obj_files = iter_expand_obj_files(input_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {input_dir}")

    suffix = str(expand_cfg.get("suffix", ""))
    offset = float(expand_cfg.get("offset", 0.0))
    print(f"Expanding support boundaries for {len(obj_files)} OBJ files into {output_dir}")
    for obj_path in obj_files:
        V, F = load_obj_mesh(str(obj_path))
        expanded_V = expand_boundary_vertices(V, F, offset_distance=offset)
        output_path = output_dir / f"{obj_path.stem}{suffix}.obj"
        save_obj_mesh(output_path, expanded_V, F)
        print(f"Expanded: {obj_path.name} -> {output_path.name}")

    return output_dir


def run_path_stage(
    *,
    mesh_dir: Path,
    output_dir: Path,
    contour_cfg: dict[str, object],
    clean_output: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    layer_meshes = iter_layer_meshes(mesh_dir)
    if not layer_meshes:
        raise FileNotFoundError(f"No layer meshes named like <main>_<sub>.obj were found in {mesh_dir}")

    spacing = float(contour_cfg["spacing"])
    outer_start_mode = str(contour_cfg.get("outer_start_mode", "boundary"))
    heat_params = HeatMethodParams(
        timestep_scale=float(contour_cfg.get("timestep_scale", 1.0)),
        use_direct_solver=not bool(contour_cfg.get("iterative", False)),
    )
    prefer_libigl = bool(contour_cfg.get("prefer_libigl", True))
    heat_backend = str(contour_cfg.get("heat_backend", "native")).strip().lower()
    if heat_backend not in {"native", "potpourri3d"}:
        raise ValueError("contours.heat_backend must be either 'native' or 'potpourri3d'.")
    max_level = contour_cfg.get("max_level")
    max_level = None if max_level is None else float(max_level)
    max_level_fraction = contour_cfg.get("max_level_fraction")
    max_level_fraction = None if max_level_fraction is None else float(max_level_fraction)
    resample_spacing = contour_cfg.get("resample_spacing")
    resample_spacing = None if resample_spacing is None else float(resample_spacing)
    sharp_angle_deg = float(contour_cfg.get("sharp_angle_deg", 35.0))
    min_contour_points = int(contour_cfg.get("min_contour_points", 4))
    mild_smooth_all = bool(contour_cfg.get("mild_smooth_all", False))
    mild_smooth_iterations = int(contour_cfg.get("mild_smooth_iterations", 8))
    mild_smooth_alpha = float(contour_cfg.get("mild_smooth_alpha", 0.2))
    smooth_level_frac_from = contour_cfg.get("smooth_level_frac_from")
    smooth_level_frac_from = None if smooth_level_frac_from is None else float(smooth_level_frac_from)
    smooth_level_frac_to = contour_cfg.get("smooth_level_frac_to")
    smooth_level_frac_to = None if smooth_level_frac_to is None else float(smooth_level_frac_to)
    smooth_iterations = int(contour_cfg.get("smooth_iterations", 30))
    smooth_alpha = float(contour_cfg.get("smooth_alpha", 0.75))
    allow_branching_boundary_fallback = bool(contour_cfg.get("allow_branching_boundary_fallback", True))
    first_layers_lowest_y_start_count = int(
        contour_cfg.get(
            "first_layers_lowest_y_start_count",
            contour_cfg.get("first_layers_lowest_z_start_count", 0),
        ),
    )
    if first_layers_lowest_y_start_count < 0:
        raise ValueError("contours.first_layers_lowest_y_start_count must be non-negative.")
    layer_workers = int(contour_cfg.get("layer_workers", 1))
    if layer_workers <= 0:
        raise ValueError("contours.layer_workers must be positive.")
    model_lift = float(contour_cfg.get("model_lift", 0.0))
    model_lift_axis = str(contour_cfg.get("model_lift_axis", "y")).strip().lower()
    if model_lift_axis not in {"x", "y", "z"}:
        raise ValueError("contours.model_lift_axis must be one of 'x', 'y', or 'z'.")

    print(f"Generating connected paths for {len(layer_meshes)} layers into {output_dir}")
    print(f"Contour layer workers: {layer_workers}")
    layer_outputs: list[dict[str, object]] = []
    main_layer_ranks = _main_layer_ranks_from_paths(layer_meshes)
    if layer_workers == 1:
        for layer_mesh in layer_meshes:
            main_index = int(layer_mesh.stem.split("_", maxsplit=1)[0])
            start_selection_mode = (
                "lowest_y"
                if main_layer_ranks[main_index] < first_layers_lowest_y_start_count
                else "lexicographic"
            )
            output_path = process_layer(
                layer_mesh,
                output_dir,
                spacing=spacing,
                heat_params=heat_params,
                prefer_libigl=prefer_libigl,
                heat_backend=heat_backend,
                max_level=max_level,
                max_level_fraction=max_level_fraction,
                resample_spacing=resample_spacing,
                sharp_angle_deg=sharp_angle_deg,
                min_contour_points=min_contour_points,
                mild_smooth_all=mild_smooth_all,
                mild_smooth_iterations=mild_smooth_iterations,
                mild_smooth_alpha=mild_smooth_alpha,
                smooth_level_frac_from=smooth_level_frac_from,
                smooth_level_frac_to=smooth_level_frac_to,
                smooth_iterations=smooth_iterations,
                smooth_alpha=smooth_alpha,
                outer_start_mode=outer_start_mode,
                start_selection_mode=start_selection_mode,
                allow_branching_boundary_fallback=allow_branching_boundary_fallback,
                model_lift=model_lift,
                model_lift_axis=model_lift_axis,
            )
            main_str, sub_str = layer_mesh.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": layer_mesh.name,
                    "path_file": output_path.name,
                }
            )
            print(f"Saved path: {output_path.name}")
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    process_layer,
                    layer_mesh,
                    output_dir,
                    spacing=spacing,
                    heat_params=heat_params,
                    prefer_libigl=prefer_libigl,
                    heat_backend=heat_backend,
                    max_level=max_level,
                    max_level_fraction=max_level_fraction,
                    resample_spacing=resample_spacing,
                    sharp_angle_deg=sharp_angle_deg,
                    min_contour_points=min_contour_points,
                    mild_smooth_all=mild_smooth_all,
                    mild_smooth_iterations=mild_smooth_iterations,
                    mild_smooth_alpha=mild_smooth_alpha,
                    smooth_level_frac_from=smooth_level_frac_from,
                    smooth_level_frac_to=smooth_level_frac_to,
                    smooth_iterations=smooth_iterations,
                    smooth_alpha=smooth_alpha,
                    outer_start_mode=outer_start_mode,
                    start_selection_mode=(
                        "lowest_y"
                        if main_layer_ranks[int(layer_mesh.stem.split("_", maxsplit=1)[0])] < first_layers_lowest_y_start_count
                        else "lexicographic"
                    ),
                    allow_branching_boundary_fallback=allow_branching_boundary_fallback,
                    model_lift=model_lift,
                    model_lift_axis=model_lift_axis,
                )
                for layer_mesh in layer_meshes
            ]
            output_paths = [future.result() for future in futures]
        output_paths.sort(key=lambda path: (int(path.stem.split("_", maxsplit=1)[0]), int(path.stem.split("_", maxsplit=1)[1])))
        for output_path in output_paths:
            main_str, sub_str = output_path.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": f"{output_path.stem}.obj",
                    "path_file": output_path.name,
                }
            )
            print(f"Saved path: {output_path.name}")

    batch_metadata = _build_batch_metadata(
        input_dir=mesh_dir,
        output_dir=output_dir,
        spacing=spacing,
        heat_params=heat_params,
        prefer_libigl=prefer_libigl,
        heat_backend=heat_backend,
        max_level=max_level,
        max_level_fraction=max_level_fraction,
        resample_spacing=resample_spacing,
        sharp_angle_deg=sharp_angle_deg,
        min_contour_points=min_contour_points,
        mild_smooth_all=mild_smooth_all,
        mild_smooth_iterations=mild_smooth_iterations,
        mild_smooth_alpha=mild_smooth_alpha,
        smooth_level_frac_from=smooth_level_frac_from,
        smooth_level_frac_to=smooth_level_frac_to,
        smooth_iterations=smooth_iterations,
        smooth_alpha=smooth_alpha,
        outer_start_mode=outer_start_mode,
        first_layers_lowest_y_start_count=first_layers_lowest_y_start_count,
        allow_branching_boundary_fallback=allow_branching_boundary_fallback,
        model_lift=model_lift,
        model_lift_axis=model_lift_axis,
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = output_dir / "batch_paths.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved path metadata: {batch_metadata_path.name}")
    return output_dir


def run_hybrid_path_stage(
    *,
    mesh_dir: Path,
    output_dir: Path,
    hybrid_cfg: dict[str, object],
    clean_output: bool,
    base_dir: Path,
    material_entries: list[dict[str, object]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    layer_meshes = iter_layer_meshes(mesh_dir)
    if not layer_meshes:
        raise FileNotFoundError(f"No layer meshes named like <main>_<sub>.obj were found in {mesh_dir}")

    spacing = float(hybrid_cfg["spacing"])
    resample_spacing_raw = hybrid_cfg.get("resample_spacing")
    resample_spacing = None if resample_spacing_raw is None else float(resample_spacing_raw)
    boundary_count = int(hybrid_cfg["boundary_count"])
    boundary_field_backend = str(hybrid_cfg.get("boundary_field_backend", "native")).strip().lower()
    if boundary_field_backend not in {"native", "potpourri3d_heat", "potpourri_fast_marching_unsigned"}:
        raise ValueError("hybrid_contours.boundary_field_backend must be one of 'native', 'potpourri3d_heat', or 'potpourri_fast_marching_unsigned'.")
    parallel_field_backend = str(hybrid_cfg.get("parallel_field_backend", "signed_heat")).strip().lower()
    if parallel_field_backend not in {"signed_heat", "fast_marching_unsigned", "vector_heat_stripe"}:
        raise ValueError("hybrid_contours.parallel_field_backend must be one of 'signed_heat', 'fast_marching_unsigned', or 'vector_heat_stripe'.")
    axis_mode = str(hybrid_cfg.get("axis_mode", "fixed")).strip().lower()
    if axis_mode not in {"fixed", "fixed_angle", "alternating_angle", "auto_tangent_alternating"}:
        raise ValueError("hybrid_contours.axis_mode must be 'fixed', 'fixed_angle', 'alternating_angle', or 'auto_tangent_alternating'.")
    axis = str(hybrid_cfg.get("axis", "x"))
    hatch_angle_degrees = float(hybrid_cfg.get("hatch_angle_degrees", 0.0))
    plane_value_raw = hybrid_cfg.get("plane_value")
    plane_value = None if plane_value_raw is None else float(plane_value_raw)
    level_set_constraint = str(hybrid_cfg.get("level_set_constraint", "None"))
    soft_level_set_weight = float(hybrid_cfg.get("soft_level_set_weight", -1.0))
    contour_max_level_fraction_raw = hybrid_cfg.get("contour_max_level_fraction", 1.0)
    contour_max_level_fraction = None if contour_max_level_fraction_raw is None else float(contour_max_level_fraction_raw)
    min_polyline_points = int(hybrid_cfg.get("min_polyline_points", 2))
    sharp_angle_deg = float(hybrid_cfg.get("sharp_angle_deg", 35.0))
    mild_smooth_all = bool(hybrid_cfg.get("mild_smooth_all", False))
    mild_smooth_iterations = int(hybrid_cfg.get("mild_smooth_iterations", 8))
    mild_smooth_alpha = float(hybrid_cfg.get("mild_smooth_alpha", 0.2))
    use_vectorized_extractor = bool(hybrid_cfg.get("use_vectorized_extractor", False))
    model_lift = float(hybrid_cfg.get("model_lift", 0.0))
    model_lift_axis = str(hybrid_cfg.get("model_lift_axis", "y")).strip().lower()
    if model_lift_axis not in {"x", "y", "z"}:
        raise ValueError("hybrid_contours.model_lift_axis must be one of 'x', 'y', or 'z'.")
    first_layers_lowest_y_start_count = int(
        hybrid_cfg.get(
            "first_layers_lowest_y_start_count",
            hybrid_cfg.get("first_layers_lowest_z_start_count", 0),
        ),
    )
    if first_layers_lowest_y_start_count < 0:
        raise ValueError("hybrid_contours.first_layers_lowest_y_start_count must be non-negative.")
    layer_workers = int(hybrid_cfg.get("layer_workers", 1))
    if layer_workers <= 0:
        raise ValueError("hybrid_contours.layer_workers must be positive.")
    material_overrides = _hybrid_material_overrides(hybrid_cfg)
    generated_top_layer_report = _run_hybrid_top_layer_classifier(
        mesh_dir=mesh_dir,
        hybrid_cfg=hybrid_cfg,
        base_dir=base_dir,
        output_dir=output_dir,
    )
    top_layer_stems, top_layer_override = _load_hybrid_top_layer_override(
        hybrid_cfg=hybrid_cfg,
        base_dir=base_dir,
        generated_report_path=generated_top_layer_report,
    )
    planar_infill_raw = hybrid_cfg.get("planar_infill")
    if planar_infill_raw is not None and not isinstance(planar_infill_raw, dict):
        raise ValueError("hybrid_contours.planar_infill must be a JSON object when provided.")
    planar_infill_base = dict(planar_infill_raw) if isinstance(planar_infill_raw, dict) else {}
    planar_infill_enabled = bool(planar_infill_base.get("enabled", False))
    ironing_raw = hybrid_cfg.get("ironing")
    if ironing_raw is not None and not isinstance(ironing_raw, dict):
        raise ValueError("hybrid_contours.ironing must be a JSON object when provided.")
    ironing_base = dict(ironing_raw) if isinstance(ironing_raw, dict) else {}
    ironing_enabled = bool(ironing_base.get("enabled", False))
    if planar_infill_enabled and str(hybrid_cfg.get("planner_backend", "")).strip().lower() != "xz_projected_2d":
        raise ValueError("planar_infill is currently supported only with planner_backend='xz_projected_2d'.")
    if ironing_enabled and str(hybrid_cfg.get("planner_backend", "")).strip().lower() != "xz_projected_2d":
        raise ValueError("ironing is currently supported only with planner_backend='xz_projected_2d'.")
    projected_width_raw = hybrid_cfg.get("projected_width_check")
    if projected_width_raw is not None and not isinstance(projected_width_raw, dict):
        raise ValueError("hybrid_contours.projected_width_check must be a JSON object when provided.")
    projected_width_cfg = dict(projected_width_raw) if isinstance(projected_width_raw, dict) else {}
    projected_width_enabled = bool(projected_width_cfg.get("enabled", False))
    if projected_width_enabled and str(hybrid_cfg.get("planner_backend", "")).strip().lower() != "xz_projected_2d":
        raise ValueError("projected_width_check is currently supported only with planner_backend='xz_projected_2d'.")

    print(f"Generating hybrid connected paths for {len(layer_meshes)} layers into {output_dir}")
    print(f"Hybrid layer workers: {layer_workers}")
    print(f"Hybrid vectorized extractor: {use_vectorized_extractor}")
    if material_overrides:
        print(f"Hybrid material overrides: {material_overrides}")
    if top_layer_override:
        print(f"Hybrid top-layer overrides: {top_layer_override} (selected layers={len(top_layer_stems)})")
    layer_outputs: list[dict[str, object]] = []
    plane_value_mode = "fixed" if plane_value is not None else "midpoint_extent"
    main_layer_ranks = _main_layer_ranks_from_paths(layer_meshes)
    solid_base_main_layers = int(planar_infill_base.get("solid_base_main_layers", 0)) if planar_infill_enabled else 0
    solid_top_main_layers = int(planar_infill_base.get("solid_top_main_layers", 0)) if planar_infill_enabled else 0
    if solid_base_main_layers < 0 or solid_top_main_layers < 0:
        raise ValueError("planar_infill solid layer counts must be non-negative.")
    ordered_mains = sorted(main_layer_ranks, key=main_layer_ranks.get)
    solid_base_mains = set(ordered_mains[:solid_base_main_layers])
    detected_top_mains = {int(stem.split("_", maxsplit=1)[0]) for stem in top_layer_stems}
    solid_top_mains: set[int] = set()
    for main in detected_top_mains:
        rank = main_layer_ranks.get(main)
        if rank is not None:
            solid_top_mains.update(ordered_mains[max(0, rank - solid_top_main_layers + 1) : rank + 1])

    ironing_target = str(ironing_base.get("target", "top_classified")).strip().lower()
    if ironing_target == "top_classified":
        ironing_mains = detected_top_mains
    elif ironing_target == "last_main_layers":
        count = int(ironing_base.get("main_layer_count", 1))
        if count < 0:
            raise ValueError("ironing.main_layer_count must be non-negative.")
        ironing_mains = set(ordered_mains[-count:]) if count else set()
    else:
        raise ValueError("ironing.target must be 'top_classified' or 'last_main_layers'.")

    projected_width_target = str(projected_width_cfg.get("target", "last_main_layers")).strip().lower()
    if projected_width_target == "last_main_layers":
        count = int(projected_width_cfg.get("main_layer_count", 3))
        if count < 0:
            raise ValueError("projected_width_check.main_layer_count must be non-negative.")
        projected_width_mains = set(ordered_mains[-count:]) if count else set()
    elif projected_width_target == "top_classified":
        projected_width_mains = detected_top_mains
    else:
        raise ValueError("projected_width_check.target must be 'last_main_layers' or 'top_classified'.")
    projected_width_max_factor = float(projected_width_cfg.get("max_factor", 2.0))
    projected_width_parallel_spacing_raw = projected_width_cfg.get("parallel_spacing")
    projected_width_parallel_spacing = (
        None
        if projected_width_parallel_spacing_raw is None
        else float(projected_width_parallel_spacing_raw)
    )
    if projected_width_enabled and projected_width_max_factor <= 0.0:
        raise ValueError("projected_width_check.max_factor must be positive.")
    if (
        projected_width_enabled
        and projected_width_parallel_spacing is not None
        and projected_width_parallel_spacing <= 0.0
    ):
        raise ValueError("projected_width_check.parallel_spacing must be positive or null.")

    def planar_infill_for_layer(layer_mesh: Path) -> dict[str, object] | None:
        attachment=hybrid_cfg.get('infill_wall_overlap')
        # Carry model controls through both serial and spawned projected workers.
        # Legacy surface planners ignore these additional dictionary keys.
        model_controls={}
        if attachment is not None:model_controls['wall_attachment_overlap']=attachment
        if 'model_half_width_inset' in hybrid_cfg:
            model_controls['model_half_width_inset']=hybrid_cfg['model_half_width_inset']
        if 'point_reduction' in hybrid_cfg:
            model_controls['point_reduction']=hybrid_cfg['point_reduction']
        dense_controls=dict(enabled=False,**model_controls) if model_controls else None
        if not planar_infill_enabled:
            return dense_controls
        main = int(layer_mesh.stem.split("_", maxsplit=1)[0])
        # Dense base and top bands retain the established hybrid planner:
        # boundary contours followed by its greedily chained inner raster.
        # Planar sparse infill is only a middle-layer replacement.
        if main in solid_base_mains or main in solid_top_mains:
            return dense_controls
        values = dict(planar_infill_base)
        values["solid"] = False
        values["band"] = "sparse"
        values.update(model_controls)
        return values

    def ironing_for_layer(layer_mesh: Path) -> dict[str, object] | None:
        if not ironing_enabled:
            return None
        main = int(layer_mesh.stem.split("_", maxsplit=1)[0])
        return dict(ironing_base) if main in ironing_mains else None

    def projected_width_check_for_layer(layer_mesh: Path) -> tuple[bool, float, float | None]:
        main = int(layer_mesh.stem.split("_", maxsplit=1)[0])
        return (
            projected_width_enabled and main in projected_width_mains,
            projected_width_max_factor,
            projected_width_parallel_spacing,
        )

    if layer_workers == 1:
        for layer_mesh in layer_meshes:
            main_index = int(layer_mesh.stem.split("_", maxsplit=1)[0])
            start_selection_mode = (
                "lowest_y"
                if main_layer_ranks[main_index] < first_layers_lowest_y_start_count
                else "first"
            )
            (
                effective_boundary_count,
                effective_parallel_spacing,
                effective_resample_spacing,
                effective_parallel_resample_spacing,
                effective_boundary_start_offset_count,
                effective_parallel_direction_mode,
                effective_disable_parallel,
                material_override_meta,
            ) = _effective_hybrid_layer_settings(
                layer_mesh=layer_mesh,
                material_entries=material_entries,
                material_overrides=material_overrides,
                boundary_count=boundary_count,
                resample_spacing=resample_spacing,
                spacing=spacing,
            )
            (
                effective_boundary_count,
                effective_parallel_spacing,
                effective_resample_spacing,
                effective_parallel_resample_spacing,
                top_layer_width_check_enabled,
                top_layer_width_check_max_factor,
                top_layer_override_meta,
            ) = _apply_hybrid_top_layer_override(
                layer_mesh=layer_mesh,
                boundary_count=boundary_count,
                spacing=spacing,
                resample_spacing=resample_spacing,
                effective_boundary_count=effective_boundary_count,
                effective_parallel_spacing=effective_parallel_spacing,
                effective_resample_spacing=effective_resample_spacing,
                effective_parallel_resample_spacing=effective_parallel_resample_spacing,
                top_layer_stems=top_layer_stems,
                top_layer_override=top_layer_override,
            )
            (
                selected_width_check,
                selected_width_factor,
                selected_width_parallel_spacing,
            ) = projected_width_check_for_layer(layer_mesh)
            if selected_width_check:
                top_layer_width_check_enabled = True
                top_layer_width_check_max_factor = selected_width_factor
                if selected_width_parallel_spacing is not None:
                    # This is both the raster pitch and its nominal deposited width.
                    # Keeping them equal prevents an intentional under-filled band.
                    effective_parallel_spacing = selected_width_parallel_spacing
            output_path, layer_meta = process_hybrid_layer(
                layer_mesh,
                output_dir,
                spacing=spacing,
                parallel_spacing=effective_parallel_spacing,
                resample_spacing=effective_resample_spacing,
                parallel_resample_spacing=effective_parallel_resample_spacing,
                boundary_count=effective_boundary_count,
                sharp_angle_deg=sharp_angle_deg,
                boundary_field_backend=boundary_field_backend,
                parallel_field_backend=parallel_field_backend,
                axis_mode=axis_mode,
                axis=axis,
                hatch_angle_degrees=hatch_angle_degrees,
                plane_value=plane_value,
                level_set_constraint=level_set_constraint,
                soft_level_set_weight=soft_level_set_weight,
                contour_max_level_fraction=contour_max_level_fraction,
                min_polyline_points=min_polyline_points,
                mild_smooth_all=mild_smooth_all,
                mild_smooth_iterations=mild_smooth_iterations,
                mild_smooth_alpha=mild_smooth_alpha,
                start_selection_mode=start_selection_mode,
                boundary_start_offset_count=effective_boundary_start_offset_count,
                parallel_direction_mode=effective_parallel_direction_mode,
                disable_parallel=effective_disable_parallel,
                top_layer_width_check_enabled=top_layer_width_check_enabled,
                top_layer_width_check_max_factor=top_layer_width_check_max_factor,
                is_top_layer=bool(top_layer_override_meta),
                use_vectorized_extractor=use_vectorized_extractor,
                model_lift=model_lift,
                model_lift_axis=model_lift_axis,
                planar_infill=planar_infill_for_layer(layer_mesh),
                **({"ironing": ironing_for_layer(layer_mesh)} if ironing_enabled else {}),
            )
            main_str, sub_str = layer_mesh.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": layer_mesh.name,
                    "path_file": output_path.name,
                    "material_path_override": material_override_meta or None,
                    "start_selection_mode": start_selection_mode,
                    "boundary_start_offset_count": effective_boundary_start_offset_count,
                    "parallel_direction_mode": effective_parallel_direction_mode,
                    "parallel_disabled": bool(effective_disable_parallel),
                    "top_layer_path_override": top_layer_override_meta or None,
                    "use_vectorized_extractor": bool(use_vectorized_extractor),
                    "planar_infill": planar_infill_for_layer(layer_mesh),
                    "ironing": ironing_for_layer(layer_mesh),
                    **layer_meta,
                }
            )
            print(f"Saved hybrid path: {output_path.name}")
    else:
        from concurrent.futures import ProcessPoolExecutor

        job_specs = []
        for layer_mesh in layer_meshes:
            (
                effective_boundary_count,
                effective_parallel_spacing,
                effective_resample_spacing,
                effective_parallel_resample_spacing,
                effective_boundary_start_offset_count,
                effective_parallel_direction_mode,
                effective_disable_parallel,
                material_override_meta,
            ) = _effective_hybrid_layer_settings(
                layer_mesh=layer_mesh,
                material_entries=material_entries,
                material_overrides=material_overrides,
                boundary_count=boundary_count,
                resample_spacing=resample_spacing,
                spacing=spacing,
            )
            (
                effective_boundary_count,
                effective_parallel_spacing,
                effective_resample_spacing,
                effective_parallel_resample_spacing,
                top_layer_width_check_enabled,
                top_layer_width_check_max_factor,
                top_layer_override_meta,
            ) = _apply_hybrid_top_layer_override(
                layer_mesh=layer_mesh,
                boundary_count=boundary_count,
                spacing=spacing,
                resample_spacing=resample_spacing,
                effective_boundary_count=effective_boundary_count,
                effective_parallel_spacing=effective_parallel_spacing,
                effective_resample_spacing=effective_resample_spacing,
                effective_parallel_resample_spacing=effective_parallel_resample_spacing,
                top_layer_stems=top_layer_stems,
                top_layer_override=top_layer_override,
            )
            (
                selected_width_check,
                selected_width_factor,
                selected_width_parallel_spacing,
            ) = projected_width_check_for_layer(layer_mesh)
            if selected_width_check:
                top_layer_width_check_enabled = True
                top_layer_width_check_max_factor = selected_width_factor
                if selected_width_parallel_spacing is not None:
                    # See the serial branch: pitch and nominal width must match.
                    effective_parallel_spacing = selected_width_parallel_spacing
            job_specs.append(
                {
                    "layer_mesh": layer_mesh,
                    "boundary_count": effective_boundary_count,
                    "parallel_spacing": effective_parallel_spacing,
                    "resample_spacing": effective_resample_spacing,
                    "parallel_resample_spacing": effective_parallel_resample_spacing,
                    "boundary_start_offset_count": effective_boundary_start_offset_count,
                    "parallel_direction_mode": effective_parallel_direction_mode,
                    "disable_parallel": effective_disable_parallel,
                    "top_layer_width_check_enabled": top_layer_width_check_enabled,
                    "top_layer_width_check_max_factor": top_layer_width_check_max_factor,
                    "is_top_layer": bool(top_layer_override_meta),
                    "use_vectorized_extractor": use_vectorized_extractor,
                    "planar_infill": planar_infill_for_layer(layer_mesh),
                    "ironing": ironing_for_layer(layer_mesh),
                    "material_override_meta": material_override_meta,
                    "top_layer_override_meta": top_layer_override_meta,
                    "start_selection_mode": (
                        "lowest_y"
                        if main_layer_ranks[int(layer_mesh.stem.split("_", maxsplit=1)[0])] < first_layers_lowest_y_start_count
                        else "first"
                    ),
                }
            )

        with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    process_hybrid_layer,
                    job["layer_mesh"],
                    output_dir,
                    spacing=spacing,
                    parallel_spacing=job["parallel_spacing"],
                    resample_spacing=job["resample_spacing"],
                    parallel_resample_spacing=job["parallel_resample_spacing"],
                    boundary_count=job["boundary_count"],
                    sharp_angle_deg=sharp_angle_deg,
                    boundary_field_backend=boundary_field_backend,
                    parallel_field_backend=parallel_field_backend,
                    axis_mode=axis_mode,
                    axis=axis,
                    hatch_angle_degrees=hatch_angle_degrees,
                    plane_value=plane_value,
                    level_set_constraint=level_set_constraint,
                    soft_level_set_weight=soft_level_set_weight,
                    contour_max_level_fraction=contour_max_level_fraction,
                    min_polyline_points=min_polyline_points,
                    mild_smooth_all=mild_smooth_all,
                    mild_smooth_iterations=mild_smooth_iterations,
                    mild_smooth_alpha=mild_smooth_alpha,
                    start_selection_mode=job["start_selection_mode"],
                    boundary_start_offset_count=job["boundary_start_offset_count"],
                    parallel_direction_mode=job["parallel_direction_mode"],
                    disable_parallel=job["disable_parallel"],
                    top_layer_width_check_enabled=job["top_layer_width_check_enabled"],
                    top_layer_width_check_max_factor=job["top_layer_width_check_max_factor"],
                    is_top_layer=job["is_top_layer"],
                    use_vectorized_extractor=job["use_vectorized_extractor"],
                    model_lift=model_lift,
                    model_lift_axis=model_lift_axis,
                    planar_infill=job["planar_infill"],
                    **({"ironing": job["ironing"]} if ironing_enabled else {}),
                )
                for job in job_specs
            ]
            results = [
                (future.result(), job["material_override_meta"], job["top_layer_override_meta"], job["start_selection_mode"])
                for future, job in zip(futures, job_specs)
            ]
        results.sort(key=lambda item: (int(item[0][0].stem.split("_", maxsplit=1)[0]), int(item[0][0].stem.split("_", maxsplit=1)[1])))
        for (output_path, layer_meta), material_override_meta, top_layer_override_meta, start_selection_mode in results:
            main_str, sub_str = output_path.stem.split("_", maxsplit=1)
            layer_outputs.append(
                {
                    "layer_main": int(main_str),
                    "layer_sub": int(sub_str),
                    "source_mesh": f"{output_path.stem}.obj",
                    "path_file": output_path.name,
                    "material_path_override": material_override_meta or None,
                    "start_selection_mode": start_selection_mode,
                    "top_layer_path_override": top_layer_override_meta or None,
                    "boundary_start_offset_count": int(layer_meta["boundary_start_offset_count"]),
                    "parallel_direction_mode": str(layer_meta["parallel_direction_mode"]),
                    "parallel_disabled": bool(layer_meta.get("parallel_disabled", False)),
                    "use_vectorized_extractor": bool(layer_meta.get("use_vectorized_extractor", use_vectorized_extractor)),
                    **layer_meta,
                }
            )
            print(f"Saved hybrid path: {output_path.name}")

    batch_metadata = _build_hybrid_batch_metadata(
        input_dir=mesh_dir,
        output_dir=output_dir,
        spacing=spacing,
        parallel_spacing=spacing,
        resample_spacing=float(resample_spacing) if resample_spacing is not None else float(spacing),
        parallel_resample_spacing=float(resample_spacing) if resample_spacing is not None else float(spacing),
        boundary_count=boundary_count,
        sharp_angle_deg=sharp_angle_deg,
        boundary_field_backend=boundary_field_backend,
        parallel_field_backend=parallel_field_backend,
        axis_mode=axis_mode,
        axis=axis,
        plane_value_mode=plane_value_mode,
        plane_value=plane_value,
        level_set_constraint=level_set_constraint,
        soft_level_set_weight=soft_level_set_weight,
        contour_max_level_fraction=contour_max_level_fraction,
        min_polyline_points=min_polyline_points,
        mild_smooth_all=mild_smooth_all,
        mild_smooth_iterations=mild_smooth_iterations,
        mild_smooth_alpha=mild_smooth_alpha,
        first_layers_lowest_y_start_count=first_layers_lowest_y_start_count,
        boundary_start_offset_count=0,
        parallel_direction_mode="connect_nearest",
        use_vectorized_extractor=use_vectorized_extractor,
        model_lift=model_lift,
        model_lift_axis=model_lift_axis,
        layer_outputs=layer_outputs,
    )
    if material_overrides:
        batch_metadata["material_overrides"] = material_overrides
        batch_metadata["width_preserved_with_material_overrides"] = True
    if top_layer_override:
        batch_metadata["top_layer_overrides"] = top_layer_override
    if generated_top_layer_report is not None:
        batch_metadata["top_layer_classifier_report"] = str(generated_top_layer_report)
    if planar_infill_enabled:
        batch_metadata["planar_infill"] = planar_infill_base
    if ironing_enabled:
        batch_metadata["ironing"] = {
            **ironing_base,
            "target_main_indices": sorted(ironing_mains),
        }
    if projected_width_enabled:
        batch_metadata["projected_width_check"] = {
            **projected_width_cfg,
            "target_main_indices": sorted(projected_width_mains),
        }
    if axis_mode in {"fixed_angle", "alternating_angle"}:
        batch_metadata["hatch_angle_degrees"] = hatch_angle_degrees
    batch_metadata_path = output_dir / "batch_paths.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved path metadata: {batch_metadata_path.name}")
    return output_dir


def run_height_stage(
    *,
    mesh_dir: Path,
    path_dir: Path,
    output_dir: Path,
    heights_cfg: dict[str, object],
    clean_output: bool,
) -> Path:
    backend = str(heights_cfg.get("backend", "nearest_triangle")).strip().lower()
    if backend == "nearest_triangle":
        from support_height_validation import resolve_height_settings
        heights_cfg = resolve_height_settings(heights_cfg)
        axis = str(heights_cfg.get("axis", "y")).strip().lower()
        if axis != "y":
            raise ValueError("heights_3axis.backend='nearest_triangle' currently supports axis='y' only.")
        max_height_raw = heights_cfg.get("max_height")
        print("3-axis height backend: nearest_triangle")
        result = run_nearest_surface_height_stage(
            argparse.Namespace(
                mesh_dir=mesh_dir,
                path_dir=path_dir,
                output_dir=output_dir,
                previous_main_window=int(heights_cfg.get("previous_main_window", 2)),
                platform_value=float(heights_cfg.get("platform_value", 0.0)),
                max_platform_height=None if max_height_raw is None else float(max_height_raw),
                nearest_direct_height=float(heights_cfg.get("nearest_direct_height", 0.3)),
                layer_workers=int(heights_cfg.get("layer_workers", 1)),
                clean_output=bool(clean_output),
                support_validation=heights_cfg.get('support_validation', {}),
            ),
        )
        decimation_cfg = heights_cfg.get("adaptive_point_decimation")
        if decimation_cfg is not None:
            if not isinstance(decimation_cfg, dict):
                raise ValueError("heights_3axis.adaptive_point_decimation must be a JSON object.")
            apply_adaptive_point_decimation(result, decimation_cfg)
        return result
    if backend != "legacy_projected_y":
        raise ValueError("heights_3axis.backend must be 'nearest_triangle' or 'legacy_projected_y'.")

    mesh_files = collect_layer_files(mesh_dir, "obj")
    path_files = collect_layer_files(path_dir, "txt")
    shared_keys = sorted(set(mesh_files) & set(path_files), key=lambda key: (key.main, key.sub))
    if not shared_keys:
        raise FileNotFoundError("No matching <main>_<sub> mesh/path pairs were found for the height stage.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    source_batch_metadata = load_existing_metadata(path_dir / "batch_paths.meta.json")
    model_lift, model_lift_axis = infer_model_lift_from_metadata(source_batch_metadata)
    reference_cache: dict[object, object] = {}
    all_mesh_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))

    axis = str(heights_cfg.get("axis", "y"))
    previous_main_window = int(heights_cfg.get("previous_main_window", 2))
    platform_value = float(heights_cfg.get("platform_value", 0.0))
    k_nearest_vertices = int(heights_cfg.get("k_nearest_vertices", 12))
    max_height_raw = heights_cfg.get("max_height")
    max_height = None if max_height_raw is None else float(max_height_raw)
    uncapped_first_layer_count = int(heights_cfg.get("uncapped_first_layer_count", 0))
    if uncapped_first_layer_count < 0:
        raise ValueError("heights_3axis.uncapped_first_layer_count must be non-negative.")
    auto_uncap_base_layers = bool(heights_cfg.get("auto_uncap_base_layers", False))
    base_layer_min_fraction = float(heights_cfg.get("base_layer_min_fraction", 0.7))
    if not (0.0 <= base_layer_min_fraction <= 1.0):
        raise ValueError("heights_3axis.base_layer_min_fraction must be in [0, 1].")
    num_workers = int(heights_cfg.get("num_workers", 1))
    tree_query_workers = int(heights_cfg.get("tree_query_workers", 1))
    layer_workers = int(heights_cfg.get("layer_workers", 1))
    slope_correction_raw = heights_cfg.get("experimental_slope_perpendicular_height_scale")
    slope_correction = dict(slope_correction_raw) if isinstance(slope_correction_raw, dict) else None

    print(f"Computing 3-axis heights for {len(shared_keys)} layer paths into {output_dir}")
    print(f"Layer workers: {layer_workers}")
    print(f"Model lift: {model_lift} along {model_lift_axis}")
    layer_outputs: list[dict[str, object]] = []
    layer_jobs = []
    main_layer_ranks = _main_layer_ranks_from_keys(shared_keys)
    for key in shared_keys:
        candidate_mains = previous_main_indices(all_mesh_keys, key.main, previous_main_window)
        reference_keys = [mesh_key for mesh_key in all_mesh_keys if mesh_key.main in candidate_mains]
        layer_jobs.append(
            {
                "key": key,
                "reference_keys": reference_keys,
            }
        )

    if layer_workers == 1:
        for job in layer_jobs:
            key = job["key"]
            reference_keys = job["reference_keys"]
            reference_meshes = []
            for reference_key in reference_keys:
                cached = reference_cache.get(reference_key)
                if cached is None:
                    cached = load_reference_mesh_3axis(
                        mesh_files[reference_key],
                        reference_key,
                        axis=axis,
                        coordinate_lift=model_lift,
                        coordinate_lift_axis=model_lift_axis,
                    )
                    reference_cache[reference_key] = cached
                reference_meshes.append(cached)

            path_rows = load_path_rows(path_files[key])
            uncapped_by_first_layers = main_layer_ranks[key.main] < uncapped_first_layer_count
            (
                heights,
                platform_mask,
                uncapped_by_base_layer,
                effective_max_height,
                slope_correction_stats,
            ) = compute_heights_with_effective_cap(
                path_rows[:, 0:3],
                reference_meshes,
                axis=axis,
                platform_value=platform_value,
                k_nearest_vertices=k_nearest_vertices,
                max_height=max_height,
                uncapped_by_first_layers=uncapped_by_first_layers,
                auto_uncap_base_layers=auto_uncap_base_layers,
                base_layer_min_fraction=base_layer_min_fraction,
                num_workers=num_workers,
                tree_query_workers=tree_query_workers,
                slope_correction=slope_correction,
            )
            platform_support_count = int(platform_mask.sum())
            platform_support_fraction = float(platform_mask.mean()) if platform_mask.size else 0.0

            output_path = output_dir / f"{key.stem}_3axis.txt"
            save_path_with_heights(output_path, path_rows, heights)
            layer_outputs.append(
                {
                    "layer_main": key.main,
                    "layer_sub": key.sub,
                    "source_mesh": mesh_files[key].name,
                    "source_path_file": path_files[key].name,
                    "output_path_file": output_path.name,
                    "source_metadata_file": "batch_paths.meta.json",
                    "source_metadata": source_batch_metadata,
                    "reference_layers": [reference_key.stem for reference_key in reference_keys],
                    "effective_max_height": effective_max_height,
                    "uncapped_by_first_layers": uncapped_by_first_layers,
                    "uncapped_by_base_layer": uncapped_by_base_layer,
                    "slope_correction_stats": slope_correction_stats,
                    "platform_support_count": platform_support_count,
                    "platform_support_fraction": platform_support_fraction,
                }
            )
            print(
                f"Saved heights: {output_path.name} "
                f"(references={len(reference_keys)}, max_height={float(heights.max()) if heights.size else 0.0:.6g})"
            )
    else:
        from batch_compute_heights import _compute_layer_heights_job
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
            futures = []
            for job in layer_jobs:
                key = job["key"]
                reference_items = [
                    (reference_key, mesh_files[reference_key])
                    for reference_key in job["reference_keys"]
                ]
                uncapped_by_first_layers = main_layer_ranks[key.main] < uncapped_first_layer_count
                futures.append(
                    executor.submit(
                        _compute_layer_heights_job,
                        key=key,
                        mesh_path=mesh_files[key],
                        path_path=path_files[key],
                        output_dir=output_dir,
                        reference_items=reference_items,
                        axis=axis,
                        platform_value=platform_value,
                        k_nearest_vertices=k_nearest_vertices,
                        max_height=max_height,
                        uncapped_by_first_layers=uncapped_by_first_layers,
                        auto_uncap_base_layers=auto_uncap_base_layers,
                        base_layer_min_fraction=base_layer_min_fraction,
                        num_workers=num_workers,
                        tree_query_workers=tree_query_workers,
                        model_lift=model_lift,
                        model_lift_axis=model_lift_axis,
                        slope_correction=slope_correction,
                    )
                )

            results = [future.result() for future in futures]
        results.sort(key=lambda item: (item["layer_key"].main, item["layer_key"].sub))
        for result in results:
            key = result["layer_key"]
            output_path = result["output_path"]
            layer_outputs.append(
                {
                    "layer_main": key.main,
                    "layer_sub": key.sub,
                    "source_mesh": mesh_files[key].name,
                    "source_path_file": path_files[key].name,
                    "output_path_file": output_path.name,
                    "source_metadata_file": "batch_paths.meta.json",
                    "source_metadata": source_batch_metadata,
                    "reference_layers": result["reference_layers"],
                    "effective_max_height": result["effective_max_height"],
                    "uncapped_by_first_layers": result["uncapped_by_first_layers"],
                    "uncapped_by_base_layer": result["uncapped_by_base_layer"],
                    "slope_correction_stats": result.get("slope_correction_stats"),
                    "platform_support_count": result["platform_support_count"],
                    "platform_support_fraction": result["platform_support_fraction"],
                }
            )
            print(
                f"Saved heights: {output_path.name} "
                f"(references={len(result['reference_layers'])}, max_height={result['max_height']:.6g})"
            )

    batch_metadata = write_height_metadata(
        mesh_dir=mesh_dir,
        path_dir=path_dir,
        output_dir=output_dir,
        axis=axis,
        previous_main_window=previous_main_window,
        platform_value=platform_value,
        k_nearest_vertices=k_nearest_vertices,
        max_height=max_height,
        uncapped_first_layer_count=uncapped_first_layer_count,
        auto_uncap_base_layers=auto_uncap_base_layers,
        base_layer_min_fraction=base_layer_min_fraction,
        num_workers=num_workers,
        tree_query_workers=tree_query_workers,
        layer_workers=layer_workers,
        slope_correction=slope_correction,
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = output_dir / "batch_3axis.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved height metadata: {batch_metadata_path.name}")
    decimation_cfg = heights_cfg.get("adaptive_point_decimation")
    if decimation_cfg is not None:
        if not isinstance(decimation_cfg, dict):
            raise ValueError("heights_3axis.adaptive_point_decimation must be a JSON object.")
        apply_adaptive_point_decimation(output_dir, decimation_cfg)
    return output_dir


def run_height_stage_5axis(
    *,
    mesh_dir: Path,
    path_dir: Path,
    output_dir: Path,
    heights_cfg: dict[str, object],
    clean_output: bool,
) -> Path:
    mesh_files = collect_layer_files(mesh_dir, "obj")
    path_files = collect_layer_files(path_dir, "txt")
    shared_keys = sorted(set(mesh_files) & set(path_files), key=lambda key: (key.main, key.sub))
    if not shared_keys:
        raise FileNotFoundError("No matching <main>_<sub> mesh/path pairs were found for the 5-axis height stage.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        clear_directory(output_dir)
        print(f"Cleared output directory: {output_dir}")
    source_batch_metadata = load_existing_metadata(path_dir / "batch_paths.meta.json")
    model_lift, model_lift_axis = infer_model_lift_from_metadata(source_batch_metadata)
    all_mesh_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))

    previous_main_window = int(heights_cfg.get("previous_main_window", 4))
    layer_workers = int(heights_cfg.get("layer_workers", 1))
    major_alignment_threshold = float(heights_cfg.get("major_alignment_threshold", 0.5))
    report_major_deviations = bool(heights_cfg.get("report_major_deviations", False))
    if layer_workers <= 0:
        raise ValueError("heights_5axis.layer_workers must be positive.")

    print(f"Computing 5-axis heights for {len(shared_keys)} layer paths into {output_dir}")
    print(f"Layer workers: {layer_workers}")
    print(f"Model lift: {model_lift} along {model_lift_axis}")
    layer_jobs = []
    for key in shared_keys:
        candidate_mains = previous_main_indices(all_mesh_keys, key.main, previous_main_window)
        reference_keys = [mesh_key for mesh_key in all_mesh_keys if mesh_key.main in candidate_mains]
        layer_jobs.append(
            {
                "key": key,
                "path_path": path_files[key],
                "reference_items": [(reference_key, mesh_files[reference_key]) for reference_key in reference_keys],
            }
        )

    if layer_workers == 1:
        results = [
            _compute_layer_heights_5axis_job(
                key=job["key"],
                path_path=job["path_path"],
                output_dir=output_dir,
                reference_items=job["reference_items"],
                major_alignment_threshold=major_alignment_threshold,
                model_lift=model_lift,
                model_lift_axis=model_lift_axis,
            )
            for job in layer_jobs
        ]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    _compute_layer_heights_5axis_job,
                    key=job["key"],
                    path_path=job["path_path"],
                    output_dir=output_dir,
                    reference_items=job["reference_items"],
                    major_alignment_threshold=major_alignment_threshold,
                    model_lift=model_lift,
                    model_lift_axis=model_lift_axis,
                )
                for job in layer_jobs
            ]
            results = [future.result() for future in futures]

    results.sort(key=lambda item: (item["layer_key"].main, item["layer_key"].sub))

    layer_outputs: list[dict[str, object]] = []
    for result in results:
        key = result["layer_key"]
        output_path = result["output_path"]
        layer_outputs.append(
            {
                "layer_main": key.main,
                "layer_sub": key.sub,
                "source_mesh": mesh_files[key].name,
                "source_path_file": path_files[key].name,
                "output_path_file": output_path.name,
                "source_metadata_file": "batch_paths.meta.json",
                "source_metadata": source_batch_metadata,
                "reference_layers": result["reference_layers"],
                "major_deviation_count": result["major_deviation_count"],
                "alignment_min": result["alignment_min"],
                "alignment_max": result["alignment_max"],
                "alignment_mean": result["alignment_mean"],
            }
        )
        print(
            f"Saved heights: {output_path.name} "
            f"(references={len(result['reference_layers'])}, max_height={result['max_height']:.6g})"
        )
        if report_major_deviations:
            print(
                f"  major_deviations={result['major_deviation_count']} "
                f"alignment_min={result['alignment_min']} "
                f"alignment_mean={result['alignment_mean']} "
                f"alignment_max={result['alignment_max']}"
            )

    batch_metadata = write_height_metadata_5axis(
        mesh_dir=mesh_dir,
        path_dir=path_dir,
        output_dir=output_dir,
        previous_main_window=previous_main_window,
        layer_workers=layer_workers,
        major_alignment_threshold=major_alignment_threshold,
        report_major_deviations=report_major_deviations,
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = output_dir / "batch_5axis.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved height metadata: {batch_metadata_path.name}")
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    config, base_dir = load_config(args.config)
    from draft_toolpaths import validate_outputs
    draft = validate_outputs(config, base_dir)

    paths_cfg = require_section(config, "paths")
    materials_cfg_raw = config.get("materials")
    if materials_cfg_raw is not None and not isinstance(materials_cfg_raw, list):
        raise ValueError("materials must be a JSON array when provided.")
    remesh_cfg = require_section(config, "remesh")
    toolpath_mode = str(config.get("toolpath_mode", "standard")).strip().lower()
    contour_cfg = require_section(config, "contours")
    hybrid_cfg = require_section(config, "hybrid_contours") if toolpath_mode == "hybrid" else {}
    expand_cfg = require_section(config, "support_mesh_expand")
    height_mode = str(config.get("height_mode", "3axis")).strip().lower()
    if height_mode == "3axis":
        heights_cfg = require_section(config, "heights_3axis")
    elif height_mode == "5axis":
        heights_cfg = require_section(config, "heights_5axis")
    else:
        raise ValueError("height_mode must be either '3axis' or '5axis'.")
    clean_output_dirs = bool(config.get("clean_output_dirs", False))
    clear_workers_between_stages = bool(config.get("clear_workers_between_stages", True))

    material_entries: list[dict[str, object]] = []
    if materials_cfg_raw:
        combined_input_surfaces = resolve_path(
            base_dir,
            paths_cfg.get("combined_input_surfaces", "temp_combined_input_surfaces"),
        )
        input_surfaces, material_entries = build_multi_material_input_stage(
            base_dir=base_dir,
            materials_cfg=materials_cfg_raw,
            output_dir=combined_input_surfaces,
            clean_output=clean_output_dirs,
        )
    else:
        input_surfaces = resolve_path(base_dir, paths_cfg["input_surfaces"])
    remeshed_surfaces = resolve_path(base_dir, paths_cfg["remeshed_surfaces"])
    toolpaths_dir = resolve_path(base_dir, paths_cfg["toolpaths"])
    heights_dir = resolve_path(base_dir, paths_cfg["heights"])
    expanded_support_dir = resolve_path(base_dir, paths_cfg["expanded_support_surfaces"])

    contour_mesh_dir = run_remesh_stage(
        input_dir=input_surfaces,
        output_dir=remeshed_surfaces,
        remesh_cfg=remesh_cfg,
        clean_output=clean_output_dirs,
    )
    if clear_workers_between_stages:
        cleanup_stage_workers(stage_name="remesh stage")
    support_mesh_dir = contour_mesh_dir if draft else run_support_expansion_stage(
        input_dir=contour_mesh_dir,
        output_dir=expanded_support_dir,
        expand_cfg=expand_cfg,
        clean_output=clean_output_dirs,
    )
    if clear_workers_between_stages:
        cleanup_stage_workers(stage_name="support expansion stage")
    if toolpath_mode == "standard":
        run_path_stage(
            mesh_dir=contour_mesh_dir,
            output_dir=toolpaths_dir,
            contour_cfg=contour_cfg,
            clean_output=clean_output_dirs,
        )
        if clear_workers_between_stages:
            cleanup_stage_workers(stage_name="standard path stage")
    elif toolpath_mode == "hybrid":
        run_hybrid_path_stage(
            mesh_dir=contour_mesh_dir,
            output_dir=toolpaths_dir,
            hybrid_cfg=hybrid_cfg,
            clean_output=clean_output_dirs,
            base_dir=base_dir,
            material_entries=material_entries or None,
        )
        if clear_workers_between_stages:
            cleanup_stage_workers(stage_name="hybrid path stage")
    else:
        raise ValueError("toolpath_mode must be either 'standard' or 'hybrid'.")
    if material_entries:
        attach_material_metadata(
            target_dir=toolpaths_dir,
            metadata_filename="batch_paths.meta.json",
            material_entries=material_entries,
        )
    if draft:
        from draft_toolpaths import write_draft
        import json
        write_draft(toolpaths_dir, json.loads((toolpaths_dir / 'batch_paths.meta.json').read_text(encoding='utf8')))
        return
    if height_mode == "3axis":
        run_height_stage(
            mesh_dir=support_mesh_dir,
            path_dir=toolpaths_dir,
            output_dir=heights_dir,
            heights_cfg=heights_cfg,
            clean_output=clean_output_dirs,
        )
        if clear_workers_between_stages:
            cleanup_stage_workers(stage_name="3-axis height stage")
        if material_entries:
            attach_material_metadata(
                target_dir=heights_dir,
                metadata_filename="batch_3axis.meta.json",
                material_entries=material_entries,
            )
    else:
        run_height_stage_5axis(
            mesh_dir=support_mesh_dir,
            path_dir=toolpaths_dir,
            output_dir=heights_dir,
            heights_cfg=heights_cfg,
            clean_output=clean_output_dirs,
        )
        if clear_workers_between_stages:
            cleanup_stage_workers(stage_name="5-axis height stage")
        if material_entries:
            attach_material_metadata(
                target_dir=heights_dir,
                metadata_filename="batch_5axis.meta.json",
                material_entries=material_entries,
            )


if __name__ == "__main__":
    main()

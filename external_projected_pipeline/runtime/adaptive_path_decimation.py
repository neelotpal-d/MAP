"""Conservative post-height point decimation for height-augmented toolpaths."""

from __future__ import annotations

from multiprocessing import get_context

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import numpy as np


def _load_tags(path: Path, count: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    values = np.full(count, "", dtype=object)
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        start, end = int(fields[0]), int(fields[1])
        if start < 0 or end < start or end > count:
            raise ValueError(f"Invalid tag range in {path.name}: {line}")
        values[start:end] = fields[2]
    return values


def _write_tags(path: Path, values: np.ndarray) -> None:
    if values.size == 0:
        path.write_text("", encoding="utf-8")
        return
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    path.write_text(
        "".join(f"{int(start)} {int(end)} {values[start]}\n" for start, end in zip(starts, ends) if values[start]),
        encoding="utf-8",
    )


def _load_break_indices(path: Path, count: int) -> list[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    values = sorted({int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})
    if any(value <= 0 or value >= count for value in values):
        raise ValueError(f"Invalid path break index in {path.name}")
    return values


def _write_long_jump_tags(path: Path, jumps: list[tuple[int, float]]) -> None:
    """Store segment events as ``destination_index length long_jump`` rows."""
    path.write_text(
        "".join(f"{index} {length:.10f} long_jump\n" for index, length in jumps),
        encoding="utf-8",
    )


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 180.0
    cosine = float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _can_remove(
    rows: np.ndarray,
    anchor: int,
    current: int,
    following: int,
    *,
    max_spacing: float,
    max_height_delta: float,
    max_normal_angle_deg: float,
    max_turn_angle_deg: float,
) -> bool:
    points = rows[:, 0:3]
    normals = rows[:, 3:6]
    heights = rows[:, 6]
    if float(np.linalg.norm(points[following] - points[anchor])) > max_spacing:
        return False
    if float(np.ptp(heights[anchor : following + 1])) > max_height_delta:
        return False
    anchor_normal = normals[anchor]
    if any(
        _angle_degrees(anchor_normal, normal) > max_normal_angle_deg
        for normal in normals[anchor + 1 : following + 1]
    ):
        return False
    return _angle_degrees(points[current] - points[anchor], points[following] - points[current]) <= max_turn_angle_deg


def _select_segment(
    rows: np.ndarray,
    tags: np.ndarray,
    *,
    max_spacing: float,
    max_height_delta: float,
    max_normal_angle_deg: float,
    max_turn_angle_deg: float,
) -> np.ndarray:
    count = rows.shape[0]
    if count <= 2:
        return np.arange(count, dtype=np.int64)
    protected = np.asarray(["boundary" in str(tag).lower() for tag in tags], dtype=bool)
    selected = [0]
    anchor = 0
    for current in range(1, count - 1):
        following = current + 1
        if protected[current] or not _can_remove(
            rows, anchor, current, following,
            max_spacing=max_spacing,
            max_height_delta=max_height_delta,
            max_normal_angle_deg=max_normal_angle_deg,
            max_turn_angle_deg=max_turn_angle_deg,
        ):
            selected.append(current)
            anchor = current
    selected.append(count - 1)
    return np.asarray(selected, dtype=np.int64)


def _apply_nozzle_clearance_lift(
    rows: np.ndarray,
    breaks: list[int],
    *,
    nozzle_tip_diameter: float,
    max_lift: float,
) -> dict[str, float | int]:
    """Lift path points for a finite nozzle tip across the local bead direction.

    The path is stored in the internal y-up coordinate system.  The tangent is
    never evaluated across a tagged path break, and the stored bead-height
    column is intentionally left unchanged: this is a nozzle-position
    correction rather than an extrusion-volume correction.
    """
    points = rows[:, 0:3]
    normals = rows[:, 3:6]
    lifts = np.zeros(rows.shape[0], dtype=np.float64)
    bounds = [0, *breaks, rows.shape[0]]
    radius = 0.5 * nozzle_tip_diameter
    for start, end in zip(bounds[:-1], bounds[1:]):
        for index in range(start, end):
            if end - start < 2:
                continue
            if index == start:
                tangent = points[index + 1] - points[index]
            elif index == end - 1:
                tangent = points[index] - points[index - 1]
            else:
                tangent = points[index + 1] - points[index - 1]
            tangent_length = float(np.linalg.norm(tangent))
            normal_length = float(np.linalg.norm(normals[index]))
            if tangent_length <= 1e-12 or normal_length <= 1e-12:
                continue
            tangent /= tangent_length
            normal = normals[index] / normal_length
            # This is the local across-bead direction in the surface tangent plane.
            across_bead = np.cross(tangent, normal)
            across_length = float(np.linalg.norm(across_bead))
            if across_length <= 1e-12:
                continue
            across_bead /= across_length
            horizontal_length = float(np.hypot(across_bead[0], across_bead[2]))
            if horizontal_length <= 1e-12:
                lift = max_lift
            else:
                lift = min(max_lift, radius * abs(float(across_bead[1])) / horizontal_length)
            lifts[index] = lift
    points[:, 1] += lifts
    nonzero = lifts[lifts > 0.0]
    return {
        "point_count": int(rows.shape[0]),
        "lifted_point_count": int(nonzero.size),
        "mean_lift": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_lift": float(lifts.max()) if lifts.size else 0.0,
    }


def _decimate_layer_job(
    output_dir: Path,
    layer_index: int,
    layer: dict[str, object],
    settings: dict[str, float | None],
) -> dict[str, object]:
    started = time.perf_counter()
    output_name = layer.get("output_path_file")
    tag_name = layer.get("path_tag_file")
    if not isinstance(output_name, str) or not isinstance(tag_name, str):
        return {"layer_index": layer_index, "skipped": True}
    path = output_dir / output_name
    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError(f"Expected x y z nx ny nz height rows in {path.name}")
    segment_lengths = np.linalg.norm(np.diff(rows[:, 0:3], axis=0), axis=1)
    long_jump_threshold = settings["long_jump_threshold"]
    long_jump_lengths = {
        int(index + 1): float(length)
        for index, length in enumerate(segment_lengths)
        if long_jump_threshold is not None and float(length) > long_jump_threshold
    }
    tags = _load_tags(output_dir / tag_name, rows.shape[0])
    if tags is None:
        return {"layer_index": layer_index, "skipped": True}
    width_name = layer.get("width_profile_file")
    widths = None
    if isinstance(width_name, str) and width_name:
        widths = np.atleast_1d(np.loadtxt(output_dir / width_name, dtype=np.float64))
        if widths.size != rows.shape[0]:
            raise ValueError(f"Width profile does not match processed path: {width_name}")
    break_name = layer.get("path_break_file")
    if long_jump_lengths and not isinstance(break_name, str):
        break_name = f"{path.stem}.breaks.txt"
    existing_breaks = _load_break_indices(output_dir / break_name, rows.shape[0]) if isinstance(break_name, str) else []
    read_elapsed = time.perf_counter() - started
    breaks = sorted({*existing_breaks, *long_jump_lengths})
    clearance_stats = None
    if bool(settings["nozzle_clearance_enabled"]):
        clearance_stats = _apply_nozzle_clearance_lift(
            rows,
            breaks,
            nozzle_tip_diameter=float(settings["nozzle_tip_diameter"]),
            max_lift=float(settings["nozzle_clearance_max_lift"]),
        )
    bounds = [0, *breaks, rows.shape[0]]
    selected_parts: list[np.ndarray] = []
    rewritten_breaks: list[int] = []
    rewritten_long_jumps: list[tuple[int, float]] = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        local_indices = _select_segment(
            rows[start:end], tags[start:end],
            max_spacing=float(settings["max_spacing"]),
            max_height_delta=float(settings["max_height_delta"]),
            max_normal_angle_deg=float(settings["max_normal_angle_deg"]),
            max_turn_angle_deg=float(settings["max_turn_angle_deg"]),
        ) + start
        if selected_parts:
            rewritten_index = sum(part.size for part in selected_parts)
            rewritten_breaks.append(rewritten_index)
            if start in long_jump_lengths:
                rewritten_long_jumps.append((rewritten_index, long_jump_lengths[start]))
        selected_parts.append(local_indices)
    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    selection_elapsed = time.perf_counter() - started - read_elapsed
    write_started = time.perf_counter()
    np.savetxt(path, rows[selected], fmt="%.10f")
    _write_tags(output_dir / tag_name, tags[selected])
    if widths is not None:
        np.savetxt(output_dir / str(width_name), widths[selected], fmt="%.10f")
    if isinstance(break_name, str) and break_name:
        break_path = output_dir / break_name
        if rewritten_breaks:
            break_path.write_text("".join(f"{value}\n" for value in rewritten_breaks), encoding="utf-8")
        elif break_path.exists():
            break_path.unlink()
    jump_path = output_dir / f"{path.stem}.jumps.txt"
    if rewritten_long_jumps:
        _write_long_jump_tags(jump_path, rewritten_long_jumps)
    elif jump_path.exists():
        jump_path.unlink()
    write_elapsed = time.perf_counter() - write_started
    return {
        "layer_index": layer_index,
        "skipped": False,
        "path_name": path.name,
        "input_count": int(rows.shape[0]),
        "output_count": int(selected.size),
        "long_jump_count": len(rewritten_long_jumps),
        "timings_seconds": {
            "read": read_elapsed,
            "selection": selection_elapsed,
            "write": write_elapsed,
            "total": time.perf_counter() - started,
        },
        "layer_update": {
            "adaptive_point_decimation": {
                "input_point_count": int(rows.shape[0]),
                "output_point_count": int(selected.size),
                "removed_point_count": int(rows.shape[0] - selected.size),
            },
            "nozzle_clearance_lift": clearance_stats,
            "path_jump_file": jump_path.name if rewritten_long_jumps else None,
            "path_jump_format": "destination_index length tag" if rewritten_long_jumps else None,
            "path_long_jump_count": len(rewritten_long_jumps),
            "path_break_file": str(break_name) if rewritten_breaks else None,
        },
    }


def apply_adaptive_point_decimation(output_dir: Path, config: dict[str, object]) -> dict[str, object]:
    """Decimate eligible processed paths and keep every sidecar index-aligned."""
    if not bool(config.get("enabled", False)):
        return {"enabled": False}
    settings: dict[str, float | None] = {
        "max_spacing": float(config.get("max_spacing", 1.0)),
        "max_height_delta": float(config.get("max_height_delta", 0.005)),
        "max_normal_angle_deg": float(config.get("max_normal_angle_deg", 1.0)),
        "max_turn_angle_deg": float(config.get("max_turn_angle_deg", 1.0)),
        "long_jump_threshold": None if config.get("long_jump_threshold") is None else float(config["long_jump_threshold"]),
    }
    clearance_cfg_raw = config.get("nozzle_clearance_lift")
    if clearance_cfg_raw is not None and not isinstance(clearance_cfg_raw, dict):
        raise ValueError("adaptive_point_decimation.nozzle_clearance_lift must be a JSON object.")
    clearance_cfg = clearance_cfg_raw if isinstance(clearance_cfg_raw, dict) else {}
    clearance_enabled = bool(clearance_cfg.get("enabled", False))
    settings["nozzle_clearance_enabled"] = float(clearance_enabled)
    settings["nozzle_tip_diameter"] = float(clearance_cfg.get("nozzle_tip_diameter", 0.4))
    settings["nozzle_clearance_max_lift"] = float(clearance_cfg.get("max_lift", 0.15))
    if min(float(settings[key]) for key in ("max_spacing", "max_height_delta", "max_normal_angle_deg", "max_turn_angle_deg")) <= 0.0:
        raise ValueError("adaptive_point_decimation tolerances must be positive.")
    if settings["long_jump_threshold"] is not None and float(settings["long_jump_threshold"]) <= 0.0:
        raise ValueError("adaptive_point_decimation.long_jump_threshold must be positive or null.")
    if clearance_enabled and (
        float(settings["nozzle_tip_diameter"]) <= 0.0 or float(settings["nozzle_clearance_max_lift"]) <= 0.0
    ):
        raise ValueError("nozzle_clearance_lift.nozzle_tip_diameter and max_lift must be positive when enabled.")
    layer_workers = max(1, int(config.get("layer_workers", 1)))
    report_timing = bool(config.get("report_timing", False))
    metadata_path = output_dir / "batch_3axis.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layers = metadata.get("layers")
    if not isinstance(layers, list):
        raise ValueError("batch_3axis.meta.json must contain a layers list.")
    jobs = [(index, layer) for index, layer in enumerate(layers) if isinstance(layer, dict)]
    print(f"[adaptive-decimation] Processing {len(jobs)} height-augmented layers with {layer_workers} workers")
    if layer_workers == 1:
        results = [_decimate_layer_job(output_dir, index, layer, settings) for index, layer in jobs]
    else:
        with ProcessPoolExecutor(max_workers=layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [executor.submit(_decimate_layer_job, output_dir, index, layer, settings) for index, layer in jobs]
            results = []
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(f"[adaptive-decimation] completed {completed}/{len(jobs)}: {result.get('path_name', 'untagged layer')}")
    total_before = total_after = skipped_without_tags = total_long_jumps = 0
    for result in results:
        if bool(result["skipped"]):
            skipped_without_tags += 1
            continue
        layer = layers[int(result["layer_index"])]
        assert isinstance(layer, dict)
        layer.update(result["layer_update"])
        total_before += int(result["input_count"])
        total_after += int(result["output_count"])
        total_long_jumps += int(result["long_jump_count"])
        print(
            f"[adaptive-decimation] {result['path_name']}: {result['input_count']} -> "
            f"{result['output_count']} points, long_jumps={result['long_jump_count']}",
        )
        if report_timing:
            timings = result["timings_seconds"]
            assert isinstance(timings, dict)
            print(
                f"[adaptive-decimation timing] {result['path_name']}: "
                f"read={float(timings['read']):.2f}s selection={float(timings['selection']):.2f}s "
                f"write={float(timings['write']):.2f}s total={float(timings['total']):.2f}s",
            )
    report = {
        "enabled": True,
        "max_spacing": settings["max_spacing"],
        "max_height_delta": settings["max_height_delta"],
        "max_normal_angle_deg": settings["max_normal_angle_deg"],
        "max_turn_angle_deg": settings["max_turn_angle_deg"],
        "long_jump_threshold": settings["long_jump_threshold"],
        "input_point_count": total_before,
        "output_point_count": total_after,
        "removed_point_count": total_before - total_after,
        "skipped_layers_without_tags": skipped_without_tags,
        "long_jump_count": total_long_jumps,
        "layer_workers": layer_workers,
        "nozzle_clearance_lift": {
            "enabled": clearance_enabled,
            "nozzle_tip_diameter": settings["nozzle_tip_diameter"],
            "max_lift": settings["nozzle_clearance_max_lift"],
        },
    }
    metadata["adaptive_point_decimation"] = report
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Adaptive post-height decimation: {total_before} -> {total_after} points "
        f"(removed {total_before - total_after}; boundary roles preserved)",
    )
    return report

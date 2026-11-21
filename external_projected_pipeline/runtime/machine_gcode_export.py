"""
Convert height-augmented 3-axis toolpaths into machine G-code.

Toolpath inputs are expected to be files named like:
    <main>_<sub>_3axis.txt

with rows:
    x y z nx ny nz h

The batch metadata file `batch_3axis.meta.json` is used to recover width and
layer mapping information. Machine behavior is configured by a machine profile
JSON plus referenced pre/post G-code files.
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PATH_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)_3axis\.txt$", re.IGNORECASE)
INTERRUPT_COMMENT = (
    ";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; "
    "Interrupt "
    ";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;"
)


@dataclass(frozen=True, slots=True)
class LayerKey:
    main: int
    sub: int

    @property
    def stem(self) -> str:
        return f"{self.main}_{self.sub}"


@dataclass(slots=True)
class MachineProfile:
    machine_name: str
    base_center: np.ndarray
    up_direction: np.ndarray
    filament_diameter: float
    alpha: float
    travel_feedrate: float | None
    print_feedrate: float | None
    travel_acceleration: float | None
    print_acceleration: float | None
    path_start_slowdown_fraction: float | None
    path_start_feedrate_scale: float | None
    first_layer_extrusion_scale: float | None
    first_layer_count: int
    first_layer_print_feedrate: float | None
    travel_lift: float
    interrupt_threshold: float
    retract_amount: float
    pre_extrude_amount: float
    pre_gcode: str
    post_gcode: str


class _StreamingLineWriter:
    """Append G-code lines directly to disk instead of retaining a large list."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="\n")
        self._pending: list[str] = []
        self._pending_size = 0

    def append(self, line: str) -> None:
        self._pending.append(line + "\n")
        self._pending_size += len(line) + 1
        if self._pending_size >= 65536:
            self._flush()

    def _flush(self) -> None:
        self._handle.write("".join(self._pending))
        self._pending.clear()
        self._pending_size = 0

    def close(self) -> None:
        if not self._handle.closed:
            self._flush()
            self._handle.close()


_DefaultLineWriter = _StreamingLineWriter


class _IndexedBatchMetadata(dict):
    """Run-local lookup; the original JSON fields and first-match rules remain."""
    def __init__(self, metadata):
        super().__init__(metadata)
        self.layers_by_key = {}
        layers = self.get('layers')
        if not isinstance(layers, list):
            self.layers_by_key = None
            return
        try:
            for layer in layers:
                if isinstance(layer, dict):
                    key = (int(layer.get('layer_main', -1)), int(layer.get('layer_sub', -1)))
                    self.layers_by_key.setdefault(key, layer)
        except (TypeError, ValueError, OverflowError):
            # Preserve the original lazy validation for malformed metadata.
            self.layers_by_key = None


def parse_layer_key(path: Path) -> LayerKey | None:
    match = PATH_PATTERN.match(path.name)
    if match is None:
        return None
    return LayerKey(main=int(match.group("main")), sub=int(match.group("sub")))


def collect_toolpath_files(folder: Path) -> dict[LayerKey, Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Toolpath folder not found: {folder}")
    files: dict[LayerKey, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        key = parse_layer_key(path)
        if key is not None:
            files[key] = path
    return files


def assert_resolved_support_heights(metadata, *, start_main, end_main):
    from height_warning_policy import check_height_outliers
    check_height_outliers(metadata, start_main=start_main, end_main=end_main)
    layers = metadata.get('layers', [])
    count = sum(row.get('support_validation', {}).get('unresolved_points', 0) for row in layers
        if start_main <= int(row['layer_main']) <= end_main)
    if not layers:
        count = metadata.get('support_validation', {}).get('unresolved_points', 0)
    if count:
        raise ValueError(f'{count} unresolved supporting-height points in main range {start_main}..{end_main}. '
            'Inspect *.height_validation.npz diagnostics; bridge extrusion has not been defined.')
    fallback=sum(row.get('support_validation',{}).get('threshold_fallback_points',0) for row in layers
        if start_main<=int(row['layer_main'])<=end_main) if layers else metadata.get('support_validation',{}).get('threshold_fallback_points',0)
    if fallback:
        print(f'[height warning] {fallback} ambiguous points use threshold extrusion heights; geometric support is unverified.')


def load_batch_metadata(toolpath_dir: Path) -> dict[str, object]:
    if (toolpath_dir / '.causal_lift_in_progress').exists():
        raise ValueError('Causal nozzle lift is incomplete in this folder; wait for completion before exporting.')
    metadata_path = toolpath_dir / "batch_3axis.meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing batch metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_machine_profile(machine_dir: Path, machine_name: str) -> MachineProfile:
    json_path = machine_dir / f"{machine_name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Machine profile not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    base_center = np.asarray(payload["base_center"], dtype=np.float64)
    up_direction = np.asarray(payload["up_direction"], dtype=np.float64)
    if base_center.shape != (3,):
        raise ValueError("machine base_center must have shape (3,).")
    if up_direction.shape != (3,):
        raise ValueError("machine up_direction must have shape (3,).")

    up_norm = np.linalg.norm(up_direction)
    if up_norm <= np.finfo(np.float64).eps:
        raise ValueError("machine up_direction must be non-zero.")
    up_direction = up_direction / up_norm
    expected_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(up_direction, expected_up, atol=1e-9):
        warnings.warn(
            f"Machine '{machine_name}' has unsupported up_direction {up_direction.tolist()}. "
            "Temporary exporter only supports [0, 0, 1].",
            stacklevel=2,
        )
        raise ValueError("Unsupported machine up_direction for current exporter.")

    pre_gcode_path = machine_dir / payload["pre_gcode_file"]
    post_gcode_path = machine_dir / payload["post_gcode_file"]
    return MachineProfile(
        machine_name=str(payload["machine_name"]),
        base_center=base_center,
        up_direction=up_direction,
        filament_diameter=float(payload["filament_diameter"]),
        alpha=float(payload["alpha"]),
        travel_feedrate=(
            None
            if payload.get("travel_feedrate") is None
            else float(payload["travel_feedrate"])
        ),
        print_feedrate=(
            None
            if payload.get("print_feedrate") is None
            else float(payload["print_feedrate"])
        ),
        travel_acceleration=(
            None
            if payload.get("travel_acceleration") is None
            else float(payload["travel_acceleration"])
        ),
        print_acceleration=(
            None
            if payload.get("print_acceleration") is None
            else float(payload["print_acceleration"])
        ),
        path_start_slowdown_fraction=(
            None
            if payload.get("path_start_slowdown_fraction") is None
            else float(payload["path_start_slowdown_fraction"])
        ),
        path_start_feedrate_scale=(
            None
            if payload.get("path_start_feedrate_scale") is None
            else float(payload["path_start_feedrate_scale"])
        ),
        first_layer_extrusion_scale=(
            None
            if payload.get("first_layer_extrusion_scale") is None
            else float(payload["first_layer_extrusion_scale"])
        ),
        first_layer_count=int(payload.get("first_layer_count", 0)),
        first_layer_print_feedrate=(
            None
            if payload.get("first_layer_print_feedrate") is None
            else float(payload["first_layer_print_feedrate"])
        ),
        travel_lift=float(payload["travel_lift"]),
        interrupt_threshold=float(payload["interrupt_threshold"]),
        retract_amount=float(payload["retract_amount"]),
        pre_extrude_amount=float(payload["pre_extrude_amount"]),
        pre_gcode=pre_gcode_path.read_text(encoding="utf-8"),
        post_gcode=post_gcode_path.read_text(encoding="utf-8"),
    )


def load_path_rows(path_file: Path) -> np.ndarray:
    if path_file.stat().st_size == 0:
        return np.empty((0, 7), dtype=np.float64)
    data = np.loadtxt(path_file, dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(f"Path file must contain x y z nx ny nz h columns: {path_file}")
    if not np.isfinite(data[:,:7]).all() or np.any(data[:,6]<0):
        raise ValueError(f'Path coordinates/normals/heights must be finite and heights nonnegative: {path_file}')
    return np.asarray(data[:, 0:7], dtype=np.float64)


def transform_points_y_up_to_z_up(points: np.ndarray, base_center: np.ndarray) -> np.ndarray:
    transformed = np.empty_like(points)
    transformed[:, 0] = points[:, 0]
    transformed[:, 1] = -points[:, 2]
    transformed[:, 2] = points[:, 1]
    transformed += base_center[None, :]
    return transformed


def export_xy_centering_offset(
    toolpath_files: dict[LayerKey, Path],
    selected_keys: list[LayerKey],
    machine: MachineProfile,
    *,
    enabled: bool,
) -> np.ndarray:
    """Center the selected model footprint on the configured machine center."""
    if not enabled:
        return np.zeros(3, dtype=np.float64)
    lower = np.full(2, np.inf, dtype=np.float64)
    upper = np.full(2, -np.inf, dtype=np.float64)
    for key in selected_keys:
        rows = load_path_rows(toolpath_files[key])
        if rows.size == 0:
            continue
        points = transform_points_y_up_to_z_up(rows[:, 0:3], machine.base_center)
        lower = np.minimum(lower, np.min(points[:, 0:2], axis=0))
        upper = np.maximum(upper, np.max(points[:, 0:2], axis=0))
    if not np.all(np.isfinite(lower)):
        raise ValueError("Cannot center an export containing no toolpath points.")
    footprint_center = 0.5 * (lower + upper)
    offset = np.zeros(3, dtype=np.float64)
    offset[0:2] = machine.base_center[0:2] - footprint_center
    print(
        f"Centering selected XY footprint [{lower[0]:.4f}, {lower[1]:.4f}].."
        f"[{upper[0]:.4f}, {upper[1]:.4f}] with offset "
        f"X={offset[0]:.4f}, Y={offset[1]:.4f}",
    )
    return offset


def filament_area(diameter: float) -> float:
    return math.pi * diameter * diameter / 4.0


def bead_width_from_spacing(spacing: float, height: float) -> float:
    """Invert our rounded bead area so its area equals spacing * height.

    Ordinary flat beads use w=s+(1-pi/4)h. Tall measured beads retain the
    existing symmetric capsule model; solve its other branch without clamping h.
    Export uses the equivalent area s*h directly to avoid this work per move.
    """
    if not math.isfinite(spacing) or not math.isfinite(height) or min(spacing, height) <= 0:
        raise ValueError("Spacing and physical height must be finite and positive.")
    corner = 1.0 - 0.25 * math.pi
    if spacing >= 0.25 * math.pi * height:
        return spacing + corner * height
    ratio = spacing / height
    return 2.0 * spacing / (1.0 + math.sqrt(1.0 - 4.0 * corner * ratio))


def model_spacing_width_for_layer(metadata: dict, key: LayerKey, *, enabled: bool) -> bool:
    """Use existing part metadata once per layer; never classify wall geometry."""
    if not enabled:
        return False
    entry = find_layer_metadata_entry(metadata, key) or {}
    source = entry.get('source_metadata')
    if not isinstance(source, dict):
        source = {}
    part = entry.get('part', source.get('part'))
    if part is not None:
        return part == 'model'
    role = str(entry.get('layer_role', source.get('layer_role', '')))
    # Unlabelled ordinary model batches remain usable with the explicit option.
    return not role.startswith('support')


def bead_cross_section_area(width: float, height: float, *, mode: str,
                            width_is_spacing: bool = False) -> float:
    """Return the deposited bead area for the selected geometric model."""
    if width <= 0.0 or height <= 0.0:
        return 0.0
    if mode == "rectangular":
        return width * height
    if mode == "rounded_rectangle":
        if width_is_spacing:
            # Exact area of the derived rounded bead, including the tall branch.
            return width * height
        # Maximal corner radius that fits both measured dimensions. For w >= h
        # this is the usual horizontal capsule; for h > w it is vertical.
        # Do not clamp measured height or silently switch to rectangular flow.
        diameter = min(width, height)
        return width * height - (1.0 - 0.25 * math.pi) * diameter * diameter
    raise ValueError("bead_cross_section must be 'rectangular' or 'rounded_rectangle'.")


def infer_width_from_metadata(batch_metadata: dict[str, object], layer_key: LayerKey) -> float:
    top_level_width = batch_metadata.get("width")
    if top_level_width is not None:
        return float(top_level_width)

    # Derived height batches retain their path metadata at this level rather
    # than duplicating it into every output-layer entry.
    source_batch_metadata = batch_metadata.get("source_path_metadata")
    if isinstance(source_batch_metadata, dict):
        source_width = source_batch_metadata.get("width")
        if source_width is not None:
            return float(source_width)
        source_spacing = source_batch_metadata.get("contour_spacing")
        if source_spacing is not None:
            return float(source_spacing)

    layers = batch_metadata.get("layers")
    if isinstance(layers, list):
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if int(layer.get("layer_main", -1)) != layer_key.main or int(layer.get("layer_sub", -1)) != layer_key.sub:
                continue
            source_metadata = layer.get("source_metadata")
            if isinstance(source_metadata, dict):
                if source_metadata.get("width") is not None:
                    return float(source_metadata["width"])
                if source_metadata.get("contour_spacing") is not None:
                    return float(source_metadata["contour_spacing"])
    raise ValueError("Unable to infer width from batch metadata.")


def infer_width_profile_from_metadata(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
    *,
    toolpath_dir: Path,
    point_count: int,
) -> np.ndarray | None:
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    if layer is None:
        return None
    profile_name = layer.get("width_profile_file")
    if not isinstance(profile_name, str) or not profile_name:
        return None
    profile_path = toolpath_dir / profile_name
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing width profile file for layer {layer_key.stem}: {profile_path}")
    if profile_path.stat().st_size == 0:
        return np.empty((0,), dtype=np.float64)
    values = np.loadtxt(profile_path, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != point_count:
        raise ValueError(
            f"Width profile for layer {layer_key.stem} has {values.shape[0]} values, expected {point_count}.",
        )
    return values


def infer_nominal_segment_lengths(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
    *,
    toolpath_dir: Path,
    point_count: int,
) -> np.ndarray | None:
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    name = layer.get("nominal_segment_length_file") if layer else None
    if not name:
        return None
    values = np.asarray(np.loadtxt(toolpath_dir / str(name)), dtype=np.float64).reshape(-1)
    if values.size != max(0, point_count - 1) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"Invalid nominal extrusion segment lengths for {layer_key.stem}")
    return values


def infer_path_tag_blocks_from_metadata(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
    *,
    toolpath_dir: Path,
    point_count: int,
) -> list[tuple[int, int, str]] | None:
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    if layer is None:
        return None
    tag_name = layer.get("path_tag_file")
    if not isinstance(tag_name, str) or not tag_name:
        return None
    tag_path = toolpath_dir / tag_name
    if not tag_path.exists():
        raise FileNotFoundError(f"Missing path-tag file for layer {layer_key.stem}: {tag_path}")
    blocks: list[tuple[int, int, str]] = []
    for line in tag_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"Invalid path-tag block in {tag_path}: {line!r}")
        blocks.append((int(fields[0]), int(fields[1]), fields[2]))
    if not blocks or blocks[0][0] != 0 or blocks[-1][1] != point_count:
        raise ValueError(f"Path-tag blocks for layer {layer_key.stem} do not cover all path points.")
    if any(start != previous_end or end <= start for (start, end, _), (_, previous_end, _) in zip(blocks[1:], blocks[:-1], strict=False)):
        raise ValueError(f"Path-tag blocks for layer {layer_key.stem} are not contiguous.")
    return blocks


def infer_path_break_indices_from_metadata(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
    *,
    toolpath_dir: Path,
    point_count: int,
) -> set[int]:
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    if layer is None:
        return set()
    break_name = layer.get("path_break_file")
    if not isinstance(break_name, str) or not break_name:
        return set()
    break_path = toolpath_dir / break_name
    if not break_path.exists():
        raise FileNotFoundError(f"Missing path-break file for layer {layer_key.stem}: {break_path}")
    values = {int(line.strip()) for line in break_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if any(index <= 0 or index >= point_count for index in values):
        raise ValueError(f"Invalid path-break index in {break_path}")
    return values


def infer_long_jump_events_from_metadata(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
    *,
    toolpath_dir: Path,
    point_count: int,
) -> dict[int, float]:
    """Read pre-decimation long-jump events keyed by destination point index."""
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    if layer is None:
        return {}
    jump_name = layer.get("path_jump_file")
    if not isinstance(jump_name, str) or not jump_name:
        return {}
    jump_path = toolpath_dir / jump_name
    if not jump_path.exists():
        raise FileNotFoundError(f"Missing path-jump file for layer {layer_key.stem}: {jump_path}")
    events: dict[int, float] = {}
    for line in jump_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3 or fields[2] != "long_jump":
            raise ValueError(f"Invalid long-jump event in {jump_path}: {line!r}")
        index = int(fields[0])
        if index <= 0 or index >= point_count:
            raise ValueError(f"Invalid long-jump index in {jump_path}: {index}")
        events[index] = float(fields[1])
    return events


def find_layer_metadata_entry(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
) -> dict[str, object] | None:
    if isinstance(batch_metadata, _IndexedBatchMetadata) and batch_metadata.layers_by_key is not None:
        return batch_metadata.layers_by_key.get((layer_key.main, layer_key.sub))
    layers = batch_metadata.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if int(layer.get("layer_main", -1)) != layer_key.main or int(layer.get("layer_sub", -1)) != layer_key.sub:
            continue
        return layer
    return None


def infer_material_index_from_metadata(
    batch_metadata: dict[str, object],
    layer_key: LayerKey,
) -> int | None:
    layer = find_layer_metadata_entry(batch_metadata, layer_key)
    if layer is None:
        return None
    material_index = layer.get("material_index")
    if material_index is None:
        return None
    return int(material_index)


def selected_layer_keys(
    all_keys: list[LayerKey],
    *,
    start_main: int,
    end_main: int,
) -> list[LayerKey]:
    if end_main < start_main:
        raise ValueError("end_main must be >= start_main.")
    return [
        key
        for key in sorted(all_keys, key=lambda item: (item.main, item.sub))
        if start_main <= key.main <= end_main
    ]


def _feedrate_suffix(feedrate: float | None) -> str:
    if feedrate is None:
        return ""
    return f" F{feedrate:.5f}"


def _append_acceleration_command(
    lines: list[str],
    *,
    acceleration: float | None,
    current_acceleration: float | None,
) -> float | None:
    if acceleration is None:
        return current_acceleration
    if current_acceleration is not None and math.isclose(current_acceleration, acceleration, rel_tol=1e-9, abs_tol=1e-9):
        return current_acceleration
    lines.append(f"M204 S{acceleration:.5f}")
    return acceleration


def main_layer_rank_lookup(selected_keys: list[LayerKey]) -> dict[int, int]:
    main_values = sorted({key.main for key in selected_keys})
    return {main: rank for rank, main in enumerate(main_values)}


def first_layer_extrusion_scale_for_layer(
    layer_key: LayerKey,
    *,
    main_layer_ranks: dict[int, int],
    first_layer_extrusion_scale: float | None,
) -> float:
    if first_layer_extrusion_scale is None:
        return 1.0
    if main_layer_ranks.get(layer_key.main, 1) == 0:
        return float(first_layer_extrusion_scale)
    return 1.0


def feedrate_for_path_fraction(
    *,
    base_feedrate: float | None,
    current_fraction: float,
    slowdown_fraction: float | None,
    slowdown_scale: float | None,
) -> float | None:
    if base_feedrate is None:
        return None
    if slowdown_fraction is None or slowdown_scale is None:
        return base_feedrate
    if slowdown_fraction <= 0.0 or slowdown_scale <= 0.0:
        return base_feedrate
    if current_fraction < slowdown_fraction:
        return float(base_feedrate) * float(slowdown_scale)
    return base_feedrate


def print_feedrate_for_layer(
    layer_key: LayerKey,
    *,
    main_layer_ranks: dict[int, int],
    print_feedrate: float | None,
    first_layer_count: int,
    first_layer_print_feedrate: float | None,
) -> float | None:
    if (
        first_layer_print_feedrate is not None
        and first_layer_count > 0
        and main_layer_ranks.get(layer_key.main, first_layer_count) < first_layer_count
    ):
        return first_layer_print_feedrate
    return print_feedrate


def append_travel_to_point(
    lines: list[str],
    point: np.ndarray,
    *,
    lift: float,
    feedrate: float | None,
    acceleration: float | None,
    current_acceleration: float | None,
) -> float | None:
    current_acceleration = _append_acceleration_command(
        lines,
        acceleration=acceleration,
        current_acceleration=current_acceleration,
    )
    feedrate_suffix = _feedrate_suffix(feedrate)
    lines.append(f"G1 X{point[0]:.5f} Y{point[1]:.5f} Z{point[2] + lift:.5f}{feedrate_suffix}")
    lines.append(f"G1 Z{point[2]:.5f}{feedrate_suffix}")
    return current_acceleration


def append_interrupt_transition(
    lines: list[str],
    current_point: np.ndarray,
    next_point: np.ndarray,
    machine: MachineProfile,
    *,
    reason: str,
    travel_feedrate: float | None,
    travel_acceleration: float | None,
    current_acceleration: float | None,
) -> float | None:
    z_clear = max(float(current_point[2]), float(next_point[2])) + machine.travel_lift
    current_acceleration = _append_acceleration_command(
        lines,
        acceleration=travel_acceleration,
        current_acceleration=current_acceleration,
    )
    feedrate_suffix = _feedrate_suffix(travel_feedrate)
    lines.append(INTERRUPT_COMMENT)
    lines.append(f"; {reason}")
    lines.append(f"G1 E{-machine.retract_amount:.5f}")
    lines.append(f"G1 Z{z_clear:.5f}{feedrate_suffix}")
    lines.append(f"G1 X{next_point[0]:.5f} Y{next_point[1]:.5f}{feedrate_suffix}")
    lines.append(f"G1 Z{z_clear:.5f}{feedrate_suffix}")
    lines.append(f"G1 Z{next_point[2]:.5f}{feedrate_suffix}")
    lines.append(f"G1 E{machine.pre_extrude_amount:.5f}")
    return current_acceleration


def emit_query(toolpath_dir: Path) -> None:
    toolpath_files = collect_toolpath_files(toolpath_dir)
    if not toolpath_files:
        raise FileNotFoundError(f"No *_3axis.txt files found in {toolpath_dir}")

    sorted_keys = sorted(toolpath_files, key=lambda key: (key.main, key.sub))
    grouped: dict[int, list[int]] = {}
    for key in sorted_keys:
        grouped.setdefault(key.main, []).append(key.sub)

    main_values = sorted(grouped)
    print(f"Toolpath folder: {toolpath_dir}")
    print(f"Main index range: {main_values[0]} -> {main_values[-1]}")
    for main in main_values:
        subs = ", ".join(str(sub) for sub in sorted(grouped[main]))
        print(f"{main}: sub {subs}")


def emit_export(
    toolpath_dir: Path,
    output_gcode: Path,
    *,
    machine_dir: Path,
    machine_name: str,
    start_main: int,
    end_main: int,
    travel_feedrate: float | None,
    print_feedrate: float | None,
    travel_acceleration: float | None,
    print_acceleration: float | None,
    path_start_slowdown_fraction: float | None,
    path_start_feedrate_scale: float | None,
    first_layer_extrusion_scale: float | None,
    first_layer_count: int,
    first_layer_print_feedrate: float | None,
    center_model: bool = False,
    bead_cross_section: str = "rounded_rectangle",
    model_width_from_spacing: bool = False,
) -> None:
    batch_metadata = load_batch_metadata(toolpath_dir)
    assert_resolved_support_heights(batch_metadata, start_main=start_main, end_main=end_main)
    machine = load_machine_profile(machine_dir, machine_name)
    toolpath_files = collect_toolpath_files(toolpath_dir)
    selected_keys = selected_layer_keys(list(toolpath_files.keys()), start_main=start_main, end_main=end_main)
    if not selected_keys:
        raise FileNotFoundError(
            f"No *_3axis.txt files found in main-index range {start_main}..{end_main}"
        )
    if first_layer_count < 0:
        raise ValueError("first_layer_count must be non-negative.")
    if path_start_slowdown_fraction is not None and not (0.0 <= path_start_slowdown_fraction <= 1.0):
        raise ValueError("path_start_slowdown_fraction must be in [0, 1] when provided.")
    if path_start_feedrate_scale is not None and path_start_feedrate_scale <= 0.0:
        raise ValueError("path_start_feedrate_scale must be positive when provided.")
    if first_layer_extrusion_scale is not None and first_layer_extrusion_scale <= 0.0:
        raise ValueError("first_layer_extrusion_scale must be positive when provided.")
    if bead_cross_section not in {"rectangular", "rounded_rectangle"}:
        raise ValueError("bead_cross_section must be 'rectangular' or 'rounded_rectangle'.")
    if model_width_from_spacing and bead_cross_section != "rounded_rectangle":
        raise ValueError("model_width_from_spacing requires rounded_rectangle bead geometry.")

    output_gcode.parent.mkdir(parents=True, exist_ok=True)
    area = filament_area(machine.filament_diameter)
    centering_offset = export_xy_centering_offset(toolpath_files, selected_keys, machine, enabled=center_model)
    lines: list[str] = []
    if machine.pre_gcode.strip():
        lines.append("; BEGIN PRE_GCODE")
        lines.extend(machine.pre_gcode.rstrip().splitlines())
        lines.append("; END PRE_GCODE")
    lines.append("; Relative extrusion mode")
    lines.append(f"; bead_cross_section={bead_cross_section}")
    if model_width_from_spacing:
        lines.append("; model_width_from_spacing=true; model paths keep their centreline positions")
    lines.append("M83")

    previous_endpoint: np.ndarray | None = None
    current_acceleration: float | None = None
    main_layer_ranks = main_layer_rank_lookup(selected_keys)

    for key in selected_keys:
        path_file = toolpath_files[key]
        rows = load_path_rows(path_file)
        width_profile = infer_width_profile_from_metadata(
            batch_metadata,
            key,
            toolpath_dir=toolpath_dir,
            point_count=rows.shape[0],
        )
        path_tag_blocks = infer_path_tag_blocks_from_metadata(
            batch_metadata,
            key,
            toolpath_dir=toolpath_dir,
            point_count=rows.shape[0],
        )
        path_break_indices = infer_path_break_indices_from_metadata(
            batch_metadata,
            key,
            toolpath_dir=toolpath_dir,
            point_count=rows.shape[0],
        )
        long_jump_events = infer_long_jump_events_from_metadata(
            batch_metadata,
            key,
            toolpath_dir=toolpath_dir,
            point_count=rows.shape[0],
        )
        # A per-point sidecar is authoritative. The scalar metadata width is
        # only a fallback for legacy batches without one.
        width = None if width_profile is not None else infer_width_from_metadata(batch_metadata, key)
        spacing_width = model_spacing_width_for_layer(batch_metadata, key, enabled=model_width_from_spacing)
        transformed_points = transform_points_y_up_to_z_up(rows[:, 0:3], machine.base_center) + centering_offset
        heights = rows[:, 6]
        nominal_segment_lengths = infer_nominal_segment_lengths(
            batch_metadata, key, toolpath_dir=toolpath_dir, point_count=rows.shape[0],
        )
        extrusion_scale = first_layer_extrusion_scale_for_layer(
            key,
            main_layer_ranks=main_layer_ranks,
            first_layer_extrusion_scale=first_layer_extrusion_scale,
        )
        layer_print_feedrate = print_feedrate_for_layer(
            key,
            main_layer_ranks=main_layer_ranks,
            print_feedrate=print_feedrate,
            first_layer_count=first_layer_count,
            first_layer_print_feedrate=first_layer_print_feedrate,
        )
        printable_lengths = np.linalg.norm(np.diff(transformed_points, axis=0), axis=1) if transformed_points.shape[0] > 1 else np.empty((0,), dtype=np.float64)
        printable_lengths = np.asarray(printable_lengths, dtype=np.float64)
        if printable_lengths.size > 0:
            printable_lengths = np.where(printable_lengths > machine.interrupt_threshold, 0.0, printable_lengths)
            printable_lengths[list(index - 1 for index in path_break_indices)] = 0.0
        total_print_length = float(np.sum(printable_lengths))

        lines.append(f";;;;;;{path_file.name};;;;;;;;;;;;")
        if layer_print_feedrate is not None:
            lines.append(f"; print_feedrate={layer_print_feedrate:.5f}")
        if print_acceleration is not None:
            lines.append(f"; print_acceleration={print_acceleration:.5f}")
        if path_start_slowdown_fraction is not None and path_start_feedrate_scale is not None:
            lines.append(
                f"; path_start_slowdown_fraction={path_start_slowdown_fraction:.5f} "
                f"path_start_feedrate_scale={path_start_feedrate_scale:.5f}"
            )
        if not math.isclose(extrusion_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            lines.append(f"; first_layer_extrusion_scale={extrusion_scale:.5f}")
        active_tag = path_tag_blocks[0][2] if path_tag_blocks else None
        next_tag_block = 1
        first_point = transformed_points[0]
        lines.append(";role: travel")
        if previous_endpoint is None:
            current_acceleration = append_travel_to_point(
                lines,
                first_point,
                lift=machine.travel_lift,
                feedrate=travel_feedrate,
                acceleration=travel_acceleration,
                current_acceleration=current_acceleration,
            )
            lines.append(f";jump: file_start {path_file.name}")
        else:
            lines.append(f";jump: file_transition {path_file.name}")
            current_acceleration = append_interrupt_transition(
                lines,
                previous_endpoint,
                first_point,
                machine,
                reason=f"File transition to {path_file.name}",
                travel_feedrate=travel_feedrate,
                travel_acceleration=travel_acceleration,
                current_acceleration=current_acceleration,
            )
        if active_tag is not None:
            lines.append(f";role: {active_tag}")

        cumulative_print_length = 0.0
        for idx in range(rows.shape[0] - 1):
            current_point = transformed_points[idx]
            next_point = transformed_points[idx + 1]
            segment_length = float(np.linalg.norm(next_point - current_point))
            resume_tag = active_tag
            if path_tag_blocks is not None and next_tag_block < len(path_tag_blocks):
                start_index, _, next_tag = path_tag_blocks[next_tag_block]
                if idx + 1 == start_index:
                    resume_tag = next_tag
                    next_tag_block += 1
            if idx + 1 in path_break_indices:
                lines.append(";role: travel")
                jump_length = long_jump_events.get(idx + 1)
                lines.append(
                    f";jump: long_jump index={idx + 1} length={jump_length:.5f}"
                    if jump_length is not None
                    else f";jump: path_break index={idx + 1}",
                )
                current_acceleration = append_interrupt_transition(
                    lines,
                    current_point,
                    next_point,
                    machine,
                    reason=f"Path break in {path_file.name}",
                    travel_feedrate=travel_feedrate,
                    travel_acceleration=travel_acceleration,
                    current_acceleration=current_acceleration,
                )
                if resume_tag is not None:
                    lines.append(f";role: {resume_tag}")
                    active_tag = resume_tag
                continue
            if segment_length > machine.interrupt_threshold:
                lines.append(";role: travel")
                lines.append(
                    f";jump: long_distance index={idx + 1} "
                    f"length={segment_length:.5f}"
                )
                current_acceleration = append_interrupt_transition(
                    lines,
                    current_point,
                    next_point,
                    machine,
                    reason=(
                        f"Long jump in {path_file.name}: "
                        f"{segment_length:.5f} > {machine.interrupt_threshold:.5f}"
                    ),
                    travel_feedrate=travel_feedrate,
                    travel_acceleration=travel_acceleration,
                    current_acceleration=current_acceleration,
                )
                if resume_tag is not None:
                    lines.append(f";role: {resume_tag}")
                    active_tag = resume_tag
                continue
            if resume_tag is not None and resume_tag != active_tag:
                lines.append(f";role: {resume_tag}")
                active_tag = resume_tag

            avg_height = 0.5 * (float(heights[idx]) + float(heights[idx + 1]))
            segment_width = (
                float(0.5 * (width_profile[idx] + width_profile[idx + 1]))
                if width_profile is not None
                else float(width)
            )
            volume_length = segment_length if nominal_segment_lengths is None else float(nominal_segment_lengths[idx])
            volume = volume_length * bead_cross_section_area(
                segment_width,
                avg_height,
                mode=bead_cross_section,
                width_is_spacing=spacing_width and active_tag != 'ironing',
            )
            extrusion = machine.alpha * extrusion_scale * volume / area
            progress_fraction = (
                cumulative_print_length / total_print_length
                if total_print_length > 0.0
                else 1.0
            )
            segment_print_feedrate = feedrate_for_path_fraction(
                base_feedrate=layer_print_feedrate,
                current_fraction=progress_fraction,
                slowdown_fraction=path_start_slowdown_fraction,
                slowdown_scale=path_start_feedrate_scale,
            )
            print_feedrate_suffix = _feedrate_suffix(segment_print_feedrate)
            current_acceleration = _append_acceleration_command(
                lines,
                acceleration=print_acceleration,
                current_acceleration=current_acceleration,
            )
            lines.append(
                f"G1 X{next_point[0]:.5f} Y{next_point[1]:.5f} Z{next_point[2]:.5f} E{extrusion:.5f}{print_feedrate_suffix}"
            )
            cumulative_print_length += segment_length

        previous_endpoint = transformed_points[-1]

    if previous_endpoint is not None:
        lines.append(
            f"G1 Z{float(previous_endpoint[2]) + 2.0 * machine.travel_lift:.5f}"
        )

    if machine.post_gcode.strip():
        lines.append("; BEGIN POST_GCODE")
        lines.extend(machine.post_gcode.rstrip().splitlines())
        lines.append("; END POST_GCODE")

    output_gcode.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported G-code: {output_gcode}")
    print(f"Machine: {machine.machine_name}")
    print(f"Main index range: {start_main} -> {end_main}")
    print(f"Included files: {len(selected_keys)}")


def _compact_bare_metadata(metadata, keys):
    """Send only fields consumed by layer conversion, never nested patch reports."""
    fields = ('layer_main', 'layer_sub', 'material_index', 'width_profile_file', 'part', 'layer_role',
              'path_tag_file', 'path_break_file', 'path_jump_file',
              'nominal_segment_length_file', 'source_metadata','surface_metric_mesh',
              'surface_hatch_directions','parallel_axis')
    raw=metadata.get('source_path_metadata',{})
    raw_by_key={(int(r.get('layer_main',-1)),int(r.get('layer_sub',-1))):r for r in raw.get('layers',[]) if isinstance(r,dict)} if isinstance(raw,dict) else {}
    layers_by_key = {}
    for layer in metadata.get('layers', []):
        if not isinstance(layer, dict):
            continue
        key = (int(layer.get('layer_main', -1)), int(layer.get('layer_sub', -1)))
        selected = {name: layer[name] for name in fields if name in layer}
        source_layer=raw_by_key.get(key,{})
        for name in ('surface_metric_mesh','surface_hatch_directions','parallel_axis'):
            if name not in selected and name in source_layer:selected[name]=source_layer[name]
        if not selected.get('surface_metric_mesh') and isinstance(raw,dict) and raw.get('input_dir') and source_layer.get('source_mesh'):
            selected['surface_metric_mesh']=str(Path(raw['input_dir'])/source_layer['source_mesh'])
        source = selected.get('source_metadata')
        if isinstance(source, dict):
            selected['source_metadata'] = {name: source[name] for name in ('width', 'contour_spacing', 'part', 'layer_role') if name in source}
        layers_by_key.setdefault(key, []).append(selected)
    shared = {name: metadata[name] for name in ('width','model_lift','model_lift_axis') if name in metadata}
    source = metadata.get('source_path_metadata')
    if isinstance(source, dict):
        shared['source_path_metadata'] = {name: source[name] for name in ('width', 'contour_spacing') if name in source}
    return {(key.main, key.sub): dict(shared, layers=layers_by_key.get((key.main, key.sub), [])) for key in keys}


def _initialize_bare_worker(context):
    global _bare_worker_context
    _bare_worker_context = context


def _run_bare_job(job):
    key, path_file, metadata, flow_layers = job
    writer = _DefaultLineWriter
    if flow_layers is not None:
        from export_projected_manifest_support import make_flow_writer
        writer = make_flow_writer(flow_layers)
    _emit_bare_layer(key, path_file, batch_metadata=metadata, writer_factory=writer, **_bare_worker_context)


def _emit_bare_layer(key, path_file, *, toolpath_dir, output_dir, batch_metadata,
                     machine, area, main_layer_ranks, centering_offset,
                     travel_feedrate, print_feedrate, path_start_slowdown_fraction,
                     path_start_feedrate_scale, first_layer_extrusion_scale,
                     first_layer_count, first_layer_print_feedrate,
                     include_material_index_in_name, bead_cross_section,
                     model_width_from_spacing=False,
                     surface_spacing_correction=False,max_surface_spacing=2.,
                     writer_factory=None):
    """Convert one independent layer using the unchanged serial calculations."""
    rows = load_path_rows(path_file)

    output_stem = key.stem
    if include_material_index_in_name:
        material_index = infer_material_index_from_metadata(batch_metadata, key)
        if material_index is None:
            raise ValueError(
                f"Layer {key.stem} has no material_index in batch metadata, so the material suffix cannot be written.",
            )
        output_stem = f"{output_stem}_{material_index}"

    output_path = output_dir / f"{output_stem}.gcode"
    if rows.shape[0] == 0:
        output_path.write_text("", encoding="utf-8")
        return

    width_profile = infer_width_profile_from_metadata(
        batch_metadata,
        key,
        toolpath_dir=toolpath_dir,
        point_count=rows.shape[0],
    )
    width = None if width_profile is not None else infer_width_from_metadata(batch_metadata, key)
    spacing_width = model_spacing_width_for_layer(batch_metadata, key, enabled=model_width_from_spacing)
    path_tag_blocks = infer_path_tag_blocks_from_metadata(
        batch_metadata,
        key,
        toolpath_dir=toolpath_dir,
        point_count=rows.shape[0],
    )
    path_break_indices = infer_path_break_indices_from_metadata(
        batch_metadata,
        key,
        toolpath_dir=toolpath_dir,
        point_count=rows.shape[0],
    )
    long_jump_events = infer_long_jump_events_from_metadata(
        batch_metadata,
        key,
        toolpath_dir=toolpath_dir,
        point_count=rows.shape[0],
    )
    transformed_points = transform_points_y_up_to_z_up(rows[:, 0:3], machine.base_center) + centering_offset
    heights = rows[:, 6]
    nominal_segment_lengths = infer_nominal_segment_lengths(
        batch_metadata, key, toolpath_dir=toolpath_dir, point_count=rows.shape[0],
    )
    corrected_spacing=np.zeros(max(0,len(rows)-1))
    if surface_spacing_correction and spacing_width and path_tag_blocks:
        from surface_hatch_metric import layer_surface_spacings
        tags=np.full(len(rows),'',dtype=object)
        for start,end,tag in path_tag_blocks:tags[start:end]=tag
        layer=next((r for r in batch_metadata.get('layers',[]) if
            (int(r.get('layer_main',-1)),int(r.get('layer_sub',-1)))==(key.main,key.sub)),{})
        layer=dict(layer)
        raw=batch_metadata.get('source_path_metadata',{})
        source_layer=next((r for r in raw.get('layers',[]) if
            (int(r.get('layer_main',-1)),int(r.get('layer_sub',-1)))==(key.main,key.sub)),{}) if isinstance(raw,dict) else {}
        for name in ('surface_metric_mesh','surface_hatch_directions','parallel_axis'):
            if name not in layer and name in source_layer:layer[name]=source_layer[name]
        if not layer.get('surface_metric_mesh') and isinstance(raw,dict) and raw.get('input_dir') and source_layer.get('source_mesh'):
            layer['surface_metric_mesh']=str(Path(raw['input_dir'])/source_layer['source_mesh'])
        if layer.get('part','model')=='model':
            profile=width_profile if width_profile is not None else np.full(len(rows),width)
            corrected_spacing,metric_report=layer_surface_spacings(rows,profile,tags,path_break_indices,layer,batch_metadata,maximum_spacing=max_surface_spacing)
            (output_dir/f'{output_stem}.surface_metric.json').write_text(json.dumps(metric_report,indent=2),encoding='utf8')
    extrusion_scale = first_layer_extrusion_scale_for_layer(
        key,
        main_layer_ranks=main_layer_ranks,
        first_layer_extrusion_scale=first_layer_extrusion_scale,
    )
    layer_print_feedrate = print_feedrate_for_layer(
        key,
        main_layer_ranks=main_layer_ranks,
        print_feedrate=print_feedrate,
        first_layer_count=first_layer_count,
        first_layer_print_feedrate=first_layer_print_feedrate,
    )
    printable_lengths = np.linalg.norm(np.diff(transformed_points, axis=0), axis=1) if transformed_points.shape[0] > 1 else np.empty((0,), dtype=np.float64)
    printable_lengths = np.asarray(printable_lengths, dtype=np.float64)
    total_print_length = float(np.sum(printable_lengths))

    bare_lines = (_StreamingLineWriter if writer_factory is None else writer_factory)(output_path)
    if spacing_width:
        bare_lines.append('; model_width_from_spacing=true; area=local_spacing*physical_height; ironing uses explicit width')
    first_point = transformed_points[0]
    active_tag: str | None = None
    next_tag_block = 1
    if path_tag_blocks:
        active_tag = path_tag_blocks[0][2]
        bare_lines.append(f";role: {active_tag}")
    feedrate_suffix = _feedrate_suffix(travel_feedrate)
    bare_lines.append(
        f"G1 X{first_point[0]:.5f} Y{first_point[1]:.5f} Z{first_point[2]:.5f} E0.00000{feedrate_suffix}"
    )

    cumulative_print_length = 0.0
    cached_print_feedrate = object()
    print_feedrate_suffix = ""
    for idx in range(rows.shape[0] - 1):
        current_point = transformed_points[idx]
        next_point = transformed_points[idx + 1]
        segment_length = float(np.linalg.norm(next_point - current_point))
        resume_tag = active_tag
        if path_tag_blocks is not None and next_tag_block < len(path_tag_blocks):
            start_index, _, next_tag = path_tag_blocks[next_tag_block]
            if idx + 1 == start_index:
                resume_tag = next_tag
                next_tag_block += 1
        if idx + 1 in path_break_indices:
            bare_lines.append(";role: travel")
            jump_length = long_jump_events.get(idx + 1)
            bare_lines.append(
                f";jump: long_jump index={idx + 1} length={jump_length:.5f}"
                if jump_length is not None
                else f";jump: path_break index={idx + 1}",
            )
            bare_lines.append(
                f"G1 X{next_point[0]:.5f} Y{next_point[1]:.5f} Z{next_point[2]:.5f} E0.00000{feedrate_suffix}",
            )
            if resume_tag is not None:
                bare_lines.append(f";role: {resume_tag}")
                active_tag = resume_tag
            continue
        if resume_tag is not None and resume_tag != active_tag:
            bare_lines.append(f";role: {resume_tag}")
            active_tag = resume_tag
        avg_height = 0.5 * (float(heights[idx]) + float(heights[idx + 1]))
        segment_width = float(0.5 * (width_profile[idx] + width_profile[idx + 1])) if width_profile is not None else width
        volume_length = segment_length if nominal_segment_lengths is None else float(nominal_segment_lengths[idx])
        if corrected_spacing[idx]>0:segment_width=float(corrected_spacing[idx])
        volume = volume_length * bead_cross_section_area(
            segment_width,
            avg_height,
            mode=bead_cross_section,
            width_is_spacing=spacing_width and active_tag != 'ironing',
        )
        extrusion = machine.alpha * extrusion_scale * volume / area
        progress_fraction = (
            cumulative_print_length / total_print_length
            if total_print_length > 0.0
            else 1.0
        )
        segment_print_feedrate = feedrate_for_path_fraction(
            base_feedrate=layer_print_feedrate,
            current_fraction=progress_fraction,
            slowdown_fraction=path_start_slowdown_fraction,
            slowdown_scale=path_start_feedrate_scale,
        )
        if segment_print_feedrate is not cached_print_feedrate:
            print_feedrate_suffix = _feedrate_suffix(segment_print_feedrate)
            cached_print_feedrate = segment_print_feedrate
        bare_lines.append(
            f"G1 X{next_point[0]:.5f} Y{next_point[1]:.5f} Z{next_point[2]:.5f} E{extrusion:.5f}{print_feedrate_suffix}"
        )
        cumulative_print_length += segment_length

    bare_lines.close()



def emit_bare_export(
    toolpath_dir: Path,
    output_dir: Path,
    *,
    machine_dir: Path,
    machine_name: str,
    start_main: int,
    end_main: int,
    travel_feedrate: float | None,
    print_feedrate: float | None,
    path_start_slowdown_fraction: float | None,
    path_start_feedrate_scale: float | None,
    first_layer_extrusion_scale: float | None,
    first_layer_count: int,
    first_layer_print_feedrate: float | None,
    include_material_index_in_name: bool,
    center_model: bool = False,
    bead_cross_section: str = "rounded_rectangle",
    model_width_from_spacing: bool = False,
    surface_spacing_correction: bool = False,
    max_surface_spacing: float = 2.,
    batch_metadata: dict | None = None,
    layer_workers: int | None = None,
) -> None:
    batch_metadata = _IndexedBatchMetadata(load_batch_metadata(toolpath_dir) if batch_metadata is None else batch_metadata)
    assert_resolved_support_heights(batch_metadata, start_main=start_main, end_main=end_main)
    machine = load_machine_profile(machine_dir, machine_name)
    toolpath_files = collect_toolpath_files(toolpath_dir)
    selected_keys = selected_layer_keys(list(toolpath_files.keys()), start_main=start_main, end_main=end_main)
    if not selected_keys:
        raise FileNotFoundError(
            f"No *_3axis.txt files found in main-index range {start_main}..{end_main}"
        )
    if first_layer_count < 0:
        raise ValueError("first_layer_count must be non-negative.")
    if path_start_slowdown_fraction is not None and not (0.0 <= path_start_slowdown_fraction <= 1.0):
        raise ValueError("path_start_slowdown_fraction must be in [0, 1] when provided.")
    if path_start_feedrate_scale is not None and path_start_feedrate_scale <= 0.0:
        raise ValueError("path_start_feedrate_scale must be positive when provided.")
    if first_layer_extrusion_scale is not None and first_layer_extrusion_scale <= 0.0:
        raise ValueError("first_layer_extrusion_scale must be positive when provided.")
    if bead_cross_section not in {"rectangular", "rounded_rectangle"}:
        raise ValueError("bead_cross_section must be 'rectangular' or 'rounded_rectangle'.")
    if model_width_from_spacing and bead_cross_section != "rounded_rectangle":
        raise ValueError("model_width_from_spacing requires rounded_rectangle bead geometry.")

    if surface_spacing_correction and (not model_width_from_spacing or not np.isfinite(max_surface_spacing) or max_surface_spacing<=0):
        raise ValueError('Surface correction requires spacing-derived model extrusion and a positive finite spacing limit.')

    output_dir.mkdir(parents=True, exist_ok=True)
    area = filament_area(machine.filament_diameter)
    main_layer_ranks = main_layer_rank_lookup(selected_keys)
    centering_offset = export_xy_centering_offset(toolpath_files, selected_keys, machine, enabled=center_model)

    context = dict(toolpath_dir=toolpath_dir, output_dir=output_dir, machine=machine,
        area=area, main_layer_ranks=main_layer_ranks, centering_offset=centering_offset,
        travel_feedrate=travel_feedrate, print_feedrate=print_feedrate,
        path_start_slowdown_fraction=path_start_slowdown_fraction,
        path_start_feedrate_scale=path_start_feedrate_scale,
        first_layer_extrusion_scale=first_layer_extrusion_scale,
        first_layer_count=first_layer_count, first_layer_print_feedrate=first_layer_print_feedrate,
        include_material_index_in_name=include_material_index_in_name, bead_cross_section=bead_cross_section,
        model_width_from_spacing=model_width_from_spacing,surface_spacing_correction=surface_spacing_correction,
        max_surface_spacing=max_surface_spacing)
    if layer_workers is None:
        workers = min(4, os.cpu_count() or 1, len(selected_keys))
        # Avoid spawn overhead on small automatic exports; explicit requests win.
        if sum(toolpath_files[key].stat().st_size for key in selected_keys) < 16 * 1024 * 1024:
            workers = 1
    else:
        if layer_workers < 1:
            raise ValueError("layer_workers must be positive.")
        workers = min(int(layer_workers), len(selected_keys))
    flow_layers = getattr(_StreamingLineWriter, '_bare_flow_layers', None)
    if (batch_metadata.layers_by_key is None
            or (_StreamingLineWriter is not _DefaultLineWriter and flow_layers is None)):
        # Keep lazy metadata errors and unsupported custom writers on their old path.
        workers = 1
    if workers == 1:
        for key in selected_keys:
            _emit_bare_layer(key, toolpath_files[key], batch_metadata=batch_metadata, **context)
    else:
        metadata_by_key = _compact_bare_metadata(batch_metadata, selected_keys)
        jobs = []
        for key in selected_keys:
            flow_layer = None
            if flow_layers is not None:
                flow_layer = {}
                if (key.main, key.sub) in flow_layers:
                    source_layer = flow_layers[(key.main, key.sub)]
                    flow_layer[(key.main, key.sub)] = {name: source_layer[name] for name in
                        ('part', 'original_main', 'original_sub', 'scalar', 'extrusion_role_scales')
                        if name in source_layer}
            jobs.append((key, toolpath_files[key], metadata_by_key[(key.main, key.sub)], flow_layer))
        print(f"[bare export] {len(jobs)} layers; workers={workers}", flush=True)
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_initialize_bare_worker, initargs=(context,), mp_context=get_context('spawn'))
        futures = []
        try:
            futures = [pool.submit(_run_bare_job, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    print(f"Exported bare G-code files: {output_dir}")
    print(f"Machine: {machine.machine_name}")
    print(f"Main index range: {start_main} -> {end_main}")
    print(f"Included files: {len(selected_keys)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query or export machine G-code from height-augmented 3-axis toolpaths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    query_parser = subparsers.add_parser(
        "query",
        help="Report the available main/sub index coverage in a toolpath folder.",
    )
    query_parser.add_argument(
        "toolpath_dir",
        type=Path,
        help="Folder containing *_3axis.txt files and batch_3axis.meta.json.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a selected main-index range to one machine G-code file.",
    )
    export_parser.add_argument(
        "toolpath_dir",
        type=Path,
        help="Folder containing *_3axis.txt files and batch_3axis.meta.json.",
    )
    export_parser.add_argument(
        "output_gcode",
        type=Path,
        help="Output G-code file path.",
    )
    export_parser.add_argument(
        "--machine-dir",
        type=Path,
        default=Path("machines"),
        help="Folder containing machine profile JSON and referenced G-code blocks.",
    )
    export_parser.add_argument(
        "--machine-name",
        type=str,
        required=True,
        help="Machine profile name, resolved as <machine-name>.json inside the machine folder.",
    )
    export_parser.add_argument(
        "--start-main",
        type=int,
        required=True,
        help="Inclusive starting main index for export.",
    )
    export_parser.add_argument(
        "--end-main",
        type=int,
        required=True,
        help="Inclusive ending main index for export.",
    )
    export_parser.add_argument(
        "--center-model",
        action="store_true",
        help="Center the selected XY toolpath footprint on the machine profile base_center before export.",
    )
    export_parser.add_argument(
        "--bead-cross-section",
        choices=("rectangular", "rounded_rectangle"),
        default="rounded_rectangle",
        help="Extrusion bead-area model. Defaults to rounded rectangle; rectangular retains the legacy width*height model.",
    )
    export_parser.add_argument(
        "--model-width-from-spacing", action="store_true",
        help="Treat model width profiles as assigned spacing and derive rounded bead width from local height. Keeps centres fixed; the bead envelope can grow. Support and ironing keep explicit widths.",
    )
    export_parser.add_argument(
        "--travel-feedrate",
        type=float,
        default=None,
        help="Optional override feedrate for non-printing transition G1 moves. Defaults to machine profile value if present.",
    )
    export_parser.add_argument(
        "--print-feedrate",
        type=float,
        default=None,
        help="Optional override feedrate for extrusion G1 moves after the first-layer override no longer applies. Defaults to machine profile value if present.",
    )
    export_parser.add_argument(
        "--travel-acceleration",
        type=float,
        default=None,
        help="Optional override acceleration for non-printing transition moves. Emits M204 S... when set or present in the machine profile.",
    )
    export_parser.add_argument(
        "--print-acceleration",
        type=float,
        default=None,
        help="Optional override acceleration for extrusion moves. Emits M204 S... when set or present in the machine profile.",
    )
    export_parser.add_argument(
        "--path-start-slowdown-fraction",
        type=float,
        default=None,
        help="Optional fraction in [0, 1] of each toolpath layer whose printable moves use a slowed print feedrate.",
    )
    export_parser.add_argument(
        "--path-start-feedrate-scale",
        type=float,
        default=None,
        help="Optional positive multiplier applied to print feedrate inside --path-start-slowdown-fraction. Example: 0.5 halves the speed.",
    )
    export_parser.add_argument(
        "--first-layer-extrusion-scale",
        type=float,
        default=None,
        help="Optional positive multiplier applied to extrusion on the first selected main layer only.",
    )
    export_parser.add_argument(
        "--first-layer-count",
        type=int,
        default=None,
        help="Optional override for how many selected main layers should use --first-layer-print-feedrate. Defaults to machine profile value if present.",
    )
    export_parser.add_argument(
        "--first-layer-print-feedrate",
        type=float,
        default=None,
        help="Optional override extrusion feedrate for the first N selected main layers. Defaults to machine profile value if present.",
    )

    export_bare_parser = subparsers.add_parser(
        "export-bare",
        help="Export one bare G-code file per selected layer with only G1 X Y Z E lines.",
    )
    export_bare_parser.add_argument(
        "--layer-workers", type=int, default=None,
        help="Parallel bare conversion processes. Default: up to four for exports >=16 MiB; 1 forces serial conversion.",
    )
    export_bare_parser.add_argument(
        "toolpath_dir",
        type=Path,
        help="Folder containing *_3axis.txt files and batch_3axis.meta.json.",
    )
    export_bare_parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder for <main>_<sub>.gcode files.",
    )
    export_bare_parser.add_argument(
        "--machine-dir",
        type=Path,
        default=Path("machines"),
        help="Folder containing machine profile JSON and referenced G-code blocks.",
    )
    export_bare_parser.add_argument(
        "--machine-name",
        type=str,
        required=True,
        help="Machine profile name, resolved as <machine-name>.json inside the machine folder.",
    )
    export_bare_parser.add_argument(
        "--start-main",
        type=int,
        required=True,
        help="Inclusive starting main index for export.",
    )
    export_bare_parser.add_argument(
        "--end-main",
        type=int,
        required=True,
        help="Inclusive ending main index for export.",
    )
    export_bare_parser.add_argument(
        "--center-model",
        action="store_true",
        help="Center the selected XY toolpath footprint on the machine profile base_center before export.",
    )
    export_bare_parser.add_argument(
        "--bead-cross-section",
        choices=("rectangular", "rounded_rectangle"),
        default="rounded_rectangle",
        help="Extrusion bead-area model. Defaults to rounded rectangle; rectangular retains the legacy width*height model.",
    )
    export_bare_parser.add_argument(
        "--model-width-from-spacing", action="store_true",
        help="Treat model width profiles as assigned spacing and derive rounded bead width from local height. Keeps centres fixed; the bead envelope can grow. Support and ironing keep explicit widths.",
    )
    export_bare_parser.add_argument('--surface-spacing-correction',action='store_true',help='Correct qualified interior hatch spacing on the saved surface; walls and links keep their existing policy.')
    export_bare_parser.add_argument('--max-surface-spacing',type=float,default=2.,help='Maximum qualified physical assigned spacing (mm); not a printer calibration.')
    export_bare_parser.add_argument(
        "--travel-feedrate",
        type=float,
        default=None,
        help="Optional feedrate attached to the initial non-printing positioning move.",
    )
    export_bare_parser.add_argument(
        "--print-feedrate",
        type=float,
        default=None,
        help="Optional override feedrate for extrusion G1 moves after the first-layer override no longer applies. Defaults to machine profile value if present.",
    )
    export_bare_parser.add_argument(
        "--path-start-slowdown-fraction",
        type=float,
        default=None,
        help="Optional fraction in [0, 1] of each toolpath layer whose printable moves use a slowed print feedrate.",
    )
    export_bare_parser.add_argument(
        "--path-start-feedrate-scale",
        type=float,
        default=None,
        help="Optional positive multiplier applied to print feedrate inside --path-start-slowdown-fraction. Example: 0.5 halves the speed.",
    )
    export_bare_parser.add_argument(
        "--first-layer-extrusion-scale",
        type=float,
        default=None,
        help="Optional positive multiplier applied to extrusion on the first selected main layer only.",
    )
    export_bare_parser.add_argument(
        "--first-layer-count",
        type=int,
        default=None,
        help="Optional override for how many selected main layers should use --first-layer-print-feedrate. Defaults to machine profile value if present.",
    )
    export_bare_parser.add_argument(
        "--first-layer-print-feedrate",
        type=float,
        default=None,
        help="Optional override extrusion feedrate for the first N selected main layers. Defaults to machine profile value if present.",
    )
    export_bare_parser.add_argument(
        "--include-material-index-in-name",
        action="store_true",
        help="Append _<material_index> to each bare export filename using batch metadata, yielding <main>_<sub>_<material>.gcode.",
    )

    return parser


def main(*, batch_metadata: dict | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "query":
        emit_query(args.toolpath_dir)
        return

    if args.command == "export":
        machine = load_machine_profile(args.machine_dir, args.machine_name)
        emit_export(
            args.toolpath_dir,
            args.output_gcode,
            machine_dir=args.machine_dir,
            machine_name=args.machine_name,
            start_main=args.start_main,
            end_main=args.end_main,
            travel_feedrate=machine.travel_feedrate if args.travel_feedrate is None else float(args.travel_feedrate),
            print_feedrate=machine.print_feedrate if args.print_feedrate is None else float(args.print_feedrate),
            travel_acceleration=(
                machine.travel_acceleration
                if args.travel_acceleration is None
                else float(args.travel_acceleration)
            ),
            print_acceleration=(
                machine.print_acceleration
                if args.print_acceleration is None
                else float(args.print_acceleration)
            ),
            path_start_slowdown_fraction=(
                machine.path_start_slowdown_fraction
                if args.path_start_slowdown_fraction is None
                else float(args.path_start_slowdown_fraction)
            ),
            path_start_feedrate_scale=(
                machine.path_start_feedrate_scale
                if args.path_start_feedrate_scale is None
                else float(args.path_start_feedrate_scale)
            ),
            first_layer_extrusion_scale=(
                machine.first_layer_extrusion_scale
                if args.first_layer_extrusion_scale is None
                else float(args.first_layer_extrusion_scale)
            ),
            first_layer_count=machine.first_layer_count if args.first_layer_count is None else int(args.first_layer_count),
            first_layer_print_feedrate=(
                machine.first_layer_print_feedrate
                if args.first_layer_print_feedrate is None
                else float(args.first_layer_print_feedrate)
            ),
            center_model=bool(args.center_model),
            bead_cross_section=str(args.bead_cross_section),
            model_width_from_spacing=bool(args.model_width_from_spacing),
        )
        return

    if args.command == "export-bare":
        machine = load_machine_profile(args.machine_dir, args.machine_name)
        emit_bare_export(
            args.toolpath_dir,
            args.output_dir,
            machine_dir=args.machine_dir,
            machine_name=args.machine_name,
            start_main=args.start_main,
            end_main=args.end_main,
            travel_feedrate=machine.travel_feedrate if args.travel_feedrate is None else float(args.travel_feedrate),
            print_feedrate=machine.print_feedrate if args.print_feedrate is None else float(args.print_feedrate),
            path_start_slowdown_fraction=(
                machine.path_start_slowdown_fraction
                if args.path_start_slowdown_fraction is None
                else float(args.path_start_slowdown_fraction)
            ),
            path_start_feedrate_scale=(
                machine.path_start_feedrate_scale
                if args.path_start_feedrate_scale is None
                else float(args.path_start_feedrate_scale)
            ),
            first_layer_extrusion_scale=(
                machine.first_layer_extrusion_scale
                if args.first_layer_extrusion_scale is None
                else float(args.first_layer_extrusion_scale)
            ),
            first_layer_count=machine.first_layer_count if args.first_layer_count is None else int(args.first_layer_count),
            first_layer_print_feedrate=(
                machine.first_layer_print_feedrate
                if args.first_layer_print_feedrate is None
                else float(args.first_layer_print_feedrate)
            ),
            include_material_index_in_name=bool(args.include_material_index_in_name),
            layer_workers=args.layer_workers,
            center_model=bool(args.center_model),
            bead_cross_section=str(args.bead_cross_section),
            model_width_from_spacing=bool(args.model_width_from_spacing),
            batch_metadata=batch_metadata,
            surface_spacing_correction=bool(args.surface_spacing_correction),
            max_surface_spacing=float(args.max_surface_spacing),
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

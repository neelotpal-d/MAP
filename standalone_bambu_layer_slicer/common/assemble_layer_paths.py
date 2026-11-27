#!/usr/bin/env python3
"""Assemble ordered layer-path G-code snippets into one multimaterial G-code.

Input snippet names must be:

    <main_index>_<sub_index>_<material_number>.gcode

or, for single-material folders:

    <main_index>_<sub_index>.gcode

where material_number is normally 0 or 1. Files without a material suffix use
single_material_default_material, which defaults to 1. The script keeps
pre_code and post_code untouched, follows the saved toolpath order (or legacy
main/sub order when no schedule exists), and emits a
compact tool-change block whenever the physical tool changes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


SNIPPET_RE = re.compile(r"^(\d+)_(\d+)_([0-9]+)\.gcode$", re.IGNORECASE)
SINGLE_MATERIAL_SNIPPET_RE = re.compile(r"^(\d+)_(\d+)\.gcode$", re.IGNORECASE)
TOOL_LINE_RE = re.compile(r"^(\s*)T([01])(\s*(?:;.*)?)$")
M620_START_RE = re.compile(r"^(\s*)M620\s+S([01])A(.*)$")
M621_END_RE = re.compile(r"^(\s*)M621\s+S([01])A(.*)$")
M620_AIR_RE = re.compile(r"^(\s*)M620\.6\s+I([01])(\s+.*)$")
G383_ACTIVE_L_RE = re.compile(r"^(\s*G383(?:\.3)?\s+.*\sL)([01])(\s*(?:;.*)?)$")
Z_WORD_RE = re.compile(r"\bZ(-?\d+(?:\.\d+)?)")
X_WORD_RE = re.compile(r"\bX(-?\d+(?:\.\d+)?)")
Y_WORD_RE = re.compile(r"\bY(-?\d+(?:\.\d+)?)")
F_WORD_RE = re.compile(r"\bF(-?\d+(?:\.\d+)?)")
E_WORD_RE = re.compile(r"\bE([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
# Common generated moves have explicit XYZ in this order. Other syntax keeps
# the established per-word parser, including relative modes and inline comments.
_DECIMAL_TOKEN = r"(-?\d+(?:\.\d+)?)"
_EXPLICIT_MOVE_RE = re.compile(
    r"[ \t]*G[01][ \t]+X" + _DECIMAL_TOKEN
    + r"[ \t]+Y" + _DECIMAL_TOKEN + r"[ \t]+Z" + _DECIMAL_TOKEN
    + r"(?:[ \t]+E" + _DECIMAL_TOKEN + r")?"
    + r"(?:[ \t]+F" + _DECIMAL_TOKEN + r")?[ \t]*"
)
ROLE_TAG_RE = re.compile(r"^;\s*role\s*:\s*(.+?)\s*$", re.IGNORECASE)
JUMP_TAG_RE = re.compile(r"^;\s*jump\s*(?::\s*(?:true|yes|1))?\s*$", re.IGNORECASE)
M204_S_RE = re.compile(r"\bS(-?\d+(?:\.\d+)?)", re.IGNORECASE)
ZERO_E_WORD_RE = re.compile(r"(?<!\S)E[+-]?0(?:\.0+)?(?=\s|$)", re.IGNORECASE)
MOTION_COMMAND_RE = re.compile(r"^\s*G[01]\b", re.IGNORECASE)
M620_10_RE = re.compile(
    r"^(\s*M620\.10\b.*?\bA)(\d+)(\b.*?\bT)(-?\d+(?:\.\d+)?)(\b.*?\bP)(-?\d+(?:\.\d+)?)(.*)$"
)
POST_PARK_Z_RE = re.compile(
    r"^(?P<prefix>[ \t]*G1[ \t]+Z)(?P<z>-?\d+(?:\.\d+)?)(?P<suffix>[ \t]+F\d+(?:\.\d+)?)(?P<comment>[ \t]*;[ \t]*lower z a little[ \t]*)$",
    re.IGNORECASE | re.MULTILINE,
)


PRE_CODE_SECTIONS = {
    "auto_extrude_cali": (
        ";===== auto extrude cali start =========================",
        ";===== auto extrude cali end =========================",
        "skip_auto_extrude_cali",
    ),
    "bed_leveling": (
        ";===== bed leveling ==================================",
        ";===== bed leveling end ================================",
        "skip_bed_leveling",
    ),
    "z_offset_cali": (
        ";===== z ofst cali start =====",
        ";===== z ofst cali end =====",
        "skip_z_offset_cali",
    ),
    "xy_offset_cali": (
        ";===== xy ofst cali start =====",
        ";===== xy ofst cali end =====",
        "skip_xy_offset_cali",
    ),
}


@dataclass(frozen=True)
class Snippet:
    path: Path
    main_index: int
    sub_index: int
    material: int
    tool: int


@dataclass
class MotionState:
    x: float | None = None
    y: float | None = None
    z: float | None = None
    previous_xy: tuple[float, float] | None = None
    # A wipe needs only the local tail of the deposited path. A deque avoids
    # copying that tail on every extrusion segment in large non-planar jobs.
    print_path: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=80))


class GCodeLineBuffer:
    """Build rewritten G-code without keeping one Python string per line."""

    def __init__(self) -> None:
        self._buffer = StringIO()

    def append(self, line: str) -> None:
        self._buffer.write(line)
        self._buffer.write("\n")

    def extend(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.append(line)

    def render(self, trailing_newline: bool) -> str:
        text = self._buffer.getvalue()
        if not trailing_newline and text.endswith("\n"):
            return text[:-1]
        return text


def iter_gcode_lines(text: str):
    """Yield logical G-code lines without allocating ``text.splitlines()``."""
    for line in StringIO(text):
        yield line.rstrip("\r\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    snippets = discover_snippets(Path(config["input_dir"]), config)
    if not snippets:
        raise SystemExit("No snippets found. Expected files like 0_0_0.gcode.")

    output_gcode = Path(config["output_gcode"])
    output_gcode.parent.mkdir(parents=True, exist_ok=True)

    assembled, summary = assemble_gcode(config, snippets)
    output_gcode.write_text(assembled, encoding=config.get("encoding", "utf-8"), newline="\n")

    summary_path = Path(config.get("summary_json", output_gcode.with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {output_gcode}")
    print(f"Wrote {summary_path}")
    print(f"Snippets: {len(snippets)}")
    print(f"Material sequence: {summary['material_sequence']}")
    print(f"Tool sequence: {summary['tool_sequence']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON config path.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["input_dir", "pre_code", "post_code", "output_gcode"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")
    return config


def discover_snippets(input_dir: Path, config: dict[str, Any]) -> list[Snippet]:
    material_to_tool = {int(k): int(v) for k, v in config.get("material_to_tool", {"0": 0, "1": 1}).items()}
    default_material = int(config.get("single_material_default_material", 1))
    allow_missing_suffix = bool(config.get("allow_missing_material_suffix", True))
    snippets: list[Snippet] = []

    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = SNIPPET_RE.match(path.name)
        if match:
            main_index, sub_index, material = (int(part) for part in match.groups())
        else:
            single_match = SINGLE_MATERIAL_SNIPPET_RE.match(path.name)
            if not single_match or not allow_missing_suffix:
                continue
            main_index, sub_index = (int(part) for part in single_match.groups())
            material = default_material
        if material not in material_to_tool:
            raise ValueError(f"{path.name}: material {material} has no material_to_tool mapping")
        snippets.append(
            Snippet(
                path=path,
                main_index=main_index,
                sub_index=sub_index,
                material=material,
                tool=material_to_tool[material],
            )
        )

    snippets = sorted(snippets, key=lambda item: (item.main_index, item.sub_index, item.path.name))
    schedule = input_dir / 'toolpath_order.json'
    if schedule.is_file():
        document = json.loads(schedule.read_text(encoding='utf8'))
        if document.get('schema') != 'map.toolpath-order.v1':
            raise ValueError('Unknown saved toolpath-order schema.')
        rows = document['layers']
        ranks = {(int(r['layer_main']), int(r['layer_sub'])): r['toolpath_order'] for r in rows}
        if len(ranks) != len(rows) or any(type(v) is not int for v in ranks.values()) or sorted(ranks.values()) != list(range(len(rows))):
            raise ValueError('Invalid saved bare-toolpath execution ranks.')
        if {(s.main_index,s.sub_index) for s in snippets} != set(ranks):
            raise ValueError('Bare files do not match their saved toolpath schedule.')
        snippets.sort(key=lambda s: ranks[(s.main_index,s.sub_index)])
    return snippets


def snippet_content_kind(path: Path, encoding: str, require_deposition: bool) -> str:
    """Stop at the first useful command/deposition; scan dry files to EOF."""
    commands = False
    position = [None, None, None]
    with path.open(encoding=encoding) as stream:
        for line in stream:
            code = line.partition(";")[0].strip()
            if not code:
                continue
            commands = True
            if not require_deposition:
                return "printing"
            if not MOTION_COMMAND_RE.match(code):
                continue
            before = position.copy()
            for i, pattern in enumerate((X_WORD_RE, Y_WORD_RE, Z_WORD_RE)):
                match = pattern.search(code)
                if match is not None:
                    position[i] = float(match[1])
            e = E_WORD_RE.search(code)
            if (e is not None and float(e[1]) > 0 and position != before
                    and None not in position):
                return "printing"
    return "nonprinting" if commands else "empty"


def assemble_gcode(config: dict[str, Any], snippets: list[Snippet]) -> tuple[str, dict[str, Any]]:
    encoding = config.get("encoding", "utf-8")
    parts: list[str] = []
    # Empty layer-path exports carry schedule metadata, not printer motion.
    # Selecting one as `current` would retract/lift for an absent layer, consume
    # its prime in a skipped XY approach, and then bypass the real next sublayer's
    # approach because its main index appears unchanged.
    skipped_empty_snippets = []
    skipped_nonprinting_snippets = []
    nonempty_snippets = []
    for snippet in snippets:
        # Generated paths have a few metadata comments before their first G1.
        # Stop at the first executable line; do not rescan large motion files.
        kind = snippet_content_kind(snippet.path, encoding, bool(config.get("skip_nonprinting_snippets", False)))
        if kind == "printing":
            nonempty_snippets.append(snippet)
        elif kind == "empty":
            skipped_empty_snippets.append(snippet.path.name)
        else:
            skipped_nonprinting_snippets.append(snippet.path.name)
    snippets = nonempty_snippets
    if not snippets:
        raise ValueError("No nonempty layer-path snippets to assemble.")

    report_assembly_progress(config, "Reading machine pre/post code")

    pre_code = read_text(Path(config["pre_code"]), encoding)
    pre_code, skipped_pre_code_sections = maybe_skip_pre_code_sections(pre_code, config)
    pre_code, flush_temperature_rewrites = maybe_rewrite_m620_10_flush_temperatures(pre_code, config)
    if config.get("rewrite_pre_code_start_material", True):
        pre_code = rewrite_logical_tool_commands(pre_code, snippets[0].tool)
    post_code = read_text(Path(config["post_code"]), encoding)
    parts.append(trim_trailing_newlines(pre_code))
    parts.append("")
    used_tools = sorted({snippet.tool for snippet in snippets})
    secondary_load_line_block = build_secondary_nozzle_load_line_block(config, snippets[0].tool, used_tools)
    if secondary_load_line_block:
        parts.append(secondary_load_line_block)
        parts.append("")
    initial_temperature_block = build_initial_temperature_block(config, snippets[0].tool)
    if initial_temperature_block:
        parts.append(initial_temperature_block)
        parts.append("")
    report_assembly_progress(config, "Scanning model bounds for the planar platform")
    stage_started = perf_counter()
    platform_info = build_planar_platform_block(config, snippets, encoding)
    report_assembly_progress(config, f"Planar-platform scan completed in {perf_counter() - stage_started:.1f}s")
    wipe_tower_bounds = validate_wipe_tower_xy_bounds(config, snippets, encoding)
    # An explicit model offset is useful when the source paths use Z=0 as
    # their geometric datum but no generated platform is being printed.
    model_z_lift = float(config.get("model_z_offset", 0.0))
    platform_z_lift = 0.0
    if platform_info:
        if "planar_platform_model_z_lift" in config:
            platform_z_lift = float(config["planar_platform_model_z_lift"])
        elif bool(config.get("planar_platform_lift_model_z", True)):
            platform_z_lift = float(config.get("planar_platform_height", 0.2))
        platform_info["summary"]["model_z_lift"] = platform_z_lift
        model_z_lift += platform_z_lift
    if platform_info:
        parts.append(platform_info["gcode"])
        parts.append("")
    wipe_tower_platform_info = build_wipe_tower_platform_block(config)
    if wipe_tower_platform_info:
        parts.append(wipe_tower_platform_info["gcode"])
        parts.append("")
    parts.append(";===== assembled layer path body start =====")
    model_acceleration = config.get("model_acceleration")
    first_layer_acceleration = config.get("first_layer_acceleration")
    remaining_layer_acceleration = config.get("remaining_layer_acceleration")
    restore_acceleration_after_model = config.get(
        "restore_acceleration_after_model",
        config.get("restore_acceleration_after_body"),
    )
    model_fan_speed = config.get("model_fan_speed")
    restore_fan_speed_after_model = config.get(
        "restore_fan_speed_after_model",
        config.get("restore_fan_speed_after_body"),
    )
    if model_z_lift > 0:
        parts.append(f"; ASSEMBLED_MODEL_Z_OFFSET {format_float(model_z_lift)}")

    current: Snippet | None = None
    transitions: list[dict[str, Any]] = []
    skipped_redundant_tool_changes: list[dict[str, Any]] = []
    layer_lifts: list[dict[str, Any]] = []
    material_sequence: list[int] = []
    tool_sequence: list[int] = []
    layer_ranges = compute_layer_ranges(snippets)
    layer_fraction_by_main = compute_layer_fraction_by_main(snippets)
    main_indices = sorted({snippet.main_index for snippet in snippets})
    layer_rank_by_main = {main_index: rank for rank, main_index in enumerate(main_indices)}
    report_assembly_progress(config, "Preparing layer Z targets and wipe-tower plan")
    stage_started = perf_counter()
    z_targets = build_z_targets(snippets, config, encoding)
    balanced_wipe_tower_plan = build_balanced_wipe_tower_plan(snippets, config)
    report_assembly_progress(config, f"Layer preparation completed in {perf_counter() - stage_started:.1f}s")
    z_normalizations: list[dict[str, Any]] = []
    feedrate_rewrites: list[dict[str, Any]] = []
    z_smoothing_rewrites: list[dict[str, Any]] = []
    extrusion_rewrites: list[dict[str, Any]] = []
    wipe_tower_state: dict[str, Any] = {"count": 0}
    balanced_wipe_tower_state: dict[str, Any] = {"printed": {}, "layer_heights": {}, "last_z": None}
    first_layer_main_index = int(config.get("first_layer_main_index", min(snippet.main_index for snippet in snippets)))
    total_main_layer_count = max(snippet.main_index for snippet in snippets) + 1
    first_snippet_main_index = min(snippet.main_index for snippet in snippets)
    config["_first_snippet_main_index"] = first_snippet_main_index
    highest_print_z_so_far: float | None = None
    motion_state = MotionState()
    activated_tools: set[int] = set()
    pressure_debt_by_tool: dict[int, float] = {}
    source_pressure_debt_by_tool: dict[int, float] = {}
    emitted_layer_markers: set[int] = set()
    previous_layer_marker_z: float | None = None

    for index, snippet in enumerate(snippets):
        report_assembly_progress(
            config,
            f"Processing snippet {index + 1}/{len(snippets)}: {snippet.path.name}",
        )
        snippet_started = perf_counter()
        snippet_text = read_text(snippet.path, encoding)
        snippet_text, z_normalization = maybe_normalize_snippet_z(snippet_text, snippet, z_targets, config)
        if z_normalization is not None:
            z_normalizations.append(z_normalization)
        if model_z_lift > 0:
            snippet_text = lift_z_words(snippet_text, model_z_lift)
        snippet_text, feedrate_rewrite = maybe_rewrite_snippet_feedrates(
            snippet_text,
            snippet,
            config,
            is_first_layer=snippet.main_index == first_layer_main_index,
            layer_rank=layer_rank_by_main[snippet.main_index],
            layer_count=len(main_indices),
        )
        if feedrate_rewrite is not None:
            feedrate_rewrites.append(feedrate_rewrite)
        snippet_text, z_smoothing_rewrite = maybe_smooth_snippet_z(snippet_text, snippet, config)
        if z_smoothing_rewrite is not None:
            z_smoothing_rewrites.append(z_smoothing_rewrite)
        snippet_text, extrusion_rewrite = maybe_rewrite_snippet_extrusion(
            snippet_text,
            snippet,
            config,
            layer_fraction_by_main,
        )
        if extrusion_rewrite is not None:
            extrusion_rewrites.append(extrusion_rewrite)
        snippet_text = maybe_strip_zero_extrusion_words(snippet_text, config)
        approach_reason: str | None = None
        snippet_first_print_z = first_z_value(snippet_text)
        snippet_max_print_z = max_z_value(snippet_text)

        if current is None:
            material_sequence.append(snippet.material)
            tool_sequence.append(snippet.tool)
            activated_tools.add(snippet.tool)
            parts.append(
                f"; ASSEMBLED_START_MATERIAL file={snippet.path.name} material={snippet.material} tool=T{snippet.tool}"
            )
        else:
            layer_changed = snippet.main_index != current.main_index
            material_changed = snippet.material != current.material
            tool_changed = snippet.tool != current.tool
            sublayer_changed = False
            if config.get("approach_on_sublayer_change", False) and not layer_changed:
                xy = first_xy_value(snippet_text)
                z = first_z_value(snippet_text)
                if xy is not None and z is not None and None not in (motion_state.x, motion_state.y, motion_state.z):
                    sublayer_changed = math.dist((motion_state.x, motion_state.y, motion_state.z), (*xy, z)) >= float(config.get("sublayer_transition_min_distance", 3.0))
            if material_changed:
                material_sequence.append(snippet.material)
            if tool_changed:
                first_activation = snippet.tool not in activated_tools
                compensation_retract = 0.0
                if not first_activation:
                    compensation_retract = tool_change_extrusion_delta(
                        config,
                        "tool_change_retract_by_tool",
                        snippet.tool,
                        "tool_change_retract",
                    )
                tool_sequence.append(snippet.tool)
                transitions.append(
                    {
                        "before_file": current.path.name,
                        "after_file": snippet.path.name,
                        "from_material": current.material,
                        "to_material": snippet.material,
                        "from_tool": current.tool,
                        "to_tool": snippet.tool,
                        "at_snippet_index": index,
                        "main_index": snippet.main_index,
                        "sub_index": snippet.sub_index,
                        "first_tool_activation": first_activation,
                        "dynamic_compensation_retract": compensation_retract,
                    }
                )
                parts.append(
                    build_tool_change_block(
                        config=config,
                        old_tool=current.tool,
                        new_tool=snippet.tool,
                        old_material=current.material,
                        new_material=snippet.material,
                        next_file=snippet.path.name,
                        first_tool_activation=first_activation,
                        dynamic_compensation_retract=compensation_retract,
                        motion_state=motion_state,
                    )
                )
                add_pressure_debt(
                    pressure_debt_by_tool,
                    current.tool,
                    tool_change_extrusion_delta(
                        config,
                        "tool_change_retract_by_tool",
                        current.tool,
                        "tool_change_retract",
                    ),
                )
                activated_tools.add(snippet.tool)
                wipe_tower_block = ""
                if not bool(config.get("enable_balanced_wipe_tower", False)):
                    wipe_tower_block = build_simple_wipe_tower_block(config, snippet_text, snippet, wipe_tower_state)
                if wipe_tower_block:
                    parts.append(wipe_tower_block)
                approach_reason = "tool_change"
            elif material_changed:
                skipped_redundant_tool_changes.append(
                    {
                        "before_file": current.path.name,
                        "after_file": snippet.path.name,
                        "from_material": current.material,
                        "to_material": snippet.material,
                        "tool": snippet.tool,
                        "at_snippet_index": index,
                        "main_index": snippet.main_index,
                        "sub_index": snippet.sub_index,
                        "reason": "same_physical_tool",
                    }
                )
            elif (layer_changed or sublayer_changed) and config.get("lift_on_layer_change", True):
                layer_change_safe_z = layer_change_lift_target_z(config, highest_print_z_so_far)
                layer_lifts.append(
                    {
                        "before_file": current.path.name,
                        "after_file": snippet.path.name,
                        "material": snippet.material,
                        "tool": snippet.tool,
                        "at_snippet_index": index,
                        "from_main_index": current.main_index,
                        "to_main_index": snippet.main_index,
                        "sub_index": snippet.sub_index,
                        "highest_print_z_so_far": highest_print_z_so_far,
                        "lift_target_z": layer_change_safe_z,
                    }
                )
                layer_change_block, layer_change_withdrawal = build_layer_change_lift_block(
                    config=config,
                    tool=snippet.tool,
                    material=snippet.material,
                    old_main_index=current.main_index,
                    new_main_index=snippet.main_index,
                    next_file=snippet.path.name,
                    lift_target_z=layer_change_safe_z,
                    motion_state=motion_state,
                )
                parts.append(layer_change_block)
                add_pressure_debt(
                    pressure_debt_by_tool,
                    snippet.tool,
                    layer_change_withdrawal,
                )
                approach_reason = "layer_change" if layer_changed else "sublayer_change"

            if (
                material_changed
                and not tool_changed
                and layer_changed
                and config.get("lift_on_layer_change", True)
            ):
                layer_change_safe_z = layer_change_lift_target_z(config, highest_print_z_so_far)
                layer_lifts.append(
                    {
                        "before_file": current.path.name,
                        "after_file": snippet.path.name,
                        "material": snippet.material,
                        "tool": snippet.tool,
                        "at_snippet_index": index,
                        "from_main_index": current.main_index,
                        "to_main_index": snippet.main_index,
                        "sub_index": snippet.sub_index,
                        "highest_print_z_so_far": highest_print_z_so_far,
                        "lift_target_z": layer_change_safe_z,
                        "reason": "same_tool_material_id_change",
                    }
                )
                layer_change_block, layer_change_withdrawal = build_layer_change_lift_block(
                    config=config,
                    tool=snippet.tool,
                    material=snippet.material,
                    old_main_index=current.main_index,
                    new_main_index=snippet.main_index,
                    next_file=snippet.path.name,
                    lift_target_z=layer_change_safe_z,
                    motion_state=motion_state,
                )
                parts.append(layer_change_block)
                add_pressure_debt(
                    pressure_debt_by_tool,
                    snippet.tool,
                    layer_change_withdrawal,
                )
                approach_reason = "layer_change"

        previous_transition_was_tool_change = approach_reason == "tool_change"
        previous_transition_was_layer_change = approach_reason == "layer_change"
        tower_pressure_debt_prime = get_pressure_debt(pressure_debt_by_tool, snippet.tool)
        balanced_wipe_tower_block = build_balanced_wipe_tower_block(
            config=config,
            snippet_text=snippet_text,
            snippet=snippet,
            plan=balanced_wipe_tower_plan,
            state=balanced_wipe_tower_state,
            highest_print_z_so_far=highest_print_z_so_far,
            previous_transition_was_tool_change=previous_transition_was_tool_change,
            previous_transition_was_layer_change=previous_transition_was_layer_change,
            pressure_debt_prime=tower_pressure_debt_prime,
        )
        if balanced_wipe_tower_block:
            parts.append(balanced_wipe_tower_block)
            consume_pressure_debt(pressure_debt_by_tool, snippet.tool, tower_pressure_debt_prime)
            tower_post_retract = balanced_wipe_tower_post_retract_amount(
                config,
                previous_transition_was_tool_change,
                previous_transition_was_layer_change,
                tower_pressure_debt_prime,
            )
            add_pressure_debt(pressure_debt_by_tool, snippet.tool, tower_post_retract)
            approach_reason = "wipe_tower"

        if approach_reason and config.get("approach_first_xy_while_lifted", True):
            approached_xy = first_xy_value(snippet_text)
            debt_prime = get_pressure_debt(pressure_debt_by_tool, snippet.tool)
            parts.append(build_approach_block(config, snippet_text, snippet, approach_reason, debt_prime))
            if approached_xy is not None:
                consume_pressure_debt(pressure_debt_by_tool, snippet.tool, debt_prime)
                motion_state.x, motion_state.y = approached_xy
                if config.get("approach_descend_before_unretract", False):
                    motion_state.z = first_z_value(snippet_text)
                motion_state.print_path.clear()

        snippet_acceleration = resolve_model_acceleration(
            config=config,
            is_first_layer=snippet.main_index == first_layer_main_index,
        )
        snippet_text, long_jump_rewrites = maybe_rewrite_long_jumps(
            snippet_text,
            config,
            motion_state,
            highest_print_z_so_far,
            snippet,
            pressure_debt_by_tool,
            snippet_acceleration,
            source_pressure_debt_by_tool,
        )
        if current is not None and config.get("move_leading_pressure_wipe_to_previous_snippet", True):
            snippet_text, transferred_wipe = transfer_leading_pressure_wipe_to_previous_snippet(
                snippet_text,
                current,
                snippet,
            )
            if transferred_wipe:
                insert_before_previous_snippet_end(parts, current, transferred_wipe)
        layer_lifts.extend(long_jump_rewrites)

        snippet_conditioning_start = build_model_conditioning_start_block(
            model_acceleration=snippet_acceleration,
            model_fan_speed=model_fan_speed,
            snippet=snippet,
        )
        if snippet_conditioning_start:
            parts.append(snippet_conditioning_start)
        scans_recovery = bool(config.get("track_source_retraction", False)) or (
            bool(config.get("lift_on_long_travel", False)) and float(config.get("long_travel_min_distance", 20.0)) > 0)
        # The scan already places recovery immediately before deposition.
        # Debt left at its end belongs AFTER this snippet, not before it.
        debt_prime = 0.0 if scans_recovery else consume_pressure_debt(pressure_debt_by_tool, snippet.tool)
        if debt_prime > 0:
            parts.append(build_pressure_debt_unretract_block(config, snippet, debt_prime))
        if snippet.main_index not in emitted_layer_markers:
            layer_marker = build_bambu_layer_marker(
                config,
                snippet,
                total_main_layer_count,
                snippet_first_print_z,
                previous_layer_marker_z,
            )
            if layer_marker:
                parts.append(layer_marker)
            emitted_layer_markers.add(snippet.main_index)
            if snippet_first_print_z is not None:
                previous_layer_marker_z = snippet_first_print_z
        parts.append(
            f"; ASSEMBLED_SNIPPET_START file={snippet.path.name} main={snippet.main_index} sub={snippet.sub_index} "
            f"material={snippet.material} tool=T{snippet.tool}"
        )
        parts.append(trim_outer_newlines(snippet_text))
        parts.append(f"; ASSEMBLED_SNIPPET_END file={snippet.path.name}")
        snippet_conditioning_end = build_model_conditioning_end_block(
            restore_acceleration=restore_acceleration_after_model,
            restore_fan_speed=restore_fan_speed_after_model,
            snippet=snippet,
        )
        if snippet_conditioning_end:
            parts.append(snippet_conditioning_end)
        if snippet_max_print_z is not None:
            highest_print_z_so_far = (
                snippet_max_print_z
                if highest_print_z_so_far is None
                else max(highest_print_z_so_far, snippet_max_print_z)
            )
        current = snippet
        report_assembly_progress(
            config,
            f"Completed snippet {index + 1}/{len(snippets)} in {perf_counter() - snippet_started:.1f}s",
        )

    parts.append(";===== assembled layer path body end =====")
    parts.append("")
    post_code, post_park_z = maybe_raise_post_code_park_z(post_code, highest_print_z_so_far, config)
    parts.append(trim_leading_newlines(post_code))
    parts.append("")

    summary = {
        "input_dir": str(Path(config["input_dir"])),
        "pre_code": str(Path(config["pre_code"])),
        "post_code": str(Path(config["post_code"])),
        "output_gcode": str(Path(config["output_gcode"])),
        "snippet_count": len(snippets),
        "skipped_empty_snippets": skipped_empty_snippets,
        "skipped_nonprinting_snippets": skipped_nonprinting_snippets,
        "first_material": snippets[0].material,
        "first_tool": snippets[0].tool,
        "last_material": snippets[-1].material,
        "last_tool": snippets[-1].tool,
        "pre_code_start_material_rewritten": bool(config.get("rewrite_pre_code_start_material", True)),
        "model_acceleration": None if model_acceleration is None else float(model_acceleration),
        "first_layer_acceleration": (
            None if first_layer_acceleration is None else float(first_layer_acceleration)
        ),
        "remaining_layer_acceleration": (
            None if remaining_layer_acceleration is None else float(remaining_layer_acceleration)
        ),
        "restore_acceleration_after_model": (
            None if restore_acceleration_after_model is None else float(restore_acceleration_after_model)
        ),
        "model_fan_speed": None if model_fan_speed is None else float(model_fan_speed),
        "restore_fan_speed_after_model": (
            None if restore_fan_speed_after_model is None else float(restore_fan_speed_after_model)
        ),
        "skipped_pre_code_sections": skipped_pre_code_sections,
        "flush_temperature_rewrites": flush_temperature_rewrites,
        "material_sequence": material_sequence,
        "tool_sequence": tool_sequence,
        "transitions": transitions,
        "skipped_redundant_tool_changes": skipped_redundant_tool_changes,
        "layer_lifts": layer_lifts,
        "z_normalizations": z_normalizations,
        "feedrate_rewrites": feedrate_rewrites,
        "z_smoothing_rewrites": z_smoothing_rewrites,
        "extrusion_rewrites": extrusion_rewrites,
        "bambu_layer_markers": len(emitted_layer_markers),
        "layer_ranges": layer_ranges,
        "planar_platform": platform_info["summary"] if platform_info else None,
        "wipe_tower_platform": wipe_tower_platform_info["summary"] if wipe_tower_platform_info else None,
        "wipe_tower_bounds": wipe_tower_bounds,
        "balanced_wipe_tower": summarize_balanced_wipe_tower_plan(balanced_wipe_tower_plan),
        "post_park_z": post_park_z,
    }
    report_assembly_progress(config, "Joining assembled G-code")
    return "\n".join(parts), summary


def report_assembly_progress(config: dict[str, Any], message: str) -> None:
    if bool(config.get("progress_reporting", False)):
        print(f"[assembler] {message}", flush=True)


def build_z_targets(snippets: list[Snippet], config: dict[str, Any], encoding: str) -> dict[int, float]:
    if not config.get("normalize_snippet_z", False):
        return {}

    reference_material = int(config.get("z_reference_material", 0))
    targets = collect_z_targets_for_material(snippets, reference_material, encoding)
    if targets:
        config["_resolved_z_reference_material"] = reference_material
        return targets

    fallback_materials: list[int] = []
    if "single_material_default_material" in config:
        fallback_materials.append(int(config["single_material_default_material"]))
    fallback_materials.extend(snippet.material for snippet in snippets)

    for fallback_material in dict.fromkeys(fallback_materials):
        if fallback_material == reference_material:
            continue
        targets = collect_z_targets_for_material(snippets, fallback_material, encoding)
        if targets:
            config["_resolved_z_reference_material"] = fallback_material
            return targets

    if "z_first_layer" in config and "z_layer_height" in config:
        first = float(config["z_first_layer"])
        height = float(config["z_layer_height"])
        return {snippet.main_index: first + snippet.main_index * height for snippet in snippets}

    available = sorted({snippet.material for snippet in snippets})
    raise ValueError(
        "normalize_snippet_z is true, but no reference-material Z values were found. "
        f"Configured z_reference_material={reference_material}; available materials={available}. "
        "Set z_reference_material or provide z_first_layer and z_layer_height."
    )


def collect_z_targets_for_material(snippets: list[Snippet], material: int, encoding: str) -> dict[int, float]:
    targets: dict[int, float] = {}
    for snippet in snippets:
        if snippet.material != material:
            continue
        first_z = first_z_value(read_text(snippet.path, encoding))
        if first_z is not None:
            targets[snippet.main_index] = first_z
    return targets


def maybe_normalize_snippet_z(
    snippet_text: str,
    snippet: Snippet,
    z_targets: dict[int, float],
    config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if not z_targets:
        return snippet_text, None
    reference_material = int(config.get("_resolved_z_reference_material", config.get("z_reference_material", 0)))
    if config.get("normalize_reference_material_z", False) is False and snippet.material == reference_material:
        return snippet_text, None
    target_z = z_targets.get(snippet.main_index)
    if target_z is None:
        return snippet_text, None
    current_z = first_z_value(snippet_text)
    if current_z is None:
        return snippet_text, None
    delta = target_z - current_z
    if abs(delta) < 1e-9:
        return snippet_text, None

    def replace(match: re.Match[str]) -> str:
        return "Z" + format_z(float(match.group(1)) + delta)

    return (
        Z_WORD_RE.sub(replace, snippet_text),
        {
            "file": snippet.path.name,
            "main_index": snippet.main_index,
            "material": snippet.material,
            "from_first_z": current_z,
            "to_first_z": target_z,
            "delta": delta,
        },
    )


def first_z_value(text: str) -> float | None:
    match = Z_WORD_RE.search(text)
    if not match:
        return None
    return float(match.group(1))


def max_z_value(text: str) -> float | None:
    max_z: float | None = None
    for match in Z_WORD_RE.finditer(text):
        z = float(match.group(1))
        max_z = z if max_z is None else max(max_z, z)
    return max_z


def build_bambu_layer_marker(
    config: dict[str, Any],
    snippet: Snippet,
    total_layer_count: int,
    layer_z: float | None,
    previous_layer_z: float | None,
) -> str:
    if not config.get("emit_bambu_layer_markers", True):
        return ""

    layer_height = float(config.get("preview_layer_height", config.get("z_layer_height", 0.2)))
    z_height_mode = str(config.get("bambu_z_height_mode", "nominal"))
    if z_height_mode == "nominal":
        layer_z = (snippet.main_index + 1) * layer_height
    if (
        config.get("infer_preview_layer_height_from_z", False)
        and layer_z is not None
        and previous_layer_z is not None
        and layer_z > previous_layer_z
    ):
        layer_height = layer_z - previous_layer_z

    lines = [
        "; CHANGE_LAYER",
    ]
    if config.get("emit_bambu_z_height_tags", False) and layer_z is not None:
        lines.append(f"; Z_HEIGHT: {format_float(layer_z)}")
    lines.extend(
        [
            f"; LAYER_HEIGHT: {format_float(layer_height)}",
            ";======== H2D 20250710 layer_change ========",
            f"; layer num/total_layer_count: {snippet.main_index + 1}/{total_layer_count}",
        ]
    )
    if config.get("emit_bambu_layer_progress_commands", False):
        lines.extend(
            [
                "; update layer progress",
                f"M73 L{snippet.main_index + 1}",
                f"M991 S0 P{snippet.main_index} ;notify layer change",
            ]
        )
    feature = config.get("preview_feature")
    feature_once = bool(config.get("preview_feature_once", False))
    feature_first_layer = int(config.get("_first_snippet_main_index", 0))
    if feature and (not feature_once or snippet.main_index == feature_first_layer):
        lines.append(f"; FEATURE: {feature}")
    line_width = config.get("preview_line_width")
    if line_width is not None:
        lines.append(f"; LINE_WIDTH: {format_float(float(line_width))}")
    return "\n".join(lines)


def resolve_model_acceleration(config: dict[str, Any], is_first_layer: bool) -> Any:
    layer_key = "first_layer_acceleration" if is_first_layer else "remaining_layer_acceleration"
    if layer_key in config:
        return config[layer_key]
    return config.get("model_acceleration")


def build_model_conditioning_start_block(
    model_acceleration: Any,
    model_fan_speed: Any,
    snippet: Snippet,
) -> str:
    lines: list[str] = []
    if model_acceleration is not None:
        acceleration = float(model_acceleration)
        lines.append(
            f"; ASSEMBLED_MODEL_ACCELERATION_START file={snippet.path.name} S{format_float(acceleration)}"
        )
        lines.append(f"M204 S{format_float(acceleration)}")
    if model_fan_speed is not None:
        fan_speed = float(model_fan_speed)
        lines.append(f"; ASSEMBLED_MODEL_FAN_START file={snippet.path.name} S{format_float(fan_speed)}")
        lines.append(f"M106 S{format_float(fan_speed)}")
    return "\n".join(lines)


def build_model_conditioning_end_block(
    restore_acceleration: Any,
    restore_fan_speed: Any,
    snippet: Snippet,
) -> str:
    lines: list[str] = []
    if restore_acceleration is not None:
        acceleration = float(restore_acceleration)
        lines.append(
            f"; ASSEMBLED_MODEL_ACCELERATION_END file={snippet.path.name} S{format_float(acceleration)}"
        )
        lines.append(f"M204 S{format_float(acceleration)}")
    if restore_fan_speed is not None:
        fan_speed = float(restore_fan_speed)
        lines.append(f"; ASSEMBLED_MODEL_FAN_END file={snippet.path.name} S{format_float(fan_speed)}")
        lines.append(f"M106 S{format_float(fan_speed)}")
    return "\n".join(lines)


def layer_change_lift_target_z(config: dict[str, Any], highest_print_z_so_far: float | None) -> float | None:
    if not bool(config.get("layer_change_lift_from_highest_z_so_far", True)):
        return None
    if highest_print_z_so_far is None:
        return None
    z_hop = float(config.get("layer_change_z_hop", config.get("z_hop", 3.0)))
    if z_hop <= 0:
        return None
    return highest_print_z_so_far + z_hop


def maybe_rewrite_long_jumps(
    snippet_text: str,
    config: dict[str, Any],
    motion_state: MotionState,
    highest_print_z_so_far: float | None,
    snippet: Snippet,
    pressure_debt_by_tool: dict[int, float],
    resume_acceleration: Any,
    source_pressure_debt_by_tool: dict[int, float] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    min_distance = float(config.get("long_travel_min_distance", 20.0))
    z_speed_limit = optional_positive_feedrate(config.get("travel_z_speed_limit"), "travel_z_speed_limit")
    lifts_enabled = bool(config.get("lift_on_long_travel", False)) and min_distance > 0
    if not lifts_enabled and not config.get("track_source_retraction", False):
        update_motion_state_from_gcode(snippet_text, motion_state)
        return snippet_text, []
    if not lifts_enabled:
        min_distance = math.inf
    if source_pressure_debt_by_tool is None:
        source_pressure_debt_by_tool = {}

    z_hop = float(config.get("long_travel_z_hop", config.get("layer_change_z_hop", config.get("z_hop", 3.0))))
    z_feedrate = float(config.get("long_travel_z_hop_feedrate", config.get("layer_change_z_hop_feedrate", config.get("z_hop_feedrate", 1200))))
    retract = float(config.get("long_travel_retract", config.get("layer_change_retract", 0.0)))
    unretract = float(config.get("long_travel_unretract", config.get("layer_change_unretract", retract)))
    retract_feedrate = float(config.get("retract_feedrate", 1800))
    unretract_feedrate = float(config.get("unretract_feedrate", retract_feedrate))
    default_travel_feedrate = float(config.get("long_travel_feedrate", config.get("approach_xy_feedrate", 30000)))
    pressure_wipe_only = bool(config.get("long_travel_pressure_wipe_only", False))
    pressure_wipe_distance = float(config.get("long_travel_pressure_wipe_distance", 1.0))
    pressure_wipe_retract = float(config.get("long_travel_pressure_wipe_retract", 0.38))
    pressure_relief = float(config.get("long_travel_pressure_relief", 0.02))
    pressure_wipe_feedrate = float(config.get("long_travel_pressure_wipe_feedrate", retract_feedrate))
    pressure_wipe_z_lift = float(config.get("long_travel_pressure_wipe_z_lift", config.get("pressure_wipe_z_lift", 0.0)))
    pressure_wipe_z_feedrate = float(
        config.get("long_travel_pressure_wipe_z_feedrate", config.get("pressure_wipe_z_feedrate", z_feedrate))
    )
    sparse_infill_jump_requires_tag = bool(config.get("sparse_infill_jump_requires_tag", True))

    absolute_xyz = True
    output = GCodeLineBuffer()
    rewrites: list[dict[str, Any]] = []
    state = MotionState(
        motion_state.x,
        motion_state.y,
        motion_state.z,
        motion_state.previous_xy,
        deque(motion_state.print_path, maxlen=80),
    )
    previous_xy = motion_state.previous_xy
    local_highest_print_z = highest_print_z_so_far
    # The first move in a snippet may immediately follow a printed edge in the
    # preceding snippet. Preserve that edge so cross-snippet travel wipes too.
    previous_move_was_printing = has_current_print_edge(state)
    modal_feedrate: float | None = None
    modal_acceleration: float | None = (
        float(resume_acceleration) if resume_acceleration is not None else None
    )
    pending_resume_acceleration: Any = None
    pending_resume_feedrate: float | None = None
    active_role: str | None = None
    next_move_has_jump_tag = False

    for line in iter_gcode_lines(snippet_text):
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            role_match = ROLE_TAG_RE.match(stripped)
            if role_match is not None:
                active_role = normalize_role_name(role_match.group(1))
                output.append(line)
                continue
            if JUMP_TAG_RE.match(stripped) is not None:
                next_move_has_jump_tag = True
                output.append(line)
                continue
            upper = stripped.upper()
            if upper.startswith("M204"):
                acceleration_match = M204_S_RE.search(stripped)
                if acceleration_match is not None:
                    modal_acceleration = float(acceleration_match.group(1))
                output.append(line)
                continue
            if upper == "G90":
                absolute_xyz = True
                output.append(line)
                continue
            if upper == "G91":
                absolute_xyz = False
                output.append(line)
                continue
            output.append(line)
            continue

        start_x = state.x
        start_y = state.y
        start_z = state.z
        parsed_move = _EXPLICIT_MOVE_RE.fullmatch(line) if absolute_xyz else None
        if parsed_move is not None:
            target_x, target_y, target_z = float(parsed_move[1]), float(parsed_move[2]), float(parsed_move[3])
            e_value = float(parsed_move[4]) if parsed_move[4] is not None else None
            line_feedrate = float(parsed_move[5]) if parsed_move[5] is not None else None
        else:
            feedrate_match = F_WORD_RE.search(line)
            line_feedrate = float(feedrate_match.group(1)) if feedrate_match else None
            target_x = resolved_axis_value(state.x, X_WORD_RE.search(line), absolute_xyz)
            target_y = resolved_axis_value(state.y, Y_WORD_RE.search(line), absolute_xyz)
            target_z = resolved_axis_value(state.z, Z_WORD_RE.search(line), absolute_xyz)
            e_match = E_WORD_RE.search(line)
            e_value = float(e_match.group(1)) if e_match else None
        moved = (start_x, start_y, start_z) != (target_x, target_y, target_z)
        is_extrusion = e_value is not None and e_value > 0 and moved
        if config.get("track_source_retraction", False) and e_value is not None:
            if e_value < 0:
                add_pressure_debt(source_pressure_debt_by_tool, snippet.tool, -e_value)
            elif e_value > 0 and not moved:
                consume_pressure_debt(source_pressure_debt_by_tool, snippet.tool, e_value)
        can_check_xy_distance = target_x is not None and target_y is not None and state.x is not None and state.y is not None
        distance = None
        if can_check_xy_distance:
            distance = math.hypot(target_x - state.x, target_y - state.y)
            if target_z is not None and state.z is not None:
                distance = math.hypot(distance, target_z - state.z)
        e_per_mm = e_value / distance if is_extrusion and distance and distance > 0 else None
        jump_kind = "extruding_move" if is_extrusion else "travel"
        sparse_infill_needs_tag = sparse_infill_jump_requires_tag and active_role == "sparse_infill"
        jump_is_allowed = not sparse_infill_needs_tag or next_move_has_jump_tag

        lift_target_z = layer_change_lift_target_z(config, local_highest_print_z)
        if (
            distance is not None
            and distance >= min_distance
            and lift_target_z is not None
            and z_hop > 0
            and jump_is_allowed
            and (not is_extrusion or next_move_has_jump_tag)
        ):
            # The injected lift changes the modal feedrate and forces G90. Preserve
            # the source command's effective state for the first resumed print move.
            resume_feedrate = line_feedrate if line_feedrate is not None else modal_feedrate
            restore_relative_xyz = not absolute_xyz
            feedrate = default_travel_feedrate
            has_wipe_path = bool(
                backward_wipe_path(
                    from_x=state.x,
                    from_y=state.y,
                    from_z=state.z,
                    print_path=state.print_path,
                    distance=pressure_wipe_distance,
                )
            )
            wipe_before_jump = pressure_wipe_only and has_wipe_path
            travel_retract = retract
            if config.get("avoid_repeated_travel_retraction", False) and not wipe_before_jump:
                # Consecutive travel-only moves must not withdraw another full
                # amount while the filament is already retracted. Source and
                # generated withdrawal remain separately tracked for recovery.
                outstanding = get_pressure_debt(pressure_debt_by_tool, snippet.tool)
                outstanding += get_pressure_debt(source_pressure_debt_by_tool, snippet.tool)
                travel_retract = max(0.0, retract - outstanding)
            if wipe_before_jump:
                add_pressure_debt(pressure_debt_by_tool, snippet.tool, pressure_wipe_retract + pressure_relief)
            elif travel_retract > 0:
                add_pressure_debt(pressure_debt_by_tool, snippet.tool, travel_retract)
            restore_feedrate_after_debt = modal_feedrate
            wipe_path = backward_wipe_path(
                from_x=state.x,
                from_y=state.y,
                from_z=state.z,
                print_path=state.print_path,
                distance=pressure_wipe_distance,
            )
            wipe_to_xy = tangential_wipe_target(
                from_x=state.x,
                from_y=state.y,
                previous_xy=previous_xy,
                distance=pressure_wipe_distance,
            )
            output.extend(
                build_long_travel_lift_block(
                    snippet=snippet,
                    distance=distance,
                    from_x=state.x,
                    from_y=state.y,
                    to_x=target_x,
                    to_y=target_y,
                    lift_target_z=lift_target_z,
                    z_feedrate=z_feedrate,
                    travel_feedrate=feedrate,
                    retract=travel_retract,
                    unretract=0.0 if retract > 0 else unretract,
                    retract_feedrate=retract_feedrate,
                    unretract_feedrate=unretract_feedrate,
                    config=config,
                    travel_acceleration=config.get(
                        "long_travel_acceleration",
                        config.get("restore_acceleration_after_model"),
                    ),
                    jump_kind=jump_kind,
                    pressure_wipe_only=wipe_before_jump,
                    pressure_wipe_retract=pressure_wipe_retract,
                    pressure_relief=pressure_relief,
                    pressure_wipe_feedrate=pressure_wipe_feedrate,
                    pressure_wipe_z_lift=pressure_wipe_z_lift,
                    pressure_wipe_z_feedrate=pressure_wipe_z_feedrate,
                    pressure_wipe_retract_before_wipe=bool(
                        config.get("long_travel_pressure_wipe_retract_before_wipe", config.get("pressure_wipe_retract_before_wipe", False))
                    ),
                    wipe_path=wipe_path,
                    wipe_to_xy=wipe_to_xy,
                    follow_wipe_z=bool(config.get("backward_wipe_follow_z", False)),
                    wipe_z_offset=float(config.get("backward_wipe_z_offset", 0.0)),
                )
            )
            wipe_end_xy = wipe_path[-1][:2] if wipe_before_jump and wipe_path else wipe_to_xy
            rewrites.append(
                {
                    "transition_kind": "long_extruding_move" if is_extrusion else "long_travel",
                    "file": snippet.path.name,
                    "main_index": snippet.main_index,
                    "sub_index": snippet.sub_index,
                    "material": snippet.material,
                    "tool": snippet.tool,
                    "distance": distance,
                    "original_e": e_value,
                    "original_e_per_mm": e_per_mm,
                    "e_zeroed_after_jump": is_extrusion,
                    "previous_move_was_printing": previous_move_was_printing,
                    "pressure_wipe_only": wipe_before_jump,
                    "pressure_wipe_retract": pressure_wipe_retract if wipe_before_jump else None,
                    "pressure_relief": pressure_relief if wipe_before_jump else None,
                    "pressure_wipe_distance": pressure_wipe_distance if wipe_before_jump else None,
                    "pressure_debt_added": pressure_wipe_retract + pressure_relief if wipe_before_jump else travel_retract,
                    "restore_feedrate_after_debt": restore_feedrate_after_debt,
                    "from_xy": [state.x, state.y],
                    "to_xy": [target_x, target_y],
                    "wipe_to_xy": list(wipe_end_xy) if wipe_before_jump and wipe_end_xy else None,
                    "wipe_path": [list(point) for point in wipe_path] if wipe_before_jump and wipe_path else None,
                    "highest_print_z_so_far": local_highest_print_z,
                    "lift_target_z": lift_target_z,
                }
            )
            if is_extrusion:
                if target_z is not None:
                    output.append(
                        f"; long-extruding jump lower to original endpoint Z{format_z(target_z)}"
                    )
                    output.append(f"G1 Z{format_z(target_z)} F{format_float(z_feedrate)}")
            else:
                # XY is already at the destination after the clearance block;
                # this command now descends vertically, even if its original
                # XYZ travel was mostly horizontal.
                if z_speed_limit is not None and target_z is not None:
                    output.append(set_or_append_feedrate(line, min(z_feedrate, 60 * z_speed_limit)))
                else:
                    output.append(line)
            if restore_relative_xyz:
                output.append("; restore relative XYZ mode after long travel")
                output.append("G91")
            if is_extrusion:
                if config.get("long_travel_acceleration", config.get("restore_acceleration_after_model")) is not None:
                    pending_resume_acceleration = modal_acceleration
                pending_resume_feedrate = resume_feedrate
            else:
                # The original travel command has now completed at the lifted
                # travel state. Return to the source modal state before its next
                # command, even when no pressure-debt prime will follow.
                if resume_feedrate is not None:
                    output.append("; restore feature feedrate after long travel")
                    output.append(f"G1 F{format_float(resume_feedrate)}")
                if (
                    config.get("long_travel_acceleration", config.get("restore_acceleration_after_model"))
                    is not None
                    and modal_acceleration is not None
                ):
                    output.extend(build_long_travel_acceleration_end_lines(modal_acceleration))
            if line_feedrate is not None:
                modal_feedrate = line_feedrate
            if start_x is not None and start_y is not None and (target_x != start_x or target_y != start_y):
                previous_xy = (start_x, start_y)
            state.x = target_x
            state.y = target_y
            if target_z is not None:
                state.z = target_z
            previous_move_was_printing = False
            if wipe_before_jump:
                state.print_path.clear()
            next_move_has_jump_tag = False
            continue

        if is_extrusion:
            debt_prime = consume_pressure_debt(pressure_debt_by_tool, snippet.tool)
            if config.get("track_source_retraction", False):
                debt_prime += consume_pressure_debt(source_pressure_debt_by_tool, snippet.tool)
            if debt_prime > 0:
                output.append(build_pressure_debt_unretract_block(config, snippet, debt_prime))
            if pending_resume_feedrate is not None:
                output.append("; restore feature feedrate after long travel")
                output.append(f"G1 F{format_float(pending_resume_feedrate)}")
                pending_resume_feedrate = None
            if pending_resume_acceleration is not None:
                output.extend(build_long_travel_acceleration_end_lines(pending_resume_acceleration))
                pending_resume_acceleration = None
        requested_feed = line_feedrate if line_feedrate is not None else modal_feedrate
        if (not is_extrusion and z_speed_limit is not None and requested_feed is not None
                and distance is not None and distance > 0 and target_z is not None and start_z is not None
                and abs(target_z - start_z) > 1e-9):
            limited_feed = 60 * float(z_speed_limit) * distance / abs(target_z - start_z)
            if requested_feed > limited_feed:
                output.append(set_or_append_feedrate(line, limited_feed))
                output.append(f"G1 F{format_float(requested_feed)} ; restore source feed after Z-limited travel")
            else:
                output.append(line)
        else:
            output.append(line)
        if line_feedrate is not None:
            modal_feedrate = line_feedrate
        if start_x is not None and start_y is not None and target_x is not None and target_y is not None:
            if target_x != start_x or target_y != start_y:
                previous_xy = (start_x, start_y)
        state.x = target_x
        state.y = target_y
        state.z = target_z
        if is_extrusion and target_z is not None:
            local_highest_print_z = (
                target_z
                if local_highest_print_z is None
                else max(local_highest_print_z, target_z)
            )
        if is_extrusion:
            append_print_segment(state, start_x, start_y, start_z, target_x, target_y, target_z)
        previous_move_was_printing = is_extrusion
        next_move_has_jump_tag = False

    motion_state.x = state.x
    motion_state.y = state.y
    motion_state.z = state.z
    motion_state.previous_xy = previous_xy
    motion_state.print_path = state.print_path
    return output.render(snippet_text.endswith("\n")), rewrites


def build_long_travel_lift_block(
    snippet: Snippet,
    distance: float,
    from_x: float,
    from_y: float,
    to_x: float,
    to_y: float,
    lift_target_z: float,
    z_feedrate: float,
    travel_feedrate: float,
    retract: float,
    unretract: float,
    retract_feedrate: float,
    unretract_feedrate: float,
    config: dict[str, Any],
    travel_acceleration: Any = None,
    jump_kind: str = "travel",
    pressure_wipe_only: bool = False,
    pressure_wipe_retract: float = 0.0,
    pressure_relief: float = 0.0,
    pressure_wipe_feedrate: float = 1800.0,
    pressure_wipe_z_lift: float = 0.0,
    pressure_wipe_z_feedrate: float = 1200.0,
    pressure_wipe_retract_before_wipe: bool = False,
    wipe_path: list[tuple[float, float, float]] | None = None,
    wipe_to_xy: tuple[float, float] | None = None,
    follow_wipe_z: bool = False,
    wipe_z_offset: float = 0.0,
) -> list[str]:
    lines = [
        f"; ASSEMBLED_LONG_TRAVEL file={snippet.path.name} main={snippet.main_index} sub={snippet.sub_index} "
        f"material={snippet.material} tool=T{snippet.tool} distance={format_float(distance)} kind={jump_kind}",
    ]
    pressure_wipe_acceleration = config.get("long_travel_pressure_wipe_acceleration")
    use_separate_wipe_acceleration = pressure_wipe_only and pressure_wipe_acceleration is not None
    if travel_acceleration is not None and not use_separate_wipe_acceleration:
        lines.extend(
            [
                f"; ASSEMBLED_LONG_TRAVEL_ACCELERATION_START S{format_float(float(travel_acceleration))}",
                f"M204 S{format_float(float(travel_acceleration))}",
            ]
        )
    if pressure_wipe_only:
        if use_separate_wipe_acceleration:
            lines.extend(
                [
                    f"; ASSEMBLED_LONG_TRAVEL_PRESSURE_WIPE_ACCELERATION S{format_float(float(pressure_wipe_acceleration))}",
                    f"M204 S{format_float(float(pressure_wipe_acceleration))}",
                ]
            )
        lines.extend(
            build_pressure_wipe_lines(
                wipe_to_xy=wipe_to_xy,
                wipe_path=wipe_path,
                wipe_retract=pressure_wipe_retract,
                pressure_relief=pressure_relief,
                feedrate=pressure_wipe_feedrate,
                z_lift=pressure_wipe_z_lift,
                z_feedrate=pressure_wipe_z_feedrate,
                retract_before_wipe=pressure_wipe_retract_before_wipe,
                follow_wipe_z=follow_wipe_z,
                wipe_z_offset=wipe_z_offset,
            )
        )
        if travel_acceleration is not None and use_separate_wipe_acceleration:
            lines.extend(
                [
                    f"; ASSEMBLED_LONG_TRAVEL_ACCELERATION_START S{format_float(float(travel_acceleration))}",
                    f"M204 S{format_float(float(travel_acceleration))}",
                ]
            )
    else:
        lines.extend(build_extrusion_delta_lines(-retract, retract_feedrate, "long-travel retract"))
    lines.extend(
        [
            f"; long-travel lift to highest_print_z_so_far + hop = Z{format_z(lift_target_z)}",
            "G90",
            f"G1 Z{format_z(lift_target_z)} F{format_float(z_feedrate)}",
            "; wait for clearance lift before XY travel",
            "M400",
            f"G1 X{format_xy(to_x)} Y{format_xy(to_y)} F{format_float(travel_feedrate)}",
            "; wait for XY travel before any descent",
            "M400",
        ]
    )
    if not pressure_wipe_only and unretract != 0:
        lines.extend(build_extrusion_delta_lines(unretract, unretract_feedrate, "long-travel unretract"))
        lines.extend(build_after_unretract_sync_lines(config))
    return lines


def build_long_travel_acceleration_end_lines(acceleration: Any) -> list[str]:
    value = float(acceleration)
    return [
        f"; ASSEMBLED_LONG_TRAVEL_ACCELERATION_END S{format_float(value)}",
        f"M204 S{format_float(value)}",
    ]


def tangential_wipe_target(
    from_x: float | None,
    from_y: float | None,
    previous_xy: tuple[float, float] | None,
    distance: float,
) -> tuple[float, float] | None:
    if from_x is None or from_y is None or previous_xy is None or distance <= 0:
        return None
    prev_x, prev_y = previous_xy
    dx = from_x - prev_x
    dy = from_y - prev_y
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    scale = distance / length
    return from_x + dx * scale, from_y + dy * scale


def append_print_segment(
    state: MotionState,
    start_x: float | None,
    start_y: float | None,
    start_z: float | None,
    target_x: float | None,
    target_y: float | None,
    target_z: float | None,
) -> None:
    if None in (start_x, start_y, start_z, target_x, target_y, target_z):
        return
    if math.hypot(target_x - start_x, target_y - start_y) <= 1e-9:
        return
    start = (start_x, start_y, start_z)
    target = (target_x, target_y, target_z)
    if state.print_path and math.dist(state.print_path[-1], start) > 1e-6:
        state.print_path.clear()
    if not state.print_path:
        state.print_path.append(start)
    state.print_path.append(target)


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def backward_wipe_path(
    from_x: float | None,
    from_y: float | None,
    from_z: float | None,
    print_path: Sequence[tuple[float, float, float]],
    distance: float,
) -> list[tuple[float, float, float]]:
    if from_x is None or from_y is None or from_z is None or distance <= 0 or len(print_path) < 2:
        return []

    current = (from_x, from_y, from_z)
    path = list(print_path)
    if math.dist(path[-1], current) > 1e-6:
        return []

    remaining = distance
    out = [current]
    cursor = current
    for previous in reversed(path[:-1]):
        segment_length = distance_xy(cursor, previous)
        if segment_length <= 1e-9:
            cursor = previous
            continue
        if segment_length <= remaining + 1e-9:
            out.append(previous)
            remaining -= segment_length
            cursor = previous
            if remaining <= 1e-9:
                break
            continue
        ratio = remaining / segment_length
        clipped = (
            cursor[0] + (previous[0] - cursor[0]) * ratio,
            cursor[1] + (previous[1] - cursor[1]) * ratio,
            cursor[2] + (previous[2] - cursor[2]) * ratio,
        )
        out.append(clipped)
        break

    return out if len(out) > 1 else []


def has_current_print_edge(state: MotionState) -> bool:
    """Return whether the nozzle is still at the end of a printable edge."""
    if state.x is None or state.y is None or state.z is None or not state.print_path:
        return False
    return math.dist(state.print_path[-1], (state.x, state.y, state.z)) <= 1e-6


def build_pressure_wipe_lines(
    wipe_to_xy: tuple[float, float] | None,
    wipe_retract: float,
    pressure_relief: float,
    feedrate: float,
    wipe_path: list[tuple[float, float] | tuple[float, float, float]] | None = None,
    z_lift: float = 0.0,
    z_feedrate: float = 1200.0,
    retract_before_wipe: bool = False,
    follow_wipe_z: bool = False,
    wipe_z_offset: float = 0.0,
) -> list[str]:
    if wipe_retract <= 0 and pressure_relief <= 0:
        return []
    lines = ["; long-travel pressure wipe", "M83"]
    if z_lift > 0:
        lines.extend(
            [
                f"; pressure-wipe z lift {format_z(z_lift)}",
                "G91",
                f"G1 Z{format_z(z_lift)} F{format_float(z_feedrate)}",
                "G90",
            ]
        )
    if wipe_retract > 0 and retract_before_wipe:
        lines.append(f"G1 E-{format_float(wipe_retract)} F{format_float(feedrate)}")
        wipe_retract = 0.0

    if wipe_path and len(wipe_path) > 1:
        total_length = sum(distance_xy(a, b) for a, b in zip(wipe_path, wipe_path[1:]))
        if total_length <= 1e-9:
            if wipe_retract > 0:
                lines.append(f"G1 E-{format_float(wipe_retract)} F{format_float(feedrate)}")
        else:
            lines.append("; WIPE_START")
            if follow_wipe_z and wipe_z_offset != 0:
                lines.append(f"; backward-wipe Z offset {format_z(wipe_z_offset)}")
            emitted = 0.0
            travelled = 0.0
            for index, (a, b) in enumerate(zip(wipe_path, wipe_path[1:])):
                segment_length = distance_xy(a, b)
                if segment_length <= 1e-9:
                    continue
                if wipe_retract > 0:
                    # Round cumulative withdrawal, rather than independently
                    # rounding every small edge. This conserves the formatted
                    # total even when the wipe contains many short segments.
                    travelled += segment_length
                    target = float(format_float(wipe_retract * min(travelled / total_length, 1.0)))
                    d_e = max(0.0, target - emitted)
                    formatted_e = format_float(d_e)
                    emitted += float(formatted_e)
                    if float(formatted_e) == 0.0:
                        lines.append(format_wipe_move(b, feedrate, None, follow_wipe_z, wipe_z_offset))
                    else:
                        lines.append(format_wipe_move(b, feedrate, f"-{formatted_e}", follow_wipe_z, wipe_z_offset))
                else:
                    lines.append(format_wipe_move(b, feedrate, None, follow_wipe_z, wipe_z_offset))
            lines.append("; WIPE_END")
    elif wipe_retract > 0 and wipe_to_xy is None:
        lines.append(f"G1 E-{format_float(wipe_retract)} F{format_float(feedrate)}")
    elif wipe_to_xy is not None:
        wipe_x, wipe_y = wipe_to_xy
        lines.append("; WIPE_START")
        if wipe_retract > 0:
            lines.append(
                f"G1 X{format_xy(wipe_x)} Y{format_xy(wipe_y)} E-{format_float(wipe_retract)} F{format_float(feedrate)}"
            )
        else:
            lines.append(f"G1 X{format_xy(wipe_x)} Y{format_xy(wipe_y)} F{format_float(feedrate)}")
        lines.append("; WIPE_END")
    if pressure_relief > 0:
        lines.append(f"G1 E-{format_float(pressure_relief)} F{format_float(feedrate)}")
    return lines


def format_wipe_move(
    point: tuple[float, float] | tuple[float, float, float],
    feedrate: float,
    extrusion: str | None,
    follow_wipe_z: bool,
    wipe_z_offset: float,
) -> str:
    line = f"G1 X{format_xy(point[0])} Y{format_xy(point[1])}"
    if follow_wipe_z and len(point) >= 3:
        line += f" Z{format_z(point[2] + wipe_z_offset)}"
    if extrusion is not None:
        line += f" E{extrusion}"
    return f"{line} F{format_float(feedrate)}"


def transfer_leading_pressure_wipe_to_previous_snippet(
    snippet_text: str,
    previous_snippet: Snippet,
    snippet: Snippet,
) -> tuple[str, str]:
    # This runs after every snippet. Inspect only the leading header; building
    # a list for every line in a 15+ MiB snippet just to inspect eight lines
    # dominates A1 assembly time.
    leading_lines: list[str] = []
    for line in iter_gcode_lines(snippet_text):
        leading_lines.append(line)
        if len(leading_lines) == 8:
            break
    if len(leading_lines) < 3:
        return snippet_text, ""

    long_travel_index = next(
        (
            index
            for index, line in enumerate(leading_lines)
            if line.startswith("; ASSEMBLED_LONG_TRAVEL ")
        ),
        None,
    )
    if long_travel_index is None:
        return snippet_text, ""
    if (
        long_travel_index + 1 >= len(leading_lines)
        or leading_lines[long_travel_index + 1] != "; long-travel pressure wipe"
    ):
        return snippet_text, ""

    long_travel_header = leading_lines[long_travel_index]
    header_start = snippet_text.find(long_travel_header)
    header_end = snippet_text.find("\n", header_start)
    wipe_start = header_end + 1
    lift_start = snippet_text.find("; long-travel lift", wipe_start)
    if header_start < 0 or header_end < 0 or lift_start < 0:
        return snippet_text, ""

    moved_header = (
        f"; ASSEMBLED_TRANSFERRED_PRESSURE_WIPE from_file={previous_snippet.path.name} "
        f"to_file={snippet.path.name}"
    )
    moved_wipe = snippet_text[wipe_start:lift_start].rstrip("\r\n")
    remaining = snippet_text[:wipe_start] + snippet_text[lift_start:]
    return remaining, f"{moved_header}\n{moved_wipe}"


def insert_before_previous_snippet_end(parts: list[str], previous_snippet: Snippet, inserted_text: str) -> None:
    end_marker = f"; ASSEMBLED_SNIPPET_END file={previous_snippet.path.name}"
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == end_marker:
            parts.insert(index, inserted_text)
            return
    parts.append(inserted_text)


def zero_e_word(line: str) -> str:
    command, comment = split_inline_comment(line)
    return E_WORD_RE.sub("E0.00000", command, count=1) + comment


def split_inline_comment(line: str) -> tuple[str, str]:
    comment_start = line.find(";")
    if comment_start < 0:
        return line, ""
    return line[:comment_start], line[comment_start:]


def update_motion_state_from_gcode(snippet_text: str, motion_state: MotionState) -> None:
    absolute_xyz = True
    for line in iter_gcode_lines(snippet_text):
        stripped = line.strip()
        upper = stripped.upper()
        if upper == "G90":
            absolute_xyz = True
            continue
        if upper == "G91":
            absolute_xyz = False
            continue
        if not stripped.startswith(("G0", "G1")):
            continue
        start_x = motion_state.x
        start_y = motion_state.y
        start_z = motion_state.z
        target_x = resolved_axis_value(motion_state.x, X_WORD_RE.search(line), absolute_xyz)
        target_y = resolved_axis_value(motion_state.y, Y_WORD_RE.search(line), absolute_xyz)
        target_z = resolved_axis_value(motion_state.z, Z_WORD_RE.search(line), absolute_xyz)
        e_match = E_WORD_RE.search(line)
        is_extrusion = e_match is not None and float(e_match.group(1)) > 0
        if start_x is not None and start_y is not None and target_x is not None and target_y is not None:
            if target_x != start_x or target_y != start_y:
                motion_state.previous_xy = (start_x, start_y)
        if is_extrusion:
            append_print_segment(motion_state, start_x, start_y, start_z, target_x, target_y, target_z)
        motion_state.x = target_x
        motion_state.y = target_y
        motion_state.z = target_z


def resolved_axis_value(current: float | None, match: re.Match[str] | None, absolute: bool) -> float | None:
    if not match:
        return current
    value = float(match.group(1))
    if absolute or current is None:
        return value
    return current + value


def maybe_rewrite_snippet_feedrates(
    snippet_text: str,
    snippet: Snippet,
    config: dict[str, Any],
    is_first_layer: bool,
    layer_rank: int,
    layer_count: int,
) -> tuple[str, dict[str, Any] | None]:
    override_key = "first_layer_feedrate" if is_first_layer else "remaining_layer_feedrate"
    multiplier_key = "first_layer_feedrate_multiplier" if is_first_layer else "remaining_layer_feedrate_multiplier"

    override = config.get(override_key)
    multiplier = config.get(multiplier_key)
    slope_limits = normalized_slope_feedrate_limits(config)
    raster_feedrate = config.get("raster_after_zigzag_feedrate")
    enable_raster_limit = bool(config.get("enable_raster_after_zigzag_feedrate_limit", False))
    enable_order_raster_limit = bool(config.get("enable_raster_after_fraction_feedrate_limit", False))
    role_feedrates = normalized_role_feedrates(config)
    role_accelerations = normalized_role_accelerations(config)
    priority_roles = normalized_role_names(
        config.get("priority_roles", ["top_layer", "top_surface", "top_raster"]),
        "priority_roles",
    )
    layer_override_roles = normalized_role_names(
        config.get("layer_override_roles", ["base_layer", "top_layer"]),
        "layer_override_roles",
    )
    bottom_layer_count = int(config.get("bottom_layer_count", 0))
    if bottom_layer_count < 0:
        raise ValueError("bottom_layer_count must be >= 0")
    # Positional base layers use the same semantic role machinery as an
    # explicit `;role: base_layer` tag. An explicit whole-layer tag wins.
    layer_override_role = find_layer_override_role(snippet_text, layer_override_roles)
    if layer_override_role is None and layer_rank < bottom_layer_count:
        layer_override_role = "base_layer"
    if (
        override is None
        and multiplier is None
        and not slope_limits
        and not enable_raster_limit
        and not enable_order_raster_limit
        and not role_feedrates
        and not role_accelerations
    ):
        return snippet_text, None

    override_value = float(override) if override is not None else None
    multiplier_value = float(multiplier) if multiplier is not None else None
    raster_feedrate_value = float(raster_feedrate) if raster_feedrate is not None else None
    order_raster_feedrate_value = (
        float(config["raster_after_fraction_feedrate"])
        if "raster_after_fraction_feedrate" in config
        else raster_feedrate_value
    )
    raster_after_fraction = float(config.get("raster_after_extrusion_fraction", 0.25))
    raster_after_min_moves = int(config.get("raster_after_min_extrusion_moves", 100))
    raster_min_reversals = int(config.get("raster_zigzag_min_reversals", 2))
    raster_min_move = float(config.get("raster_zigzag_min_move", 0.2))
    if not 0.0 <= raster_after_fraction <= 1.0:
        raise ValueError("raster_after_extrusion_fraction must be between 0 and 1")
    total_extrusion_moves = count_positive_extrusion_moves(snippet_text) if enable_order_raster_limit else 0
    changed = 0
    slope_changed = 0
    raster_changed = 0
    order_raster_changed = 0
    original_values: list[float] = []
    new_values: list[float] = []
    slope_samples: list[dict[str, float]] = []
    raster_samples: list[dict[str, float]] = []
    order_raster_samples: list[dict[str, float]] = []
    role_samples: list[dict[str, Any]] = []
    acceleration_samples: list[dict[str, Any]] = []
    role_changed = 0
    layer_changed = 0
    role_acceleration_changed = 0
    layer_acceleration_changed = 0
    output_lines = GCodeLineBuffer()
    absolute_xyz = True
    current_x: float | None = None
    current_y: float | None = None
    current_z: float | None = None
    modal_feedrate: float | None = None
    zigzag_axis: str | None = None
    zigzag_sign: int = 0
    zigzag_reversals = 0
    raster_mode = False
    extrusion_move_index = 0
    # Untagged source paths are conservatively treated as boundary loops.
    # An explicit `;role:` comment replaces this modal role for later moves.
    untagged_role_value = config.get("untagged_role", "boundary_loop")
    active_role = None if untagged_role_value is None else normalize_role_name(str(untagged_role_value))
    role_candidates = [] if active_role is None else [active_role]
    role_marker_run_open = False
    fallback_feedrate = None if active_role is None else role_feedrates.get(active_role)
    fallback_acceleration = None if active_role is None else role_accelerations.get(active_role)
    base_acceleration = resolve_model_acceleration(config, is_first_layer)
    modal_acceleration = None if base_acceleration is None else float(base_acceleration)

    override_text = format_float(override_value) if override_value is not None and math.isfinite(override_value) else None
    needs_motion_geometry = bool(slope_limits) or (enable_raster_limit and raster_feedrate_value is not None)

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        old_value = float(match.group(1))
        new_value = override_value if override_value is not None else old_value * multiplier_value
        assert new_value is not None
        changed += 1
        if len(original_values) < 8:
            original_values.append(old_value)
            new_values.append(new_value)
        return "F" + (override_text if override_text is not None else format_float(new_value))

    for line in iter_gcode_lines(snippet_text):
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            role_match = ROLE_TAG_RE.match(stripped)
            if role_match is not None:
                tagged_role = normalize_role_name(role_match.group(1))
                if role_marker_run_open:
                    role_candidates.append(tagged_role)
                else:
                    role_candidates = [tagged_role]
                    role_marker_run_open = True
                active_role = select_role_by_priority(role_candidates, priority_roles)
                output_lines.append(line)
                continue
            upper = stripped.upper()
            if upper.startswith("M204"):
                acceleration_match = M204_S_RE.search(stripped)
                if acceleration_match is not None:
                    modal_acceleration = float(acceleration_match.group(1))
                output_lines.append(line)
                continue
            if upper == "G90":
                absolute_xyz = True
                output_lines.append(line)
                continue
            if upper == "G91":
                absolute_xyz = False
                output_lines.append(line)
                continue
            output_lines.append(line)
            continue

        command, comment = split_inline_comment(line)
        parsed_move = _EXPLICIT_MOVE_RE.fullmatch(command) if needs_motion_geometry and absolute_xyz else None
        if parsed_move is not None:
            target_x, target_y, target_z = float(parsed_move[1]), float(parsed_move[2]), float(parsed_move[3])
            e_value = float(parsed_move[4]) if parsed_move[4] is not None else None
        else:
            if needs_motion_geometry:
                target_x = resolved_axis_value(current_x, X_WORD_RE.search(command), absolute_xyz)
                target_y = resolved_axis_value(current_y, Y_WORD_RE.search(command), absolute_xyz)
                target_z = resolved_axis_value(current_z, Z_WORD_RE.search(command), absolute_xyz)
            else:
                # XYZ has no consumer in this pass when both geometric limits
                # are disabled. The ordered travel pass still resolves motion.
                target_x = target_y = target_z = None
            e_match = E_WORD_RE.search(command)
            e_value = float(e_match.group(1)) if e_match else None
        is_extrusion = e_value is not None and e_value > 0
        if is_extrusion:
            extrusion_move_index += 1
        if override is not None or multiplier is not None:
            if override_text is not None:
                if len(original_values) < 8:
                    for match in F_WORD_RE.finditer(command):
                        if len(original_values) >= 8:
                            break
                        original_values.append(float(match[1]))
                        new_values.append(override_value)
                command, replacements = F_WORD_RE.subn("F" + override_text, command)
                changed += replacements
                if replacements:
                    modal_feedrate = float(F_WORD_RE.search(command).group(1))
            else:
                previous_changed = changed
                command = F_WORD_RE.sub(replace, command)
                if changed != previous_changed:
                    modal_feedrate = float(F_WORD_RE.search(command).group(1))  # type: ignore[union-attr]

        feature_feedrate = None
        feature_source = None
        if layer_override_role is not None:
            feature_feedrate = role_feedrates.get(layer_override_role, fallback_feedrate)
            feature_source = layer_override_role
        elif active_role is not None and active_role in role_feedrates:
            feature_feedrate = role_feedrates[active_role]
            feature_source = active_role
        elif fallback_feedrate is not None:
            feature_feedrate = fallback_feedrate
            feature_source = active_role or "untagged_role"

        if is_extrusion and feature_feedrate is not None:
            current_feedrate = float(F_WORD_RE.search(command).group(1)) if F_WORD_RE.search(command) else modal_feedrate
            if current_feedrate is None or abs(current_feedrate - feature_feedrate) > 1e-9:
                command = set_or_append_feedrate(command, feature_feedrate)
                modal_feedrate = feature_feedrate
                role_changed += 1
                if len(role_samples) < 12:
                    role_samples.append(
                        {
                            "role": active_role,
                            "source": feature_source,
                            "old_feedrate": 0.0 if current_feedrate is None else current_feedrate,
                            "new_feedrate": feature_feedrate,
                        }
                    )

        acceleration_source = None
        if layer_override_role is not None:
            feature_acceleration = role_accelerations.get(layer_override_role, fallback_acceleration)
            acceleration_source = layer_override_role
        elif active_role is not None and active_role in role_accelerations:
            feature_acceleration = role_accelerations[active_role]
            acceleration_source = active_role
        elif fallback_acceleration is not None:
            feature_acceleration = fallback_acceleration
            acceleration_source = active_role or "untagged_role"
        else:
            feature_acceleration = modal_acceleration if base_acceleration is None else float(base_acceleration)
            acceleration_source = "base"

        if (
            is_extrusion
            and feature_acceleration is not None
            and (modal_acceleration is None or abs(modal_acceleration - feature_acceleration) > 1e-9)
        ):
            output_lines.append(
                f"; ASSEMBLED_FEATURE_ACCELERATION role={active_role or 'none'} source={acceleration_source} S{format_float(feature_acceleration)}"
            )
            output_lines.append(f"M204 S{format_float(feature_acceleration)}")
            modal_acceleration = feature_acceleration
            if acceleration_source != "base":
                role_acceleration_changed += 1
            if len(acceleration_samples) < 12:
                acceleration_samples.append(
                    {
                        "role": active_role,
                        "source": acceleration_source,
                        "new_acceleration": feature_acceleration,
                    }
                )

        slope_cap = None
        slope = None
        if (
            is_extrusion
            and needs_motion_geometry
            and current_x is not None
            and current_y is not None
            and current_z is not None
            and target_x is not None
            and target_y is not None
            and target_z is not None
        ):
            dxy = math.hypot(target_x - current_x, target_y - current_y)
            if dxy > 1e-9:
                dx = target_x - current_x
                dy = target_y - current_y
                slope = abs(target_z - current_z) / dxy
                if slope_limits:
                    slope_cap = slope_feedrate_cap(slope, slope_limits)
                if enable_raster_limit and raster_feedrate_value is not None:
                    axis, sign = dominant_axis_direction(dx, dy, raster_min_move)
                    if axis is not None and sign != 0:
                        if zigzag_axis == axis and zigzag_sign != 0 and sign != zigzag_sign:
                            zigzag_reversals += 1
                        elif zigzag_axis != axis:
                            zigzag_reversals = 0
                        zigzag_axis = axis
                        zigzag_sign = sign
                        if zigzag_reversals >= raster_min_reversals:
                            raster_mode = True

        if slope_cap is not None:
            current_feedrate = float(F_WORD_RE.search(command).group(1)) if F_WORD_RE.search(command) else modal_feedrate
            if current_feedrate is None or current_feedrate > slope_cap + 1e-9:
                command = set_or_append_feedrate(command, slope_cap)
                modal_feedrate = slope_cap
                slope_changed += 1
                if slope is not None and len(slope_samples) < 8:
                    slope_samples.append(
                        {
                            "slope": slope,
                            "old_feedrate": 0.0 if current_feedrate is None else current_feedrate,
                            "new_feedrate": slope_cap,
                        }
                    )
        if raster_mode and raster_feedrate_value is not None and is_extrusion:
            current_feedrate = float(F_WORD_RE.search(command).group(1)) if F_WORD_RE.search(command) else modal_feedrate
            if current_feedrate is None or current_feedrate > raster_feedrate_value + 1e-9:
                command = set_or_append_feedrate(command, raster_feedrate_value)
                modal_feedrate = raster_feedrate_value
                raster_changed += 1
                if len(raster_samples) < 8:
                    raster_samples.append(
                        {
                            "old_feedrate": 0.0 if current_feedrate is None else current_feedrate,
                            "new_feedrate": raster_feedrate_value,
                            "zigzag_reversals": float(zigzag_reversals),
                        }
                    )
        if (
            enable_order_raster_limit
            and order_raster_feedrate_value is not None
            and is_extrusion
            and total_extrusion_moves >= raster_after_min_moves
            and extrusion_move_index / total_extrusion_moves >= raster_after_fraction
        ):
            current_feedrate = float(F_WORD_RE.search(command).group(1)) if F_WORD_RE.search(command) else modal_feedrate
            if current_feedrate is None or current_feedrate > order_raster_feedrate_value + 1e-9:
                command = set_or_append_feedrate(command, order_raster_feedrate_value)
                modal_feedrate = order_raster_feedrate_value
                order_raster_changed += 1
                if len(order_raster_samples) < 8:
                    order_raster_samples.append(
                        {
                            "old_feedrate": 0.0 if current_feedrate is None else current_feedrate,
                            "new_feedrate": order_raster_feedrate_value,
                            "extrusion_fraction": extrusion_move_index / total_extrusion_moves,
                        }
                    )
        if F_WORD_RE.search(command):
            modal_feedrate = float(F_WORD_RE.search(command).group(1))  # type: ignore[union-attr]

        output_lines.append(command + comment)
        role_marker_run_open = False
        current_x = target_x
        current_y = target_y
        current_z = target_z

    if (
        changed == 0
        and slope_changed == 0
        and raster_changed == 0
        and order_raster_changed == 0
        and role_changed == 0
        and layer_changed == 0
        and role_acceleration_changed == 0
        and layer_acceleration_changed == 0
    ):
        return snippet_text, None

    return (
        output_lines.render(snippet_text.endswith("\n")),
        {
            "file": snippet.path.name,
            "main_index": snippet.main_index,
            "material": snippet.material,
            "layer_kind": "first" if is_first_layer else "remaining",
            "layer_rank": layer_rank,
            "layer_count": layer_count,
            "override_feedrate": override_value,
            "feedrate_multiplier": multiplier_value,
            "changed_count": changed,
            "slope_limited_count": slope_changed,
            "slope_feedrate_limits": slope_limits,
            "raster_limited_count": raster_changed,
            "raster_after_zigzag_feedrate": raster_feedrate_value,
            "raster_zigzag_min_reversals": raster_min_reversals,
            "order_raster_limited_count": order_raster_changed,
            "raster_after_fraction_feedrate": order_raster_feedrate_value,
            "raster_after_extrusion_fraction": raster_after_fraction,
            "raster_after_min_extrusion_moves": raster_after_min_moves,
            "sample_original_feedrates": original_values,
            "sample_new_feedrates": new_values,
            "sample_slope_limited_feedrates": slope_samples,
            "sample_raster_limited_feedrates": raster_samples,
            "sample_order_raster_limited_feedrates": order_raster_samples,
            "role_feedrates": role_feedrates,
            "role_accelerations": role_accelerations,
            "priority_roles": priority_roles,
            "layer_override_role": layer_override_role,
            "layer_override_roles": layer_override_roles,
            "role_feedrate_changed_count": role_changed,
            "layer_feedrate_changed_count": layer_changed,
            "role_acceleration_changed_count": role_acceleration_changed,
            "layer_acceleration_changed_count": layer_acceleration_changed,
            "bottom_layer_count": bottom_layer_count,
            "sample_role_feedrates": role_samples,
            "sample_role_accelerations": acceleration_samples,
        },
    )


def count_positive_extrusion_moves(snippet_text: str) -> int:
    count = 0
    for line in iter_gcode_lines(snippet_text):
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            continue
        e_match = E_WORD_RE.search(split_inline_comment(line)[0])
        if e_match is not None and float(e_match.group(1)) > 0:
            count += 1
    return count


def dominant_axis_direction(dx: float, dy: float, min_move: float) -> tuple[str | None, int]:
    if abs(dx) >= abs(dy):
        if abs(dx) < min_move:
            return None, 0
        return "x", 1 if dx > 0 else -1
    if abs(dy) < min_move:
        return None, 0
    return "y", 1 if dy > 0 else -1


def normalize_role_name(value: str) -> str:
    """Normalize a source `;role:` value into a stable JSON mapping key."""
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalized_role_names(raw_values: Any, field_name: str) -> list[str]:
    if not isinstance(raw_values, list):
        raise ValueError(f"{field_name} must be a list of role names")
    names = [normalize_role_name(str(value)) for value in raw_values]
    if any(not name for name in names):
        raise ValueError(f"{field_name} cannot contain an empty role name")
    return names


def select_role_by_priority(candidates: list[str], priority_roles: list[str]) -> str | None:
    """Choose the highest-priority tag, otherwise preserve the final source tag."""
    for role in priority_roles:
        if role in candidates:
            return role
    return candidates[-1] if candidates else None


def find_layer_override_role(snippet_text: str, layer_override_roles: list[str]) -> str | None:
    """Return a semantic layer tag that applies to every extrusion in this snippet."""
    tags = {
        normalize_role_name(match.group(1))
        for line in iter_gcode_lines(snippet_text)
        if (match := ROLE_TAG_RE.match(line.lstrip())) is not None
    }
    for role in layer_override_roles:
        if role in tags:
            return role
    return None


def normalized_role_feedrates(config: dict[str, Any]) -> dict[str, float]:
    return normalized_role_values(config.get("role_feedrates", {}), "role_feedrates")


def normalized_role_accelerations(config: dict[str, Any]) -> dict[str, float]:
    return normalized_role_values(config.get("role_accelerations", {}), "role_accelerations")


def normalized_role_values(raw_values: Any, field_name: str) -> dict[str, float]:
    if raw_values is None:
        return {}
    if not isinstance(raw_values, dict):
        raise ValueError(f"{field_name} must be an object mapping role names to positive values")
    result: dict[str, float] = {}
    for role, value in raw_values.items():
        normalized_role = normalize_role_name(str(role))
        if not normalized_role:
            raise ValueError(f"{field_name} cannot contain an empty role name")
        resolved_value = optional_positive_feedrate(value, f"{field_name}.{role}")
        if resolved_value is None:
            raise ValueError(f"{field_name}.{role} cannot be null")
        result[normalized_role] = resolved_value
    return result


def optional_positive_feedrate(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    feedrate = float(value)
    if feedrate <= 0:
        raise ValueError(f"{field_name} must be > 0 when set")
    return feedrate


def normalized_slope_feedrate_limits(config: dict[str, Any]) -> list[dict[str, float]]:
    if not bool(config.get("enable_slope_feedrate_limit", False)):
        return []
    raw_limits = config.get("slope_feedrate_limits", [])
    limits: list[dict[str, float]] = []
    for item in raw_limits:
        if not isinstance(item, dict):
            raise ValueError("slope_feedrate_limits entries must be objects")
        slope = float(item["slope"])
        feedrate = float(item["feedrate"])
        if slope < 0:
            raise ValueError("slope_feedrate_limits slope must be >= 0")
        if feedrate <= 0:
            raise ValueError("slope_feedrate_limits feedrate must be > 0")
        limits.append({"slope": slope, "feedrate": feedrate})
    limits.sort(key=lambda item: item["slope"])
    return limits


def slope_feedrate_cap(slope: float, limits: list[dict[str, float]]) -> float | None:
    cap = None
    for item in limits:
        if slope >= item["slope"]:
            cap = item["feedrate"]
        else:
            break
    return cap


def set_or_append_feedrate(command: str, feedrate: float) -> str:
    replacement = "F" + format_float(feedrate)
    if F_WORD_RE.search(command):
        return F_WORD_RE.sub(replacement, command)
    return command.rstrip() + " " + replacement


def maybe_smooth_snippet_z(
    snippet_text: str,
    snippet: Snippet,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    if not bool(config.get("enable_z_smoothing", False)):
        return snippet_text, None

    window = int(config.get("z_smoothing_window", 5))
    passes = int(config.get("z_smoothing_passes", 1))
    min_run = int(config.get("z_smoothing_min_run", 25))
    strength = float(config.get("z_smoothing_strength", 1.0))
    target_feedrate = config.get("z_smoothing_target_feedrate")
    target_feedrate_value = None if target_feedrate is None else float(target_feedrate)
    preserve_endpoints = bool(config.get("z_smoothing_preserve_endpoints", True))
    max_xy_step = config.get("z_smoothing_max_xy_step", 1.0)
    max_xy_step_value = None if max_xy_step is None else float(max_xy_step)
    max_z_step = config.get("z_smoothing_max_z_step", 0.12)
    max_z_step_value = None if max_z_step is None else float(max_z_step)
    max_delta = config.get("z_smoothing_max_delta", 0.05)
    max_delta_value = None if max_delta is None else float(max_delta)
    max_direction_change = config.get("z_smoothing_max_direction_change_degrees", 30.0)
    max_direction_change_value = None if max_direction_change is None else float(max_direction_change)

    if window < 3 or window % 2 == 0:
        raise ValueError("z_smoothing_window must be an odd integer >= 3")
    if passes < 1:
        raise ValueError("z_smoothing_passes must be >= 1")
    if min_run < 3:
        raise ValueError("z_smoothing_min_run must be >= 3")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("z_smoothing_strength must be between 0 and 1")

    lines = snippet_text.splitlines()
    output_lines = list(lines)
    absolute_xyz = True
    current_x: float | None = None
    current_y: float | None = None
    current_z: float | None = None
    modal_feedrate: float | None = None
    current_run: list[dict[str, Any]] = []
    runs: list[list[dict[str, Any]]] = []

    def flush_run() -> None:
        nonlocal current_run
        if len(current_run) >= min_run:
            runs.append(current_run)
        current_run = []

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        upper = stripped.upper()
        if upper == "G90":
            absolute_xyz = True
            flush_run()
            continue
        if upper == "G91":
            absolute_xyz = False
            flush_run()
            continue
        if not stripped.startswith(("G0", "G1")):
            flush_run()
            continue

        command, _comment = split_inline_comment(line)
        feedrate_match = F_WORD_RE.search(command)
        if feedrate_match:
            modal_feedrate = float(feedrate_match.group(1))
        target_x = resolved_axis_value(current_x, X_WORD_RE.search(command), absolute_xyz)
        target_y = resolved_axis_value(current_y, Y_WORD_RE.search(command), absolute_xyz)
        target_z = resolved_axis_value(current_z, Z_WORD_RE.search(command), absolute_xyz)
        e_match = E_WORD_RE.search(command)
        is_positive_extrusion = e_match is not None and float(e_match.group(1)) > 0
        has_explicit_z = Z_WORD_RE.search(command) is not None
        feedrate_matches = (
            target_feedrate_value is None
            or (modal_feedrate is not None and abs(modal_feedrate - target_feedrate_value) < 1e-6)
        )
        step_fits_run = True
        if current_run:
            previous = current_run[-1]
            dxy = math.hypot(target_x - previous["x"], target_y - previous["y"]) if target_x is not None and target_y is not None else 0.0
            dz = abs(target_z - previous["z"]) if target_z is not None else 0.0
            direction_fits = True
            if max_direction_change_value is not None and len(current_run) >= 2 and dxy > 1e-9:
                before = current_run[-2]
                prev_dx = previous["x"] - before["x"]
                prev_dy = previous["y"] - before["y"]
                prev_len = math.hypot(prev_dx, prev_dy)
                if prev_len > 1e-9:
                    dot = (prev_dx * (target_x - previous["x"]) + prev_dy * (target_y - previous["y"])) / (
                        prev_len * dxy
                    )
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                    direction_fits = angle <= max_direction_change_value
            step_fits_run = (
                (max_xy_step_value is None or dxy <= max_xy_step_value)
                and (max_z_step_value is None or dz <= max_z_step_value)
                and direction_fits
            )
        can_smooth = (
            absolute_xyz
            and is_positive_extrusion
            and has_explicit_z
            and target_x is not None
            and target_y is not None
            and target_z is not None
            and feedrate_matches
            and step_fits_run
        )
        if can_smooth:
            current_run.append(
                {
                    "line_index": index,
                    "command": command,
                    "z": target_z,
                    "x": target_x,
                    "y": target_y,
                    "feedrate": modal_feedrate,
                }
            )
        else:
            flush_run()
            if (
                absolute_xyz
                and is_positive_extrusion
                and has_explicit_z
                and target_x is not None
                and target_y is not None
                and target_z is not None
                and feedrate_matches
            ):
                current_run.append(
                    {
                        "line_index": index,
                        "command": command,
                        "z": target_z,
                        "x": target_x,
                        "y": target_y,
                        "feedrate": modal_feedrate,
                    }
                )

        current_x = target_x
        current_y = target_y
        current_z = target_z

    flush_run()

    if not runs:
        return snippet_text, None

    changed_points = 0
    total_abs_delta = 0.0
    max_abs_delta = 0.0
    sample_adjustments: list[dict[str, float]] = []

    for run in runs:
        original = [point["z"] for point in run]
        smoothed = smooth_values(original, window, passes, preserve_endpoints)
        for point, old_z, smooth_z in zip(run, original, smoothed):
            new_z = old_z + (smooth_z - old_z) * strength
            delta = new_z - old_z
            if max_delta_value is not None and abs(delta) > max_delta_value:
                delta = math.copysign(max_delta_value, delta)
                new_z = old_z + delta
            if abs(delta) < 1e-9:
                continue
            line_index = point["line_index"]
            command, comment = split_inline_comment(output_lines[line_index])
            command = set_z_word(command, new_z)
            output_lines[line_index] = command + comment
            changed_points += 1
            abs_delta = abs(delta)
            total_abs_delta += abs_delta
            max_abs_delta = max(max_abs_delta, abs_delta)
            if len(sample_adjustments) < 8:
                sample_adjustments.append(
                    {
                        "line_index": float(line_index + 1),
                        "old_z": old_z,
                        "new_z": new_z,
                        "delta": delta,
                    }
                )

    if changed_points == 0:
        return snippet_text, None

    return (
        "\n".join(output_lines) + ("\n" if snippet_text.endswith("\n") else ""),
        {
            "file": snippet.path.name,
            "main_index": snippet.main_index,
            "sub_index": snippet.sub_index,
            "material": snippet.material,
            "tool": snippet.tool,
            "window": window,
            "passes": passes,
            "min_run": min_run,
            "strength": strength,
            "target_feedrate": target_feedrate_value,
            "preserve_endpoints": preserve_endpoints,
            "max_xy_step": max_xy_step_value,
            "max_z_step": max_z_step_value,
            "max_delta": max_delta_value,
            "max_direction_change_degrees": max_direction_change_value,
            "smoothed_run_count": len(runs),
            "changed_point_count": changed_points,
            "mean_abs_delta_z": total_abs_delta / changed_points,
            "max_abs_delta_z": max_abs_delta,
            "sample_adjustments": sample_adjustments,
        },
    )


def smooth_values(values: list[float], window: int, passes: int, preserve_endpoints: bool) -> list[float]:
    half_window = window // 2
    smoothed = list(values)
    for _ in range(passes):
        source = smoothed
        target = list(source)
        for index in range(len(source)):
            if preserve_endpoints and (index == 0 or index == len(source) - 1):
                continue
            lo = max(0, index - half_window)
            hi = min(len(source), index + half_window + 1)
            target[index] = sum(source[lo:hi]) / (hi - lo)
        smoothed = target
    return smoothed


def set_z_word(command: str, z: float) -> str:
    replacement = "Z" + format_z(z)
    if Z_WORD_RE.search(command):
        return Z_WORD_RE.sub(replacement, command)
    return command.rstrip() + " " + replacement


def maybe_rewrite_snippet_extrusion(
    snippet_text: str,
    snippet: Snippet,
    config: dict[str, Any],
    layer_fraction_by_main: dict[int, float],
) -> tuple[str, dict[str, Any] | None]:
    multiplier = resolve_extrusion_multiplier(config, snippet, layer_fraction_by_main)
    if multiplier is None:
        return snippet_text, None

    multiplier_value = float(multiplier)
    if abs(multiplier_value - 1.0) < 1e-12:
        return snippet_text, None

    changed = 0
    original_values: list[float] = []
    new_values: list[float] = []
    output_lines = GCodeLineBuffer()

    for line in iter_gcode_lines(snippet_text):
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            output_lines.append(line)
            continue

        command, comment = split_inline_comment(line)

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            old_value = float(match.group(1))
            # Keep pressure-control moves exact; flow correction is for
            # positive material deposition only.
            if old_value <= 0:
                return match.group(0)
            new_value = old_value * multiplier_value
            changed += 1
            if len(original_values) < 8:
                original_values.append(old_value)
                new_values.append(new_value)
            return "E" + format_float(new_value)

        output_lines.append(E_WORD_RE.sub(replace, command) + comment)

    if changed == 0:
        return snippet_text, None

    return (
        output_lines.render(snippet_text.endswith("\n")),
        {
            "file": snippet.path.name,
            "main_index": snippet.main_index,
            "layer_fraction": layer_fraction_by_main.get(snippet.main_index),
            "material": snippet.material,
            "extrusion_multiplier": multiplier_value,
            "changed_count": changed,
            "sample_original_e": original_values,
            "sample_new_e": new_values,
        },
    )


def resolve_extrusion_multiplier(
    config: dict[str, Any],
    snippet: Snippet,
    layer_fraction_by_main: dict[int, float],
) -> float | None:
    multiplier = config.get("extrusion_multiplier")
    schedule = config.get("extrusion_multiplier_schedule")
    if not schedule:
        return None if multiplier is None else float(multiplier)

    layer_fraction = layer_fraction_by_main.get(snippet.main_index, 0.0)
    selected = multiplier
    for entry in sorted(schedule, key=extrusion_schedule_start_fraction):
        start_fraction = extrusion_schedule_start_fraction(entry)
        if layer_fraction + 1e-12 >= start_fraction:
            selected = entry.get("multiplier", selected)
    return None if selected is None else float(selected)


def extrusion_schedule_start_fraction(entry: dict[str, Any]) -> float:
    if "start_fraction" in entry:
        return float(entry["start_fraction"])
    if "start_percent" in entry:
        return float(entry["start_percent"]) / 100.0
    return 0.0


def build_planar_platform_block(config: dict[str, Any], snippets: list[Snippet], encoding: str) -> dict[str, Any] | None:
    if not bool(config.get("enable_planar_platform", False)):
        return None

    bbox_source = str(config.get("planar_platform_bbox_source", "all_snippets"))
    if bbox_source == "all_snippets":
        bbox_snippets = snippets
    elif bbox_source == "first_main_layer":
        first_main_index = min(snippet.main_index for snippet in snippets)
        bbox_snippets = [snippet for snippet in snippets if snippet.main_index == first_main_index]
    else:
        raise ValueError(
            "planar_platform_bbox_source must be 'all_snippets' or 'first_main_layer'"
        )

    inferred = infer_snippet_xy_bbox_and_first_z(bbox_snippets, encoding)
    if inferred is None:
        raise ValueError("Cannot build planar platform: no printable XY/Z moves found in snippets")
    min_x, min_y, max_x, max_y, first_z = inferred

    margin = float(config.get("planar_platform_margin", 2.0))
    height = float(config.get("planar_platform_height", 0.2))
    spacing = float(config.get("planar_platform_line_spacing", 0.45))
    line_width = float(config.get("planar_platform_line_width", 0.45))
    feedrate = float(config.get("planar_platform_feedrate", 900))
    travel_feedrate = float(config.get("planar_platform_travel_feedrate", 6000))
    extrusion_scale = float(config.get("planar_platform_extrusion_scale", 0.85))
    filament_diameter = float(config.get("planar_platform_filament_diameter", 1.75))

    if height <= 0:
        raise ValueError("planar_platform_height must be > 0")
    if spacing <= 0:
        raise ValueError("planar_platform_line_spacing must be > 0")
    if line_width <= 0:
        raise ValueError("planar_platform_line_width must be > 0")
    if filament_diameter <= 0:
        raise ValueError("planar_platform_filament_diameter must be > 0")

    platform_z = float(config.get("planar_platform_z", height))
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Cannot build planar platform: invalid XY bounds")

    gcode = build_rect_platform_gcode(
        marker="ASSEMBLED_PLANAR_PLATFORM",
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        platform_z=platform_z,
        height=height,
        spacing=spacing,
        line_width=line_width,
        feedrate=feedrate,
        travel_feedrate=travel_feedrate,
        extrusion_scale=extrusion_scale,
        filament_diameter=filament_diameter,
    )
    return {
        "gcode": gcode,
        "summary": {
            "bbox": [min_x, min_y, max_x, max_y],
            "z": platform_z,
            "height": height,
            "line_spacing": spacing,
            "line_width": line_width,
            "model_z_lift": height,
        },
    }


def build_wipe_tower_platform_block(config: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(config.get("enable_planar_platform", False)):
        return None
    if not bool(config.get("enable_balanced_wipe_tower", False)):
        return None
    if not bool(config.get("enable_wipe_tower_platform", True)):
        return None

    rect = balanced_wipe_tower_xy_rect(config)
    min_x, min_y, max_x, max_y = [float(value) for value in rect["bbox"]]
    margin = float(config.get("wipe_tower_platform_margin", config.get("planar_platform_margin", 2.0)))
    height = float(config.get("wipe_tower_platform_height", config.get("planar_platform_height", 0.2)))
    platform_z = float(config.get("wipe_tower_platform_z", config.get("planar_platform_z", height)))
    spacing = float(config.get("wipe_tower_platform_line_spacing", config.get("planar_platform_line_spacing", 0.45)))
    line_width = float(config.get("wipe_tower_platform_line_width", config.get("planar_platform_line_width", 0.45)))
    feedrate = float(config.get("wipe_tower_platform_feedrate", config.get("planar_platform_feedrate", 900)))
    travel_feedrate = float(config.get("wipe_tower_platform_travel_feedrate", config.get("planar_platform_travel_feedrate", 6000)))
    extrusion_scale = float(config.get("wipe_tower_platform_extrusion_scale", config.get("planar_platform_extrusion_scale", 0.85)))
    filament_diameter = float(config.get("wipe_tower_platform_filament_diameter", config.get("planar_platform_filament_diameter", 1.75)))

    if margin < 0:
        raise ValueError("wipe_tower_platform_margin must be >= 0")
    min_x -= margin
    min_y -= margin
    max_x += margin
    max_y += margin
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Cannot build wipe tower platform: invalid XY bounds")
    if height <= 0:
        raise ValueError("wipe_tower_platform_height must be > 0")
    if spacing <= 0:
        raise ValueError("wipe_tower_platform_line_spacing must be > 0")
    if line_width <= 0:
        raise ValueError("wipe_tower_platform_line_width must be > 0")
    if filament_diameter <= 0:
        raise ValueError("wipe_tower_platform_filament_diameter must be > 0")

    gcode = build_rect_platform_gcode(
        marker="ASSEMBLED_WIPE_TOWER_PLATFORM",
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        platform_z=platform_z,
        height=height,
        spacing=spacing,
        line_width=line_width,
        feedrate=feedrate,
        travel_feedrate=travel_feedrate,
        extrusion_scale=extrusion_scale,
        filament_diameter=filament_diameter,
    )
    return {
        "gcode": gcode,
        "summary": {
            "bbox": [min_x, min_y, max_x, max_y],
            "tower_bbox": rect["bbox"],
            "z": platform_z,
            "height": height,
            "line_spacing": spacing,
            "line_width": line_width,
            "margin": margin,
        },
    }


def build_rect_platform_gcode(
    marker: str,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    platform_z: float,
    height: float,
    spacing: float,
    line_width: float,
    feedrate: float,
    travel_feedrate: float,
    extrusion_scale: float,
    filament_diameter: float,
) -> str:
    filament_area = math.pi * (filament_diameter / 2.0) ** 2
    line_len = max_x - min_x
    line_e = line_len * line_width * height / filament_area * extrusion_scale
    step_e = spacing * line_width * height / filament_area * extrusion_scale

    y_values: list[float] = []
    y = min_y
    while y <= max_y + 1e-6:
        y_values.append(min(y, max_y))
        y += spacing
    if y_values[-1] < max_y:
        y_values.append(max_y)

    lines = [
        f"; {marker}_START",
        f"; bbox={format_xy(min_x)},{format_xy(min_y)},{format_xy(max_x)},{format_xy(max_y)} "
        f"z={format_z(platform_z)} height={format_float(height)}",
        "G90",
        "M83",
        f"G1 X{format_xy(min_x)} Y{format_xy(y_values[0])} F{format_float(travel_feedrate)}",
        f"G1 Z{format_z(platform_z)} F1200",
    ]

    current_x = min_x
    for index, row_y in enumerate(y_values):
        next_x = max_x if abs(current_x - min_x) < 1e-9 else min_x
        lines.append(f"G1 X{format_xy(next_x)} E{format_float(line_e)} F{format_float(feedrate)}")
        current_x = next_x
        if index != len(y_values) - 1:
            next_y = y_values[index + 1]
            lines.append(f"G1 Y{format_xy(next_y)} E{format_float(step_e)} F{format_float(feedrate)}")

    lines.extend(["M400", f"; {marker}_END"])
    return "\n".join(lines)


def infer_snippet_xy_bbox_and_first_z(snippets: list[Snippet], encoding: str) -> tuple[float, float, float, float, float] | None:
    min_x: float | None = None
    max_x: float | None = None
    min_y: float | None = None
    max_y: float | None = None
    first_z: float | None = None

    for snippet in snippets:
        text = read_text(snippet.path, encoding)
        for line in iter_gcode_lines(text):
            stripped = line.lstrip()
            if not stripped.startswith(("G0", "G1")):
                continue
            x_match = X_WORD_RE.search(line)
            y_match = Y_WORD_RE.search(line)
            z_match = Z_WORD_RE.search(line)
            if x_match:
                x_value = float(x_match.group(1))
                min_x = x_value if min_x is None else min(min_x, x_value)
                max_x = x_value if max_x is None else max(max_x, x_value)
            if y_match:
                y_value = float(y_match.group(1))
                min_y = y_value if min_y is None else min(min_y, y_value)
                max_y = y_value if max_y is None else max(max_y, y_value)
            if first_z is None and z_match:
                first_z = float(z_match.group(1))

    if min_x is None or max_x is None or min_y is None or max_y is None or first_z is None:
        return None
    return min_x, min_y, max_x, max_y, first_z


def validate_wipe_tower_xy_bounds(config: dict[str, Any], snippets: list[Snippet], encoding: str) -> dict[str, Any] | None:
    if not bool(config.get("validate_wipe_tower_outside_model_bbox", True)):
        return None
    if not (bool(config.get("enable_simple_wipe_tower", False)) or bool(config.get("enable_balanced_wipe_tower", False))):
        return None

    inferred = infer_snippet_xy_bbox_and_first_z(snippets, encoding)
    if inferred is None:
        raise ValueError("Cannot validate wipe tower bounds: no printable XY/Z moves found in snippets")

    model_min_x, model_min_y, model_max_x, model_max_y, _first_z = inferred
    clearance = float(config.get("wipe_tower_model_clearance", 2.0))
    if clearance < 0:
        raise ValueError("wipe_tower_model_clearance must be >= 0")

    expanded_model = (
        model_min_x - clearance,
        model_min_y - clearance,
        model_max_x + clearance,
        model_max_y + clearance,
    )
    rects: list[dict[str, Any]] = []

    if bool(config.get("enable_simple_wipe_tower", False)):
        rects.append(simple_wipe_tower_xy_rect(config, snippets))

    if bool(config.get("enable_balanced_wipe_tower", False)):
        rects.append(balanced_wipe_tower_xy_rect(config))

    bed_bounds = configured_bed_bounds(config)
    for rect in rects:
        rect_bbox = tuple(rect["bbox"])
        if rects_overlap(rect_bbox, expanded_model):
            raise ValueError(
                f"{rect['name']} overlaps model XY bbox plus clearance. "
                f"tower_bbox={format_bbox(rect_bbox)} model_bbox={format_bbox((model_min_x, model_min_y, model_max_x, model_max_y))} "
                f"clearance={format_float(clearance)}"
            )
        if bed_bounds is not None and not rect_inside(rect_bbox, bed_bounds):
            raise ValueError(
                f"{rect['name']} is outside configured bed bounds. "
                f"tower_bbox={format_bbox(rect_bbox)} bed_bounds={format_bbox(bed_bounds)}"
            )

    return {
        "model_bbox": [model_min_x, model_min_y, model_max_x, model_max_y],
        "expanded_model_bbox": list(expanded_model),
        "clearance": clearance,
        "bed_bounds": list(bed_bounds) if bed_bounds is not None else None,
        "tower_rects": rects,
    }


def simple_wipe_tower_xy_rect(config: dict[str, Any], snippets: list[Snippet]) -> dict[str, Any]:
    x_start = float(config.get("simple_wipe_tower_x_start", 250.0))
    x_end = float(config.get("simple_wipe_tower_x_end", 310.0))
    y_start = float(config.get("simple_wipe_tower_y_start", 40.0))
    row_spacing = float(config.get("simple_wipe_tower_row_spacing", 0.8))
    rows_per_change = int(config.get("simple_wipe_tower_rows_per_change", 3))
    line_width = float(config.get("simple_wipe_tower_line_width", 0.45))

    if row_spacing <= 0:
        raise ValueError("simple_wipe_tower_row_spacing must be > 0")
    if rows_per_change <= 0:
        raise ValueError("simple_wipe_tower_rows_per_change must be > 0")
    if line_width <= 0:
        raise ValueError("simple_wipe_tower_line_width must be > 0")

    blocks_per_main: dict[int, int] = {}
    current: Snippet | None = None
    for snippet in snippets:
        if current is not None and snippet.material != current.material:
            blocks_per_main[snippet.main_index] = blocks_per_main.get(snippet.main_index, 0) + 1
        current = snippet

    max_blocks_on_main = max(blocks_per_main.values(), default=1)
    max_y = y_start + (max_blocks_on_main * rows_per_change - 1) * row_spacing
    half_width = line_width / 2.0
    return {
        "name": "simple_wipe_tower",
        "bbox": [
            min(x_start, x_end) - half_width,
            y_start - half_width,
            max(x_start, x_end) + half_width,
            max_y + half_width,
        ],
        "max_blocks_on_main": max_blocks_on_main,
    }


def balanced_wipe_tower_xy_rect(config: dict[str, Any]) -> dict[str, Any]:
    x = float(config.get("balanced_wipe_tower_x", config.get("simple_wipe_tower_x_start", 250.0)))
    y = float(config.get("balanced_wipe_tower_y", config.get("simple_wipe_tower_y_start", 40.0)))
    width = float(config.get("balanced_wipe_tower_width", 45.0))
    depth = float(config.get("balanced_wipe_tower_depth", 12.0))
    line_width = float(config.get("balanced_wipe_tower_line_width", 0.45))
    if width <= 0:
        raise ValueError("balanced_wipe_tower_width must be > 0")
    if depth <= 0:
        raise ValueError("balanced_wipe_tower_depth must be > 0")
    if line_width <= 0:
        raise ValueError("balanced_wipe_tower_line_width must be > 0")

    half_width = line_width / 2.0
    return {
        "name": "balanced_wipe_tower",
        "bbox": [x - half_width, y - half_width, x + width + half_width, y + depth + half_width],
        "width": width,
        "depth": depth,
    }


def configured_bed_bounds(config: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = config.get("bed_bounds")
    if isinstance(raw, list) and len(raw) == 4:
        return tuple(float(value) for value in raw)  # type: ignore[return-value]

    keys = ("bed_min_x", "bed_min_y", "bed_max_x", "bed_max_y")
    if all(key in config for key in keys):
        return tuple(float(config[key]) for key in keys)  # type: ignore[return-value]

    return None


def rects_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def rect_inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return ",".join(format_xy(value) for value in bbox)


def lift_z_words(gcode_text: str, lift: float) -> str:
    def replace(match: re.Match[str]) -> str:
        return "Z" + format_z(float(match.group(1)) + lift)

    return Z_WORD_RE.sub(replace, gcode_text)


def maybe_raise_post_code_park_z(
    post_code: str,
    highest_print_z: float | None,
    config: dict[str, Any],
) -> tuple[str, float | None]:
    if not bool(config.get("raise_post_park_z", False)) or highest_print_z is None:
        return post_code, None

    minimum_z = float(config.get("post_park_z_minimum", 0.0))
    clearance = float(config.get("post_park_z_clearance", 0.0))
    park_z = max(minimum_z, highest_print_z + clearance)

    def replace(match: re.Match[str]) -> str:
        return (
            f"{match.group('prefix')}{format_z(park_z)}{match.group('suffix')}"
            f"{match.group('comment')}; ASSEMBLED_DYNAMIC_PARK_Z"
        )

    rewritten, replacements = POST_PARK_Z_RE.subn(replace, post_code, count=1)
    return (rewritten, park_z) if replacements else (post_code, None)


def maybe_strip_zero_extrusion_words(gcode_text: str, config: dict[str, Any]) -> str:
    if not bool(config.get("strip_zero_extrusion_words", False)):
        return gcode_text

    output: list[str] = []
    for line in StringIO(gcode_text):
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        content = line[: -len(newline)] if newline else line
        command, separator, comment = content.partition(";")
        if (
            MOTION_COMMAND_RE.match(command)
            and ZERO_E_WORD_RE.search(command)
            and (X_WORD_RE.search(command) or Y_WORD_RE.search(command) or Z_WORD_RE.search(command))
        ):
            command = ZERO_E_WORD_RE.sub("", command).rstrip()
        output.append(command + (separator + comment if separator else "") + newline)
    return "".join(output)


def maybe_skip_pre_code_sections(gcode_text: str, config: dict[str, Any]) -> tuple[str, list[str]]:
    skipped: list[str] = []
    output = gcode_text
    for section_name, (start_marker, end_marker, flag_name) in PRE_CODE_SECTIONS.items():
        if not config.get(flag_name, False):
            continue
        output = remove_marked_section(output, start_marker, end_marker, section_name)
        skipped.append(section_name)
    return output, skipped


def remove_marked_section(gcode_text: str, start_marker: str, end_marker: str, section_name: str) -> str:
    lines = gcode_text.splitlines()
    keep: list[str] = []
    skipping = False
    found_start = False
    found_end = False

    for line in lines:
        if not skipping and line.strip() == start_marker:
            found_start = True
            skipping = True
            keep.append(f"; ASSEMBLER_SKIPPED_START {section_name}")
            continue
        if skipping and line.strip() == end_marker:
            found_end = True
            skipping = False
            keep.append(f"; ASSEMBLER_SKIPPED_END {section_name}")
            continue
        if not skipping:
            keep.append(line)

    if not found_start:
        raise ValueError(f"Cannot skip pre-code section {section_name}: start marker not found")
    if not found_end:
        raise ValueError(f"Cannot skip pre-code section {section_name}: end marker not found")

    return "\n".join(keep) + ("\n" if gcode_text.endswith("\n") else "")


def maybe_rewrite_m620_10_flush_temperatures(gcode_text: str, config: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not config.get("rewrite_m620_10_flush_temperatures", False):
        return gcode_text, []

    temperatures = {
        int(tool): float(temp)
        for tool, temp in config.get("tool_temperatures", {"0": 220, "1": 220}).items()
    }
    rewritten: list[dict[str, Any]] = []
    output_lines: list[str] = []

    for line_number, line in enumerate(gcode_text.splitlines(), start=1):
        match = M620_10_RE.match(line)
        if not match:
            output_lines.append(line)
            continue

        tool = int(match.group(2))
        old_flush_temperature = float(match.group(4))
        print_temperature = float(match.group(6))
        new_flush_temperature = temperatures.get(tool, print_temperature)

        if abs(old_flush_temperature - new_flush_temperature) < 1e-9:
            output_lines.append(line)
            continue

        output_lines.append(
            f"{match.group(1)}{match.group(2)}{match.group(3)}{format_float(new_flush_temperature)}"
            f"{match.group(5)}{match.group(6)}{match.group(7)}"
        )
        rewritten.append(
            {
                "line": line_number,
                "tool": tool,
                "old_flush_temperature": old_flush_temperature,
                "new_flush_temperature": new_flush_temperature,
                "print_temperature": print_temperature,
            }
        )

    return "\n".join(output_lines) + ("\n" if gcode_text.endswith("\n") else ""), rewritten


def rewrite_logical_tool_commands(gcode_text: str, target_tool: int) -> str:
    """Rewrite safe logical T0/T1 selectors to target_tool.

    M620.17 calibration slot lines are deliberately not matched by these
    patterns. G383/G383.3 active L markers are allowed to follow the first tool.
    """
    if target_tool not in (0, 1):
        return gcode_text
    source_tool = 1 - target_tool
    lines = []
    for line in gcode_text.splitlines():
        lines.append(rewrite_logical_tool_line(line, source_tool, target_tool))
    return "\n".join(lines) + ("\n" if gcode_text.endswith("\n") else "")


def rewrite_logical_tool_line(line: str, source_tool: int, target_tool: int) -> str:
    source = str(source_tool)
    target = str(target_tool)

    match = TOOL_LINE_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}T{target}{match.group(3)}"

    match = M620_START_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M620 S{target}A{match.group(3)}"

    match = M621_END_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M621 S{target}A{match.group(3)}"

    match = M620_AIR_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M620.6 I{target}{match.group(3)}"

    match = G383_ACTIVE_L_RE.match(line)
    if match and ("G383.3" in line or "O1" in line) and match.group(2) == source:
        return f"{match.group(1)}{target}{match.group(3)}"

    return line


def build_tool_change_block(
    config: dict[str, Any],
    old_tool: int,
    new_tool: int,
    old_material: int,
    new_material: int,
    next_file: str,
    first_tool_activation: bool,
    dynamic_compensation_retract: float,
    motion_state: MotionState,
) -> str:
    z_hop = float(config.get("z_hop", 3.0))
    z_feedrate = float(config.get("z_hop_feedrate", 1200))
    retract = tool_change_extrusion_delta(config, "tool_change_retract_by_tool", old_tool, "tool_change_retract")
    retract_feedrate = float(config.get("retract_feedrate", 1800))
    unretract = tool_change_extrusion_delta(config, "tool_change_unretract_by_tool", new_tool, "tool_change_unretract")
    unretract_feedrate = float(config.get("unretract_feedrate", retract_feedrate))
    wait_after_tool_ms = int(config.get("wait_after_tool_ms", 0))
    enable_air_printing_detect = bool(config.get("enable_air_printing_detect", True))
    lower_after_tool_change = bool(config.get("lower_after_tool_change", False))
    approach_first_xy = bool(config.get("approach_first_xy_while_lifted", True))
    force_tool_temperatures = bool(config.get("set_tool_temperature_on_change", False))
    dynamic_compensation = bool(config.get("dynamic_extrusion_compensation_on_tool_change", False))

    lines = [
        f"; NOZZLE_CHANGE_START OF{old_tool} NF{new_tool}",
        f"; ASSEMBLED_MATERIAL_CHANGE old_material={old_material} new_material={new_material} next_file={next_file}",
        f"; ASSEMBLED_DYNAMIC_COMPENSATION first_tool_activation={int(first_tool_activation)} R{format_float(dynamic_compensation_retract)}",
        "M400",
    ]
    pressure_wipe_retract = 0.0
    pressure_relief = 0.0
    if bool(config.get("tool_change_pressure_wipe", False)):
        pressure_wipe_retract = float(
            config.get("tool_change_pressure_wipe_retract", config.get("long_travel_pressure_wipe_retract", 0.38))
        )
        pressure_relief = float(
            config.get("tool_change_pressure_relief", config.get("long_travel_pressure_relief", 0.02))
        )
        pressure_wipe_feedrate = float(
            config.get("tool_change_pressure_wipe_feedrate", config.get("long_travel_pressure_wipe_feedrate", retract_feedrate))
        )
        pressure_wipe_z_lift = float(
            config.get(
                "tool_change_pressure_wipe_z_lift",
                config.get("long_travel_pressure_wipe_z_lift", config.get("pressure_wipe_z_lift", 0.0)),
            )
        )
        pressure_wipe_z_feedrate = float(
            config.get(
                "tool_change_pressure_wipe_z_feedrate",
                config.get("long_travel_pressure_wipe_z_feedrate", config.get("pressure_wipe_z_feedrate", z_feedrate)),
            )
        )
        wipe_distance = float(
            config.get("tool_change_pressure_wipe_distance", config.get("long_travel_pressure_wipe_distance", 1.0))
        )
        wipe_to_xy = tangential_wipe_target(
            from_x=motion_state.x,
            from_y=motion_state.y,
            previous_xy=motion_state.previous_xy,
            distance=wipe_distance,
        )
        wipe_path = backward_wipe_path(
            from_x=motion_state.x,
            from_y=motion_state.y,
            from_z=motion_state.z,
            print_path=motion_state.print_path,
            distance=wipe_distance,
        )
        lines.extend(
            [
                f"; ASSEMBLED_TOOL_CHANGE_PRESSURE_WIPE old_tool=T{old_tool} new_tool=T{new_tool}",
                *build_pressure_wipe_lines(
                    wipe_to_xy=wipe_to_xy,
                    wipe_retract=pressure_wipe_retract,
                    pressure_relief=pressure_relief,
                    feedrate=pressure_wipe_feedrate,
                    wipe_path=wipe_path,
                    z_lift=pressure_wipe_z_lift,
                    z_feedrate=pressure_wipe_z_feedrate,
                    retract_before_wipe=bool(
                        config.get("tool_change_pressure_wipe_retract_before_wipe", config.get("pressure_wipe_retract_before_wipe", False))
                    ),
                    follow_wipe_z=bool(config.get("backward_wipe_follow_z", False)),
                    wipe_z_offset=float(config.get("backward_wipe_z_offset", 0.0)),
                ),
            ]
        )
        if wipe_path:
            motion_state.x, motion_state.y, motion_state.z = wipe_path[-1]
            motion_state.print_path.clear()
    if bool(config.get("tool_change_pressure_wipe_reduces_retract", True)):
        retract = max(0.0, retract - pressure_wipe_retract - pressure_relief)
    lines.extend(build_extrusion_delta_lines(-retract, retract_feedrate, "tool-change retract"))
    if z_hop > 0:
        lines.extend(
            [
                "G91",
                f"G1 Z{format_float(z_hop)} F{format_float(z_feedrate)}",
                "G90",
            ]
        )
    lines.extend(build_tool_change_wipe_lines(config, "before_switch", old_tool))
    lines.extend(
        [
            f"M620 S{new_tool}A",
            "M400",
            f"T{new_tool}",
            "M400",
        ]
    )
    if wait_after_tool_ms > 0:
        lines.append(f"M400 P{wait_after_tool_ms}")
    if dynamic_compensation:
        lines.extend(build_dynamic_extrusion_compensation_lines(config, new_tool, dynamic_compensation_retract))
    lines.append(f"M621 S{new_tool}A")
    if enable_air_printing_detect:
        lines.append(f"M620.6 I{new_tool} W1 ;enable ams air printing detect")
    if force_tool_temperatures:
        lines.extend(build_tool_temperature_lines(config, old_tool, new_tool))
    lines.extend(build_tool_change_wipe_lines(config, "after_switch", new_tool))
    if not approach_first_xy:
        lines.extend(build_extrusion_delta_lines(unretract, unretract_feedrate, "tool-change unretract"))
        lines.extend(build_after_unretract_sync_lines(config))
    if z_hop > 0 and lower_after_tool_change:
        lines.extend(
            [
                "G91",
                f"G1 Z-{format_float(z_hop)} F{format_float(z_feedrate)}",
                "G90",
            ]
        )
    lines.extend(
        [
            f"; NOZZLE_CHANGE_END OF{old_tool} NF{new_tool}",
            "",
        ]
    )
    return "\n".join(lines)


def build_dynamic_extrusion_compensation_lines(
    config: dict[str, Any],
    tool: int,
    retract_length: float | None = None,
) -> list[str]:
    feedrate = tool_change_extrusion_delta(
        config,
        "dynamic_extrusion_compensation_feedrate_by_tool",
        tool,
        "dynamic_extrusion_compensation_feedrate",
    )
    amount = float(config.get("dynamic_extrusion_compensation_a", 0.4))
    if retract_length is None:
        retract_length = tool_change_extrusion_delta(
            config,
            "dynamic_extrusion_compensation_retract_by_tool",
            tool,
            "dynamic_extrusion_compensation_retract_length",
        )
    use_m620_10_retract = bool(config.get("dynamic_extrusion_compensation_m620_10_retract", True))
    lines = ["; dynamic extrusion compensation after tool switch"]
    if use_m620_10_retract:
        lines.append(f"M620.10 R{format_float(retract_length)}")
    lines.extend(
        [
            "M628 S0",
            "M629",
            "M400",
            f"M983.3 F{format_float(feedrate)} A{format_float(amount)} R{format_float(retract_length)}",
            "M400",
        ]
    )
    return lines


def build_tool_change_wipe_lines(config: dict[str, Any], phase: str, tool: int) -> list[str]:
    enabled_key = {
        "before_switch": "wipe_nozzle_before_tool_change",
        "after_switch": "wipe_nozzle_after_tool_change",
    }[phase]
    if not bool(config.get(enabled_key, False)):
        return []

    command_key = {
        "before_switch": "tool_change_wipe_before_commands",
        "after_switch": "tool_change_wipe_after_commands",
    }[phase]
    commands = config.get(command_key, ["G150.1"])
    if isinstance(commands, str):
        commands = [commands]

    label = "before switching away from" if phase == "before_switch" else "after switching to"
    lines = [f"; wipe nozzle {label} T{tool}"]
    lines.extend(str(command) for command in commands)
    if bool(config.get("m400_after_tool_change_wipe", True)):
        lines.append("M400")
    return lines


def build_balanced_wipe_tower_plan(snippets: list[Snippet], config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if not bool(config.get("enable_balanced_wipe_tower", False)):
        return {}

    min_lines = int(config.get("balanced_wipe_tower_min_infill_lines", 5))
    first_lines = int(config.get("balanced_wipe_tower_first_material_lines_on_change", 2))
    if min_lines <= 0:
        raise ValueError("balanced_wipe_tower_min_infill_lines must be > 0")
    if first_lines <= 0:
        raise ValueError("balanced_wipe_tower_first_material_lines_on_change must be > 0")
    if first_lines >= min_lines:
        raise ValueError("balanced_wipe_tower_first_material_lines_on_change must be less than balanced_wipe_tower_min_infill_lines")

    grouped: dict[int, list[Snippet]] = {}
    for snippet in snippets:
        grouped.setdefault(snippet.main_index, []).append(snippet)

    plan: dict[int, dict[str, Any]] = {}
    for main_index, layer_snippets in grouped.items():
        ordered_materials: list[int] = []
        for snippet in sorted(layer_snippets, key=lambda item: (item.sub_index, item.path.name)):
            if snippet.material not in ordered_materials:
                ordered_materials.append(snippet.material)
        if not ordered_materials:
            continue

        total_lines = max(min_lines, len(ordered_materials) * first_lines)
        assignments: dict[int, list[int]] = {}
        next_line = 0
        if len(ordered_materials) == 1:
            assignments[ordered_materials[0]] = list(range(total_lines))
        else:
            for material in ordered_materials[:-1]:
                assignments[material] = list(range(next_line, next_line + first_lines))
                next_line += first_lines
            assignments[ordered_materials[-1]] = list(range(next_line, total_lines))

        plan[main_index] = {
            "materials": ordered_materials,
            "wall_material": ordered_materials[0],
            "total_lines": total_lines,
            "assignments": assignments,
        }

    return plan


def build_balanced_wipe_tower_block(
    config: dict[str, Any],
    snippet_text: str,
    snippet: Snippet,
    plan: dict[int, dict[str, Any]],
    state: dict[str, Any],
    highest_print_z_so_far: float | None,
    previous_transition_was_tool_change: bool = False,
    previous_transition_was_layer_change: bool = False,
    pressure_debt_prime: float = 0.0,
) -> str:
    if not plan:
        return ""

    layer_plan = plan.get(snippet.main_index)
    if layer_plan is None:
        return ""
    assignments: dict[int, list[int]] = layer_plan["assignments"]
    line_indexes = assignments.get(snippet.material)
    if line_indexes is None:
        return ""

    printed = state.setdefault("printed", {})
    layer_printed = printed.setdefault(snippet.main_index, set())
    if snippet.material in layer_printed:
        return ""
    layer_printed.add(snippet.material)

    z = first_z_value(snippet_text)
    if z is None:
        return (
            f"; ASSEMBLED_BALANCED_WIPE_TOWER_SKIPPED file={snippet.path.name} "
            "no_first_z_found\n"
        )

    layer_height = balanced_wipe_tower_layer_height(config, state, snippet.main_index, z)
    x = float(config.get("balanced_wipe_tower_x", config.get("simple_wipe_tower_x_start", 250.0)))
    y = float(config.get("balanced_wipe_tower_y", config.get("simple_wipe_tower_y_start", 40.0)))
    width = float(config.get("balanced_wipe_tower_width", 45.0))
    depth = float(config.get("balanced_wipe_tower_depth", 12.0))
    spacing = float(config.get("balanced_wipe_tower_line_spacing", 0.45))
    line_width = float(config.get("balanced_wipe_tower_line_width", 0.45))
    filament_diameter = float(config.get("balanced_wipe_tower_filament_diameter", 1.75))
    extrusion_scale = float(config.get("balanced_wipe_tower_extrusion_scale", 1.0))
    feedrate = float(config.get("balanced_wipe_tower_feedrate", 1800))
    wall_feedrate = float(config.get("balanced_wipe_tower_wall_feedrate", feedrate))
    travel_feedrate = float(config.get("balanced_wipe_tower_travel_feedrate", config.get("approach_xy_feedrate", 30000)))
    z_feedrate = float(config.get("layer_change_z_hop_feedrate", config.get("z_hop_feedrate", 1200)))
    retract = float(config.get("layer_change_retract", config.get("retract_length", 0.8)))
    retract_feedrate = float(config.get("layer_change_retract_feedrate", config.get("retract_feedrate", 1800)))
    unretract = float(config.get("layer_change_unretract", retract))
    unretract_feedrate = float(
        config.get("layer_change_unretract_feedrate", config.get("unretract_feedrate", retract_feedrate))
    )
    if pressure_debt_prime > 0:
        retract = 0.0
        unretract, _, _ = pressure_debt_prime_amount(config, pressure_debt_prime)
    if previous_transition_was_tool_change:
        retract = 0.0
        unretract_feedrate = float(config.get("unretract_feedrate", retract_feedrate))
        if pressure_debt_prime <= 0:
            unretract = 0.0
    elif previous_transition_was_layer_change:
        retract = 0.0
        if pressure_debt_prime <= 0:
            unretract = 0.0

    if previous_transition_was_tool_change and pressure_debt_prime > 0:
        unretract_feedrate = float(config.get("unretract_feedrate", retract_feedrate))

    if previous_transition_was_layer_change and pressure_debt_prime > 0:
        unretract_feedrate = float(
            config.get("layer_change_unretract_feedrate", config.get("unretract_feedrate", retract_feedrate))
        )
    post_retract = float(config.get("balanced_wipe_tower_post_retract", retract))
    print_wall = snippet.material == int(layer_plan["wall_material"])

    if width <= 0:
        raise ValueError("balanced_wipe_tower_width must be > 0")
    if depth <= 0:
        raise ValueError("balanced_wipe_tower_depth must be > 0")
    if spacing <= 0:
        raise ValueError("balanced_wipe_tower_line_spacing must be > 0")
    if line_width <= 0:
        raise ValueError("balanced_wipe_tower_line_width must be > 0")
    if filament_diameter <= 0:
        raise ValueError("balanced_wipe_tower_filament_diameter must be > 0")

    total_lines = int(layer_plan["total_lines"])
    fill_full_depth = bool(config.get("balanced_wipe_tower_fill_full_depth", True))
    required_depth = (total_lines - 1) * spacing + line_width
    if not fill_full_depth and required_depth > depth + 1e-6:
        raise ValueError(
            "balanced wipe tower depth is too small for requested infill lines. "
            f"required_depth={format_float(required_depth)} depth={format_float(depth)}"
        )
    if fill_full_depth and depth < line_width:
        raise ValueError("balanced_wipe_tower_depth must be at least balanced_wipe_tower_line_width")

    safe_z = layer_change_lift_target_z(config, highest_print_z_so_far)
    z_hop = float(config.get("layer_change_z_hop", config.get("z_hop", 3.0)))
    if safe_z is None:
        safe_z = z + z_hop
    else:
        safe_z = max(safe_z, z + z_hop)

    inset = max(line_width / 2.0, 0.01)
    filament_area = math.pi * (filament_diameter / 2.0) ** 2
    infill_len = max(0.0, width - 2.0 * inset)
    line_e = infill_len * line_width * layer_height / filament_area * extrusion_scale
    wall_e_x = width * line_width * layer_height / filament_area * extrusion_scale
    wall_e_y = depth * line_width * layer_height / filament_area * extrusion_scale

    x0 = x
    x1 = x + width
    y0 = y
    y1 = y + depth
    if fill_full_depth:
        if total_lines == 1:
            infill_y_values = [y + depth / 2.0]
        else:
            infill_y_min = y + inset
            infill_y_max = y + depth - inset
            infill_y_values = [
                infill_y_min + (infill_y_max - infill_y_min) * index / (total_lines - 1)
                for index in range(total_lines)
            ]
    else:
        infill_y0 = y + (depth - ((total_lines - 1) * spacing)) / 2.0
        infill_y_values = [infill_y0 + index * spacing for index in range(total_lines)]
    alternate = bool(config.get("balanced_wipe_tower_alternate_direction", True))
    left_to_right = not alternate or snippet.main_index % 2 == 0

    lines = [
        f"; ASSEMBLED_BALANCED_WIPE_TOWER_START file={snippet.path.name} main={snippet.main_index} "
        f"material={snippet.material} tool=T{snippet.tool}",
        f"; wall_material={layer_plan['wall_material']} print_wall={int(print_wall)} "
        f"line_indexes={','.join(str(index) for index in line_indexes)}",
        f"; after_tool_change={int(previous_transition_was_tool_change)} "
        f"after_layer_change={int(previous_transition_was_layer_change)} "
        f"pre_travel_retract={format_float(retract)} "
        f"pre_print_unretract={format_float(unretract)}",
        f"; tower_bbox={format_xy(x0)},{format_xy(y0)},{format_xy(x1)},{format_xy(y1)} z={format_z(z)}",
        "M400",
    ]
    if retract > 0:
        lines.extend(build_extrusion_delta_lines(-retract, retract_feedrate, "balanced-wipe-tower pre-travel retract"))
    lines.extend(
        [
            "G90",
            f"G1 Z{format_z(safe_z)} F{format_float(z_feedrate)}",
            "; wait for clearance lift before XY travel",
            "M400",
            f"G1 X{format_xy(x0)} Y{format_xy(y0)} F{format_float(travel_feedrate)}",
            "; wait for XY travel before tower descent",
            "M400",
            f"G1 Z{format_z(z)} F{format_float(z_feedrate)}",
        ]
    )
    if unretract > 0:
        lines.extend(build_extrusion_delta_lines(unretract, unretract_feedrate, "balanced-wipe-tower pre-print unretract"))
        lines.extend(build_after_unretract_sync_lines(config))
    lines.append("M83")

    previous_print_xy: tuple[float, float] | None = None
    if print_wall:
        previous_print_xy = (x0, y1)
        lines.extend(
            [
                f"G1 X{format_xy(x1)} Y{format_xy(y0)} E{format_float(wall_e_x)} F{format_float(wall_feedrate)}",
                f"G1 X{format_xy(x1)} Y{format_xy(y1)} E{format_float(wall_e_y)} F{format_float(wall_feedrate)}",
                f"G1 X{format_xy(x0)} Y{format_xy(y1)} E{format_float(wall_e_x)} F{format_float(wall_feedrate)}",
                f"G1 X{format_xy(x0)} Y{format_xy(y0)} E{format_float(wall_e_y)} F{format_float(wall_feedrate)}",
            ]
        )

    current_x = x0
    current_y = y0
    for offset, line_index in enumerate(line_indexes):
        line_y = infill_y_values[line_index]
        start_x = x0 + inset if left_to_right else x1 - inset
        end_x = x1 - inset if left_to_right else x0 + inset
        if offset % 2 == 1:
            start_x, end_x = end_x, start_x
        if abs(current_x - start_x) > 1e-6 or abs(current_y - line_y) > 1e-6:
            lines.append(f"G1 X{format_xy(start_x)} Y{format_xy(line_y)} F{format_float(travel_feedrate)}")
        previous_print_xy = (start_x, line_y)
        lines.append(f"G1 X{format_xy(end_x)} E{format_float(line_e)} F{format_float(feedrate)}")
        current_x = end_x
        current_y = line_y

    if post_retract > 0 and bool(config.get("balanced_wipe_tower_post_pressure_wipe", False)):
        post_wipe_retract = min(
            post_retract,
            float(
                config.get(
                    "balanced_wipe_tower_post_pressure_wipe_retract",
                    config.get("layer_change_pressure_wipe_retract", 0.38),
                )
            ),
        )
        post_pressure_relief = min(
            max(0.0, post_retract - post_wipe_retract),
            float(
                config.get(
                    "balanced_wipe_tower_post_pressure_relief",
                    config.get("layer_change_pressure_relief", 0.02),
                )
            ),
        )
        post_wipe_distance = float(
            config.get(
                "balanced_wipe_tower_post_pressure_wipe_distance",
                config.get("layer_change_pressure_wipe_distance", 1.0),
            )
        )
        post_wipe_feedrate = float(
            config.get(
                "balanced_wipe_tower_post_pressure_wipe_feedrate",
                config.get("layer_change_pressure_wipe_feedrate", retract_feedrate),
            )
        )
        post_wipe_z_lift = float(
            config.get(
                "balanced_wipe_tower_post_pressure_wipe_z_lift",
                config.get("layer_change_pressure_wipe_z_lift", config.get("pressure_wipe_z_lift", 0.0)),
            )
        )
        post_wipe_z_feedrate = float(
            config.get(
                "balanced_wipe_tower_post_pressure_wipe_z_feedrate",
                config.get("layer_change_pressure_wipe_z_feedrate", config.get("pressure_wipe_z_feedrate", z_feedrate)),
            )
        )
        wipe_to_xy = tangential_wipe_target(
            from_x=current_x,
            from_y=current_y,
            previous_xy=previous_print_xy,
            distance=post_wipe_distance,
        )
        lines.append("; ASSEMBLED_BALANCED_WIPE_TOWER_POST_PRESSURE_WIPE")
        lines.extend(
            build_pressure_wipe_lines(
                wipe_to_xy=wipe_to_xy,
                wipe_retract=post_wipe_retract,
                pressure_relief=post_pressure_relief,
                feedrate=post_wipe_feedrate,
                z_lift=post_wipe_z_lift,
                z_feedrate=post_wipe_z_feedrate,
                retract_before_wipe=bool(
                    config.get(
                        "balanced_wipe_tower_post_pressure_wipe_retract_before_wipe",
                        config.get("pressure_wipe_retract_before_wipe", False),
                    )
                ),
            )
        )
        remaining_retract = max(0.0, post_retract - post_wipe_retract - post_pressure_relief)
        lines.extend(
            build_extrusion_delta_lines(
                -remaining_retract,
                retract_feedrate,
                "balanced-wipe-tower post-print remaining retract",
            )
        )
    else:
        lines.extend(build_extrusion_delta_lines(-post_retract, retract_feedrate, "balanced-wipe-tower post-print retract"))
    lines.extend(
        [
            "G90",
            f"G1 Z{format_z(safe_z)} F{format_float(z_feedrate)}",
            "M400",
            "; ASSEMBLED_BALANCED_WIPE_TOWER_END",
            "",
        ]
    )
    return "\n".join(lines)


def balanced_wipe_tower_post_retract_amount(
    config: dict[str, Any],
    previous_transition_was_tool_change: bool,
    previous_transition_was_layer_change: bool,
    pressure_debt_prime: float,
) -> float:
    if "balanced_wipe_tower_post_retract" in config:
        return float(config.get("balanced_wipe_tower_post_retract", 0.0))
    if pressure_debt_prime > 0:
        return 0.0
    if previous_transition_was_tool_change or previous_transition_was_layer_change:
        return 0.0
    return float(config.get("layer_change_retract", config.get("retract_length", 0.8)))


def balanced_wipe_tower_layer_height(
    config: dict[str, Any],
    state: dict[str, Any],
    main_index: int,
    z: float,
) -> float:
    layer_heights = state.setdefault("layer_heights", {})
    if main_index in layer_heights:
        return float(layer_heights[main_index])

    fallback = float(config.get("balanced_wipe_tower_effective_layer_height", 0.2))
    last_z = state.get("last_z")
    if last_z is not None and z > float(last_z) + 1e-6:
        layer_height = z - float(last_z)
    else:
        layer_height = fallback
    if layer_height <= 0:
        raise ValueError("balanced wipe tower layer height must be > 0")
    layer_heights[main_index] = layer_height
    state["last_z"] = z
    return layer_height


def summarize_balanced_wipe_tower_plan(plan: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if not plan:
        return None
    return {
        str(main_index): {
            "materials": layer_plan["materials"],
            "wall_material": layer_plan["wall_material"],
            "total_lines": layer_plan["total_lines"],
            "assignments": {
                str(material): line_indexes
                for material, line_indexes in layer_plan["assignments"].items()
            },
        }
        for main_index, layer_plan in sorted(plan.items())
    }


def build_simple_wipe_tower_block(
    config: dict[str, Any],
    snippet_text: str,
    snippet: Snippet,
    state: dict[str, Any],
) -> str:
    if not bool(config.get("enable_simple_wipe_tower", False)):
        return ""

    z = first_z_value(snippet_text)
    if z is None:
        return (
            f"; ASSEMBLED_SIMPLE_WIPE_TOWER_SKIPPED file={snippet.path.name} "
            "no_first_z_found\n"
        )

    change_idx = int(state.get("count", 0))
    state["count"] = change_idx + 1
    layer_counts = state.setdefault("layer_counts", {})
    layer_key = int(snippet.main_index)
    layer_idx = int(layer_counts.get(layer_key, 0))
    layer_counts[layer_key] = layer_idx + 1

    x_start = float(config.get("simple_wipe_tower_x_start", 250.0))
    x_end = float(config.get("simple_wipe_tower_x_end", 310.0))
    y_start = float(config.get("simple_wipe_tower_y_start", 40.0))
    row_spacing = float(config.get("simple_wipe_tower_row_spacing", 0.8))
    rows_per_change = int(config.get("simple_wipe_tower_rows_per_change", 3))
    z_hop = float(config.get("z_hop", 3.0))
    z_feedrate = float(config.get("z_hop_feedrate", 1200))
    travel_feedrate = float(config.get("simple_wipe_tower_travel_feedrate", config.get("approach_xy_feedrate", 30000)))
    line_width = float(config.get("simple_wipe_tower_line_width", 0.45))
    layer_height = float(config.get("simple_wipe_tower_effective_layer_height", 0.2))
    filament_diameter = float(config.get("simple_wipe_tower_filament_diameter", 1.75))
    max_speed = float(config.get("simple_wipe_tower_max_feedrate", 4800))
    direction_left_to_right = (snippet.main_index + layer_idx) % 2 == 0

    filament_area = math.pi * (filament_diameter / 2.0) ** 2
    line_len = abs(x_end - x_start)
    line_e = line_len * line_width * layer_height / filament_area
    step_e = row_spacing * line_width * layer_height / filament_area
    speed_map = [
        0.33 * max_speed,
        0.375 * max_speed,
        0.458 * max_speed,
        0.875 * max_speed,
        max_speed,
    ]

    y = y_start + layer_idx * rows_per_change * row_spacing
    start_x = x_start if direction_left_to_right else x_end
    end_x = x_end if direction_left_to_right else x_start

    lines = [
        f"; ASSEMBLED_SIMPLE_WIPE_TOWER_START file={snippet.path.name} material={snippet.material} tool=T{snippet.tool}",
        f"; tower_change_index={change_idx} tower_layer_row_index={layer_idx} main_index={snippet.main_index}",
        f"; tower x={format_xy(x_start)}..{format_xy(x_end)} y={format_xy(y)} rows={rows_per_change}",
        "G90",
        f"G1 Z{format_z(z + z_hop)} F{format_float(z_feedrate)}",
        f"G1 X{format_xy(start_x)} Y{format_xy(y)} F{format_float(travel_feedrate)}",
        f"G1 Z{format_z(z)} F{format_float(z_feedrate)}",
        "M83",
    ]

    current_x = start_x
    current_y = y
    for row in range(rows_per_change):
        next_x = end_x if current_x == start_x else start_x
        speed = speed_map[min(row, len(speed_map) - 1)]
        lines.append(f"G1 X{format_xy(next_x)} E{format_float(line_e)} F{format_float(speed)}")
        current_x = next_x
        if row != rows_per_change - 1:
            current_y += row_spacing
            lines.append(f"G1 Y{format_xy(current_y)} E{format_float(step_e)} F{format_float(speed)}")

    lines.extend(
        [
            "G91",
            f"G1 Z{format_float(z_hop)} F{format_float(z_feedrate)}",
            "G90",
            "M400",
            "; ASSEMBLED_SIMPLE_WIPE_TOWER_END",
            "",
        ]
    )
    return "\n".join(lines)


def tool_change_extrusion_delta(config: dict[str, Any], override_key: str, tool: int, fallback_key: str) -> float:
    overrides = config.get(override_key, {})
    if isinstance(overrides, dict):
        if str(tool) in overrides:
            return float(overrides[str(tool)])
        if tool in overrides:
            return float(overrides[tool])
    fallback = config.get(fallback_key, config.get("retract_length", 0.8))
    return float(fallback)


def build_tool_temperature_lines(config: dict[str, Any], old_tool: int, new_tool: int) -> list[str]:
    temperatures = {
        int(tool): float(temp)
        for tool, temp in config.get("tool_temperatures", {"0": 220, "1": 220}).items()
    }
    wait_for_temperature = bool(config.get("wait_tool_temperature_on_change", False))
    lines = ["; enforce tool temperatures after tool switch"]
    for tool in (new_tool, old_tool):
        if tool not in temperatures:
            continue
        command = "M109" if wait_for_temperature and tool == new_tool else "M104"
        if command == "M109":
            lines.append(f"{command} S{format_float(temperatures[tool])}")
        else:
            lines.append(f"{command} T{tool} S{format_float(temperatures[tool])} N0")
    return lines


def build_initial_temperature_block(config: dict[str, Any], first_tool: int) -> str:
    if not bool(config.get("set_initial_temperature_before_body", False)):
        return ""
    temperatures = {
        int(tool): float(temp)
        for tool, temp in config.get("tool_temperatures", {"0": 220, "1": 220}).items()
    }
    if first_tool not in temperatures:
        return ""

    wait_for_first_tool = bool(config.get("wait_initial_tool_temperature", True))
    lines = ["; enforce initial tool temperatures before assembled print body"]
    for tool in sorted(temperatures):
        lines.append(f"M104 T{tool} S{format_float(temperatures[tool])} N0")
    if wait_for_first_tool:
        lines.extend(
            [
                f"T{first_tool}",
                f"M109 S{format_float(temperatures[first_tool])}",
            ]
        )
    return "\n".join(lines)


def build_secondary_nozzle_load_line_block(config: dict[str, Any], first_tool: int, used_tools: list[int]) -> str:
    if not bool(config.get("draw_secondary_nozzle_load_line_before_body", False)):
        return ""
    initialize_single_tool_secondary = bool(config.get("draw_secondary_nozzle_load_line_for_single_tool", False))
    if len(set(used_tools)) < 2 and not initialize_single_tool_secondary:
        return ""

    tools = sorted(
        int(tool)
        for tool in config.get("tool_temperatures", {"0": 220, "1": 220})
        if int(tool) in (0, 1) and (initialize_single_tool_secondary or int(tool) in set(used_tools))
    )
    secondary_tools = [tool for tool in tools if tool != first_tool]
    if not secondary_tools:
        return ""

    temperatures = {
        int(tool): float(temp)
        for tool, temp in config.get("tool_temperatures", {"0": 220, "1": 220}).items()
    }
    x_start = float(config.get("secondary_nozzle_load_line_x_start", 250.0))
    x_end = float(config.get("secondary_nozzle_load_line_x_end", 290.0))
    y = float(config.get("secondary_nozzle_load_line_y", 2.0))
    z = float(config.get("secondary_nozzle_load_line_z", 5.0))
    pre_extrude_z_lift = float(config.get("secondary_nozzle_load_line_pre_extrude_z_lift", 0.8))
    end_z_lift = float(config.get("secondary_nozzle_load_line_end_z_lift", 0.4))
    return_switch_z_lift = float(config.get("secondary_nozzle_load_line_return_switch_z_lift", 3.0))
    travel_feedrate = float(config.get("secondary_nozzle_load_line_travel_feedrate", 60000))
    z_feedrate = float(config.get("z_hop_feedrate", 1200))
    extrude_before_line = float(config.get("secondary_nozzle_load_line_intro_extrude", 5.0))
    extrude_on_line = float(config.get("secondary_nozzle_load_line_extrude", 10.0))
    extrude_feedrate = float(config.get("secondary_nozzle_load_line_extrude_feedrate", 623.623))

    lines = ["; assembled secondary nozzle load line start"]
    for tool in secondary_tools:
        temperature = temperatures.get(tool)
        lines.extend(
            [
                f"; load line for secondary tool T{tool}",
                "M400",
                "G90",
                f"G1 Z{format_float(z)} F{format_float(z_feedrate)}",
                f"M620 S{tool}A",
                "M400",
                f"T{tool}",
                "M400",
                f"M621 S{tool}A",
            ]
        )
        if temperature is not None:
            lines.extend(
                [
                    f"M104 T{tool} S{format_float(temperature)} N0",
                    f"M109 S{format_float(temperature)}",
                ]
            )
        lines.extend(
            [
                "G29.2 S1 ; ensure z comp turn on",
                "G90",
                "M83",
                f"G1 Z{format_float(z)} F{format_float(z_feedrate)}",
                f"G1 X270 Y{format_float(y)} F{format_float(travel_feedrate)}",
                "G28.14 R0",
                "G29.2 S0",
                "G91",
                f"G1 Z{format_float(pre_extrude_z_lift)} F{format_float(z_feedrate)}",
                "G90",
                f"G1 X{format_float(x_start)} F{format_float(travel_feedrate)}",
                f"G1 E{format_float(extrude_before_line)} F{format_float(extrude_feedrate)}",
                f"G1 X{format_float(x_end)} E{format_float(extrude_on_line)} F{format_float(extrude_feedrate)}",
                "G91",
                f"G3 Z{format_float(end_z_lift)} I1.217 J0 P1 F{format_float(travel_feedrate)}",
                "G90",
                "M83",
                "G29.2 S1 ; ensure z comp turn on",
            ]
        )

    lines.extend(
        [
            f"; return to starting tool T{first_tool}",
            "G91",
            f"G1 Z{format_float(return_switch_z_lift)} F{format_float(z_feedrate)}",
            "G90",
            f"M620 S{first_tool}A",
            "M400",
            f"T{first_tool}",
            "M400",
            f"M621 S{first_tool}A",
        ]
    )
    first_temperature = temperatures.get(first_tool)
    if first_temperature is not None:
        lines.append(f"M104 T{first_tool} S{format_float(first_temperature)} N0")
    lines.append("; assembled secondary nozzle load line end")
    return "\n".join(lines)


def build_layer_change_lift_block(
    config: dict[str, Any],
    tool: int,
    material: int,
    old_main_index: int,
    new_main_index: int,
    next_file: str,
    lift_target_z: float | None = None,
    motion_state: MotionState | None = None,
) -> tuple[str, float]:
    z_hop = float(config.get("layer_change_z_hop", config.get("z_hop", 3.0)))
    z_feedrate = float(config.get("layer_change_z_hop_feedrate", config.get("z_hop_feedrate", 1200)))
    retract = float(config.get("layer_change_retract", config.get("retract_length", 0.8)))
    retract_feedrate = float(config.get("layer_change_retract_feedrate", config.get("retract_feedrate", 1800)))
    unretract = float(config.get("layer_change_unretract", retract))
    unretract_feedrate = float(
        config.get("layer_change_unretract_feedrate", config.get("unretract_feedrate", retract_feedrate))
    )
    lower_after_layer_change = bool(config.get("lower_after_layer_change", False))
    approach_first_xy = bool(config.get("approach_first_xy_while_lifted", True))

    lines = [
        f"; ASSEMBLED_LAYER_CHANGE from_main={old_main_index} to_main={new_main_index} "
        f"material={material} tool=T{tool} next_file={next_file}",
        "M400",
    ]
    pressure_wipe_retract = 0.0
    pressure_relief = 0.0
    if bool(config.get("layer_change_pressure_wipe", config.get("long_travel_pressure_wipe_only", False))):
        pressure_wipe_retract = float(
            config.get("layer_change_pressure_wipe_retract", config.get("long_travel_pressure_wipe_retract", 0.38))
        )
        pressure_relief = float(
            config.get("layer_change_pressure_relief", config.get("long_travel_pressure_relief", 0.02))
        )
        pressure_wipe_feedrate = float(
            config.get("layer_change_pressure_wipe_feedrate", config.get("long_travel_pressure_wipe_feedrate", retract_feedrate))
        )
        pressure_wipe_z_lift = float(
            config.get(
                "layer_change_pressure_wipe_z_lift",
                config.get("long_travel_pressure_wipe_z_lift", config.get("pressure_wipe_z_lift", 0.0)),
            )
        )
        pressure_wipe_z_feedrate = float(
            config.get(
                "layer_change_pressure_wipe_z_feedrate",
                config.get("long_travel_pressure_wipe_z_feedrate", config.get("pressure_wipe_z_feedrate", z_feedrate)),
            )
        )
        wipe_distance = float(
            config.get("layer_change_pressure_wipe_distance", config.get("long_travel_pressure_wipe_distance", 1.0))
        )
        wipe_to_xy = None
        if motion_state is not None and has_current_print_edge(motion_state):
            wipe_path = backward_wipe_path(
                from_x=motion_state.x,
                from_y=motion_state.y,
                from_z=motion_state.z,
                print_path=motion_state.print_path,
                distance=wipe_distance,
            )
            wipe_to_xy = tangential_wipe_target(
                from_x=motion_state.x,
                from_y=motion_state.y,
                previous_xy=motion_state.previous_xy,
                distance=wipe_distance,
            )
        else:
            wipe_path = []
        if wipe_path:
            pressure_wipe_acceleration = config.get("layer_change_pressure_wipe_acceleration")
            if pressure_wipe_acceleration is not None:
                lines.extend(
                    [
                        f"; ASSEMBLED_LAYER_CHANGE_PRESSURE_WIPE_ACCELERATION S{format_float(float(pressure_wipe_acceleration))}",
                        f"M204 S{format_float(float(pressure_wipe_acceleration))}",
                    ]
                )
            lines.extend(
                [
                    f"; ASSEMBLED_LAYER_CHANGE_PRESSURE_WIPE tool=T{tool}",
                    *build_pressure_wipe_lines(
                        wipe_to_xy=wipe_to_xy,
                        wipe_retract=pressure_wipe_retract,
                        pressure_relief=pressure_relief,
                        feedrate=pressure_wipe_feedrate,
                        wipe_path=wipe_path,
                        z_lift=pressure_wipe_z_lift,
                        z_feedrate=pressure_wipe_z_feedrate,
                        retract_before_wipe=bool(
                            config.get("layer_change_pressure_wipe_retract_before_wipe", config.get("pressure_wipe_retract_before_wipe", False))
                        ),
                        follow_wipe_z=bool(config.get("backward_wipe_follow_z", False)),
                        wipe_z_offset=float(config.get("backward_wipe_z_offset", 0.0)),
                    ),
                ]
            )
            motion_state.x, motion_state.y, motion_state.z = wipe_path[-1]
            motion_state.print_path.clear()
            if pressure_wipe_acceleration is not None:
                travel_acceleration = config.get(
                    "layer_change_travel_acceleration", config.get("long_travel_acceleration")
                )
                if travel_acceleration is not None:
                    lines.extend(
                        [
                            f"; ASSEMBLED_LAYER_CHANGE_TRAVEL_ACCELERATION S{format_float(float(travel_acceleration))}",
                            f"M204 S{format_float(float(travel_acceleration))}",
                        ]
                    )
        else:
            pressure_wipe_retract = 0.0
            pressure_relief = 0.0
    if bool(config.get("layer_change_pressure_wipe_reduces_retract", True)):
        retract = max(0.0, retract - pressure_wipe_retract - pressure_relief)
    total_withdrawal = pressure_wipe_retract + pressure_relief + retract
    lines.extend(build_extrusion_delta_lines(-retract, retract_feedrate, "layer-change retract"))
    if z_hop > 0:
        if lift_target_z is not None:
            lines.extend(
                [
                    f"; layer-change lift to highest_print_z_so_far + hop = Z{format_z(lift_target_z)}",
                    "G90",
                    f"G1 Z{format_z(lift_target_z)} F{format_float(z_feedrate)}",
                    "; wait for clearance lift before XY travel",
                    "M400",
                ]
            )
        else:
            lines.extend(
                [
                    "G91",
                    f"G1 Z{format_float(z_hop)} F{format_float(z_feedrate)}",
                    "G90",
                    "; wait for clearance lift before XY travel",
                    "M400",
                ]
            )
    if z_hop > 0 and lower_after_layer_change:
        lines.extend(
            [
                "G91",
                f"G1 Z-{format_float(z_hop)} F{format_float(z_feedrate)}",
                "G90",
            ]
        )
    if not approach_first_xy:
        lines.extend(build_extrusion_delta_lines(unretract, unretract_feedrate, "layer-change unretract"))
        lines.extend(build_after_unretract_sync_lines(config))
    lines.append("")
    return "\n".join(lines), total_withdrawal


def build_approach_block(
    config: dict[str, Any],
    snippet_text: str,
    snippet: Snippet,
    reason: str,
    pressure_debt_prime: float | None = None,
) -> str:
    first_xy = first_xy_value(snippet_text)
    if first_xy is None:
        return (
            f"; ASSEMBLED_APPROACH_SKIPPED reason={reason} file={snippet.path.name} "
            "no_first_xy_found\n"
        )

    x, y = first_xy
    feedrate = float(config.get("approach_xy_feedrate", 30000))
    prime = float(config.get("approach_pre_extrude", 0.0))
    prime_feedrate = float(config.get("approach_pre_extrude_feedrate", config.get("unretract_feedrate", 1800)))
    if pressure_debt_prime is not None:
        adjusted_prime, _, _ = pressure_debt_prime_amount(config, pressure_debt_prime)
        prime += adjusted_prime
    elif reason == "tool_change":
        if not (
            bool(config.get("enable_simple_wipe_tower", False))
            and bool(config.get("simple_wipe_tower_satisfies_unretract", True))
        ):
            prime += tool_change_extrusion_delta(
                config,
                "tool_change_unretract_by_tool",
                snippet.tool,
                "tool_change_unretract",
            )
    elif reason == "layer_change":
        prime += float(config.get("layer_change_unretract", config.get("retract_length", 0.8)))
    elif reason == "wipe_tower":
        prime = float(config.get("wipe_tower_approach_pre_extrude", config.get("approach_pre_extrude", 0.0)))
        prime += float(
            config.get(
                "wipe_tower_approach_unretract",
                config.get("layer_change_unretract", config.get("retract_length", 0.8)),
            )
        )

    lines = [
        f"; ASSEMBLED_APPROACH reason={reason} file={snippet.path.name} material={snippet.material} tool=T{snippet.tool}",
        "G90",
        f"G1 X{format_xy(x)} Y{format_xy(y)} F{format_float(feedrate)}",
        "; wait for XY travel before any descent in the snippet",
        "M400",
    ]
    if config.get("approach_descend_before_unretract", False):
        first_z = first_z_value(snippet_text)
        if first_z is None:
            raise ValueError(f"Approach requires a first Z position: {snippet.path.name}")
        z_feed = float(config.get("approach_z_feedrate", 1200.0))
        if not math.isfinite(z_feed) or z_feed <= 0:
            raise ValueError("approach_z_feedrate must be finite and positive")
        lines.extend(["; descend to the first path height before restoring pressure",
                      f"G1 Z{format_z(first_z)} F{format_float(z_feed)}", "M400"])
    if prime != 0:
        label = "approach pressure-debt unretract" if pressure_debt_prime else "approach pre-extrude"
        lines.extend(build_extrusion_delta_lines(prime, prime_feedrate, label))
        lines.extend(build_after_unretract_sync_lines(config))
    lines.append("")
    return "\n".join(lines)


def add_pressure_debt(debts: dict[int, float], tool: int, amount: float) -> None:
    if amount <= 0:
        return
    debts[tool] = debts.get(tool, 0.0) + amount


def get_pressure_debt(debts: dict[int, float], tool: int) -> float:
    return max(0.0, debts.get(tool, 0.0))


def consume_pressure_debt(debts: dict[int, float], tool: int, amount: float | None = None) -> float:
    owed = debts.get(tool, 0.0)
    if owed <= 0:
        debts.pop(tool, None)
        return 0.0
    payment = owed if amount is None else min(owed, max(0.0, amount))
    remaining = owed - payment
    if remaining > 1e-6:
        debts[tool] = remaining
    else:
        debts.pop(tool, None)
    return payment


def pressure_debt_prime_amount(config: dict[str, Any], debt: float) -> tuple[float, float, float]:
    factor = float(config.get("pressure_debt_prime_factor", 1.0))
    offset = float(config.get("pressure_debt_prime_offset", 0.0))
    restart_extra = float(config.get("pressure_debt_restart_extra", 0.0))
    return max(0.0, debt * factor + offset) + restart_extra, factor, offset


def build_pressure_debt_unretract_block(config: dict[str, Any], snippet: Snippet, amount: float) -> str:
    feedrate = float(config.get("approach_pre_extrude_feedrate", config.get("unretract_feedrate", 1800)))
    restart_extra = float(config.get("pressure_debt_restart_extra", 0.0))
    prime, factor, offset = pressure_debt_prime_amount(config, amount)
    lines = [
        f"; ASSEMBLED_PRESSURE_DEBT_UNRETRACT file={snippet.path.name} material={snippet.material} "
        f"tool=T{snippet.tool} E{format_float(prime)} debt={format_float(amount)} "
        f"prime_factor={format_float(factor)} prime_offset={format_float(offset)} "
        f"restart_extra={format_float(restart_extra)}",
    ]
    lines.extend(build_extrusion_delta_lines(prime, feedrate, "pressure-debt unretract before print"))
    lines.extend(build_after_unretract_sync_lines(config))
    return "\n".join(lines)


def first_xy_value(text: str) -> tuple[float, float] | None:
    for line in iter_gcode_lines(text):
        if not line.lstrip().startswith(("G0", "G1")):
            continue
        x_match = X_WORD_RE.search(line)
        y_match = Y_WORD_RE.search(line)
        if x_match and y_match:
            return float(x_match.group(1)), float(y_match.group(1))
    return None


def build_extrusion_delta_lines(distance: float, feedrate: float, label: str) -> list[str]:
    if distance == 0:
        return []
    return [
        f"; {label}",
        "M83",
        f"G1 E{format_float(distance)} F{format_float(feedrate)}",
    ]


def build_after_unretract_sync_lines(config: dict[str, Any]) -> list[str]:
    if not config.get("m400_after_unretract", True):
        return []
    wait_ms = int(config.get("wait_after_unretract_ms", 0))
    if wait_ms > 0:
        return [f"M400 P{wait_ms}"]
    return ["M400"]


def compute_layer_ranges(snippets: list[Snippet]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start_layer = snippets[0].main_index
    current_layer = snippets[0].main_index
    current_material = snippets[0].material
    current_tool = snippets[0].tool

    for snippet in snippets[1:]:
        if snippet.material != current_material:
            ranges.append(
                {
                    "material": current_material,
                    "tool": current_tool,
                    "start_main_index": start_layer,
                    "end_main_index": current_layer,
                }
            )
            start_layer = snippet.main_index
            current_material = snippet.material
            current_tool = snippet.tool
        current_layer = snippet.main_index

    ranges.append(
        {
            "material": current_material,
            "tool": current_tool,
            "start_main_index": start_layer,
            "end_main_index": current_layer,
        }
    )
    return ranges


def compute_layer_fraction_by_main(snippets: list[Snippet]) -> dict[int, float]:
    main_indices = sorted({snippet.main_index for snippet in snippets})
    if not main_indices:
        return {}
    if len(main_indices) == 1:
        return {main_indices[0]: 0.0}
    denominator = len(main_indices) - 1
    return {main_index: index / denominator for index, main_index in enumerate(main_indices)}


def read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def trim_outer_newlines(text: str) -> str:
    return text.strip("\r\n")


def trim_trailing_newlines(text: str) -> str:
    return text.rstrip("\r\n")


def trim_leading_newlines(text: str) -> str:
    return text.lstrip("\r\n")


def format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def format_z(value: float) -> str:
    return f"{value:.5f}"


def format_xy(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    main()

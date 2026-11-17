#!/usr/bin/env python3
"""Assemble single-material bare layer-path snippets into an A1 G-code file.

This is deliberately a thin A1 wrapper around the established H2D path
assembler.  The motion rewriting code is shared, but the A1 contract is much
smaller: exactly one material (0) and one physical tool (T0).  An A1 AMS
material change is a load, cut, purge, and wipe operation, so it must not be
treated as an H2D-like nozzle switch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
H2D_HELPERS = HERE.parent / "common"
sys.path.insert(0, str(H2D_HELPERS))

import assemble_layer_paths as core  # noqa: E402


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config["progress_reporting"] = True
    snippets = core.discover_snippets(Path(config["input_dir"]), config)
    if not snippets:
        raise SystemExit("No snippets found. Expected files like 0_0.gcode or 0_0_0.gcode.")

    invalid = [snippet for snippet in snippets if snippet.material != 0 or snippet.tool != 0]
    if invalid:
        names = ", ".join(item.path.name for item in invalid[:5])
        raise ValueError(
            "A1 v1 only accepts material 0 on T0. "
            f"Found unsupported material/tool assignment in: {names}"
        )

    input_mebibytes = sum(item.path.stat().st_size for item in snippets) / (1024 * 1024)
    print(
        f"Assembling {len(snippets)} snippets ({input_mebibytes:.1f} MiB input). "
        "Long-travel planning is enabled."
    )
    output_gcode = Path(config["output_gcode"])
    output_gcode.parent.mkdir(parents=True, exist_ok=True)
    assembled, summary = core.assemble_gcode(config, snippets)
    print("[assembler] Writing assembled G-code", flush=True)
    output_gcode.write_text(assembled, encoding=config.get("encoding", "utf-8"), newline="\n")
    del assembled
    cleanup = dict(config.get("motion_cleanup", {}))
    cleanup.setdefault('travel_z_speed_limit', config.get('travel_z_speed_limit'))
    if cleanup.get("enabled", False):
        from motion_cleanup import clean_file_in_place
        print("[assembler] Cleaning tiny motion/extrusion commands", flush=True)
        summary["motion_cleanup"] = clean_file_in_place(output_gcode, cleanup)

    summary_path = Path(config.get("summary_json", output_gcode.with_suffix(".summary.json")))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    print("[assembler] Writing assembly summary", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {output_gcode}")
    print(f"Wrote {summary_path}")
    print(f"Snippets: {summary['snippet_count']} (skipped {len(summary.get('skipped_empty_snippets', []))} empty)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON config path.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = core.load_config(path)
    config = dict(config)
    config["material_to_tool"] = {"0": 0}
    config["single_material_default_material"] = 0
    config["allow_missing_material_suffix"] = True

    # Never inject H2D-specific dual-nozzle start commands into an A1 job.
    config["draw_secondary_nozzle_load_line_before_body"] = False
    config["set_initial_temperature_before_body"] = False
    config["rewrite_pre_code_start_material"] = False
    configure_a1_pre_code_skips(config)
    return config


def configure_a1_pre_code_skips(config: dict[str, Any]) -> None:
    """Teach the shared remover the precise banners emitted by the A1 sample."""
    pre_code = Path(config["pre_code"]).read_text(encoding=config.get("encoding", "utf-8"))
    marker_prefixes = {
        "auto_extrude_cali": (
            ";===== auto extrude cali start",
            ";===== auto extrude cali end",
        ),
        "bed_leveling": (
            ";===== bed leveling",
            ";===== bed leveling end",
        ),
    }
    for section_name, (start_marker, end_marker, config_key) in core.PRE_CODE_SECTIONS.items():
        if not config.get(config_key, False):
            continue
        if start_marker in pre_code and end_marker in pre_code:
            continue
        prefixes = marker_prefixes.get(section_name)
        if prefixes:
            start_line = find_marker_line(pre_code, prefixes[0])
            end_line = find_marker_line(pre_code, prefixes[1])
            if start_line and end_line:
                # The shared remover compares exact, stripped individual lines.
                # The escaped template text lives inside a longer config line,
                # so it cannot match this standalone banner.
                core.PRE_CODE_SECTIONS[section_name] = (
                    start_line,
                    end_line,
                    config_key,
                )
                continue
        config[config_key] = False
        print(f"A1 reference lacks identifiable {section_name} markers; leaving it enabled.")


def find_marker_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return None


if __name__ == "__main__":
    raise SystemExit(main())

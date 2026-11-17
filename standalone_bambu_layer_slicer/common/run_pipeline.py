#!/usr/bin/env python3
"""Run assemble -> extracted 3MF folder -> pack from one JSON file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    assemble_config = dict(config.get("assemble", {}))
    three_mf_config = dict(config.get("three_mf", {}))
    pack_config = dict(config.get("pack", {}))

    if config.get("run_assemble", True):
        assemble_config_path = write_assemble_config(base_dir, assemble_config, config)
        run([sys.executable, str(HERE / "assemble_layer_paths.py"), "--config", str(assemble_config_path)])

    if config.get("run_three_mf", True):
        run(make_three_mf_command(base_dir, three_mf_config, assemble_config))

    if config.get("run_pack", True):
        output_folder = resolve(base_dir, three_mf_config.get("output", "GeneratedPackedJob"))
        output_archive = resolve(base_dir, pack_config.get("output", f"{output_folder}.gcode.3mf"))
        run([sys.executable, str(HERE / "pack_3mf_folder.py"), str(output_folder), str(output_archive)])

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline JSON config.")
    return parser.parse_args()


def write_assemble_config(base_dir: Path, assemble_config: dict[str, Any], config: dict[str, Any]) -> Path:
    required = ["input_dir", "pre_code", "post_code", "output_gcode"]
    missing = [key for key in required if key not in assemble_config]
    if missing:
        raise ValueError(f"assemble config missing required fields: {', '.join(missing)}")

    resolved = dict(assemble_config)
    for key in ["input_dir", "pre_code", "post_code", "output_gcode", "summary_json"]:
        if key in resolved:
            resolved[key] = str(resolve(base_dir, resolved[key]))

    path = resolve(base_dir, config.get("generated_assemble_config", ".generated_assemble_config.json"))
    path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return path


def make_three_mf_command(
    base_dir: Path,
    three_mf_config: dict[str, Any],
    assemble_config: dict[str, Any],
) -> list[str]:
    required = ["template", "obj", "output", "center"]
    missing = [key for key in required if key not in three_mf_config]
    if missing:
        raise ValueError(f"three_mf config missing required fields: {', '.join(missing)}")

    gcode = three_mf_config.get("gcode", assemble_config.get("output_gcode"))
    if not gcode:
        raise ValueError("three_mf.gcode is missing and assemble.output_gcode is not set")

    command = [
        sys.executable,
        str(HERE / "make_h2d_3mf_folder.py"),
        "--template",
        str(resolve(base_dir, three_mf_config["template"])),
        "--obj",
        str(resolve(base_dir, three_mf_config["obj"])),
        "--gcode",
        str(resolve(base_dir, gcode)),
        "--output",
        str(resolve(base_dir, three_mf_config["output"])),
        "--center",
        str(three_mf_config["center"]),
    ]

    optional_pairs = {
        "--obj-up": "obj_up",
        "--material-layout": "material_layout",
        "--object-name": "object_name",
        "--first-filament": "first_filament",
        "--plate": "plate",
        "--first-layer-count": "first_layer_count",
        "--first-layer-feedrate": "first_layer_feedrate",
        "--remaining-layer-feedrate": "remaining_layer_feedrate",
        "--first-layer-feedrate-multiplier": "first_layer_feedrate_multiplier",
        "--remaining-layer-feedrate-multiplier": "remaining_layer_feedrate_multiplier",
        "--platform-margin": "platform_margin",
        "--platform-line-spacing": "platform_line_spacing",
        "--platform-line-width": "platform_line_width",
        "--platform-layer-height": "platform_layer_height",
        "--platform-feedrate": "platform_feedrate",
        "--platform-travel-feedrate": "platform_travel_feedrate",
        "--platform-extrusion-scale": "platform_extrusion_scale",
        "--bed-leveling-margin": "bed_leveling_margin",
    }
    for flag, key in optional_pairs.items():
        if key in three_mf_config and three_mf_config[key] is not None:
            command.extend([flag, str(three_mf_config[key])])

    if three_mf_config.get("minimal_template", False):
        command.append("--minimal-template")
    if three_mf_config.get("blank_pngs", True):
        command.append("--blank-pngs")
    if three_mf_config.get("gcode_preview_mode", False):
        command.append("--gcode-preview-mode")
    if three_mf_config.get("preview_spiral_mode", False):
        command.append("--preview-spiral-mode")
    if three_mf_config.get("no_wipe_tower_metadata", False):
        command.append("--no-wipe-tower-metadata")
    if three_mf_config.get("register_all_layout_filaments", False):
        command.append("--register-all-layout-filaments")
    if three_mf_config.get("skip_auto_extrude_cali", False):
        command.append("--skip-auto-extrude-cali")
    if three_mf_config.get("skip_bed_leveling", False):
        command.append("--skip-bed-leveling")
    if three_mf_config.get("skip_z_offset_cali", False):
        command.append("--skip-z-offset-cali")
    if three_mf_config.get("skip_xy_offset_cali", False):
        command.append("--skip-xy-offset-cali")
    if three_mf_config.get("enable_planar_platform", False):
        command.append("--enable-planar-platform")
    if three_mf_config.get("platform_lift_model_z", False):
        command.append("--platform-lift-model-z")
    if three_mf_config.get("overwrite", True):
        command.append("--overwrite")

    return command


def resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def run(command: list[str]) -> None:
    print(" ".join(quote(part) for part in command))
    subprocess.run(command, check=True)


def quote(value: str) -> str:
    if " " in value:
        return f'"{value}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())

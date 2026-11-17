#!/usr/bin/env python3
"""Run the normal pipeline with long-travel pressure wipe instead of retract/unretract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_pipeline import HERE, make_three_mf_command, quote, resolve, write_assemble_config


DEFAULT_OVERRIDES = {
    "long_travel_pressure_wipe_only": True,
    "long_travel_pressure_wipe_retract": 0.38,
    "long_travel_pressure_relief": 0.02,
    "long_travel_pressure_wipe_distance": 1.0,
    "long_travel_pressure_wipe_feedrate": 8000,
    "wipe_tower_approach_pre_extrude": 0.0,
    "wipe_tower_approach_unretract": 0.0,
    "long_travel_retract": 0.0,
    "long_travel_unretract": 0.0,
}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    assemble_config = dict(config.get("assemble", {}))
    apply_missing_defaults(assemble_config, DEFAULT_OVERRIDES)

    three_mf_config = dict(config.get("three_mf", {}))
    pack_config = dict(config.get("pack", {}))
    if args.output_suffix:
        apply_output_suffix(assemble_config, three_mf_config, pack_config, config, args.output_suffix)

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


def apply_missing_defaults(config: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        config.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Pipeline JSON config.")
    parser.add_argument(
        "--output-suffix",
        default="_pressure_wipe",
        help="Suffix for generated assemble G-code/summary files. Use an empty string to keep config outputs.",
    )
    return parser.parse_args()


def apply_output_suffix(
    assemble_config: dict[str, Any],
    three_mf_config: dict[str, Any],
    pack_config: dict[str, Any],
    config: dict[str, Any],
    suffix: str,
) -> None:
    for key in ("output_gcode", "summary_json"):
        value = assemble_config.get(key)
        if value:
            path = Path(value)
            assemble_config[key] = str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))
    if assemble_config.get("output_gcode"):
        three_mf_config["gcode"] = assemble_config["output_gcode"]
    for target in ((three_mf_config, "output"), (pack_config, "output")):
        section, key = target
        value = section.get(key)
        if value:
            path = Path(value)
            section[key] = str(with_output_suffix(path, suffix))
    generated = config.get("generated_assemble_config")
    if generated:
        path = Path(generated)
        config["generated_assemble_config"] = str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def with_output_suffix(path: Path, suffix: str) -> Path:
    if path.name.endswith(".gcode.3mf"):
        base = path.name[: -len(".gcode.3mf")]
        return path.with_name(f"{base}{suffix}.gcode.3mf")
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def run(command: list[str]) -> None:
    print(" ".join(quote(part) for part in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())

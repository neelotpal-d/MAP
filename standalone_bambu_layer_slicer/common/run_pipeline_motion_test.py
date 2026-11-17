#!/usr/bin/env python3
"""Run a low-acceleration, constant-feedrate motion diagnostic pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_pipeline import HERE, make_three_mf_command, quote, resolve, write_assemble_config
from run_pipeline_pressure_wipe import DEFAULT_OVERRIDES as PRESSURE_WIPE_OVERRIDES, apply_missing_defaults


MOTION_TEST_OVERRIDES = {
    "first_layer_feedrate": 3000,
    "remaining_layer_feedrate": 3000,
    "first_layer_feedrate_multiplier": None,
    "remaining_layer_feedrate_multiplier": None,
    "enable_slope_feedrate_limit": False,
    "enable_raster_after_zigzag_feedrate_limit": False,
    "enable_raster_after_fraction_feedrate_limit": False,
    "model_acceleration": 500,
    "first_layer_acceleration": 500,
    "remaining_layer_acceleration": 500,
    "enable_z_smoothing": False,
}

FEATURE_MOTION_KEYS = (
    "untagged_role",
    "layer_override_roles",
    "role_feedrates",
    "role_accelerations",
    "bottom_layer_count",
    "sparse_infill_jump_requires_tag",
)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    assemble_config = dict(config.get("assemble", {}))
    motion_config = dict(config.get("motion_test", {}))
    apply_missing_defaults(assemble_config, PRESSURE_WIPE_OVERRIDES)
    assemble_config.update(MOTION_TEST_OVERRIDES)
    apply_motion_config(assemble_config, motion_config)
    apply_motion_cli_overrides(assemble_config, args)

    three_mf_config = dict(config.get("three_mf", {}))
    pack_config = dict(config.get("pack", {}))
    output_suffix = args.output_suffix
    if output_suffix is None:
        output_suffix = str(motion_config.get("output_suffix", "_motion_test"))
    apply_output_suffix(assemble_config, three_mf_config, pack_config, config, output_suffix)

    if config.get("run_assemble", True):
        assemble_config_path = write_assemble_config(base_dir, assemble_config, config)
        run([sys.executable, str(HERE / "assemble_layer_paths.py"), "--config", str(assemble_config_path)])

    if args.assemble_only:
        return 0

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
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix for generated assemble G-code/summary/folder/archive outputs.",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Only write the assembled G-code and summary JSON.",
    )
    parser.add_argument(
        "--feedrate",
        type=float,
        default=None,
        help="Override motion_test.feedrate.",
    )
    parser.add_argument(
        "--acceleration",
        type=float,
        default=None,
        help="Override motion_test.acceleration.",
    )
    parser.add_argument(
        "--enable-z-smoothing",
        action="store_true",
        help="Override motion_test.enable_z_smoothing to true.",
    )
    parser.add_argument(
        "--disable-z-smoothing",
        action="store_true",
        help="Override motion_test.enable_z_smoothing to false.",
    )
    parser.add_argument(
        "--z-smoothing-target-feedrate",
        type=float,
        default=None,
        help="Override motion_test.z_smoothing_target_feedrate.",
    )
    parser.add_argument("--z-smoothing-window", type=int, default=None, help="Override motion_test.z_smoothing_window.")
    parser.add_argument("--z-smoothing-passes", type=int, default=None, help="Override motion_test.z_smoothing_passes.")
    parser.add_argument("--z-smoothing-strength", type=float, default=None, help="Override motion_test.z_smoothing_strength.")
    parser.add_argument("--z-smoothing-max-delta", type=float, default=None, help="Override motion_test.z_smoothing_max_delta.")
    return parser.parse_args()


def apply_motion_config(assemble_config: dict[str, Any], motion_config: dict[str, Any]) -> None:
    feedrate = float(motion_config.get("feedrate", 3000))
    acceleration = float(motion_config.get("acceleration", 500))
    assemble_config.update(
        {
            "first_layer_feedrate": feedrate,
            "remaining_layer_feedrate": feedrate,
            "model_acceleration": acceleration,
            "first_layer_acceleration": acceleration,
            "remaining_layer_acceleration": acceleration,
            "enable_z_smoothing": bool(motion_config.get("enable_z_smoothing", False)),
            "z_smoothing_target_feedrate": motion_config.get("z_smoothing_target_feedrate", feedrate),
            "z_smoothing_window": int(motion_config.get("z_smoothing_window", 5)),
            "z_smoothing_passes": int(motion_config.get("z_smoothing_passes", 1)),
            "z_smoothing_strength": float(motion_config.get("z_smoothing_strength", 1.0)),
            "z_smoothing_max_delta": float(motion_config.get("z_smoothing_max_delta", 0.05)),
            "z_smoothing_max_direction_change_degrees": float(
                motion_config.get("z_smoothing_max_direction_change_degrees", 30.0)
            ),
        }
    )
    for key in FEATURE_MOTION_KEYS:
        if key in motion_config:
            assemble_config[key] = motion_config[key]


def apply_motion_cli_overrides(assemble_config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.feedrate is not None:
        assemble_config["first_layer_feedrate"] = args.feedrate
        assemble_config["remaining_layer_feedrate"] = args.feedrate
        if assemble_config.get("enable_z_smoothing", False) and args.z_smoothing_target_feedrate is None:
            assemble_config["z_smoothing_target_feedrate"] = args.feedrate
    if args.acceleration is not None:
        assemble_config["model_acceleration"] = args.acceleration
        assemble_config["first_layer_acceleration"] = args.acceleration
        assemble_config["remaining_layer_acceleration"] = args.acceleration
    if args.enable_z_smoothing:
        assemble_config["enable_z_smoothing"] = True
    if args.disable_z_smoothing:
        assemble_config["enable_z_smoothing"] = False
    if args.z_smoothing_target_feedrate is not None:
        assemble_config["z_smoothing_target_feedrate"] = args.z_smoothing_target_feedrate
    if args.z_smoothing_window is not None:
        assemble_config["z_smoothing_window"] = args.z_smoothing_window
    if args.z_smoothing_passes is not None:
        assemble_config["z_smoothing_passes"] = args.z_smoothing_passes
    if args.z_smoothing_strength is not None:
        assemble_config["z_smoothing_strength"] = args.z_smoothing_strength
    if args.z_smoothing_max_delta is not None:
        assemble_config["z_smoothing_max_delta"] = args.z_smoothing_max_delta


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
    for section, key in ((three_mf_config, "output"), (pack_config, "output")):
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

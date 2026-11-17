#!/usr/bin/env python3
"""Run the A1 pipeline with constant feedrate/acceleration motion overrides."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from run_pipeline import main as run_pipeline_main


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    motion = dict(config.get("motion_test", {}))
    assemble = dict(config.get("assemble", {}))
    feedrate = args.feedrate if args.feedrate is not None else float(motion.get("feedrate", 3000))
    acceleration = args.acceleration if args.acceleration is not None else float(motion.get("acceleration", 500))
    assemble.update({
        "first_layer_feedrate": feedrate,
        "remaining_layer_feedrate": feedrate,
        "first_layer_acceleration": acceleration,
        "remaining_layer_acceleration": acceleration,
        "model_acceleration": acceleration,
        "enable_slope_feedrate_limit": False,
        "enable_raster_after_zigzag_feedrate_limit": False,
        "enable_raster_after_fraction_feedrate_limit": False,
        "enable_z_smoothing": bool(motion.get("enable_z_smoothing", False)),
        "z_smoothing_target_feedrate": motion.get("z_smoothing_target_feedrate", feedrate),
        "z_smoothing_window": int(motion.get("z_smoothing_window", 5)),
        "z_smoothing_passes": int(motion.get("z_smoothing_passes", 1)),
        "z_smoothing_strength": float(motion.get("z_smoothing_strength", 1.0)),
        "z_smoothing_max_delta": float(motion.get("z_smoothing_max_delta", 0.05)),
        "z_smoothing_max_direction_change_degrees": float(
            motion.get("z_smoothing_max_direction_change_degrees", 30.0)
        ),
    })
    for key in (
        "untagged_role",
        "layer_override_roles",
        "role_feedrates",
        "role_accelerations",
        "bottom_layer_count",
        "sparse_infill_jump_requires_tag",
    ):
        if key in motion:
            assemble[key] = motion[key]
    if args.enable_z_smoothing:
        assemble["enable_z_smoothing"] = True
    if args.disable_z_smoothing:
        assemble["enable_z_smoothing"] = False

    config["assemble"] = assemble
    temp_config = config_path.with_name(".generated_a1_motion_config.json")
    temp_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    sys.argv = [sys.argv[0], "--config", str(temp_config)]
    if args.assemble_only:
        sys.argv.append("--assemble-only")
    return run_pipeline_main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="A1 pipeline JSON config.")
    parser.add_argument("--assemble-only", action="store_true", help="Stop after writing assembled G-code.")
    parser.add_argument("--feedrate", type=float, help="Override motion_test.feedrate.")
    parser.add_argument("--acceleration", type=float, help="Override motion_test.acceleration.")
    parser.add_argument("--enable-z-smoothing", action="store_true")
    parser.add_argument("--disable-z-smoothing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

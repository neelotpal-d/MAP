#!/usr/bin/env python3
"""Run A1 reference extraction -> assembly -> extracted package -> packing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
H2D_HELPERS = HERE.parent / "common"


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    assemble_config = dict(config.get("assemble", {}))
    package_config = dict(config.get("package", {}))
    pack_config = dict(config.get("pack", {}))

    print("[pipeline] Preparing A1 reference code", flush=True)
    prepare_reference(base_dir, config, assemble_config)
    if config.get("run_assemble", True):
        print("[pipeline] Starting assembly", flush=True)
        generated = write_assemble_config(base_dir, assemble_config, config)
        run([sys.executable, str(HERE / "assemble_a1_layer_paths.py"), "--config", str(generated)])
    if config.get("verify_motion", True):
        print("[pipeline] Verifying A1 motion before packaging", flush=True)
        gcode = package_config.get("gcode", assemble_config.get("output_gcode"))
        if not gcode:
            raise ValueError("A1 verification requires assemble.output_gcode or package.gcode")
        report = resolve(base_dir, config.get("motion_verification_report", "motion_verification.json"))
        run([sys.executable, str(H2D_HELPERS / "verify_a1_motion.py"),
             str(resolve(base_dir, gcode)), str(report), "--maximum-z-speed",
             str(assemble_config.get("travel_z_speed_limit", 20.))])
    if args.assemble_only:
        return 0
    if config.get("run_package", True):
        print("[pipeline] Creating 3MF folder", flush=True)
        run(make_package_command(base_dir, config, package_config, assemble_config))
    if config.get("run_pack", True):
        print("[pipeline] Packing final 3MF archive", flush=True)
        folder = resolve(base_dir, package_config["output"])
        archive = resolve(base_dir, pack_config.get("output", f"{folder}.gcode.3mf"))
        run([sys.executable, str(H2D_HELPERS / "pack_3mf_folder.py"), str(folder), str(archive)])
    return 0


def prepare_reference(base_dir: Path, config: dict[str, Any], assemble: dict[str, Any]) -> None:
    reference = config.get("reference_3mf")
    if not reference:
        return
    pre = assemble.get("pre_code")
    post = assemble.get("post_code")
    if not pre or not post:
        raise ValueError("assemble.pre_code and assemble.post_code are required with reference_3mf")
    run([
        sys.executable,
        str(HERE / "prepare_a1_reference.py"),
        "--reference", str(resolve(base_dir, reference)),
        "--pre-output", str(resolve(base_dir, pre)),
        "--post-output", str(resolve(base_dir, post)),
    ])


def write_assemble_config(base_dir: Path, assemble: dict[str, Any], config: dict[str, Any]) -> Path:
    required = ["input_dir", "pre_code", "post_code", "output_gcode"]
    missing = [key for key in required if key not in assemble]
    if missing:
        raise ValueError(f"assemble config missing: {', '.join(missing)}")
    resolved = dict(assemble)
    for key in ("input_dir", "pre_code", "post_code", "output_gcode", "summary_json"):
        if key in resolved:
            resolved[key] = str(resolve(base_dir, resolved[key]))
    path = resolve(base_dir, config.get("generated_assemble_config", ".generated_a1_assemble_config.json"))
    path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    return path


def make_package_command(
    base_dir: Path, config: dict[str, Any], package: dict[str, Any], assemble: dict[str, Any]
) -> list[str]:
    required = ["reference", "output"]
    missing = [key for key in required if key not in package]
    if missing:
        raise ValueError(f"package config missing: {', '.join(missing)}")
    gcode = package.get("gcode", assemble.get("output_gcode"))
    if not gcode:
        raise ValueError("package.gcode is missing and assemble.output_gcode is not set")
    command = [
        sys.executable, str(HERE / "make_a1_3mf_folder.py"),
        "--reference", str(resolve(base_dir, package["reference"])),
        "--gcode", str(resolve(base_dir, gcode)),
        "--output", str(resolve(base_dir, package["output"])),
    ]
    if package.get("bbox"):
        command.extend(["--bbox", str(package["bbox"])])
    if package.get("filament_color"):
        command.extend(["--filament-color", str(package["filament_color"])])
    if 'bed_leveling_margin' in package:
        command.extend(['--bed-leveling-margin', str(package['bed_leveling_margin'])])
    if package.get("spiral_mode_preview", False):
        command.append("--spiral-mode-preview")
    if package.get("overwrite", True):
        command.append("--overwrite")
    return command


def resolve(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def quote(value: str) -> str:
    return f'"{value}"' if " " in value else value


def run(command: list[str]) -> None:
    print(" ".join(quote(item) for item in command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="A1 pipeline JSON config.")
    parser.add_argument("--assemble-only", action="store_true", help="Stop after writing assembled G-code.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

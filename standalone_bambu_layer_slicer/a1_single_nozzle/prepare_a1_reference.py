#!/usr/bin/env python3
"""Extract reusable A1 startup and shutdown G-code from a .gcode.3mf file."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


GCODE_ENTRY = "Metadata/plate_1.gcode"
FIRST_LAYER_MARKER = ";VT0"
END_MARKER = "; MACHINE_END_GCODE_START"


def main() -> int:
    args = parse_args()
    with zipfile.ZipFile(args.reference) as archive:
        try:
            gcode = archive.read(GCODE_ENTRY).decode("utf-8")
        except KeyError as exc:
            raise ValueError(f"{args.reference} does not contain {GCODE_ENTRY}") from exc

    first_layer = gcode.find(FIRST_LAYER_MARKER)
    end = gcode.find(END_MARKER)
    if first_layer < 0:
        raise ValueError(f"Could not find first-layer marker {FIRST_LAYER_MARKER!r}")
    if end < 0:
        raise ValueError(f"Could not find machine-end marker {END_MARKER!r}")
    if end <= first_layer:
        raise ValueError("Machine-end marker occurs before the first layer marker")

    first_change_layer = gcode.find("; CHANGE_LAYER", first_layer)
    if first_change_layer < 0 or first_change_layer >= end:
        raise ValueError("Could not find the first CHANGE_LAYER marker after the A1 body prologue")

    # Bambu's own A1 output starts the printable body with VT0 plus a small
    # coordinate/extrusion-mode prologue.  Keep it with the pre-code, but do
    # not copy the reference object's first-layer moves that follow it.
    body_prologue = gcode[first_layer:first_change_layer].rstrip()
    pre = gcode[:first_layer].rstrip() + "\n" + body_prologue + "\n"
    post = gcode[end:].lstrip() + "\n"
    args.pre_output.parent.mkdir(parents=True, exist_ok=True)
    args.post_output.parent.mkdir(parents=True, exist_ok=True)
    args.pre_output.write_text(pre, encoding="utf-8", newline="\n")
    args.post_output.write_text(post, encoding="utf-8", newline="\n")
    print(f"Wrote {args.pre_output}")
    print(f"Wrote {args.post_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Known-good A1 .gcode.3mf file.")
    parser.add_argument("--pre-output", required=True, type=Path, help="Startup G-code output path.")
    parser.add_argument("--post-output", required=True, type=Path, help="Shutdown G-code output path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

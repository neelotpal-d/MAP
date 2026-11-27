#!/usr/bin/env python3
"""Extract reusable A1 startup and shutdown G-code from a .gcode.3mf file."""

from __future__ import annotations

import argparse
import math
import re
import zipfile
from pathlib import Path


GCODE_ENTRY = "Metadata/plate_1.gcode"
FIRST_LAYER_MARKER = ";VT0"
END_MARKER = "; MACHINE_END_GCODE_START"


def raise_shutdown_clearance(text, minimum_z, build_size):
    """Keep compiled reference-object end moves above this job's actual top."""
    if not math.isfinite(minimum_z) or not math.isfinite(build_size) or not 0<=minimum_z<=build_size:
        raise ValueError('Shutdown clearance exceeds the selected build height.')
    absolute=True;position=None;output=[]
    word=re.compile(r'(?<![A-Za-z])Z([-+]?(?:\d+(?:\.\d*)?|\.\d+))')
    for line in text.splitlines(keepends=True):
        code=line.partition(';')[0].strip()
        if code=='G90':absolute=True
        elif code=='G91':absolute=False
        if re.match(r'^G[01](?=\s)',code):
            match=word.search(code)
            if match:
                value=float(match[1])
                if absolute:target=max(value,minimum_z)
                else:
                    if position is None:raise ValueError('Shutdown relative Z has no known absolute position.')
                    target=max(position+value,minimum_z)
                if target>build_size+1e-6:raise ValueError('Shutdown Z exceeds the selected build height.')
                replacement=target if absolute else target-position
                if abs(replacement-value)>1e-8:
                    line=word.sub(lambda m:'Z'+format(replacement,'.5f'),line,count=1).rstrip('\r\n')+' ; MAP shutdown clearance\n'
                position=target
        output.append(line)
    if position is None:raise ValueError('Startup reference has no identifiable shutdown Z move.')
    return ''.join(output)


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
    if args.minimum_shutdown_z is not None:
        post=raise_shutdown_clearance(post,args.minimum_shutdown_z,args.build_size)
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
    parser.add_argument('--minimum-shutdown-z',type=float)
    parser.add_argument('--build-size',type=float,default=256.)
    parser.add_argument("--pre-output", required=True, type=Path, help="Startup G-code output path.")
    parser.add_argument("--post-output", required=True, type=Path, help="Shutdown G-code output path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Pack an extracted 3MF folder into a .3mf/.gcode.3mf archive.

PowerShell Compress-Archive can store entry names with backslashes on Windows.
3MF relationships use forward-slash paths such as /3D/3dmodel.model, so this
packer always writes POSIX-style archive names.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


REQUIRED_ROOT_ENTRIES = ["[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack an extracted 3MF folder with forward-slash ZIP paths.")
    parser.add_argument("folder", type=Path, help="Extracted package folder")
    parser.add_argument("output", type=Path, help="Output .3mf or .gcode.3mf file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder
    output = args.output

    if not folder.is_dir():
        raise FileNotFoundError(f"folder does not exist: {folder}")

    for required in REQUIRED_ROOT_ENTRIES:
        if not (folder / Path(required)).exists():
            raise FileNotFoundError(f"required package entry missing: {required}")

    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(folder).as_posix()
            zf.write(path, arcname)

    with zipfile.ZipFile(output, "r") as zf:
        names = set(zf.namelist())
        for required in REQUIRED_ROOT_ENTRIES:
            if required not in names:
                raise RuntimeError(f"packed archive is missing required entry: {required}")
        bad_names = [name for name in names if "\\" in name]
        if bad_names:
            raise RuntimeError(f"packed archive contains backslash paths: {bad_names[:3]}")

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

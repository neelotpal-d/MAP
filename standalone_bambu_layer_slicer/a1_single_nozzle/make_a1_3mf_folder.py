#!/usr/bin/env python3
"""Create an extracted A1 G-code-first 3MF folder from a reference package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


GCODE_ENTRY = Path("Metadata/plate_1.gcode")
MD5_ENTRY = Path("Metadata/plate_1.gcode.md5")
PLATE_JSON_ENTRY = Path("Metadata/plate_1.json")
PROJECT_SETTINGS_ENTRY = Path("Metadata/project_settings.config")
BODY_START = ';===== assembled layer path body start ====='
BODY_END = ';===== assembled layer path body end ====='
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    with zipfile.ZipFile(args.reference) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in reference archive: {member.filename}")
            target = output / member_path
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

    info = {'bbox': parse_bbox(args.bbox) if args.bbox else None,
            'bed_leveling_margin': args.bed_leveling_margin}
    checksum = copy_gcode_with_checksum(args.gcode, output / GCODE_ENTRY,
                                        spiral_mode_preview=args.spiral_mode_preview,
                                        job_metadata=info)
    (output / MD5_ENTRY).write_text(checksum, encoding="ascii", newline="\n")
    update_plate_json(output / PLATE_JSON_ENTRY, args)
    update_job_metadata(output, info)
    if args.spiral_mode_preview:
        update_nonplanar_preview_metadata(output / PROJECT_SETTINGS_ENTRY)
    print(f"Wrote extracted A1 package {output}")
    print('A1 job metadata: ' + json.dumps(info), flush=True)
    return 0


def copy_gcode_with_checksum(source: Path, target: Path, *, spiral_mode_preview: bool,
                             job_metadata: dict | None = None) -> str:
    """Save identical G-code and hash it without materializing extra full-size copies."""
    digest = hashlib.md5()
    if not spiral_mode_preview and job_metadata is None:
        if source.resolve() == target.resolve():
            with source.open('rb') as incoming:
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest().upper()
        with source.open('rb') as incoming, target.open('wb') as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b''):
                outgoing.write(block)
                digest.update(block)
        return digest.hexdigest().upper()
    # Keep the exact Unicode regex and newline semantics of set_gcode_spiral_mode.
    # One decoded string is needed; substitution/re-encoding no longer creates
    # several additional copies of the complete assembled job.
    with source.open(encoding='utf-8', newline='') as incoming:
        text = incoming.read()
    matches = list(re.finditer(r"(?m)^; spiral_mode = [01]\s*$", text)) if spiral_mode_preview else []
    if spiral_mode_preview and len(matches) != 1:
        raise ValueError(
            "Expected exactly one embedded '; spiral_mode = 0/1' setting in the A1 G-code header; "
            f"found {len(matches)}."
        )
    replacements = [(match.start(), match.end(), '; spiral_mode = 1') for match in matches]
    if job_metadata is not None:
        header_end, header = corrected_job_header(text, job_metadata)
        # Apply preview settings inside the small replacement header as well.
        if spiral_mode_preview:
            header = re.sub(r'(?m)^; spiral_mode = [01]\s*$', '; spiral_mode = 1', header)
        replacements = [(0, header_end, header)] + [row for row in replacements if row[0] >= header_end]
    with target.open('wb') as outgoing:
        def write_range(start, end):
            for offset in range(start, end, 1024 * 1024):
                block = text[offset:min(offset + 1024 * 1024, end)].encode('utf-8')
                outgoing.write(block)
                digest.update(block)
        offset = 0
        for start, end, value in sorted(replacements):
            write_range(offset, start)
            replacement = value.encode('utf8')
            outgoing.write(replacement)
            digest.update(replacement)
            offset = end
        write_range(offset, len(text))
    return digest.hexdigest().upper()


def corrected_job_header(text: str, info: dict) -> tuple[int, str]:
    """Use the assembled job, never the reference object's probe area or totals.

    Scan the existing decoded string without copying the multi-GB body. Time is
    deliberately unavailable (zero); this packager does not simulate firmware.
    A1 assembly uses absolute millimetre XYZ and relative E. Fail explicitly on
    other body modes rather than reporting a misleading material total/bounds.
    """
    start, end = text.find(BODY_START), text.find(BODY_END)
    if start < 0 or end <= start or text.find(BODY_START, start + 1) >= 0:
        raise ValueError('A1 packaging requires exactly one assembled layer path body.')
    mode = re.compile(r'(?m)^[ \t]*(M82|G91|G20)(?=[ \t;\r\n]|$)')
    if mode.search(text, start, end):
        raise ValueError('A1 job metadata requires absolute millimetre XYZ and relative extrusion (G90/M83).')
    layer_count = sum(1 for _ in re.compile(r'(?m)^; CHANGE_LAYER[ \t]*\r?$').finditer(text, start, end))
    if not layer_count:
        raise ValueError('A1 assembled body has no CHANGE_LAYER markers.')
    e_words = re.compile(r'(?m)^G[01][ \t]+[^;\r\n]*?\bE(' + NUMBER + r')(?=[ \t;\r\n]|$)')
    length = math.fsum(float(match[1]) for match in e_words.finditer(text, start, end))
    if not math.isfinite(length) or length < 0:
        raise ValueError('Invalid net relative extrusion in A1 assembled body.')
    header = text[:start]
    def setting(name):
        match = re.search(r'(?m)^; ' + name + r'\s*[:=]\s*(' + NUMBER + r')', header)
        if match is None or not math.isfinite(float(match[1])) or float(match[1]) <= 0:
            raise ValueError(f'A1 reference is missing a positive {name}.')
        return float(match[1])
    diameter, density = setting('filament_diameter'), setting('filament_density')
    volume = length * math.pi * (diameter / 2)**2 / 1000  # cm^3
    bbox = info.get('bbox')
    if bbox is None:
        # Standalone callers may omit --bbox. Derive the complete body footprint,
        # including platform/travels; app callers already provide these bounds.
        lower, upper, position = [math.inf]*2, [-math.inf]*2, [None]*2
        for match in re.compile(r'(?m)^G[01][ \t]+([^;\r\n]*)').finditer(text, start, end):
            for axis, value in re.findall(r'([XY])(' + NUMBER + r')(?=[ \t]|$)', match[1]):
                position['XY'.index(axis)] = float(value)
            if all(value is not None for value in position):
                for i, value in enumerate(position):
                    lower[i], upper[i] = min(lower[i], value), max(upper[i], value)
        bbox = lower + upper
    bbox = parse_bbox(','.join(map(str, bbox)))
    margin = float(info.get('bed_leveling_margin', 5.))
    if not math.isfinite(margin) or margin < 0 or any(value < 0 or value > 256 for value in bbox):
        raise ValueError('A1 probe footprint must fit 0..256 mm; margin must be finite and nonnegative.')
    probe = [max(0., bbox[0]-margin), max(0., bbox[1]-margin),
             min(256., bbox[2]+margin), min(256., bbox[3]+margin)]
    line = f'G29 A1 X{probe[0]:.5f} Y{probe[1]:.5f} I{probe[2]-probe[0]:.5f} J{probe[3]-probe[1]:.5f}'
    header, count = re.subn(r'(?m)^[ \t]*G29[ \t]+A1(?=[ \t])[^\r\n]*', line, header)
    if count != 1:
        raise ValueError(f'Expected one executable A1 bed-probing command; found {count}.')
    for name, value in (('model printing time', '0s; total estimated time: 0s'),
                        ('total layer number', str(layer_count)),
                        ('total filament length [mm]', f'{length:.2f}'),
                        ('total filament volume [cm^3]', f'{volume:.2f}'),
                        ('total filament weight [g]', f'{volume*density:.2f}')):
        header, count = re.subn(r'(?m)^; ' + re.escape(name) + r'\s*:[^\r\n]*',
                                '; ' + name + ': ' + value, header)
        if count != 1:
            raise ValueError(f'Expected one reference header value for {name}; found {count}.')
    header = re.sub(r'(?m)^M73[ \t]+[^\r\n]*', 'M73 P0 R0', header)
    header = '; Custom path execution time unavailable; reference estimates removed.\n' + header
    info.update(bbox=bbox, probe_bbox=probe, layer_count=layer_count,
                filament_length_mm=length, filament_volume_cm3=volume,
                filament_weight_g=volume*density, execution_time_estimated=False,
                support_used=text.find(';part: support', start, end) >= 0)
    return start, header


def update_job_metadata(output: Path, info: dict) -> None:
    path = output / PLATE_JSON_ENTRY
    data = json.loads(path.read_text(encoding='utf8'))
    data['bbox_all'] = info['bbox']
    data['first_layer_time'] = 0
    for entry in data.get('bbox_objects', []):
        entry.update(bbox=info['bbox'], name='Custom layer paths')
        entry.pop('area', None)  # area of the reference object is unrelated
    path.write_text(json.dumps(data, separators=(',', ':')), encoding='utf8')
    path = output / 'Metadata/slice_info.config'
    tree = ET.parse(path)
    for row in tree.findall('.//plate/metadata'):
        key = row.get('key')
        if key in ('prediction', 'first_layer_time'):
            row.set('value', '0')
        elif key == 'weight':
            row.set('value', f"{info['filament_weight_g']:.2f}")
        elif key == 'support_used':
            row.set('value', str(info['support_used']).lower())
    for row in tree.findall('.//plate/object'):
        row.set('name', 'Custom layer paths')
    for row in tree.findall('.//plate/filament'):
        row.set('used_m', f"{info['filament_length_mm']/1000:.5f}")
        row.set('used_g', f"{info['filament_weight_g']:.2f}")
        row.set('used_for_support', str(info['support_used']).lower())
    for row in tree.findall('.//layer_filament_list'):
        row.set('filament_list', '0')
        row.set('layer_ranges', f"0 {info['layer_count']-1}")
    tree.write(path, encoding='utf8', xml_declaration=True)
    (output / 'Metadata/custom_path_job.json').write_text(json.dumps(info, indent=2), encoding='utf8')


def update_plate_json(path: Path, args: argparse.Namespace) -> None:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if args.bbox is not None:
        bbox = parse_bbox(args.bbox)
        data["bbox_all"] = bbox
        for entry in data.get("bbox_objects", []):
            entry["bbox"] = bbox
    if args.filament_color is not None:
        data["filament_colors"] = [args.filament_color]
    data["filament_ids"] = [0]
    data["first_extruder"] = 0
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8", newline="\n")


def update_nonplanar_preview_metadata(path: Path) -> None:
    """Group continuous-Z paths at CHANGE_LAYER markers in Bambu's preview."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["spiral_mode"] = "1"
    path.write_text(json.dumps(data, indent=4), encoding="utf-8", newline="\n")


def set_gcode_spiral_mode(gcode: bytes) -> bytes:
    """Synchronize the G-code header parsed by GCodeProcessor with package metadata."""
    text = gcode.decode("utf-8")
    updated, replacements = re.subn(
        r"(?m)^; spiral_mode = [01]\s*$",
        "; spiral_mode = 1",
        text,
    )
    if replacements != 1:
        raise ValueError(
            "Expected exactly one embedded '; spiral_mode = 0/1' setting in the A1 G-code header; "
            f"found {replacements}."
        )
    return updated.encode("utf-8")


def parse_bbox(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("--bbox must be min_x,min_y,max_x,max_y")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Known-good A1 .gcode.3mf package.")
    parser.add_argument("--gcode", required=True, type=Path, help="Assembled G-code to embed.")
    parser.add_argument("--output", required=True, type=Path, help="Output extracted 3MF directory.")
    parser.add_argument("--bbox", help="Optional model bounds: min_x,min_y,max_x,max_y.")
    parser.add_argument('--bed-leveling-margin', type=float, default=5., help='Probe margin around this job footprint (mm).')
    parser.add_argument("--filament-color", help="Optional plate metadata color, e.g. #000000.")
    parser.add_argument(
        "--spiral-mode-preview",
        action="store_true",
        help="Use CHANGE_LAYER markers to group continuous-Z paths in Bambu Studio's preview.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

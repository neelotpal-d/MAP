#!/usr/bin/env python3
"""
Build an extracted Bambu H2D .gcode.3mf-style folder from:
  - an extracted template folder, such as Untitled/
  - a replacement OBJ model
  - a replacement plate_1.gcode

The script assumes the material/nozzle assignment stays the same as the template:
filament 1 -> left, filament 2 -> right.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable
from zlib import crc32


TOOL_RE = re.compile(r"^\s*T(\d+)\s*(?:;.*)?$")
TOOL_SUB_RE = re.compile(r"^(\s*)T([01])(\s*(?:;.*)?)$")
M620_START_RE = re.compile(r"^(\s*)M620\s+S([01])A(.*)$")
M621_END_RE = re.compile(r"^(\s*)M621\s+S([01])A(.*)$")
M620_AIR_RE = re.compile(r"^(\s*)M620\.6\s+I([01])(\s+.*)$")
M104_TOOL_RE = re.compile(r"^(\s*)M104(\s+.*\s)T([01])(\s*(?:;.*)?)$")
G151_TOOL_RE = re.compile(r"^(\s*)G151(\s+P)([01])(\s+.*)$")
G383_ACTIVE_L_RE = re.compile(r"^(\s*G383(?:\.3)?\s+.*\sL)([01])(\s*(?:;.*)?)$")
NOZZLE_CHANGE_RE = re.compile(r"NOZZLE_CHANGE_START\s+OF(\d+)\s+NF(\d+)")
LAYER_RE = re.compile(r";\s*layer num/total_layer_count:\s*(\d+)\s*/\s*(\d+)")
ASSEMBLED_SNIPPET_RE = re.compile(
    r";\s*ASSEMBLED_SNIPPET_START\b.*\bmain=(\d+)\b.*\btool=T(\d+)\b"
)
ASSEMBLED_SNIPPET_START_RE = re.compile(r"^;\s*ASSEMBLED_SNIPPET_START\b.*\bmain=(\d+)\b")
ASSEMBLED_SNIPPET_END_RE = re.compile(r"^;\s*ASSEMBLED_SNIPPET_END\b")
GCODE_XY_E_RE = re.compile(r"\b([XYE])(-?\d+(?:\.\d+)?)")
G29_AREA_RE = re.compile(r"^\s*G29\s+A[12]\s+O\b")
F_WORD_RE = re.compile(r"\bF(-?\d+(?:\.\d+)?)")
Z_WORD_RE = re.compile(r"\bZ(-?\d+(?:\.\d+)?)")
E_WORD_RE = re.compile(r"\bE(-?\d+(?:\.\d+)?)")
FIRST_PRINT_RE = re.compile(r"^\s*;\s*(CHANGE_LAYER|VT0\b)")
HEADER_LIST_RE = re.compile(r"^;\s*([^:]+?)\s*:\s*(.+)$")
HEADER_EQUALS_RE = re.compile(r"^;\s*([^=]+?)\s*=\s*(.+)$")
REQUIRED_GCODE_HEADER_KEYS = [
    "default_nozzle_volume_type",
    "extruder_max_nozzle_count",
    "extruder_nozzle_stats",
    "extruder_printable_area",
    "extruder_printable_height",
    "filament_map",
    "filament_map_2",
    "filament_map_mode",
    "filament_nozzle_map",
    "filament_colour",
    "filament_colour_type",
    "filament_settings_id",
    "filament_type",
    "filament_volume_map",
    "master_extruder_id",
    "nozzle_diameter",
    "nozzle_volume_type",
    "physical_extruder_map",
    "print_extruder_id",
    "print_extruder_variant",
    "printer_extruder_id",
    "printer_extruder_variant",
    "printer_model",
    "printer_settings_id",
]
DEFAULT_GCODE_HEADER_VALUES = {
    "default_nozzle_volume_type": "Standard,Standard",
    "extruder_max_nozzle_count": "1,1",
    "extruder_nozzle_stats": "Standard#1;Standard#1",
    "extruder_printable_area": "0x0,325x0,325x320,0x320#25x0,350x0,350x320,25x320",
    "extruder_printable_height": "320,325",
    "filament_map": "1,2",
    "filament_map_2": "0,1",
    "filament_map_mode": "Auto For Flush",
    "filament_nozzle_map": "0,1",
    "filament_colour": "#000000;#FFFFFF",
    "filament_colour_type": "1;1",
    "filament_settings_id": "\"Bambu PLA Basic @BBL H2D\";\"Bambu PLA Basic @BBL H2D\"",
    "filament_type": "PLA;PLA",
    "filament_volume_map": "0,0",
    "master_extruder_id": "2",
    "nozzle_diameter": "0.4,0.4",
    "nozzle_volume_type": "Standard,Standard",
    "physical_extruder_map": "1,0",
    "print_extruder_id": "1,2",
    "print_extruder_variant": "\"Direct Drive Standard\";\"Direct Drive Standard\"",
    "printer_extruder_id": "1,2",
    "printer_extruder_variant": "\"Direct Drive Standard\";\"Direct Drive Standard\"",
    "printer_model": "Bambu Lab H2D",
    "printer_settings_id": "Bambu Lab H2D 0.4 nozzle",
}
MATERIAL_LAYOUTS = {
    "black_left_white_right": {
        "filament_map": "1,2",
        "filament_maps_xml": "1 2",
        "filament_map_2": "0,1",
        "filament_nozzle_map": "0,1",
        "filament_volume_map": "0,0",
        "colors": ["#000000", "#FFFFFF"],
        "types": ["PLA", "PLA"],
        "group_ids": ["0", "1"],
    },
    "black_left_white_right_yellow_left": {
        "filament_map": "1,2,1",
        "filament_maps_xml": "1 2 1",
        "filament_map_2": "0,1,0",
        "filament_nozzle_map": "0,1,0",
        "filament_volume_map": "0,0,0",
        "colors": ["#000000", "#FFFFFF", "#FFFF00"],
        "types": ["PLA", "PLA", "PLA"],
        "group_ids": ["0", "1", "0"],
    },
    "black_right_white_left": {
        "filament_map": "2,1",
        "filament_maps_xml": "2 1",
        "filament_map_2": "1,0",
        "filament_nozzle_map": "1,0",
        "filament_volume_map": "0,0",
        "colors": ["#000000", "#FFFFFF"],
        "types": ["PLA", "PLA"],
        "group_ids": ["1", "0"],
    },
    "black_right_white_left_yellow_right": {
        "filament_map": "2,1,2",
        "filament_maps_xml": "2 1 2",
        "filament_map_2": "1,0,1",
        "filament_nozzle_map": "1,0,1",
        "filament_volume_map": "0,0,0",
        "colors": ["#000000", "#FFFFFF", "#FFFF00"],
        "types": ["PLA", "PLA", "PLA"],
        "group_ids": ["1", "0", "1"],
    },
    "white_left": {
        "filament_map": "1,1",
        "filament_maps_xml": "1 1",
        "filament_map_2": "0,0",
        "filament_nozzle_map": "0,0",
        "filament_volume_map": "0,0",
        "colors": ["#000000", "#FFFFFF"],
        "types": ["PLA", "PLA"],
        "group_ids": ["0", "0"],
    },
}
GCODE_SECTIONS = {
    "auto_extrude_cali": (
        ";===== auto extrude cali start =========================",
        ";===== auto extrude cali end =========================",
        "skip_auto_extrude_cali",
    ),
    "bed_leveling": (
        ";===== bed leveling ==================================",
        ";===== bed leveling end ================================",
        "skip_bed_leveling",
    ),
    "z_offset_cali": (
        ";===== z ofst cali start =====",
        ";===== z ofst cali end =====",
        "skip_z_offset_cali",
    ),
    "xy_offset_cali": (
        ";===== xy ofst cali start =====",
        ";===== xy ofst cali end =====",
        "skip_xy_offset_cali",
    ),
}

EMPTY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)
CONTENT_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
 <Default Extension="config" ContentType="text/xml"/>
 <Default Extension="gcode" ContentType="text/x.gcode"/>
 <Default Extension="json" ContentType="application/json"/>
 <Default Extension="png" ContentType="image/png"/>
</Types>
"""
RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>
 <Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>
 <Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>
</Relationships>
"""
MODEL_SETTINGS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value=""/>
    <metadata key="locked" value="false"/>
    <metadata key="filament_map_mode" value="Auto For Flush"/>
    <metadata key="filament_maps" value="1 2"/>
    <metadata key="filament_volume_maps" value="0 0"/>
    <metadata key="gcode_file" value="Metadata/plate_1.gcode"/>
    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>
    <metadata key="thumbnail_no_light_file" value="Metadata/plate_no_light_1.png"/>
    <metadata key="top_file" value="Metadata/top_1.png"/>
    <metadata key="pick_file" value="Metadata/pick_1.png"/>
    <metadata key="pattern_bbox_file" value="Metadata/plate_1.json"/>
  </plate>
</config>
"""
MODEL_SETTINGS_RELS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="plate_1.gcode" Id="rel-1" Type="http://schemas.bambulab.com/package/2021/gcode"/>
</Relationships>
"""
CUT_INFORMATION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <plate>
    <metadata key="index" value="1"/>
  </plate>
</config>
"""
SLICE_INFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <header>
    <header_item key="X-BBL-Client-Type" value="slicer"/>
    <header_item key="X-BBL-Client-Version" value="02.05.00.66"/>
  </header>
  <plate>
    <metadata key="index" value="1"/>
    <metadata key="extruder_type" value="0 0"/>
    <metadata key="nozzle_volume_type" value="0 0"/>
    <metadata key="printer_model_id" value="O1D"/>
    <metadata key="nozzle_diameters" value="0.4,0.4"/>
    <metadata key="timelapse_type" value="0"/>
    <metadata key="prediction" value="0"/>
    <metadata key="weight" value="0"/>
    <metadata key="first_layer_time" value="0"/>
    <metadata key="outside" value="false"/>
    <metadata key="support_used" value="false"/>
    <metadata key="label_object_enabled" value="false"/>
    <metadata key="filament_maps" value="1 2"/>
    <metadata key="limit_filament_maps" value="0 0"/>
    <object identify_id="1" name="model.obj" skipped="false"/>
    <filament id="1" tray_info_idx="GFA00" type="PLA" color="#000000" used_m="0" used_g="0" used_for_object="true" used_for_support="false" group_id="0" nozzle_diameter="0.40" volume_type="Standard"/>
    <filament id="2" tray_info_idx="GFA00" type="PLA" color="#FFFFFF" used_m="0" used_g="0" used_for_object="true" used_for_support="false" group_id="1" nozzle_diameter="0.40" volume_type="Standard"/>
  </plate>
</config>
"""


@dataclass
class ObjMesh:
    vertices: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]


@dataclass
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) / 2.0

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def shifted_to_center(self, center_x: float, center_y: float) -> "BBox":
        dx = center_x - self.center_x
        dy = center_y - self.center_y
        return BBox(self.min_x + dx, self.min_y + dy, self.max_x + dx, self.max_y + dy)

    def as_list(self) -> list[float]:
        return [self.min_x, self.min_y, self.max_x, self.max_y]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an extracted H2D 3MF/package folder from OBJ + G-code."
    )
    parser.add_argument("--config", type=Path, help="JSON config file with template/obj/gcode/output/etc.")
    parser.add_argument("--template", type=Path, help="Extracted template folder, e.g. Untitled")
    parser.add_argument(
        "--minimal-template",
        action="store_true",
        help="Create a small package structure without copying an extracted Bambu template.",
    )
    parser.add_argument("--obj", type=Path, help="Replacement OBJ model")
    parser.add_argument("--gcode", type=Path, help="Replacement plate_1.gcode")
    parser.add_argument("--output", type=Path, help="Output folder to create")
    parser.add_argument(
        "--center",
        default=None,
        help="Target XY center for the OBJ bounding box, e.g. 175,160. If omitted, keeps OBJ coordinates.",
    )
    parser.add_argument(
        "--obj-up",
        choices=["z", "y", "x"],
        default=None,
        help="Coordinate up-axis of the OBJ. Use y or x to rotate model geometry into Bambu/3MF Z-up.",
    )
    parser.add_argument("--object-name", default=None, help="Object name to write into metadata")
    parser.add_argument(
        "--gcode-preview-mode",
        action="store_true",
        help="Write an empty 3MF model so Bambu Studio opens the embedded plate G-code in Preview mode.",
    )
    parser.add_argument(
        "--preview-spiral-mode",
        action="store_true",
        help="Set project_settings.config spiral_mode=1 so Bambu Studio groups preview layers by CHANGE_LAYER markers.",
    )
    parser.add_argument(
        "--preview-object-ids",
        action="store_true",
        help="Insert ; OBJECT_ID comments before preview feature blocks.",
    )
    parser.add_argument(
        "--material-layout",
        choices=sorted(MATERIAL_LAYOUTS),
        default=None,
        help="Material/nozzle mapping preset. Default: black_left_white_right.",
    )
    parser.add_argument(
        "--first-filament",
        type=int,
        choices=[1, 2],
        help="Force the first filament in filament_sequence.json. 1 uses logical T0, 2 uses logical T1; physical side depends on filament_map.",
    )
    parser.add_argument(
        "--no-wipe-tower-metadata",
        action="store_true",
        help="Remove wipe tower from plate_1.json metadata. This does not remove wipe-tower moves from G-code.",
    )
    parser.add_argument(
        "--register-all-layout-filaments",
        action="store_true",
        help="Register every filament in the selected material layout, even if the G-code body only prints with one.",
    )
    parser.add_argument("--first-layer-count", type=int, help="Number of initial layers using first-layer feedrate settings.")
    parser.add_argument("--first-layer-feedrate", type=float, help="Replace G0/G1 F values on the first N layers.")
    parser.add_argument("--remaining-layer-feedrate", type=float, help="Replace G0/G1 F values after the first N layers.")
    parser.add_argument("--first-layer-feedrate-multiplier", type=float, help="Scale G0/G1 F values on the first N layers.")
    parser.add_argument("--remaining-layer-feedrate-multiplier", type=float, help="Scale G0/G1 F values after the first N layers.")
    parser.add_argument("--skip-auto-extrude-cali", action="store_true", help="Remove the marked auto extrusion calibration block from G-code.")
    parser.add_argument("--skip-bed-leveling", action="store_true", help="Remove the marked bed leveling block from G-code.")
    parser.add_argument("--bed-leveling-margin", type=float, help="XY margin around the deposited model projection for the G29 probe rectangle.")
    parser.add_argument("--skip-z-offset-cali", action="store_true", help="Remove the marked Z offset calibration block from G-code.")
    parser.add_argument("--skip-xy-offset-cali", action="store_true", help="Remove the marked XY offset calibration block from G-code.")
    parser.add_argument("--enable-planar-platform", action="store_true", help="Insert a filled first-layer platform before the model body.")
    parser.add_argument("--platform-margin", type=float, help="XY margin around the OBJ bbox for the planar platform.")
    parser.add_argument("--platform-line-spacing", type=float, help="Distance between platform infill lines.")
    parser.add_argument("--platform-line-width", type=float, help="Extrusion width used for platform E calculation.")
    parser.add_argument("--platform-layer-height", type=float, help="Layer height used for platform E calculation.")
    parser.add_argument("--platform-feedrate", type=float, help="Feedrate for platform extrusion moves.")
    parser.add_argument("--platform-travel-feedrate", type=float, help="Feedrate for platform travel moves.")
    parser.add_argument("--platform-extrusion-scale", type=float, help="Multiplier for calculated platform extrusion.")
    parser.add_argument(
        "--platform-lift-model-z",
        action="store_true",
        help="Shift model/body Z moves up by platform_layer_height after inserting the platform.",
    )
    parser.add_argument(
        "--blank-pngs",
        action="store_true",
        help="Replace package PNG previews with 1x1 blank placeholders.",
    )
    parser.add_argument("--plate-png", type=Path, help="Replacement Metadata/plate_1.png preview")
    parser.add_argument("--plate-small-png", type=Path, help="Replacement Metadata/plate_1_small.png preview")
    parser.add_argument("--plate-no-light-png", type=Path, help="Replacement Metadata/plate_no_light_1.png preview")
    parser.add_argument("--top-png", type=Path, help="Replacement Metadata/top_1.png preview")
    parser.add_argument("--pick-png", type=Path, help="Replacement Metadata/pick_1.png preview")
    parser.add_argument("--plate", default="plate_1", help="Plate key/name; default: plate_1")
    parser.add_argument("--overwrite", action="store_true", help="Replace output folder if it already exists")
    args = parser.parse_args()
    return merge_json_config(args)


def merge_json_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        if not args.minimal_template:
            require_arg(args.template, "--template or --minimal-template")
        require_arg(args.obj, "--obj")
        require_arg(args.gcode, "--gcode")
        require_arg(args.output, "--output")
        return apply_default_options(args)

    config_path = args.config.resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent

    def get_value(name: str, default: object = None) -> object:
        cli_value = getattr(args, name)
        if cli_value not in (None, False):
            return cli_value
        return raw.get(name, default)

    args.minimal_template = bool(args.minimal_template or raw.get("minimal_template", False))
    template_value = get_value("template")
    args.template = None if args.minimal_template and template_value is None else resolve_json_path(template_value, base_dir, "template")
    args.obj = resolve_json_path(get_value("obj"), base_dir, "obj")
    args.gcode = resolve_json_path(get_value("gcode"), base_dir, "gcode")
    args.output = resolve_json_path(get_value("output"), base_dir, "output")
    args.center = get_value("center")
    args.obj_up = str(get_value("obj_up", args.obj_up or "z")).lower()
    if args.obj_up not in ("z", "y", "x"):
        raise ValueError("obj_up must be 'z', 'y', or 'x'")
    args.material_layout = str(get_value("material_layout", args.material_layout or "black_left_white_right"))
    if args.material_layout not in MATERIAL_LAYOUTS:
        raise ValueError(f"material_layout must be one of: {', '.join(sorted(MATERIAL_LAYOUTS))}")
    args.object_name = get_value("object_name")
    args.gcode_preview_mode = bool(args.gcode_preview_mode or raw.get("gcode_preview_mode", False))
    args.preview_spiral_mode = bool(args.preview_spiral_mode or raw.get("preview_spiral_mode", False))
    args.preview_object_ids = bool(args.preview_object_ids or raw.get("preview_object_ids", False))
    args.first_filament = get_value("first_filament")
    if args.first_filament is not None:
        args.first_filament = int(args.first_filament)
        if args.first_filament not in (1, 2):
            raise ValueError("first_filament must be 1 or 2")
    args.plate = str(get_value("plate", args.plate))
    args.overwrite = bool(args.overwrite or raw.get("overwrite", False))
    args.no_wipe_tower_metadata = bool(args.no_wipe_tower_metadata or raw.get("no_wipe_tower_metadata", False))
    args.register_all_layout_filaments = bool(args.register_all_layout_filaments or raw.get("register_all_layout_filaments", False))
    args.first_layer_count = optional_int(get_value("first_layer_count", 1))
    if args.first_layer_count < 1:
        raise ValueError("first_layer_count must be >= 1")
    args.first_layer_feedrate = optional_float(get_value("first_layer_feedrate"))
    args.remaining_layer_feedrate = optional_float(get_value("remaining_layer_feedrate"))
    args.first_layer_feedrate_multiplier = optional_float(get_value("first_layer_feedrate_multiplier"))
    args.remaining_layer_feedrate_multiplier = optional_float(get_value("remaining_layer_feedrate_multiplier"))
    args.skip_auto_extrude_cali = bool(args.skip_auto_extrude_cali or raw.get("skip_auto_extrude_cali", False))
    args.skip_bed_leveling = bool(args.skip_bed_leveling or raw.get("skip_bed_leveling", False))
    args.bed_leveling_margin = optional_float(get_value("bed_leveling_margin", 5.0))
    args.skip_z_offset_cali = bool(args.skip_z_offset_cali or raw.get("skip_z_offset_cali", False))
    args.skip_xy_offset_cali = bool(args.skip_xy_offset_cali or raw.get("skip_xy_offset_cali", False))
    args.enable_planar_platform = bool(args.enable_planar_platform or raw.get("enable_planar_platform", False))
    args.platform_margin = optional_float(get_value("platform_margin", 2.0))
    args.platform_line_spacing = optional_float(get_value("platform_line_spacing", 0.45))
    args.platform_line_width = optional_float(get_value("platform_line_width", 0.45))
    args.platform_layer_height = optional_float(get_value("platform_layer_height", 0.2))
    args.platform_feedrate = optional_float(get_value("platform_feedrate", 900))
    args.platform_travel_feedrate = optional_float(get_value("platform_travel_feedrate", 6000))
    args.platform_extrusion_scale = optional_float(get_value("platform_extrusion_scale", 0.85))
    args.platform_lift_model_z = bool(args.platform_lift_model_z or raw.get("platform_lift_model_z", False))
    args.blank_pngs = bool(args.blank_pngs or raw.get("blank_pngs", False))
    for attr in ["plate_png", "plate_small_png", "plate_no_light_png", "top_png", "pick_png"]:
        value = get_value(attr)
        setattr(args, attr, None if value is None else resolve_json_path(value, base_dir, attr))
    return apply_default_options(args)


def apply_default_options(args: argparse.Namespace) -> argparse.Namespace:
    args.first_layer_count = args.first_layer_count or 1
    args.platform_margin = 2.0 if args.platform_margin is None else args.platform_margin
    args.platform_line_spacing = 0.45 if args.platform_line_spacing is None else args.platform_line_spacing
    args.platform_line_width = 0.45 if args.platform_line_width is None else args.platform_line_width
    args.platform_layer_height = 0.2 if args.platform_layer_height is None else args.platform_layer_height
    args.platform_feedrate = 900 if args.platform_feedrate is None else args.platform_feedrate
    args.platform_travel_feedrate = 6000 if args.platform_travel_feedrate is None else args.platform_travel_feedrate
    args.platform_extrusion_scale = 0.85 if args.platform_extrusion_scale is None else args.platform_extrusion_scale
    args.bed_leveling_margin = 5.0 if args.bed_leveling_margin is None else args.bed_leveling_margin
    if args.bed_leveling_margin < 0:
        raise ValueError("bed_leveling_margin must be >= 0")
    return args


def require_arg(value: object, name: str) -> None:
    if value is None:
        raise SystemExit(f"error: {name} is required unless --config is used")


def resolve_json_path(value: object, base_dir: Path, name: str) -> Path:
    if value is None:
        raise SystemExit(f"error: config is missing required field: {name}")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_int(value: object) -> int:
    return int(value)


def ensure_inputs(args: argparse.Namespace) -> None:
    paths = [(args.obj, "OBJ"), (args.gcode, "G-code")]
    if not args.minimal_template:
        paths.insert(0, (args.template, "template"))
    for attr, label in [
        ("plate_png", "plate PNG"),
        ("plate_small_png", "small plate PNG"),
        ("plate_no_light_png", "no-light plate PNG"),
        ("top_png", "top PNG"),
        ("pick_png", "pick PNG"),
    ]:
        path = getattr(args, attr, None)
        if path is not None:
            paths.append((path, label))
    for path, label in paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not args.minimal_template and not args.template.is_dir():
        raise NotADirectoryError(f"template is not a directory: {args.template}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output folder already exists: {args.output} (use --overwrite)")


def parse_center(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--center must be formatted as X,Y")
    return float(parts[0]), float(parts[1])


def parse_obj(path: Path) -> ObjMesh:
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def parse_index(token: str) -> int:
        raw = token.split("/")[0]
        idx = int(raw)
        if idx < 0:
            return len(vertices) + idx
        return idx - 1

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
            elif fields[0] == "f" and len(fields) >= 4:
                face = [parse_index(token) for token in fields[1:]]
                for i in range(1, len(face) - 1):
                    triangles.append((face[0], face[i], face[i + 1]))

    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    if not triangles:
        raise ValueError(f"OBJ contains vertices but no faces/triangles: {path}")
    return ObjMesh(vertices=vertices, triangles=triangles)


def bbox_from_vertices(vertices: Iterable[tuple[float, float, float]]) -> BBox:
    xs: list[float] = []
    ys: list[float] = []
    for x, y, _z in vertices:
        xs.append(x)
        ys.append(y)
    return BBox(min(xs), min(ys), max(xs), max(ys))


def convert_mesh_to_z_up(mesh: ObjMesh, obj_up: str | None) -> ObjMesh:
    if obj_up in (None, "z"):
        return mesh
    if obj_up not in {"x", "y"}:
        raise ValueError("obj_up must be 'z', 'y', or 'x'")

    if obj_up == "y":
        # OBJ Y-up -> Bambu/3MF Z-up. This is a +90 degree rotation around X:
        # (x, y, z) -> (x, -z, y). Triangle winding is flipped to preserve normals.
        vertices = [(x, -z, y) for x, y, z in mesh.vertices]
        triangles = [(a, c, b) for a, b, c in mesh.triangles]
    else:
        # OBJ X-up -> Bambu/3MF Z-up with a cyclic axis remap:
        # (x, y, z) -> (y, z, x). This preserves handedness and triangle winding.
        vertices = [(y, z, x) for x, y, z in mesh.vertices]
        triangles = list(mesh.triangles)
    return ObjMesh(vertices=vertices, triangles=triangles)


def translate_mesh_xy(mesh: ObjMesh, dx: float, dy: float) -> ObjMesh:
    vertices = [(x + dx, y + dy, z) for x, y, z in mesh.vertices]
    return ObjMesh(vertices=vertices, triangles=list(mesh.triangles))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_gcode_header_values(gcode_text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in gcode_text.splitlines()[:500]:
        match = HEADER_EQUALS_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        values[key] = value
    return values


def template_gcode_header_values(args: argparse.Namespace) -> dict[str, str]:
    if args.minimal_template or args.template is None:
        return dict(DEFAULT_GCODE_HEADER_VALUES)
    template_gcode = args.template / "Metadata" / "plate_1.gcode"
    if not template_gcode.exists():
        return dict(DEFAULT_GCODE_HEADER_VALUES)
    values = dict(DEFAULT_GCODE_HEADER_VALUES)
    values.update(extract_gcode_header_values(read_text(template_gcode)))
    values.update(layout_gcode_header_values(args.material_layout))
    return values


def layout_gcode_header_values(material_layout: str) -> dict[str, str]:
    layout = MATERIAL_LAYOUTS[material_layout]
    colors = layout["colors"]
    types = layout["types"]
    return {
        "filament_map": layout["filament_map"],
        "filament_map_2": layout["filament_map_2"],
        "filament_nozzle_map": layout["filament_nozzle_map"],
        "filament_volume_map": layout["filament_volume_map"],
        "filament_colour": ";".join(colors),
        "filament_type": ";".join(types),
    }


def effective_layout_values(material_layout: str, sequence: list[int]) -> dict[str, str | list[str]]:
    layout = MATERIAL_LAYOUTS[material_layout]
    used_indices = sorted({max(0, item - 1) for item in sequence})
    if not used_indices:
        used_indices = list(range(len(layout["colors"])))

    def select_csv(key: str) -> list[str]:
        values = str(layout[key]).split(",")
        return values

    def select_list(key: str) -> list[str]:
        return list(layout[key])

    filament_map = select_csv("filament_map")
    filament_map_2 = select_csv("filament_map_2")
    filament_nozzle_map = select_csv("filament_nozzle_map")
    filament_volume_map = select_csv("filament_volume_map")
    colors = select_list("colors")
    types = select_list("types")
    group_ids = select_list("group_ids")
    return {
        "filament_map": ",".join(filament_map),
        "filament_maps_xml": " ".join(filament_map),
        "filament_map_2": ",".join(filament_map_2),
        "filament_nozzle_map": ",".join(filament_nozzle_map),
        "filament_volume_map": ",".join(filament_volume_map),
        "filament_volume_maps_xml": " ".join(filament_volume_map),
        "filament_colour": ";".join(colors),
        "filament_type": ";".join(types),
        "colors": colors,
        "types": types,
        "group_ids": group_ids,
        "used_indices": used_indices,
    }


def ensure_required_gcode_header(gcode_text: str, required_values: dict[str, str]) -> str:
    existing = extract_gcode_header_values(gcode_text)
    missing_lines = []
    for key in REQUIRED_GCODE_HEADER_KEYS:
        if key not in existing and key in required_values:
            missing_lines.append(f"; {key} = {required_values[key]}")

    if not missing_lines:
        return gcode_text

    header = [
        "; HEADER_BLOCK_START",
        "; BambuStudio 02.05.00.66",
        "; generated_bambu_package_header: 1",
        "; HEADER_BLOCK_END",
        "",
        "; CONFIG_BLOCK_START",
        *missing_lines,
        "; CONFIG_BLOCK_END",
        "",
    ]
    return "\n".join(header) + gcode_text


def extract_template_gcode_header_prefix(template_dir: Path | None) -> str | None:
    if template_dir is None:
        return None
    template_gcode = template_dir / "Metadata" / "plate_1.gcode"
    if not template_gcode.exists():
        return None

    lines = read_text(template_gcode).splitlines()
    prefix: list[str] = []
    saw_header = False
    saw_config = False
    for line in lines:
        prefix.append(line)
        stripped = line.strip()
        if stripped == "; HEADER_BLOCK_START":
            saw_header = True
        elif stripped == "; CONFIG_BLOCK_START":
            saw_config = True
        elif stripped == "; CONFIG_BLOCK_END":
            if saw_header and saw_config:
                return "\n".join(prefix).rstrip() + "\n\n"
            return None
    return None


def gcode_has_config_header(gcode_text: str) -> bool:
    scan = gcode_text[:20000]
    return "; HEADER_BLOCK_START" in scan and "; CONFIG_BLOCK_START" in scan


def ensure_bambu_gcode_header(
    gcode_text: str,
    required_values: dict[str, str],
    template_dir: Path | None,
) -> str:
    if gcode_has_config_header(gcode_text):
        return ensure_required_gcode_header(gcode_text, required_values)

    template_prefix = extract_template_gcode_header_prefix(template_dir)
    if template_prefix is not None:
        return template_prefix + gcode_text.lstrip()

    return ensure_required_gcode_header(gcode_text, required_values)


def enforce_gcode_header_layout(gcode_text: str, material_layout: str) -> str:
    replacements = layout_gcode_header_values(material_layout)
    return replace_gcode_header_values(gcode_text, replacements)


def enforce_gcode_header_effective_layout(gcode_text: str, material_layout: str, sequence: list[int]) -> str:
    effective = effective_layout_values(material_layout, sequence)
    replacements = {
        "filament_map": str(effective["filament_map"]),
        "filament_map_2": str(effective["filament_map_2"]),
        "filament_nozzle_map": str(effective["filament_nozzle_map"]),
        "filament_volume_map": str(effective["filament_volume_map"]),
        "filament_colour": str(effective["filament_colour"]),
        "filament_colour_type": ";".join("1" for _ in str(effective["filament_colour"]).split(";")),
        "filament_settings_id": ";".join(
            '"Bambu PLA Basic @BBL H2D"' for _ in str(effective["filament_colour"]).split(";")
        ),
        "filament_type": str(effective["filament_type"]),
        "activate_air_filtration": ",".join("0" for _ in str(effective["filament_colour"]).split(";")),
    }
    return replace_gcode_header_values(gcode_text, replacements)


def enforce_gcode_header_preview_settings(gcode_text: str, args: argparse.Namespace) -> str:
    if not bool(args.preview_spiral_mode):
        return gcode_text
    return replace_gcode_header_values(gcode_text, {"spiral_mode": "1"})


def replace_gcode_header_values(gcode_text: str, replacements: dict[str, str]) -> str:
    lines = gcode_text.splitlines()
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        match = HEADER_EQUALS_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        if key in replacements:
            lines[idx] = f"; {key} = {replacements[key]}"
            seen.add(key)

    missing = [key for key in replacements if key not in seen]
    if missing:
        insert = [f"; {key} = {replacements[key]}" for key in missing]
        insert_at = None
        for idx, line in enumerate(lines):
            if line.strip() == "; CONFIG_BLOCK_START":
                insert_at = idx + 1
                break
        if insert_at is None:
            lines = insert + [""] + lines
        else:
            lines[insert_at:insert_at] = insert

    return "\n".join(lines) + ("\n" if gcode_text.endswith("\n") else "")


def first_normal_tool(gcode_text: str) -> int | None:
    for line in gcode_text.splitlines():
        match = TOOL_RE.match(line)
        if match:
            tool_id = int(match.group(1))
            if tool_id in (0, 1):
                return tool_id
    return None


def has_real_toolchanges(gcode_text: str) -> bool:
    return NOZZLE_CHANGE_RE.search(gcode_text) is not None


def rewrite_single_tool_gcode_start(gcode_text: str, first_filament: int | None) -> tuple[str, list[str]]:
    """Make single-tool G-code agree with first_filament.

    This deliberately rewrites only T0/T1-style single-tool jobs. If the file has
    real NOZZLE_CHANGE_START blocks, the tool order is a slicer decision and this
    function only reports mismatches.
    """
    if first_filament is None:
        return gcode_text, []
    target_tool = first_filament - 1
    messages: list[str] = []
    current_first = first_normal_tool(gcode_text)

    if current_first == target_tool:
        messages.append(f"G-code first tool already matches first_filament={first_filament} (T{target_tool}).")
        return gcode_text, messages

    if has_real_toolchanges(gcode_text):
        messages.append(
            "WARNING: first_filament does not match first T command, but G-code has real NOZZLE_CHANGE_START blocks; "
            "left tool-change order unchanged."
        )
        return gcode_text, messages

    source_tool = 1 - target_tool
    lines = gcode_text.splitlines()
    changed = 0

    for index, line in enumerate(lines):
        # For single-tool G-code, rewrite all normal T0/T1 and associated Bambu
        # selector commands. Physical calibration slot lines such as M620.17
        # intentionally remain exactly as slicer emitted; only the active G383
        # logical filament marker is allowed to follow the selected first tool.
        # Special T1001/T65535/T65279 are intentionally ignored.
        new_line = replace_tool_line(line, source_tool, target_tool)
        if new_line != line:
            lines[index] = new_line
            changed += 1

    messages.append(
        f"Rewrote single-tool G-code from T{source_tool}/filament {source_tool + 1} "
        f"to T{target_tool}/filament {first_filament}; changed {changed} lines."
    )
    return "\n".join(lines) + ("\n" if gcode_text.endswith("\n") else ""), messages


def replace_tool_line(line: str, source_tool: int, target_tool: int) -> str:
    source = str(source_tool)
    target = str(target_tool)

    match = TOOL_SUB_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}T{target}{match.group(3)}"

    match = M620_START_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M620 S{target}A{match.group(3)}"

    match = M621_END_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M621 S{target}A{match.group(3)}"

    match = M620_AIR_RE.match(line)
    if match and match.group(2) == source:
        return f"{match.group(1)}M620.6 I{target}{match.group(3)}"

    match = M104_TOOL_RE.match(line)
    if match and match.group(3) == source:
        return f"{match.group(1)}M104{match.group(2)}T{target}{match.group(4)}"

    match = G151_TOOL_RE.match(line)
    if match and match.group(3) == source:
        return f"{match.group(1)}G151{match.group(2)}{target}{match.group(4)}"

    match = G383_ACTIVE_L_RE.match(line)
    if match and ("G383.3" in line or "O1" in line) and match.group(2) == source:
        return f"{match.group(1)}{target}{match.group(3)}"

    return line

def parse_gcode_metadata(gcode_text: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for line in gcode_text.splitlines()[:400]:
        if line.startswith("; model printing time:"):
            metadata["prediction"] = parse_total_estimated_seconds(line)
        match = HEADER_LIST_RE.match(line) or HEADER_EQUALS_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        metadata[key] = value
    return metadata


def parse_total_estimated_seconds(line: str) -> int | None:
    match = re.search(r"total estimated time:\s*([^;]+)$", line)
    if not match:
        return None
    return parse_duration_seconds(match.group(1).strip())


def parse_duration_seconds(value: str) -> int | None:
    total = 0
    for amount, unit in re.findall(r"(\d+)\s*([hms])", value):
        number = int(amount)
        if unit == "h":
            total += number * 3600
        elif unit == "m":
            total += number * 60
        elif unit == "s":
            total += number
    return total or None


def parse_number_list(value: object, default: list[float]) -> list[float]:
    if not isinstance(value, str):
        return default
    parts = [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
    try:
        return [float(part) for part in parts]
    except ValueError:
        return default


def parse_string_list(value: object, default: list[str]) -> list[str]:
    if not isinstance(value, str):
        return default
    parts = [part.strip().strip('"') for part in value.split(";") if part.strip()]
    return parts or default


def derive_filament_sequence(gcode_text: str) -> list[int]:
    sequence: list[int] = []
    first_tool: int | None = None

    for line in gcode_text.splitlines():
        change = NOZZLE_CHANGE_RE.search(line)
        if change:
            sequence.append(int(change.group(2)) + 1)
            continue
        tool = TOOL_RE.match(line)
        if tool and first_tool is None:
            tool_id = int(tool.group(1))
            if tool_id < 1000:
                first_tool = tool_id

    if first_tool is not None:
        return [first_tool + 1] + sequence
    return sequence or [1]


def apply_first_filament_override(sequence: list[int], first_filament: int | None) -> list[int]:
    if first_filament is None:
        return sequence
    if not sequence:
        return [first_filament]
    overridden = list(sequence)
    overridden[0] = first_filament
    return overridden


def maybe_register_all_layout_filaments(sequence: list[int], material_layout: str, enabled: bool) -> list[int]:
    return sequence


def describe_used_filaments(sequence: list[int]) -> str:
    used = sorted(set(sequence))
    if used == [1]:
        return "single-tool package: only filament 1 / T0 will be registered as used"
    if used == [2]:
        return "single-tool package: only filament 2 / T1 will be registered as used"
    return "multi-tool package: filaments " + ",".join(str(item) for item in used) + " will be registered as used"


def derive_layer_filament_ranges(gcode_text: str) -> dict[tuple[int, ...], list[int]]:
    layer_tools: dict[int, set[int]] = {}
    current_layer: int | None = None
    current_tool: int | None = None

    for line in gcode_text.splitlines():
        assembled = ASSEMBLED_SNIPPET_RE.search(line)
        if assembled:
            current_layer = int(assembled.group(1))
            current_tool = int(assembled.group(2))
            if current_tool < 1000:
                layer_tools.setdefault(current_layer, set()).add(current_tool)
            continue

        layer = LAYER_RE.search(line)
        if layer:
            current_layer = int(layer.group(1)) - 1
            layer_tools.setdefault(current_layer, set())
            if current_tool is not None:
                layer_tools[current_layer].add(current_tool)
            continue

        tool = TOOL_RE.match(line)
        if tool:
            tool_id = int(tool.group(1))
            if tool_id < 1000:
                current_tool = tool_id
                if current_layer is not None:
                    layer_tools.setdefault(current_layer, set()).add(current_tool)

    grouped: dict[tuple[int, ...], list[int]] = {}
    for layer_id in sorted(layer_tools):
        tools = tuple(sorted(layer_tools[layer_id]))
        if tools:
            grouped.setdefault(tools, []).append(layer_id)
    return grouped


def infer_total_layer_count(gcode_text: str) -> int | None:
    max_assembled_layer: int | None = None
    for line in gcode_text.splitlines():
        layer = LAYER_RE.search(line)
        if layer:
            return int(layer.group(2))
        assembled = ASSEMBLED_SNIPPET_RE.search(line)
        if assembled:
            max_assembled_layer = max(max_assembled_layer or 0, int(assembled.group(1)))
    if max_assembled_layer is not None:
        return max_assembled_layer + 1
    return None


def infer_max_z_height(gcode_text: str) -> float | None:
    max_z: float | None = None
    for match in Z_WORD_RE.finditer(print_body_for_layer_inference(gcode_text)):
        z = float(match.group(1))
        max_z = z if max_z is None else max(max_z, z)
    return max_z


def replace_gcode_header_colon_values(gcode_text: str, replacements: dict[str, str]) -> str:
    lines = gcode_text.splitlines()
    for idx, line in enumerate(lines):
        match = HEADER_LIST_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        if key in replacements:
            lines[idx] = f"; {key}: {replacements[key]}"
    return "\n".join(lines) + ("\n" if gcode_text.endswith("\n") else "")


def update_gcode_header_geometry_summary(gcode_text: str) -> str:
    replacements: dict[str, str] = {}
    total_layer_count = infer_total_layer_count(gcode_text)
    max_z = infer_max_z_height(gcode_text)
    if total_layer_count is not None:
        replacements["total layer number"] = str(total_layer_count)
    if max_z is not None:
        replacements["max_z_height"] = f"{max_z:.2f}"
    if not replacements:
        return gcode_text
    return replace_gcode_header_colon_values(gcode_text, replacements)


def normalize_single_tool_layer_ranges(
    layer_ranges: dict[tuple[int, ...], list[int]],
    sequence: list[int],
    total_layer_count: int | None,
) -> dict[tuple[int, ...], list[int]]:
    if len(set(sequence)) != 1 or total_layer_count is None or total_layer_count <= 0:
        return layer_ranges
    tool_id = sequence[0] - 1
    return {(tool_id,): list(range(total_layer_count))}


def compress_ranges(values: list[int]) -> str:
    if not values:
        return ""
    values = sorted(set(values))
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start} {prev}")
        start = prev = value
    ranges.append(f"{start} {prev}")
    return ",".join(ranges)


def safe_replace_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(dst)
        resolved = dst.resolve()
        cwd = Path.cwd().resolve()
        if cwd not in resolved.parents and resolved != cwd:
            raise ValueError(f"refusing to overwrite outside current workspace: {dst}")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_output_folder(args: argparse.Namespace) -> None:
    if args.minimal_template:
        create_minimal_template(args.output, args.overwrite)
    else:
        safe_replace_tree(args.template, args.output, args.overwrite)


def create_minimal_template(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(output_dir)
        resolved = output_dir.resolve()
        cwd = Path.cwd().resolve()
        if cwd not in resolved.parents and resolved != cwd:
            raise ValueError(f"refusing to overwrite outside current workspace: {output_dir}")
        shutil.rmtree(output_dir)

    (output_dir / "3D").mkdir(parents=True)
    (output_dir / "Metadata" / "_rels").mkdir(parents=True)
    (output_dir / "_rels").mkdir(parents=True)
    (output_dir / "[Content_Types].xml").write_text(CONTENT_TYPES_XML, encoding="utf-8")
    (output_dir / "_rels" / ".rels").write_text(RELS_XML, encoding="utf-8")
    (output_dir / "Metadata" / "model_settings.config").write_text(MODEL_SETTINGS_XML, encoding="utf-8")
    (output_dir / "Metadata" / "_rels" / "model_settings.config.rels").write_text(MODEL_SETTINGS_RELS_XML, encoding="utf-8")
    (output_dir / "Metadata" / "cut_information.xml").write_text(CUT_INFORMATION_XML, encoding="utf-8")
    (output_dir / "Metadata" / "slice_info.config").write_text(SLICE_INFO_XML, encoding="utf-8")
    (output_dir / "Metadata" / "plate_1.json").write_text(
        json.dumps(
            {
                "bbox_all": [0, 0, 0, 0],
                "bbox_objects": [],
                "bed_type": "textured_plate",
                "filament_colors": ["#000000", "#FFFFFF"],
                "filament_ids": [0, 1],
                "first_extruder": 0,
                "first_layer_time": 0,
                "is_seq_print": False,
                "nozzle_diameter": 0.4,
                "version": 2,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    for name in ["plate_1.png", "plate_1_small.png", "plate_no_light_1.png", "top_1.png", "pick_1.png"]:
        (output_dir / "Metadata" / name).write_bytes(EMPTY_PNG)


def write_gcode_and_md5(
    gcode_path: Path,
    output_dir: Path,
    required_header_values: dict[str, str],
    first_filament: int | None,
    material_layout: str,
    args: argparse.Namespace,
    obj_bbox: BBox,
) -> str:
    dest = output_dir / "Metadata" / "plate_1.gcode"
    gcode_text = read_text(gcode_path)
    gcode_text = ensure_bambu_gcode_header(gcode_text, required_header_values, args.template)
    gcode_text = enforce_gcode_header_layout(gcode_text, material_layout)
    gcode_text, sanity_messages = rewrite_single_tool_gcode_start(gcode_text, first_filament)
    for message in sanity_messages:
        print(message)
    gcode_text, skipped_sections = maybe_skip_gcode_sections(gcode_text, args)
    if skipped_sections:
        print(f"skipped G-code sections: {', '.join(skipped_sections)}")
    gcode_text, feedrate_report = rewrite_gcode_feedrates(gcode_text, args)
    if feedrate_report["changed_count"]:
        print(
            "feedrate rewrite: "
            f"{feedrate_report['changed_count']} G0/G1 F values changed "
            f"(first_layer={feedrate_report['first_layer_changed']}, "
            f"remaining={feedrate_report['remaining_layer_changed']})"
        )
    gcode_text, platform_inserted = maybe_insert_planar_platform(gcode_text, args, obj_bbox)
    if platform_inserted:
        print("inserted planar platform layer")
    gcode_text = update_gcode_header_geometry_summary(gcode_text)
    data = gcode_text.encode("utf-8")
    dest.write_bytes(data)
    digest = hashlib.md5(data).hexdigest()
    (output_dir / "Metadata" / "plate_1.gcode.md5").write_text(digest, encoding="ascii")
    return gcode_text


def rewrite_generated_gcode_effective_layout(output_dir: Path, material_layout: str, sequence: list[int]) -> str:
    path = output_dir / "Metadata" / "plate_1.gcode"
    gcode_text = read_text(path)
    gcode_text = enforce_gcode_header_effective_layout(gcode_text, material_layout, sequence)
    return write_generated_gcode_text(output_dir, gcode_text)


def write_generated_gcode_text(output_dir: Path, gcode_text: str) -> str:
    path = output_dir / "Metadata" / "plate_1.gcode"
    data = gcode_text.encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.md5(data).hexdigest()
    (output_dir / "Metadata" / "plate_1.gcode.md5").write_text(digest, encoding="ascii")
    return gcode_text


def rewrite_bed_leveling_area(gcode_text: str, margin: float, fallback_bbox: BBox) -> tuple[str, BBox, int]:
    """Match Bambu's G29 probe rectangle to the generated model projection."""
    projected_bounds = model_print_projection_bounds(gcode_text) or fallback_bbox
    bounds = expanded_bbox(projected_bounds, margin)
    values = {
        "X": bounds.min_x,
        "Y": bounds.min_y,
        "I": bounds.width,
        "J": bounds.height,
    }
    replacements = 0

    def replace_area_line(match: re.Match[str]) -> str:
        nonlocal replacements
        rewritten = match.group(0)
        changed = True
        for word, value in values.items():
            rewritten, count = re.subn(
                rf"\b{word}-?\d+(?:\.\d+)?",
                f"{word}{format_float(value)}",
                rewritten,
                count=1,
            )
            changed = changed and count == 1
        replacements += int(changed)
        return rewritten if changed else match.group(0)

    rewritten = re.sub(
        r"^\s*G29\s+A[12]\s+O[^\r\n]*",
        replace_area_line,
        gcode_text,
        flags=re.MULTILINE,
    )
    return rewritten, bounds, replacements


def model_print_projection_bounds(gcode_text: str) -> BBox | None:
    """Return the XY projection of deposited assembled-model moves only."""
    active_main: int | None = None
    absolute_xy = True
    x: float | None = None
    y: float | None = None
    bounds: BBox | None = None

    for line in StringIO(print_body_for_layer_inference(gcode_text)):
        line = line.rstrip("\r\n")
        start = ASSEMBLED_SNIPPET_START_RE.match(line)
        if start:
            active_main = int(start.group(1))
            continue
        if ASSEMBLED_SNIPPET_END_RE.match(line):
            active_main = None
            continue
        stripped = line.lstrip()
        if stripped == "G90":
            absolute_xy = True
            continue
        if stripped == "G91":
            absolute_xy = False
            continue
        if not stripped.startswith(("G0", "G1")):
            continue

        values = {name: float(value) for name, value in GCODE_XY_E_RE.findall(line)}
        old_x, old_y = x, y
        if absolute_xy:
            x = values.get("X", x)
            y = values.get("Y", y)
        else:
            x = (x or 0.0) + values.get("X", 0.0)
            y = (y or 0.0) + values.get("Y", 0.0)
        if active_main is None or not stripped.startswith("G1"):
            continue
        if values.get("E", 0.0) <= 0 or None in (x, y, old_x, old_y):
            continue

        move_bounds = BBox(min(old_x, x), min(old_y, y), max(old_x, x), max(old_y, y))
        bounds = move_bounds if bounds is None else BBox(
            min(bounds.min_x, move_bounds.min_x),
            min(bounds.min_y, move_bounds.min_y),
            max(bounds.max_x, move_bounds.max_x),
            max(bounds.max_y, move_bounds.max_y),
        )
    return bounds


def update_project_settings_bed_leveling_area(output_dir: Path, bounds: BBox) -> None:
    path = output_dir / "Metadata" / "project_settings.config"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    start_gcode = data.get("machine_start_gcode")
    if not isinstance(start_gcode, str):
        return
    rewritten, _unused, replacements = rewrite_bed_leveling_area(start_gcode, 0.0, bounds)
    if replacements:
        data["machine_start_gcode"] = rewritten
        path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def inject_preview_object_ids(gcode_text: str, object_id: int) -> str:
    if "; OBJECT_ID:" in gcode_text:
        return gcode_text

    lines = gcode_text.splitlines()
    output: list[str] = []
    pending_object_id = False
    for line in lines:
        output.append(line)
        if line == "; CHANGE_LAYER":
            pending_object_id = True
            continue
        if pending_object_id and line.startswith("; FEATURE:"):
            output.insert(len(output) - 1, f"; OBJECT_ID: {object_id}")
            pending_object_id = False
            continue
        if pending_object_id and line.startswith("; ASSEMBLED_SNIPPET_START"):
            output.insert(len(output) - 1, f"; OBJECT_ID: {object_id}")
            pending_object_id = False
            continue
    return "\n".join(output) + ("\n" if gcode_text.endswith("\n") else "")


def maybe_insert_planar_platform(gcode_text: str, args: argparse.Namespace, obj_bbox: BBox) -> tuple[str, bool]:
    if not bool(args.enable_planar_platform):
        return gcode_text, False

    if "ASSEMBLED_PLANAR_PLATFORM_START" in gcode_text:
        print("skipped generator planar platform: assembled platform already exists")
        return gcode_text, False

    z = first_print_z(gcode_text)
    if z is None:
        raise ValueError("Cannot insert planar platform: no printable Z move found in G-code")

    if bool(args.platform_lift_model_z):
        gcode_text = lift_model_z_for_platform(gcode_text, float(args.platform_layer_height))

    platform = build_planar_platform_gcode(args, obj_bbox, z)
    start_marker = ";===== assembled layer path body start ====="
    marker_index = gcode_text.find(start_marker)
    if marker_index != -1:
        return gcode_text[:marker_index] + platform + gcode_text[marker_index:], True

    first_print_index = first_print_move_index(gcode_text)
    if first_print_index is None:
        raise ValueError("Cannot insert planar platform: no insertion point found in G-code")
    return gcode_text[:first_print_index] + platform + gcode_text[first_print_index:], True


def lift_model_z_for_platform(gcode_text: str, lift: float) -> str:
    if lift <= 0:
        return gcode_text

    start_marker = ";===== assembled layer path body start ====="
    end_marker = ";===== assembled layer path body end ====="
    start = gcode_text.find(start_marker)
    end = gcode_text.find(end_marker)

    if start != -1 and end != -1 and end > start:
        body_start = gcode_text.find("\n", start)
        body_start = start if body_start == -1 else body_start + 1
        lifted_body = lift_z_words(gcode_text[body_start:end], lift)
        return (
            gcode_text[:body_start]
            + f"; PLANAR_PLATFORM_MODEL_Z_LIFT {format_float(lift)}\n"
            + lifted_body
            + gcode_text[end:]
        )

    first_print_index = first_print_move_index(gcode_text)
    if first_print_index is None:
        raise ValueError("Cannot lift model Z for platform: no first print move found")
    return (
        gcode_text[:first_print_index]
        + f"; PLANAR_PLATFORM_MODEL_Z_LIFT {format_float(lift)}\n"
        + lift_z_words(gcode_text[first_print_index:], lift)
    )


def lift_z_words(gcode_text: str, lift: float) -> str:
    def replace(match: re.Match[str]) -> str:
        return "Z" + format_float(float(match.group(1)) + lift)

    return Z_WORD_RE.sub(replace, gcode_text)


def build_planar_platform_gcode(args: argparse.Namespace, obj_bbox: BBox, z: float) -> str:
    margin = float(args.platform_margin)
    spacing = float(args.platform_line_spacing)
    line_width = float(args.platform_line_width)
    layer_height = float(args.platform_layer_height)
    feedrate = float(args.platform_feedrate)
    travel_feedrate = float(args.platform_travel_feedrate)
    extrusion_scale = float(args.platform_extrusion_scale)
    filament_diameter = 1.75
    filament_area = math.pi * (filament_diameter / 2.0) ** 2

    if spacing <= 0:
        raise ValueError("platform_line_spacing must be > 0")
    if line_width <= 0 or layer_height <= 0:
        raise ValueError("platform_line_width and platform_layer_height must be > 0")

    min_x = obj_bbox.min_x - margin
    max_x = obj_bbox.max_x + margin
    min_y = obj_bbox.min_y - margin
    max_y = obj_bbox.max_y + margin
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Cannot insert planar platform: invalid object bbox")

    line_len = max_x - min_x
    line_e = line_len * line_width * layer_height / filament_area * extrusion_scale
    step_e = spacing * line_width * layer_height / filament_area * extrusion_scale
    y_values: list[float] = []
    y = min_y
    while y <= max_y + 1e-6:
        y_values.append(min(y, max_y))
        y += spacing
    if y_values[-1] < max_y:
        y_values.append(max_y)

    lines = [
        "; PLANAR_PLATFORM_START",
        f"; bbox={format_float(min_x)},{format_float(min_y)},{format_float(max_x)},{format_float(max_y)} z={format_float(z)}",
        "G90",
        "M83",
        f"G1 X{format_float(min_x)} Y{format_float(y_values[0])} F{format_float(travel_feedrate)}",
        f"G1 Z{format_float(z)} F1200",
    ]

    current_x = min_x
    for index, row_y in enumerate(y_values):
        next_x = max_x if current_x == min_x else min_x
        lines.append(f"G1 X{format_float(next_x)} E{format_float(line_e)} F{format_float(feedrate)}")
        current_x = next_x
        if index != len(y_values) - 1:
            next_y = y_values[index + 1]
            lines.append(f"G1 Y{format_float(next_y)} E{format_float(step_e)} F{format_float(feedrate)}")

    lines.extend(["M400", "; PLANAR_PLATFORM_END", ""])
    return "\n".join(lines)


def first_print_z(gcode_text: str) -> float | None:
    scan_text = print_body_for_layer_inference(gcode_text)
    for line in scan_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            continue
        z_match = Z_WORD_RE.search(line)
        e_match = E_WORD_RE.search(line)
        if z_match and e_match and float(e_match.group(1)) >= 0:
            return float(z_match.group(1))
    return None


def first_print_move_index(gcode_text: str) -> int | None:
    offset = 0
    for line in gcode_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("G0", "G1")) and Z_WORD_RE.search(line) and E_WORD_RE.search(line):
            return offset
        offset += len(line)
    return None


def maybe_skip_gcode_sections(gcode_text: str, args: argparse.Namespace) -> tuple[str, list[str]]:
    skipped: list[str] = []
    output = gcode_text
    for section_name, (start_marker, end_marker, flag_name) in GCODE_SECTIONS.items():
        if not bool(getattr(args, flag_name, False)):
            continue
        output, status = remove_marked_section(output, start_marker, end_marker, section_name)
        skipped.append(f"{section_name}:{status}")
    return output, skipped


def remove_marked_section(gcode_text: str, start_marker: str, end_marker: str, section_name: str) -> tuple[str, str]:
    lines = gcode_text.splitlines()
    keep: list[str] = []
    skipping = False
    found_start = False
    found_end = False

    for line in lines:
        if not skipping and line.strip() == start_marker:
            found_start = True
            skipping = True
            keep.append(f"; GENERATOR_SKIPPED_START {section_name}")
            continue
        if skipping and line.strip() == end_marker:
            found_end = True
            skipping = False
            keep.append(f"; GENERATOR_SKIPPED_END {section_name}")
            continue
        if not skipping:
            keep.append(line)

    if not found_start:
        already_skipped = (
            f"; GENERATOR_SKIPPED_START {section_name}" in gcode_text
            or f"; ASSEMBLER_SKIPPED_START {section_name}" in gcode_text
        )
        return gcode_text, "already_skipped" if already_skipped else "not_found"
    if not found_end:
        raise ValueError(f"Cannot skip G-code section {section_name}: end marker not found")

    return "\n".join(keep) + ("\n" if gcode_text.endswith("\n") else ""), "removed"


def rewrite_gcode_feedrates(gcode_text: str, args: argparse.Namespace) -> tuple[str, dict[str, int]]:
    first_override = args.first_layer_feedrate
    remaining_override = args.remaining_layer_feedrate
    first_multiplier = args.first_layer_feedrate_multiplier
    remaining_multiplier = args.remaining_layer_feedrate_multiplier
    if all(value is None for value in [first_override, remaining_override, first_multiplier, remaining_multiplier]):
        return gcode_text, {"changed_count": 0, "first_layer_changed": 0, "remaining_layer_changed": 0}

    first_layer_count = int(args.first_layer_count)
    inferred_z_threshold = infer_first_layers_z_threshold(gcode_text, first_layer_count)
    current_layer: int | None = None
    layer_markers_seen = False
    first_z: float | None = None
    current_z: float | None = None
    changed_count = 0
    first_changed = 0
    remaining_changed = 0
    output_lines: list[str] = []

    for line in gcode_text.splitlines():
        layer_match = LAYER_RE.search(line)
        if layer_match:
            layer_markers_seen = True
            current_layer = int(layer_match.group(1))
            output_lines.append(line)
            continue

        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            output_lines.append(line)
            continue

        z_match = Z_WORD_RE.search(line)
        if z_match:
            current_z = float(z_match.group(1))
            if first_z is None:
                first_z = current_z

        if layer_markers_seen:
            is_first_layer = current_layer is None or current_layer < first_layer_count
        elif inferred_z_threshold is not None and current_z is not None:
            is_first_layer = current_z <= inferred_z_threshold + 1e-6
        elif first_z is None or current_z is None:
            is_first_layer = True
        else:
            is_first_layer = current_z <= first_z + 1e-6
        override = first_override if is_first_layer else remaining_override
        multiplier = first_multiplier if is_first_layer else remaining_multiplier
        if override is None and multiplier is None:
            output_lines.append(line)
            continue

        changed_line = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed_count, first_changed, remaining_changed, changed_line
            old_value = float(match.group(1))
            new_value = override if override is not None else old_value * multiplier
            assert new_value is not None
            changed_count += 1
            if is_first_layer:
                first_changed += 1
            else:
                remaining_changed += 1
            changed_line = True
            return "F" + format_float(new_value)

        rewritten = F_WORD_RE.sub(replace, line)
        output_lines.append(rewritten if changed_line else line)

    return (
        "\n".join(output_lines) + ("\n" if gcode_text.endswith("\n") else ""),
        {
            "changed_count": changed_count,
            "first_layer_changed": first_changed,
            "remaining_layer_changed": remaining_changed,
        },
    )


def infer_first_layers_z_threshold(gcode_text: str, first_layer_count: int) -> float | None:
    z_values: list[float] = []
    seen_z_keys: set[int] = set()
    current_z: float | None = None
    scan_text = print_body_for_layer_inference(gcode_text)

    for line in scan_text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(("G0", "G1")):
            continue

        z_match = Z_WORD_RE.search(line)
        if z_match:
            current_z = float(z_match.group(1))

        e_match = E_WORD_RE.search(line)
        if current_z is None or e_match is None:
            continue
        if float(e_match.group(1)) <= 0:
            continue
        z_key = round(current_z * 1_000_000)
        if z_key not in seen_z_keys:
            seen_z_keys.add(z_key)
            z_values.append(current_z)

    if not z_values:
        return None
    z_values.sort()
    index = min(max(first_layer_count, 1), len(z_values)) - 1
    return z_values[index]


def print_body_for_layer_inference(gcode_text: str) -> str:
    start_marker = ";===== assembled layer path body start ====="
    end_marker = ";===== assembled layer path body end ====="
    start = gcode_text.find(start_marker)
    end = gcode_text.find(end_marker)
    if start == -1 or end == -1 or end <= start:
        return gcode_text
    return gcode_text[start:end]


def format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def update_preview_pngs(args: argparse.Namespace) -> None:
    replacements = {
        "plate_png": "plate_1.png",
        "plate_small_png": "plate_1_small.png",
        "plate_no_light_png": "plate_no_light_1.png",
        "top_png": "top_1.png",
        "pick_png": "pick_1.png",
    }
    metadata_dir = args.output / "Metadata"

    if args.blank_pngs:
        for dest_name in replacements.values():
            (metadata_dir / dest_name).write_bytes(EMPTY_PNG)

    for attr, dest_name in replacements.items():
        src = getattr(args, attr, None)
        if src is not None:
            shutil.copy2(src, metadata_dir / dest_name)


def update_model_settings(output_dir: Path, material_layout: str, sequence: list[int]) -> None:
    path = output_dir / "Metadata" / "model_settings.config"
    tree = ET.parse(path)
    root = tree.getroot()
    plate = root.find("plate")
    if plate is None:
        raise ValueError("model_settings.config is missing <plate>")
    layout = effective_layout_values(material_layout, sequence)
    set_metadata(plate, "filament_maps", layout["filament_maps_xml"])
    set_metadata(plate, "filament_volume_maps", layout["filament_volume_maps_xml"])
    indent_xml(root)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def update_project_settings(output_dir: Path, args: argparse.Namespace, material_layout: str, sequence: list[int]) -> None:
    path = output_dir / "Metadata" / "project_settings.config"
    data = json.loads(path.read_text(encoding="utf-8"))
    effective = effective_layout_values(material_layout, sequence)
    data["filament_map"] = str(effective["filament_map"]).split(",")
    data["filament_map_2"] = str(effective["filament_map_2"]).split(",")
    data["filament_nozzle_map"] = str(effective["filament_nozzle_map"]).split(",")
    data["filament_volume_map"] = str(effective["filament_volume_map"]).split(",")
    data["filament_colour"] = list(MATERIAL_LAYOUTS[material_layout]["colors"][: len(data["filament_map"])])
    data["filament_type"] = list(MATERIAL_LAYOUTS[material_layout]["types"][: len(data["filament_map"])])
    data["filament_map_mode"] = "Manual"
    if bool(args.preview_spiral_mode):
        data["spiral_mode"] = "1"
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    print("project settings: synchronized filament maps")
    if bool(args.preview_spiral_mode):
        print("project settings: set spiral_mode=1 for Bambu Studio logical-layer preview")


def write_filament_sequence(output_dir: Path, plate: str, sequence: list[int]) -> None:
    path = output_dir / "Metadata" / "filament_sequence.json"
    path.write_text(json.dumps({plate: {"sequence": sequence}}, separators=(",", ":")), encoding="utf-8")


def update_plate_json(
    output_dir: Path,
    obj_name: str,
    obj_bbox: BBox,
    gcode_meta: dict[str, object],
    sequence: list[int],
    keep_wipe_tower_metadata: bool,
    material_layout: str,
    args: argparse.Namespace,
) -> None:
    path = output_dir / "Metadata" / "plate_1.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    old_objects = data.get("bbox_objects", [])
    wipe_objects = [item for item in old_objects if item.get("name") == "wipe_tower"] if keep_wipe_tower_metadata else []
    object_id = crc32(obj_name.encode("utf-8")) & 0x7FFFFFFF
    model_object = {
        "area": obj_bbox.area,
        "bbox": obj_bbox.as_list(),
        "id": object_id,
        "layer_height": infer_layer_height(gcode_meta, data.get("bbox_objects", [])),
        "name": obj_name,
    }
    extra_bboxes: list[BBox] = []
    if bool(args.enable_planar_platform):
        platform_bbox = expanded_bbox(obj_bbox, float(args.platform_margin))
        extra_bboxes.append(platform_bbox)
    objects = [model_object] + wipe_objects

    all_boxes = [BBox(*item["bbox"]) for item in objects if "bbox" in item] + extra_bboxes
    data["bbox_objects"] = objects
    data["bbox_all"] = union_bbox(all_boxes).as_list() if all_boxes else obj_bbox.as_list()
    used_filament_ids = sorted({max(0, item - 1) for item in sequence})
    all_colors = MATERIAL_LAYOUTS[material_layout]["colors"]
    data["filament_colors"] = [all_colors[idx] for idx in used_filament_ids if idx < len(all_colors)]
    data["filament_ids"] = used_filament_ids
    data["first_extruder"] = max(0, sequence[0] - 1) if sequence else 0
    data["nozzle_diameter"] = parse_number_list(gcode_meta.get("nozzle_diameter"), [data.get("nozzle_diameter", 0.4)])[0]
    if "curr_bed_type" in gcode_meta:
        data["bed_type"] = normalize_bed_type(str(gcode_meta["curr_bed_type"]))

    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def expanded_bbox(bbox: BBox, margin: float) -> BBox:
    return BBox(bbox.min_x - margin, bbox.min_y - margin, bbox.max_x + margin, bbox.max_y + margin)


def infer_layer_height(gcode_meta: dict[str, object], old_objects: list[dict[str, object]]) -> float:
    if "layer_height" in gcode_meta:
        try:
            return float(str(gcode_meta["layer_height"]))
        except ValueError:
            pass
    for item in old_objects:
        if item.get("name") != "wipe_tower" and "layer_height" in item:
            return float(item["layer_height"])
    return 0.2


def union_bbox(boxes: list[BBox]) -> BBox:
    return BBox(
        min(box.min_x for box in boxes),
        min(box.min_y for box in boxes),
        max(box.max_x for box in boxes),
        max(box.max_y for box in boxes),
    )


def normalize_bed_type(value: str) -> str:
    lowered = value.strip().strip('"').lower()
    if "textured" in lowered:
        return "textured_plate"
    if "cool" in lowered:
        return "cool_plate"
    if "engineering" in lowered:
        return "eng_plate"
    return lowered.replace(" ", "_")


def set_metadata(plate: ET.Element, key: str, value: object) -> None:
    for child in plate.findall("metadata"):
        if child.attrib.get("key") == key:
            child.set("value", str(value))
            return
    ET.SubElement(plate, "metadata", {"key": key, "value": str(value)})


def update_slice_info(
    output_dir: Path,
    obj_name: str,
    gcode_meta: dict[str, object],
    layer_ranges: dict[tuple[int, ...], list[int]],
    material_layout: str,
    sequence: list[int],
) -> None:
    path = output_dir / "Metadata" / "slice_info.config"
    tree = ET.parse(path)
    root = tree.getroot()
    plate = root.find("plate")
    if plate is None:
        raise ValueError("slice_info.config is missing <plate>")

    weights = parse_number_list(gcode_meta.get("total filament weight [g]"), [])
    lengths_mm = parse_number_list(gcode_meta.get("total filament length [mm]"), [])
    layout = effective_layout_values(material_layout, sequence)
    colors = layout["colors"]
    types = layout["types"]
    nozzles = parse_number_list(gcode_meta.get("nozzle_diameter"), [0.4, 0.4])

    if weights:
        set_metadata(plate, "weight", f"{sum(weights):.2f}")
    if gcode_meta.get("prediction"):
        set_metadata(plate, "prediction", gcode_meta["prediction"])
    set_metadata(plate, "nozzle_diameters", ",".join(trim_float(value) for value in nozzles))
    set_metadata(plate, "filament_maps", layout["filament_maps_xml"])
    set_metadata(plate, "limit_filament_maps", " ".join("0" for _ in layout["filament_maps_xml"].split()))

    for object_node in list(plate.findall("object")):
        plate.remove(object_node)
    ET.SubElement(plate, "object", {"identify_id": str(crc32(obj_name.encode()) & 0x7FFFFFFF), "name": obj_name, "skipped": "false"})

    used_filament_indices = sorted({max(0, item - 1) for item in sequence})
    for filament in list(plate.findall("filament")):
        try:
            filament_idx = max(0, int(filament.get("id", "0")) - 1)
        except ValueError:
            filament_idx = -1
        if used_filament_indices and filament_idx not in used_filament_indices:
            plate.remove(filament)

    for row_idx, filament in enumerate(plate.findall("filament")):
        try:
            filament_idx = max(0, int(filament.get("id", str(row_idx + 1))) - 1)
        except ValueError:
            filament_idx = row_idx

        if filament_idx < len(types):
            filament.set("type", types[filament_idx])
        if filament_idx < len(colors):
            filament.set("color", colors[filament_idx])
        if filament_idx < len(lengths_mm):
            filament.set("used_m", f"{lengths_mm[filament_idx] / 1000.0:.2f}")
        if filament_idx < len(weights):
            filament.set("used_g", f"{weights[filament_idx]:.2f}")
        elif row_idx < len(weights):
            filament.set("used_g", f"{weights[row_idx]:.2f}")
        if filament_idx < len(nozzles):
            filament.set("nozzle_diameter", f"{nozzles[filament_idx]:.2f}")
        used_indices = layout["used_indices"]
        if filament_idx in used_indices:
            effective_idx = used_indices.index(filament_idx)
        else:
            effective_idx = row_idx
        group_ids = layout["group_ids"]
        if effective_idx < len(group_ids):
            filament.set("group_id", group_ids[effective_idx])

    old_layer_lists = plate.find("layer_filament_lists")
    if old_layer_lists is not None:
        plate.remove(old_layer_lists)
    if layer_ranges:
        layer_root = ET.SubElement(plate, "layer_filament_lists")
        for tools, layers in sorted(layer_ranges.items()):
            ET.SubElement(
                layer_root,
                "layer_filament_list",
                {
                    "filament_list": " ".join(str(tool) for tool in tools),
                    "layer_ranges": compress_ranges(layers),
                },
            )

    indent_xml(root)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def trim_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_model_xml(output_dir: Path, mesh: ObjMesh, title: str) -> None:
    ns_core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    ns_bambu = "http://schemas.bambulab.com/package/2021"
    ns_prod = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
    ET.register_namespace("", ns_core)
    ET.register_namespace("BambuStudio", ns_bambu)
    ET.register_namespace("p", ns_prod)

    model = ET.Element(
        f"{{{ns_core}}}model",
        {
            "unit": "millimeter",
            "{http://www.w3.org/XML/1998/namespace}lang": "en-US",
            "xmlns:BambuStudio": ns_bambu,
            "xmlns:p": ns_prod,
            "requiredextensions": "p",
        },
    )
    for name, value in [
        ("Application", "BambuStudio-02.05.00.66"),
        ("BambuStudio:3mfVersion", "1"),
        ("CreationDate", date.today().isoformat()),
        ("ModificationDate", date.today().isoformat()),
        ("Title", title),
    ]:
        ET.SubElement(model, f"{{{ns_core}}}metadata", {"name": name}).text = value

    resources = ET.SubElement(model, f"{{{ns_core}}}resources")
    obj = ET.SubElement(resources, f"{{{ns_core}}}object", {"id": "1", "type": "model"})
    mesh_el = ET.SubElement(obj, f"{{{ns_core}}}mesh")
    vertices_el = ET.SubElement(mesh_el, f"{{{ns_core}}}vertices")
    for x, y, z in mesh.vertices:
        ET.SubElement(vertices_el, f"{{{ns_core}}}vertex", {"x": trim_float(x), "y": trim_float(y), "z": trim_float(z)})
    triangles_el = ET.SubElement(mesh_el, f"{{{ns_core}}}triangles")
    for v1, v2, v3 in mesh.triangles:
        ET.SubElement(triangles_el, f"{{{ns_core}}}triangle", {"v1": str(v1), "v2": str(v2), "v3": str(v3)})
    build = ET.SubElement(model, f"{{{ns_core}}}build")
    ET.SubElement(build, f"{{{ns_core}}}item", {"objectid": "1"})

    indent_xml(model)
    ET.ElementTree(model).write(output_dir / "3D" / "3dmodel.model", encoding="UTF-8", xml_declaration=True)


def write_empty_model_xml(output_dir: Path, title: str) -> None:
    today = date.today().isoformat()
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" requiredextensions="p">
 <metadata name="Application">BambuStudio-02.05.00.66</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="Copyright"></metadata>
 <metadata name="CreationDate">{today}</metadata>
 <metadata name="Description"></metadata>
 <metadata name="Designer"></metadata>
 <metadata name="DesignerCover"></metadata>
 <metadata name="DesignerUserId"></metadata>
 <metadata name="License"></metadata>
 <metadata name="ModificationDate">{today}</metadata>
 <metadata name="Origin"></metadata>
 <metadata name="ProfileCover"></metadata>
 <metadata name="ProfileDescription"></metadata>
 <metadata name="ProfileTitle"></metadata>
 <metadata name="Title">{title}</metadata>
 <resources>
 </resources>
 <build/>
</model>
"""
    (output_dir / "3D" / "3dmodel.model").write_text(text, encoding="utf-8")


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def main() -> int:
    args = parse_args()
    ensure_inputs(args)

    center = parse_center(args.center)
    obj_name = args.object_name or args.obj.name
    mesh = parse_obj(args.obj)
    mesh = convert_mesh_to_z_up(mesh, args.obj_up)
    obj_bbox = bbox_from_vertices(mesh.vertices)
    if center is not None:
        dx = center[0] - obj_bbox.center_x
        dy = center[1] - obj_bbox.center_y
        mesh = translate_mesh_xy(mesh, dx, dy)
        obj_bbox = bbox_from_vertices(mesh.vertices)

    prepare_output_folder(args)
    if not args.gcode_preview_mode:
        shutil.copy2(args.obj, args.output / "Metadata" / obj_name)
    update_preview_pngs(args)

    gcode_text = write_gcode_and_md5(
        args.gcode,
        args.output,
        template_gcode_header_values(args),
        args.first_filament,
        args.material_layout,
        args,
        obj_bbox,
    )
    sequence = apply_first_filament_override(derive_filament_sequence(gcode_text), args.first_filament)
    sequence = maybe_register_all_layout_filaments(sequence, args.material_layout, args.register_all_layout_filaments)
    gcode_text = rewrite_generated_gcode_effective_layout(args.output, args.material_layout, sequence)
    gcode_text = enforce_gcode_header_preview_settings(gcode_text, args)
    gcode_text, bed_leveling_bounds, g29_replacements = rewrite_bed_leveling_area(
        gcode_text,
        float(args.bed_leveling_margin),
        obj_bbox,
    )
    if g29_replacements:
        print(
            "bed leveling area: "
            f"X{format_float(bed_leveling_bounds.min_x)}..{format_float(bed_leveling_bounds.max_x)} "
            f"Y{format_float(bed_leveling_bounds.min_y)}..{format_float(bed_leveling_bounds.max_y)}"
        )
    gcode_text = write_generated_gcode_text(args.output, gcode_text)
    if args.preview_object_ids:
        preview_object_id = crc32(obj_name.encode("utf-8")) & 0x7FFFFFFF
        gcode_text = inject_preview_object_ids(gcode_text, preview_object_id)
        gcode_text = write_generated_gcode_text(args.output, gcode_text)
    gcode_meta = parse_gcode_metadata(gcode_text)
    layer_ranges = derive_layer_filament_ranges(gcode_text)
    layer_ranges = normalize_single_tool_layer_ranges(layer_ranges, sequence, infer_total_layer_count(gcode_text))

    write_filament_sequence(args.output, args.plate, sequence)
    update_model_settings(args.output, args.material_layout, sequence)
    update_project_settings(args.output, args, args.material_layout, sequence)
    if g29_replacements:
        update_project_settings_bed_leveling_area(args.output, bed_leveling_bounds)
    update_plate_json(
        args.output,
        obj_name,
        obj_bbox,
        gcode_meta,
        sequence,
        not args.no_wipe_tower_metadata,
        args.material_layout,
        args,
    )
    update_slice_info(args.output, obj_name, gcode_meta, layer_ranges, args.material_layout, sequence)
    if args.gcode_preview_mode:
        write_empty_model_xml(args.output, obj_name)
    else:
        write_model_xml(args.output, mesh, obj_name)

    print(f"Wrote extracted package folder: {args.output}")
    print(f"Filament sequence: {sequence}")
    print(describe_used_filaments(sequence))
    print(f"Object bbox: {obj_bbox.as_list()}")
    print(f"Model mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    if args.gcode_preview_mode:
        print("3MF model geometry omitted for Bambu Studio G-code preview mode")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

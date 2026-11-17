"""Bare machine export with manifest support roles and contact infill flow.

Uses the normal machine exporter for geometry and bead volume. Only the tagged
contact-infill E values receive the separately recorded role flow multiplier.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import machine_gcode_export as machine_export
from compatible_support_schedule import write_json


E_TOKEN = re.compile(r'(?<!\S)E([-+0-9.eE]+)(?=\s|$)')
OriginalWriter = machine_export._StreamingLineWriter


def make_flow_writer(layers_by_key: dict):
    class FlowWriter(OriginalWriter):
        def __init__(self, path: Path):
            indices = path.stem.split('_')
            self.layer = layers_by_key[(int(indices[0]), int(indices[1]))]
            self.scales = self.layer.get('extrusion_role_scales', {})
            self.role = ''
            self.scale = float(self.scales.get(self.role, 1.0))
            super().__init__(path)
            super().append(f';part: {self.layer["part"]}')
            super().append(f';source: {self.layer["part"]}/{self.layer["original_main"]}_{self.layer["original_sub"]}')
            super().append(f';scalar: {self.layer["scalar"]:.12g}')

        def append(self, line: str):
            if line.startswith(';role:'):
                self.role = line.split(':', 1)[1].strip()
                self.scale = float(self.scales.get(self.role, 1.0))
            if line.startswith('G1 ') and self.scale != 1.0:
                line = E_TOKEN.sub(lambda match: f'E{float(match[1]) * self.scale:.5f}', line)
            super().append(line)
    FlowWriter._bare_flow_layers = layers_by_key
    return FlowWriter


def main():
    args = machine_export.build_parser().parse_args()
    if args.command != 'export-bare':
        raise ValueError('Use export-bare with this support-flow exporter.')
    metadata = machine_export.load_batch_metadata(args.toolpath_dir)
    if metadata.get('support_model', {}).get('same_material') is not True:
        raise ValueError('Expected a processed batch from the manifest support/model pipeline.')
    layers = {(int(e['layer_main']), int(e['layer_sub'])): e for e in metadata['layers']}
    for layer in layers.values():
        for scale in layer.get('extrusion_role_scales', {}).values():
            if not 0 < float(scale) <= 1:
                raise ValueError('Role flow multipliers must be in (0, 1].')
    machine_export._StreamingLineWriter = make_flow_writer(layers)
    try:
        machine_export.main(batch_metadata=metadata)
    finally:
        machine_export._StreamingLineWriter = OriginalWriter
    selected = [entry for key, entry in sorted(layers.items()) if args.start_main <= key[0] <= args.end_main]
    export_layers = []
    for entry in selected:
        stem = f'{entry["layer_main"]}_{entry["layer_sub"]}'
        if args.include_material_index_in_name:
            stem += f'_{entry["material_index"]}'
        export_layers.append(dict(
            file=f'{stem}.gcode', main_index=entry['layer_main'], sub_index=entry['layer_sub'],
            part=entry['part'], original_main=entry['original_main'], original_sub=entry['original_sub'],
            scalar=entry['scalar'], layer_role=entry['layer_role'],
            is_platform_layer=entry['is_platform_layer'], is_contact_layer=entry['is_contact_layer'],
            is_model_base_layer=entry.get('is_model_base_layer',False),
            is_planar_base_layer=entry.get('is_planar_base_layer',False),physical_height=entry.get('physical_height'),
            extrusion_role_scales=entry.get('extrusion_role_scales', {}),
        ))
    schedule = dict(
        material_count=1, order='numeric (main_index, sub_index), support before model',
        role_flow_applied=True, bead_cross_section=args.bead_cross_section, layers=export_layers,
    )
    if args.model_width_from_spacing:
        schedule['model_width_from_spacing'] = True
        schedule['model_width_rule'] = 'Match rounded bead area to assigned spacing times physical height; ironing keeps explicit width.'
    write_json(args.output_dir / 'print_schedule.json', schedule)
    print('Contact infill flow applied; export print_schedule.json saved.')


if __name__ == '__main__':
    main()

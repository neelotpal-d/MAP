"""Manifest-driven fork of projected hybrid planning for same-material supports."""
from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from pathlib import Path
import shutil
import re
import os
import sys
from tempfile import TemporaryDirectory
import time

from compatible_support_schedule import (
    load_schedule, build_combined_input, prepare_output, read_json, resolve, write_json,
)


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def run_mesh_stage(name, function, input_dir, output_dir, settings, clean, sources):
    reuse_dir = Path(settings.get('reuse_existing_output') or output_dir).resolve()
    if settings.get('reuse_existing') and reuse_dir.is_dir():
        expected = {p.name for p in input_dir.glob('*.obj')}
        present = {p.name for p in reuse_dir.glob('*.obj')}
        if expected and expected == present:
            # A reused sub-index must still refer to the same upstream part/scalar.
            marker = reuse_dir / 'manifest_schedule_keys.json'
            if marker.is_file() and read_json(marker) == sources:
                print(f'[{name}] Reused {reuse_dir}', flush=True)
                return reuse_dir
        print(f'[{name}] Saved output absent or schedule differs; generating', flush=True)
    prepare_output(output_dir, clean=False, protected=[input_dir])
    keyword = 'remesh_cfg' if name == 'remesh' else 'expand_cfg'
    result = function(input_dir=input_dir, output_dir=output_dir, clean_output=clean, **{keyword: settings})
    if result != input_dir:
        write_json(result / 'manifest_schedule_keys.json', sources)
    return result


def run(config_path: Path) -> Path:
    print('[startup] Loading manifest support/model projected planner...', flush=True)
    import run_surface_pipeline as pipeline
    from adaptive_path_decimation import apply_adaptive_point_decimation
    from batch_generate_projected_hybrid_paths import process_projected_hybrid_layer
    from batch_generate_projected_manifest_support_paths import process_manifest_support_layer, support_profile,support_sampling
    from batch_compute_manifest_support_heights import run_manifest_height_stage

    config = read_json(config_path)
    base = config_path.resolve().parent
    from draft_toolpaths import validate_outputs, write_draft
    draft = validate_outputs(config, base)
    if config.get('height_mode', '3axis') != '3axis':
        raise ValueError('This projected support fork requires height_mode=3axis.')
    for stage_name in ('remesh', 'support_mesh_expand'):
        settings = config.get(stage_name, {})
        if settings.get('suffix', ''):
            raise ValueError(f'{stage_name}.suffix must be empty to preserve manifest layer correspondence.')
        if settings.get('reuse_existing_output'):
            settings['reuse_existing_output'] = str(resolve(base, settings['reuse_existing_output']))
    entries, export_metadata, protected = load_schedule(base, config['compatible_inputs'])
    height_settings = config.setdefault('heights_3axis', {})
    validation = height_settings.setdefault('support_validation', {})
    if validation.get('enabled') and not draft:
        original = validation.get('model_solid') or export_metadata.get('model_volume_source')
        if original:
            root = resolve(base, config['compatible_inputs']['export_root'])
            validation['model_solid'] = str(resolve(root, original))
            if not Path(validation['model_solid']).is_file():
                raise FileNotFoundError('Original model solid for height continuity: ' + validation['model_solid'])
        else:
            print('[height validation] No original model solid supplied; model/material continuity will be unavailable.', flush=True)
    paths = {name: resolve(base, config['paths'][name]) for name in
             ('combined_input_surfaces', 'remeshed_surfaces', 'expanded_support_surfaces', 'toolpaths', 'heights')}
    values = list(paths.values())
    for index, path in enumerate(values):
        prepare_output(path, clean=False, protected=protected + values[:index] + values[index + 1:])
    clean = bool(config.get('clean_output_dirs', True))
    support_cfg = deepcopy(config['support_paths'])
    support_cfg['short_fragment_filter']=config.get('short_fragment_filter',{})
    hybrid = deepcopy(config['hybrid_contours'])
    if isinstance(hybrid.get('point_reduction'),dict):
        hybrid['point_reduction']['fragment_filter']=config.get('short_fragment_filter',{})
    hybrid['planner_backend'] = 'xz_projected_2d'
    hybrid.setdefault('boundary_count', 3)
    hybrid.setdefault('layer_workers', 1)
    hybrid.setdefault('axis_mode', 'auto_tangent_alternating')
    for entry in entries:
        if entry['part'] == 'support':
            profile=support_profile(entry, support_cfg, float(hybrid['spacing']))
            support_sampling(support_cfg,dict(hybrid,spacing=profile['width']))
    if not clean:
        scheduled={Path(entry['combined_mesh']).stem for entry in entries}
        stale=[path.name for path in paths['toolpaths'].glob('*.txt')
            if re.fullmatch(r'\d+_\d+\.txt',path.name) and path.stem not in scheduled]
        if stale:
            raise ValueError('Existing toolpaths contain layers outside this schedule: '
                f'{sorted(stale)[:10]}. Choose a fresh output folder or use clean_output_dirs=true.')
    prepare_output(paths['combined_input_surfaces'], clean=clean, protected=protected)
    build_combined_input(entries, paths['combined_input_surfaces'],
                         workers=max(1,int(os.environ.get('PROJECTED_IO_WORKERS','1'))),
                         sublayer_order=config['compatible_inputs'].get('sublayer_order', 'index'))
    schedule_keys = [(e['combined_mesh'], e['part'], e['scalar'], e['original_sub']) for e in entries]
    # JSON round-trip shape: tuples become lists.
    schedule_keys = [list(item) for item in schedule_keys]
    print(f'[schedule] {len(entries)} layers, {sum(e["part"] == "support" for e in entries)} support; one material', flush=True)
    contours = run_mesh_stage('remesh', pipeline.run_remesh_stage, paths['combined_input_surfaces'],
                              paths['remeshed_surfaces'], config.get('remesh', {'enabled': False}), clean, schedule_keys)
    active_entries = [e for e in entries if (contours / e['combined_mesh']).is_file()]
    references = contours if draft else run_mesh_stage('support-expansion', pipeline.run_support_expansion_stage, contours,
                                paths['expanded_support_surfaces'], config.get('support_mesh_expand', {'enabled': False}), clean, schedule_keys)
    output = paths['toolpaths']
    prepare_output(output, clean=clean, protected=[contours, references, *protected])
    layers = []
    model_entries = [e for e in active_entries if e['part'] == 'model']
    if model_entries:
        # Use the established model planner, with top classification restricted to models.
        with TemporaryDirectory(prefix='model_planning_', dir=output) as temporary:
            model_view = Path(temporary)
            for entry in model_entries:
                shutil.copy2(contours / entry['combined_mesh'], model_view / entry['combined_mesh'])
            original_processor = pipeline.process_hybrid_layer
            pipeline.process_hybrid_layer = process_projected_hybrid_layer
            try:
                pipeline.run_hybrid_path_stage(mesh_dir=model_view, output_dir=output, hybrid_cfg=hybrid,
                                               clean_output=False, base_dir=base, material_entries=None)
            finally:
                pipeline.process_hybrid_layer = original_processor
            layers.extend(read_json(output / 'batch_paths.meta.json')['layers'])
    supports = [e for e in active_entries if e['part'] == 'support']
    workers = max(1, int(support_cfg.get('layer_workers', hybrid['layer_workers'])))
    from support_hole_repair import prepare_repair,verify_repair_sources
    repair_metadata=dict(export_metadata)
    export_root=resolve(base,config['compatible_inputs']['export_root'])
    for name in ('support_field_source','model_volume_source'):
        if repair_metadata.get(name):repair_metadata[name]=str(resolve(export_root,repair_metadata[name]))
    if float(hybrid.get('model_lift',0)) and support_cfg.get('hole_repair',{}).get('enabled',True):
        repair_contexts={};repair_report=dict(status='unavailable',reason='Bake the model lift into the saved field/layers before field-backed hole repair.')
    else:
        repair_contexts,repair_report=prepare_repair(supports,contours,output,support_cfg,repair_metadata,
            height_settings,base_dir=base,workers=workers)
    print(f'[support repair] {repair_report["status"]}: {repair_report.get("approved_hole_occurrences",0)} approved hole occurrences',flush=True)
    print(f'[support paths] {len(supports)} layers; workers={workers}', flush=True)
    def record(entry, result):
        path, layer = result
        layers.append(dict(layer, layer_main=entry['layer_main'], layer_sub=entry['layer_sub']))
        print(f'[support paths] Saved {path.name}: {layer["layer_role"]}', flush=True)
    if workers == 1:
        for entry in supports:
            try:
                source=Path(repair_report.get('repaired_meshes',{}).get(entry['combined_mesh'],contours/entry['combined_mesh']))
                record(entry, process_manifest_support_layer(source, output, entry=entry,
                                                             support_config=support_cfg, hybrid_config=hybrid,
                                                             repair_context=repair_contexts.get(entry['combined_mesh'])))
            except Exception as exc:
                raise RuntimeError(f"Support planning failed for {entry['combined_mesh']} "
                    f"(source {entry['source_mesh']}): {exc}") from exc
    else:
        pool=ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
        try:
            futures = {pool.submit(process_manifest_support_layer, Path(repair_report.get('repaired_meshes',{}).get(e['combined_mesh'],contours/e['combined_mesh'])), output,
                                   entry=e, support_config=support_cfg, hybrid_config=hybrid,
                                   repair_context=repair_contexts.get(e['combined_mesh'])): e for e in supports}
            for future in as_completed(futures):
                entry=futures[future]
                try:record(entry, future.result())
                except Exception as exc:
                    for pending in futures:pending.cancel()
                    raise RuntimeError(f"Support planning failed for {entry['combined_mesh']} "
                        f"(source {entry['source_mesh']}): {exc}") from exc
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
    verify_repair_sources(repair_report)
    lookup = {(e['layer_main'], e['layer_sub']): e for e in active_entries}
    support_levels=sorted({e['scalar'] for e in supports if not e.get('is_platform_layer') and not e.get('is_planar_base_layer')})
    previous_support={b:a for a,b in zip(support_levels[:-1],support_levels[1:])}
    for layer in layers:
        entry = lookup[(layer['layer_main'], layer['layer_sub'])]
        layer.update(entry)
        layer['original_source_mesh'] = entry['source_mesh']
        layer['source_mesh'] = entry['combined_mesh']
        if entry['part']=='support':
            layer['support_hole_repair_context']=repair_contexts.get(entry['combined_mesh'])
            layer['previous_support_scalar']=previous_support.get(entry['scalar'])
            layer['support_surface_reference_file']=repair_report.get('repaired_meshes',{}).get(entry['combined_mesh'])
        if entry['part'] == 'model':
            layer['layer_role'] = 'model'
            layer['surface_metric_mesh']=str((contours/entry['combined_mesh']).resolve())
    layers.sort(key=lambda layer: (layer['layer_main'], layer['layer_sub']))
    from path_fragment_filter import filter_layer_fragments
    fragment_config = config.get('short_fragment_filter', {})
    reports = filter_layer_fragments(output, layers, fragment_config, workers=workers)
    if fragment_config.get('enabled', False):
        print(f'[fragment filter] Removed {sum(r.get("removed_runs", 0) for r in reports)} short raster runs, '
              f'{sum(r.get("removed_points", 0) for r in reports)} points; boundaries preserved', flush=True)
    empty=[f"{layer['layer_main']}_{layer['layer_sub']}" for layer in layers if not layer.get('path_point_count')]
    if len(empty)==len(layers):
        raise ValueError('No printable toolpaths remain at the selected width/spacing; inspect the layer footprints and contour settings.')
    if empty:print(f'[paths] {len(empty)} scheduled parts have no printable centreline; retained in metadata, excluded as height supports.',flush=True)
    from path_backing_audit import repair_support_seams
    backing=(dict(status='not_evaluated', reason='Draft paths only', changes=[]) if draft else
        repair_support_seams(output,layers,width=float(support_cfg.get('width',.5)),
        nominal=float(height_settings.get('nominal_layer_height',.2)),
        bed=float(height_settings.get('platform_value',0)),threads=min(workers,4),
        repair=bool(support_cfg.get('backing_repair',{}).get('enabled',True)),
        check=bool(support_cfg.get('backing_check',{}).get('enabled',False))))
    if backing['status']=='not_evaluated':
        print('[support backing] Check disabled; continuing without evaluation.',flush=True)
    else:
        prefix='WARNING: ' if backing['after']['findings'] else ''
        print(f'{prefix}[support backing] {backing["status"]}; {len(backing["changes"])} seam relocations; '
              f'{len(backing["after"]["findings"])} remaining findings; continuing (see support_backing.json)',flush=True)
    metadata = dict(batch_kind='projected_manifest_support_model_paths', output_format='x y z nx ny nz',
                    input_dir=str(contours), output_dir=str(output), width=float(hybrid['spacing']),
                    model_lift=float(hybrid.get('model_lift', 0.0)), model_lift_axis=hybrid.get('model_lift_axis', 'y'),
                    width_profile_mode='per_point_sidecar', path_role_tag_convention='layer2gcode.path-role-tags.v1',
                    support_model=dict(same_material=True, ordering='support by scalar then model per main index',
                                       platform_over_contact_priority=True, settings=support_cfg),
                    compatible_export=export_metadata, layers=layers)
    metadata['short_fragment_filter'] = fragment_config
    metadata['support_hole_repair']=repair_report
    metadata['support_backing']=backing
    metadata['empty_path_layers']=empty
    write_json(output / 'batch_paths.meta.json', metadata)
    write_json(output / 'print_schedule.json', dict(material_count=1, layers=layers))
    if draft:
        write_draft(output, metadata)
        verify_repair_sources(repair_report)
        return output
    height_config = config.get('heights_3axis', {})
    if height_config.get('backend', 'nearest_triangle') != 'nearest_triangle':
        raise ValueError('Manifest support heights require backend=nearest_triangle.')
    if any(row.get('support_reference_additions') or row.get('support_surface_reference_file') for row in layers):
        from support_hole_repair import assemble_height_references
        references=assemble_height_references(references,output/'height_references',layers)
        print('[height references] Included field-backed support-hole masks; original layer surfaces preserved.',flush=True)
    result = run_manifest_height_stage(mesh_dir=references, path_dir=output, output_dir=paths['heights'],
                                       config=height_config, clean=clean)
    verify_repair_sources(repair_report)
    apply_adaptive_point_decimation(result, height_config.get('adaptive_point_decimation', {}))
    print(f'[done] Processed paths: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    base = args.config.resolve().parent
    log_path = resolve(base, config.get('log_file', 'projected_manifest_support.log'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print(f'[run] Config: {args.config.resolve()}', flush=True)
        started = time.perf_counter()
        try:
            result = run(args.config)
            write_json(result / 'pipeline_config.json', config)
            print(f'[timing] Total {time.perf_counter() - started:.2f} s', flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()

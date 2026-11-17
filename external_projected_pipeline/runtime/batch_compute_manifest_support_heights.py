"""Nearest-triangle heights with earlier same-main support/model sublayers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from bisect import bisect_left
from io import StringIO
import multiprocessing
import os
from pathlib import Path
import numpy as np

from batch_compute_heights import (
    collect_layer_files, infer_model_lift_from_metadata,
)
from height_output_sidecars import copy_height_sidecars, write_height_metadata
from batch_compute_projected_nearest_heights import _compute_nearest_height_group, _split_contiguous_groups
from compatible_support_schedule import prepare_output, read_json
from contact_support_height import correct_contact_heights, validate_contact_height_config
from support_height_validation import resolve_height_settings, build_reference_catalog, HeightValidator


def _height_native_thread_budget(processes):
    try:
        explicit = int(os.environ.get('IGL_NUM_THREADS', '0'))
    except ValueError:
        explicit = 0
    return explicit if explicit > 0 else max(1, (os.cpu_count() or 1) // max(1, processes))


def _initialize_height_worker(native_threads):
    # libigl's std::thread pool has its own cached limit, independent of OMP.
    # Spawned workers set it before constructing/querying any geometry trees.
    os.environ['IGL_NUM_THREADS'] = str(native_threads)


def support_reference_keys(keys, current, previous_main_window: int):
    """Window is measured in schedule indices; gaps never pull in older layers."""
    return [key for key in keys if
            current.main - previous_main_window <= key.main < current.main
            or (key.main == current.main and key.sub < current.sub)]


class ManifestReferenceIndex:
    """Ordered schedule ranges; scalar/role eligibility is unchanged."""
    def __init__(self, keys, source_layers, previous_main_window):
        if previous_main_window < 0:
            raise ValueError('Reference window must be nonnegative.')
        self.keys=sorted(keys,key=lambda k:(k.main,k.sub))
        self.coordinates=[(k.main,k.sub) for k in self.keys]
        self.mains=[k.main for k in self.keys]
        self.sources=source_layers
        self.window=previous_main_window
        self.bases=[k for k in self.keys if source_layers[(k.main,k.sub)].get('is_planar_base_layer')]
        self.base_mains=[k.main for k in self.bases]

    def references(self,key):
        current=self.sources[(key.main,key.sub)]
        lo=bisect_left(self.mains,key.main-self.window)
        hi=bisect_left(self.coordinates,(key.main,key.sub))
        bases=self.bases[:bisect_left(self.base_mains,key.main)]
        candidates=sorted(set(self.keys[lo:hi])|set(bases),key=lambda k:(k.main,k.sub))
        result=[]
        for ref in candidates:
            source=self.sources[(ref.main,ref.sub)]
            if source.get('path_point_count')==0:
                continue
            if current.get('part')=='model' and ref.main==key.main and source.get('part')=='model':
                continue
            if (current.get('part')=='support' and source.get('part')=='support'
                    and abs(float(current['scalar'])-float(source['scalar']))<=1e-9):
                continue
            result.append(ref)
        return result


def _round_height_rows(rows):
    """Reproduce the saved ten-decimal values without reopening an artifact."""
    stream = StringIO()
    np.savetxt(stream, rows, fmt='%.10f')
    return np.loadtxt(StringIO(stream.getvalue()), ndmin=2)


def _compute_manifest_height_group(jobs, *, contact_config, validation_config=None, validation_catalog=None, **kwargs):
    validator = (HeightValidator(validation_config, validation_catalog or [], contact_config=contact_config,
        platform_value=kwargs['platform_value'], max_platform_height=kwargs['max_platform_height'],
        lift=kwargs['model_lift'], lift_axis=kwargs['model_lift_axis'])
        if validation_config and validation_config.get('enabled') else None)
    def finalize(job, result, path_rows, heights, surfaces, trees, nearest_queries):
        source = job['source_layer']
        entries = job['reference_metadata']
        physical = source.get('is_planar_base_layer', False)
        model_on_support = (contact_config.get('enabled', False) and source.get('part') == 'model'
                            and any(e.get('part') == 'support' and e.get('is_contact_layer') for e in entries))
        support_on_model = (contact_config.get('enabled', False) and source.get('part') == 'support'
                            and source.get('is_model_base_layer', False)
                            and not source.get('is_platform_layer', False) and not source.get('is_planar_base_layer', False)
                            and any(e.get('part') == 'model' for e in entries))
        contact = model_on_support or support_on_model
        if not (physical or contact or validator):
            return False
        if physical:
            current = float(source['physical_height'])
            previous = max([float(e['physical_height']) for e in entries
                if e.get('is_planar_base_layer') and float(e['physical_height']) < current - 1e-9]
                or [float(kwargs['platform_value'])])
            thickness = current - previous
            if thickness <= 0:
                raise ValueError('Physical platform-base layers must ascend above the platform.')
            result.update(height_min=thickness, height_max=thickness, physical_planar_thickness=thickness)
        if not len(path_rows):
            return False
        rows = _round_height_rows(np.column_stack((path_rows[:, :6], heights)))
        if physical:
            rows[:, 6] = thickness
        contact_selection = None
        if contact:
            if physical:
                rows = _round_height_rows(rows)
            references = [(v, f, entry, tree) for (v, f), tree, entry in zip(surfaces, trees, entries)]
            # Contact formerly queried the rounded text-file coordinates. Reuse
            # only when that decimal boundary leaves the query points identical.
            queries = (nearest_queries[:len(references)]
                if np.array_equal(rows[:, :3], path_rows[:, :3]) else None)
            contact_selection = np.zeros(len(rows), dtype=bool) if validator is not None else None
            result['contact_support_height'] = correct_contact_heights(rows, references, contact_config,
                source_layer=source, nearest_queries=queries, selected_contact_mask=contact_selection)
            result['height_min'] = float(rows[:, 6].min())
            result['height_max'] = float(rows[:, 6].max())
        if validator:
            recent = {(ref.main, ref.sub): (v, f, entry, tree)
                for (ref, _), (v, f), entry, tree in zip(job['reference_items'], surfaces, entries, trees)}
            platform_mask = result['_platform_mask']
            cleared = 0
            if contact_selection is not None:
                # A successful positive contact correction replaces the old
                # fixed-Y/bed fallback. Its stale flag must not trigger another
                # geometry search. The validator still checks excessive heights.
                resolved_contact = contact_selection & np.isfinite(rows[:, 6]) & (rows[:, 6] > validator.tolerance)
                cleared = int(np.count_nonzero(platform_mask & resolved_contact))
                platform_mask = platform_mask & ~resolved_contact
            from anchored_hole_heights import qualify
            bridge_points,bridge=qualify(rows,source,job['path_path'].parent,platform_mask,
                trigger=validator.config['max_bead_height'],max_height=validator.config.get('max_verified_support_height',.5),
                output_path=result['output_path'])
            platform_mask=platform_mask&~bridge_points
            result['anchored_hole_height']=bridge
            result['support_validation'] = validator.validate(rows, source=source, key=job['key'], recent=recent,
                platform_mask=platform_mask, output_path=result['output_path'])
            result['support_validation']['contact_platform_flags_cleared'] = cleared
            result.update(height_min=float(rows[:, 6].min()), height_max=float(rows[:, 6].max()))
        np.savetxt(result['output_path'], rows, fmt='%.10f')
        return True

    return _compute_nearest_height_group(jobs, finalize_layer=finalize, **kwargs)


def run_manifest_height_stage(*, mesh_dir: Path, path_dir: Path, output_dir: Path,
                              config: dict, clean: bool) -> Path:
    metadata = read_json(path_dir / 'batch_paths.meta.json')
    config = resolve_height_settings(config)
    meshes = collect_layer_files(mesh_dir, 'obj')
    paths = collect_layer_files(path_dir, 'txt')
    scheduled = {(layer['layer_main'], layer['layer_sub']) for layer in metadata['layers']}
    keys = sorted((key for key in paths if (key.main, key.sub) in scheduled), key=lambda key: (key.main, key.sub))
    if len(keys) != len(scheduled) or len(paths) != len(keys):
        raise ValueError('Toolpath files and manifest schedule differ; use clean_output_dirs=true to regenerate.')
    if not keys or any(key not in meshes for key in keys):
        raise ValueError('Height stage requires a matching mesh for each scheduled toolpath.')
    prepare_output(output_dir, clean=clean, protected=[mesh_dir, path_dir])
    window = int(config.get('previous_main_window', 2))
    workers = int(config.get('layer_workers', 1))
    nearest_direct = float(config.get('nearest_direct_height', 0.3))
    cap = config.get('max_height')
    progress=float(config.get('progress_interval_seconds',10.))
    if not np.isfinite(progress) or progress<0:raise ValueError('Height progress interval must be finite and nonnegative.')
    if window < 0 or workers < 1 or nearest_direct < 0 or (cap is not None and float(cap) <= 0):
        raise ValueError('Invalid nearest-height window, workers, threshold or platform cap.')
    model_lift, lift_axis = infer_model_lift_from_metadata(metadata)
    contact_config = validate_contact_height_config(config.get('contact_support_height', {}))
    validation_config = config['support_validation']
    validation_catalog = (build_reference_catalog(meshes, paths, metadata['layers'], lift=model_lift, lift_axis=lift_axis,
        bounds_mesh_dir=metadata.get('input_dir'))
        if validation_config.get('enabled') else None)
    source_layers = {(layer['layer_main'], layer['layer_sub']): layer for layer in metadata['layers']}
    references = ManifestReferenceIndex(keys,source_layers,window).references
    jobs = []
    for key in keys:
        reference_keys = references(key)
        jobs.append(dict(key=key, path_path=paths[key],
            reference_items=[(ref, meshes[ref]) for ref in reference_keys],
            source_layer=source_layers[(key.main, key.sub)],
            reference_metadata=[source_layers[(ref.main, ref.sub)] for ref in reference_keys]))
    groups = _split_contiguous_groups(jobs, workers)
    kwargs = dict(output_dir=output_dir, platform_value=float(config.get('platform_value', 0.0)),
                  max_platform_height=None if cap is None else float(cap), nearest_direct_height=nearest_direct,
                  model_lift=model_lift, model_lift_axis=lift_axis, contact_config=contact_config,progress_interval=progress)
    kwargs.update(validation_config=validation_config, validation_catalog=validation_catalog)
    print(f'[heights] {len(keys)} layers; {len(groups)} processes; includes earlier same-main sublayers', flush=True)
    if contact_config.get('enabled', False):
        print(f'[contact heights] Direction={contact_config["direction_mode"]}; gap={contact_config["gap"]:.6g} mm; '
              f'max query={contact_config["max_query_distance"]:.6g} mm', flush=True)
    results = []
    if len(groups) == 1:
        results = _compute_manifest_height_group(groups[0], **kwargs)
    else:
        native_threads = _height_native_thread_budget(len(groups))
        print(f'[heights] libigl native threads per process: {native_threads}', flush=True)
        with ProcessPoolExecutor(max_workers=len(groups), mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_height_worker, initargs=(native_threads,)) as pool:
            futures = [pool.submit(_compute_manifest_height_group, group, **kwargs) for group in groups]
            for future in as_completed(futures):
                results.extend(future.result())
                print(f'[heights] Completed {len(results)}/{len(keys)} layers', flush=True)
    results.sort(key=lambda item: (item['layer_key'].main, item['layer_key'].sub))
    sidecars = copy_height_sidecars(metadata, [result['layer_key'] for result in results],
        path_dir, output_dir, workers=workers)
    layers = []
    for result, copied in zip(results, sidecars):
        key = result['layer_key']
        layer = dict(source_layers[(key.main, key.sub)])
        layer.update(copied)
        layer.update(source_mesh=meshes[key].name, source_path_file=paths[key].name,
                     output_path_file=result['output_path'].name,
                     reference_layers=[ref.stem for ref in result['reference_keys']],
                     height_measurement='nearest_triangle_surface_euclidean')
        for name in ('platform_support_count', 'platform_support_fraction', 'height_min', 'height_max',
                     'nearest_direct_count', 'fixed_y_check_count'):
            layer[name] = result[name]
        if 'physical_planar_thickness' in result:
            layer['physical_layer_height']=result['physical_planar_thickness']
            layer['height_measurement']='physical_planar_base_thickness'
        if 'contact_support_height' in result:
            layer['contact_support_height'] = result['contact_support_height']
            if result['contact_support_height']['selected_contact_points']:
                layer['height_measurement'] = f'nearest_triangle_with_contact_{contact_config["direction_mode"]}_gap'
            print(f'[contact heights] {key.stem}: selected={result["contact_support_height"]["selected_contact_points"]}, '
                  f'changed={result["contact_support_height"]["changed_points"]}', flush=True)
        if 'support_validation' in result:
            layer['support_validation'] = result['support_validation']
        if 'anchored_hole_height' in result:
            layer['anchored_hole_height']=result['anchored_hole_height']
        layers.append(layer)
        print(f'[heights] Saved {layer["output_path_file"]}, references={layer["reference_layers"]}, max={layer["height_max"]:.6g}', flush=True)
    summary = dict(validation_config)
    summary['intentional_hole_bridge_points']=sum(r.get('anchored_hole_height',{}).get('qualified_points',0) for r in results)
    if validation_config.get('enabled'):
        for name in ('checked_points', 'recovered_points', 'unresolved_points', 'normal_miss_fallback_points',
                     'threshold_fallback_points','unverified_support_points','invalid_height_points','geometrically_verified_points',
                     'normal_miss_actual_points', 'normal_miss_cap_points', 'clamped_height_points',
                     'uncapped_model_normal_points', 'above_cap_model_normal_points', 'contact_platform_flags_cleared',
                     'verified_support_normal_points','support_normal_over_limit_points',
                     'recent_reference_reuses', 'older_reference_cache_hits', 'older_reference_mesh_loads',
                     'older_reference_cache_evictions', 'older_reference_load_seconds', 'older_reference_tree_build_seconds'):
            summary[name] = sum(r.get('support_validation', {}).get(name, 0) for r in results)
        print(f'[height validation] {summary}', flush=True)
        if summary.get('threshold_fallback_points'):
            print(f"[height warning] {summary['threshold_fallback_points']} ambiguous heights assigned the threshold; geometric support remains unverified.",flush=True)
        if summary.get('support_normal_over_limit_points'):
            print(f"[height warning] {summary['support_normal_over_limit_points']} verified support normal gaps exceed "
                  f"{validation_config['max_verified_support_height']} mm; add support layers before machine export.",flush=True)
    write_height_metadata(output_dir / 'batch_3axis.meta.json', dict(
        batch_kind='3axis_manifest_support_model_heights',
        height_mode='nearest_triangle_with_contact_correction' if contact_config.get('enabled', False) else 'nearest_triangle_surface_euclidean',
        output_format='x y z nx ny nz height', input_mesh_dir=str(mesh_dir), input_path_dir=str(path_dir),
        output_dir=str(output_dir), previous_main_window=window, previous_window_units='schedule_main_index_steps',
        earlier_same_main_sublayers=True, model_lift=model_lift, model_lift_axis=lift_axis,
        max_platform_height=cap, platform_value=kwargs['platform_value'], nearest_direct_height=nearest_direct,
        nearest_triangle_layer_group_processes=len(groups), width=metadata['width'],
        width_profile_mode='per_point_sidecar', source_path_metadata=metadata,
        support_model=metadata['support_model'], layers=layers,
        contact_support_height=contact_config,
        support_validation=summary,
    ))
    write_height_metadata(output_dir / 'print_schedule.json', dict(material_count=1, layers=layers))
    return output_dir

"""Package adapter for complete model choices without changing source runners."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

# Direct execution receives the same pre-import budget as the public API.
# Spawned workers inherit it; importing this adapter never edits a host's env.
if __name__=='__main__' and len(sys.argv)==2:
    import os
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from execution_settings import resolve_execution,execution_environment
    bootstrap_config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    bootstrap_execution=resolve_execution(bootstrap_config)
    os.environ.update(execution_environment(bootstrap_execution))
    print('[execution] '+json.dumps(bootstrap_execution),flush=True)

import run_surface_pipeline as pipeline
from batch_generate_projected_hybrid_paths import process_projected_hybrid_layer


def install_model_controls(variant):
    original_stage = pipeline.run_hybrid_path_stage
    original_settings = pipeline._effective_hybrid_layer_settings
    original_top_loader = pipeline._load_hybrid_top_layer_override
    state = {}

    def settings(**kwargs):
        values = list(original_settings(**kwargs))
        config = state['config']
        applied = dict(values[7])
        overrides = applied.get('override', {})
        base_fields = {'parallel_spacing': 1, 'parallel_resample_spacing': 3,
                       'boundary_start_offset_count': 4, 'parallel_direction_mode': 5, 'disable_parallel': 6}
        for name, slot in base_fields.items():
            if name in config and name not in overrides and f'{name}_factor' not in overrides:
                if name != 'disable_parallel' or 'disable_parallel_after_first_layer_count' not in overrides:
                    values[slot] = config[name]
        fields = {'boundary_count': 0, 'parallel_spacing': 1, 'resample_spacing': 2,
                  'parallel_resample_spacing': 3, **base_fields}
        main = int(kwargs['layer_mesh'].stem.split('_')[0])
        ranks = state['ranks']
        fraction = (ranks[main] + 1) / max(1, len(ranks))
        active_rules = []
        rules = sorted(config.get('progress_overrides', []), key=lambda rule: float(rule.get('start_fraction', 0.0)))
        for rule in rules:
            start = float(rule.get('start_fraction', 0.0))
            if not 0.0 <= start <= 1.0:
                raise ValueError('progress_overrides.start_fraction must lie in [0, 1].')
            if fraction + 1e-12 >= start:
                for name, slot in fields.items():
                    factor = rule.get(f'{name}_factor')
                    if factor is not None and name not in rule and name in {'boundary_count', 'parallel_spacing', 'resample_spacing', 'parallel_resample_spacing'}:
                        base = kwargs['boundary_count'] if name == 'boundary_count' else kwargs['spacing']
                        scaled = float(base) * float(factor)
                        values[slot] = int(round(scaled)) if name == 'boundary_count' else scaled
                for name, slot in fields.items():
                    if name in rule:
                        values[slot] = rule[name]
                active_rules.append(rule)
        # The base runner applies the numeric top overrides after this function.
        # Extend the same priority to controls absent from its older parser.
        if kwargs['layer_mesh'].stem in state.get('top_stems', set()):
            top = config.get('top_layer_overrides', {})
            for name in ('disable_parallel', 'parallel_direction_mode', 'boundary_start_offset_count'):
                if name in top:
                    values[fields[name]] = top[name]
        if values[0] < 0 or values[4] < 0:
            raise ValueError('Boundary counts/offset counts must be non-negative.')
        if values[5] not in {'connect_nearest', 'one_direction'}:
            raise ValueError('parallel_direction_mode must be connect_nearest or one_direction.')
        if active_rules:
            applied['progress_overrides_applied'] = active_rules
            applied['main_layer_progress_fraction'] = fraction
        values[7] = applied
        return tuple(values)

    def top_loader(**kwargs):
        selected, override = original_top_loader(**kwargs)
        infill = state['config'].get('planar_infill', {})
        # The regular planner expands each selected top main into the requested
        # dense band. Select only the final main to produce exactly N last layers.
        if (infill.get('enabled') and infill.get('solid_top_target') == 'last_main_layers'
                and int(infill.get('solid_top_main_layers', 0)) > 0):
            last = max(state['ranks'])
            selected |= {path.stem for path in state['meshes'] if int(path.stem.split('_')[0]) == last}
        state['top_stems'] = selected
        return selected, override

    def stage(**kwargs):
        hybrid = deepcopy(kwargs['hybrid_cfg'])
        if isinstance(hybrid.get('point_reduction'),dict):
            hybrid['point_reduction']['fragment_filter']=state.get('root',{}).get('short_fragment_filter',{})
        meshes = pipeline.iter_layer_meshes(kwargs['mesh_dir'])
        mains = sorted({int(path.stem.split('_')[0]) for path in meshes})
        state.update(config=hybrid, meshes=meshes, ranks={main: rank for rank, main in enumerate(mains)})
        result = original_stage(**dict(kwargs, hybrid_cfg=hybrid))
        if variant == 'projected':
            from path_fragment_filter import filter_short_fragments
            path = kwargs['output_dir'] / 'batch_paths.meta.json'
            metadata = json.loads(path.read_text(encoding='utf-8'))
            config = state['root'].get('short_fragment_filter', {})
            reports = [filter_short_fragments(kwargs['output_dir'], layer, config) for layer in metadata['layers']]
            metadata['short_fragment_filter'] = config
            metadata['packaged_model_controls'] = {'progress_overrides': hybrid.get('progress_overrides', []),
                                                   'disable_parallel': hybrid.get('disable_parallel', False)}
            path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            if config.get('enabled'):
                print(f'[fragment filter] Removed {sum(r.get("removed_runs", 0) for r in reports)} runs', flush=True)
        return result

    pipeline._effective_hybrid_layer_settings = settings
    pipeline._load_hybrid_top_layer_override = top_loader
    pipeline.run_hybrid_path_stage = stage
    pipeline.process_hybrid_layer = process_projected_hybrid_layer
    return state


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: projected_entry.py CONFIG.json')
    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    variant = config.get('pipeline_variant', 'support-model' if 'compatible_inputs' in config else 'projected')
    if config.get('planning_approach') == 'surface_field' or config.get('tool_motion') == 'layer_normal':
        from surface_field_pipeline import run
        run(config_path)
        return
    state = install_model_controls(variant)
    state['root'] = config
    if variant == 'support-model':
        from run_surface_pipeline_projected_manifest_support import main as support_main
        support_main()
    else:
        from run_surface_pipeline_projected_hybrid import _reuse_existing_stage, _reuse_existing_hybrid_paths
        pipeline.run_remesh_stage = _reuse_existing_stage('Remesh stage', pipeline.run_remesh_stage)
        pipeline.run_support_expansion_stage = _reuse_existing_stage('Support-expansion stage', pipeline.run_support_expansion_stage)
        pipeline.run_hybrid_path_stage = _reuse_existing_hybrid_paths(pipeline.run_hybrid_path_stage)
        original_heights = pipeline.run_height_stage
        def height_stage(**kwargs):
            settings = deepcopy(kwargs['heights_cfg'])
            validation = settings.get('support_validation', {})
            if validation.get('model_solid'):
                value = Path(validation['model_solid'])
                validation['model_solid'] = str((config_path.parent / value).resolve())
            return original_heights(**dict(kwargs, heights_cfg=settings))
        pipeline.run_height_stage = height_stage
        pipeline.main()


if __name__ == '__main__':
    main()

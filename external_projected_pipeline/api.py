"""Config/subprocess boundary suitable for embedding in another slicer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import queue
from pathlib import Path
import subprocess
import sys
import time
import threading
from typing import Callable

if __package__:
    from .execution_settings import resolve_execution,execution_environment
    from .output_lock import lock_outputs
    from .process_tree import ProcessTree, ActiveCompute
else:
    from execution_settings import resolve_execution,execution_environment
    from output_lock import lock_outputs
    from process_tree import ProcessTree, ActiveCompute


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / 'runtime'
_HEARTBEAT_SECONDS=15.


@dataclass(frozen=True)
class RunResult:
    returncode: int
    log_file: str
    report_file: str
    outputs: dict


class PipelineError(RuntimeError):
    def __init__(self, result: RunResult):
        self.result = result
        super().__init__(f'Pipeline exited with code {result.returncode}. Log: {result.log_file}')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def set_value(config: dict, dotted_key: str, value):
    parts = dotted_key.split('.')
    current = config
    for key in parts[:-1]:
        current = current.setdefault(key, {})
        if not isinstance(current, dict):
            raise ValueError(f'Cannot set nested config key: {dotted_key}')
    current[parts[-1]] = value


def create_config(config_path, *, pipeline='projected', output_root, input_surfaces=None,
                  export_root=None, model_pattern='hybrid', spacing=0.5, resample_spacing=0.5,
                  boundary_count=3, density=0.25, layer_workers=2, axis_mode='auto_tangent_alternating',
                  axis='x', hatch_angle_degrees=0.0, clean_output=False, overrides=None,
                  overwrite=False) -> Path:
    """Create an editable run contract. Relative CLI/API inputs resolve from cwd."""
    if pipeline not in {'projected', 'support-model'}:
        raise ValueError('pipeline must be projected or support-model.')
    if model_pattern not in {'hybrid', 'sparse', 'contours', 'raster'}:
        raise ValueError('Unknown model_pattern.')
    if (not all(math.isfinite(float(v)) for v in (spacing,resample_spacing,density,hatch_angle_degrees))
            or isinstance(layer_workers,bool) or not isinstance(layer_workers,int)
            or isinstance(boundary_count,bool) or not isinstance(boundary_count,int)
            or spacing <= 0 or resample_spacing <= 0 or layer_workers < 1 or boundary_count < 0 or not 0 < density <= 1):
        raise ValueError('Invalid spacing, worker count, contour count or density.')
    if axis_mode not in {'fixed', 'fixed_angle', 'alternating_angle', 'auto_tangent_alternating'} or axis not in {'x', 'z'}:
        raise ValueError('Invalid projected raster axis/mode.')
    target = Path(config_path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f'Config exists; use overwrite=True/--overwrite-config: {target}')
    config = _read(HERE / 'configs' / f'{pipeline}.example.json')
    root = Path(output_root).resolve()
    config['pipeline_variant'] = pipeline
    config['clean_output_dirs'] = bool(clean_output)
    config['logs_dir'] = str(root / 'logs')
    config['log_file'] = str(root / 'pipeline.log')
    for name in ('combined_input_surfaces', 'remeshed_surfaces', 'expanded_support_surfaces', 'toolpaths', 'heights'):
        config['paths'][name] = str(root / name)
    if pipeline == 'projected':
        if input_surfaces is None:
            raise ValueError('input_surfaces is required for projected.')
        config['paths']['input_surfaces'] = str(Path(input_surfaces).resolve())
    else:
        if export_root is None:
            raise ValueError('export_root is required for support-model.')
        config['compatible_inputs']['export_root'] = str(Path(export_root).resolve())
    hybrid = config['hybrid_contours']
    hybrid.update(spacing=float(spacing), resample_spacing=float(resample_spacing), boundary_count=boundary_count,
                  axis_mode=axis_mode, axis=axis, hatch_angle_degrees=hatch_angle_degrees,
                  disable_parallel=model_pattern == 'contours', layer_workers=layer_workers)
    if model_pattern == 'raster':
        hybrid['boundary_count'] = 0
    hybrid['planar_infill']['enabled'] = model_pattern == 'sparse'
    hybrid['planar_infill']['density'] = density
    config['remesh']['file_workers'] = layer_workers
    config['heights_3axis']['layer_workers'] = layer_workers
    if pipeline == 'support-model':
        config['support_paths']['layer_workers'] = layer_workers
    for stage in ('remesh', 'support_mesh_expand'):
        config[stage]['reuse_existing_output'] = config['paths'][
            'remeshed_surfaces' if stage == 'remesh' else 'expanded_support_surfaces']
    hybrid['top_layer_classifier']['output_json'] = str(root / 'top_layers_report.json')
    for key, value in (overrides or {}).items():
        set_value(config, key, value)
    # Validate nominal thickness without importing native libraries into the host.
    nominal = config['heights_3axis'].get('nominal_layer_height')
    if nominal is not None and (not math.isfinite(float(nominal)) or float(nominal) <= 0):
        raise ValueError('nominal_layer_height must be positive and finite.')
    execution=resolve_execution(config)
    config.setdefault('execution',{}).update(
        native_threads_per_worker=execution['native_threads_per_worker'],
        blas_threads_per_process=execution['blas_threads_per_process'])
    config['execution'].setdefault('keep_patch_artifacts',False)
    config['execution'].setdefault('io_workers',4)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write(target, config)
    return target


def _execute(command, *, cwd, logs_dir, outputs, label, on_output, check, config=None,execution=None):
    started=time.perf_counter()
    logs = Path(logs_dir).resolve()
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log_path = logs / f'{label}_{stamp}.log'
    report_path = logs / f'{label}_{stamp}.json'
    report = dict(status='running', command=list(map(str, command)), cwd=str(cwd), outputs=outputs,
                  log_file=str(log_path), start_utc=stamp)
    environment=None
    if execution is not None:
        environment=execution_environment(execution)
        report['execution']=execution
    if config is not None:
        snapshot = logs / f'config_{stamp}.json'
        _write(snapshot, config)
        report['config_snapshot'] = str(snapshot)
    _write(report_path, report)
    try:
        with ActiveCompute(), log_path.open('w', encoding='utf-8') as log:
            log.write(json.dumps(report) + '\n')
            with subprocess.Popen(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  encoding='utf-8', errors='replace', text=True, bufsize=1,env=environment,
                                  start_new_session=os.name!='nt') as process,ProcessTree(process) as tree:
                messages=queue.Queue()
                def consume():
                    try:
                        for line in process.stdout:messages.put(line)
                    finally:messages.put(None)
                reader=threading.Thread(target=consume,daemon=True);reader.start()
                try:
                    stream_closed=False
                    while not stream_closed or process.poll() is None:
                        try:
                            if stream_closed:
                                process.wait(timeout=_HEARTBEAT_SECONDS)
                                break
                            line=messages.get(timeout=_HEARTBEAT_SECONDS)
                        except subprocess.TimeoutExpired:
                            line=f'[{label}] Running; elapsed {time.perf_counter()-started:.1f} s\n'
                        except queue.Empty:
                            if process.poll() is not None:
                                tree.stop();break
                            line=f'[{label}] Running; elapsed {time.perf_counter()-started:.1f} s\n'
                        if line is None:
                            stream_closed=True
                            continue
                        log.write(line)
                        log.flush()
                        if on_output:
                            on_output(line)
                    code = process.wait()
                except BaseException:
                    tree.stop()
                    process.wait(timeout=5)
                    raise
                finally:reader.join(timeout=1)
    except BaseException as exc:
        report.update(status='failed', error=str(exc),wall_seconds=time.perf_counter()-started)
        _write(report_path, report)
        raise
    report.update(status='complete' if code == 0 else 'failed', returncode=code,wall_seconds=time.perf_counter()-started)
    _write(report_path, report)
    result = RunResult(code, str(log_path), str(report_path), outputs)
    if code and check:
        raise PipelineError(result)
    return result


def run_config(config_path, *, log_dir=None, on_output: Callable[[str], None] | None = None, check=True) -> RunResult:
    """Run in a separate interpreter to isolate sibling imports and worker globals."""
    path = Path(config_path).resolve()
    config = _read(path)
    variant = config.get('pipeline_variant', 'support-model' if 'compatible_inputs' in config else 'projected')
    if variant not in {'projected', 'support-model'}:
        raise ValueError(f'Unknown pipeline_variant: {variant}')
    experimental = config.get('planning_approach') == 'surface_field' or config.get('tool_motion') == 'layer_normal'
    if experimental:
        if config.get('height_mode') != '5axis' or config.get('tool_motion') != 'layer_normal':
            raise ValueError('Experimental surface planning requires layer-normal 5axis diagnostics.')
    elif config.get('height_mode') != '3axis' or config.get('heights_3axis', {}).get('backend') != 'nearest_triangle':
        raise ValueError('This package requires 3axis nearest_triangle heights.')
    outputs = {key: str((path.parent / value).resolve()) for key, value in config['paths'].items()}
    inputs = [Path(outputs['input_surfaces'])] if 'input_surfaces' in outputs else []
    for material in config.get('materials', []):
        inputs.append((path.parent / material.get('input_surfaces', material.get('input_layers_dir', ''))).resolve())
    if 'compatible_inputs' in config:
        inputs.append((path.parent / config['compatible_inputs']['export_root']).resolve())
    generated = [Path(value) for key, value in outputs.items() if key != 'input_surfaces']
    for index, output in enumerate(generated):
        for protected in [*inputs, HERE, path, *generated[:index], *generated[index + 1:]]:
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError(f'Generated folder overlaps an input/config/package or another output: {output}, {protected}')
        if output == Path(output.anchor):
            raise ValueError(f'Generated folder cannot be a filesystem root: {output}')
    logs = log_dir or path.parent / config.get('logs_dir', 'logs')
    execution=resolve_execution(config)
    with lock_outputs(generated):
        return _execute([sys.executable, '-u', str(RUNTIME / 'projected_entry.py'), str(path)], cwd=path.parent,
                        logs_dir=logs, outputs=outputs, label='pipeline', on_output=on_output, check=check, config=config,
                        execution=execution)


def export_bare(toolpath_dir, output_dir, *, machine_name='bambuH2D', machine_dir=None,
                start_main=None, end_main=None, options=(), log_dir=None, on_output=None, check=True) -> RunResult:
    source, target = Path(toolpath_dir).resolve(), Path(output_dir).resolve()
    if (source / 'batch_5axis.meta.json').exists():
        raise ValueError('Experimental layer-normal paths cannot be exported by this fixed-axis machine exporter.')
    metadata = _read(source / 'batch_3axis.meta.json')
    if metadata.get('machine_export_supported') is False:
        raise ValueError('This experimental path batch is for inspection, not machine export.')
    if __package__:
        from .runtime.height_warning_policy import check_height_outliers
    else:
        from runtime.height_warning_policy import check_height_outliers
    check_height_outliers(metadata, start_main=start_main, end_main=end_main)
    issues = metadata.get('support_validation', {}).get('unresolved_points', 0)
    if issues:
        raise ValueError(f'{issues} unresolved supporting-height points remain. Inspect the per-layer '
            '*.height_validation.npz diagnostics before machine export; no bridge thickness is assumed.')
    mains = [int(layer['layer_main']) for layer in metadata['layers']]
    if not mains:
        raise ValueError('Processed batch is empty.')
    support = metadata.get('support_model', {}).get('same_material') is True
    script = 'export_projected_manifest_support.py' if support else 'machine_gcode_export.py'
    command = [sys.executable, '-u', str(RUNTIME / script), 'export-bare', str(source), str(target),
               '--machine-dir', str(Path(machine_dir).resolve() if machine_dir else HERE / 'machines'),
               '--machine-name', machine_name, '--start-main', str(min(mains) if start_main is None else start_main),
               '--end-main', str(max(mains) if end_main is None else end_main), *map(str, options)]
    return _execute(command, cwd=source, logs_dir=log_dir or target.parent / 'logs',
                    outputs={'bare_paths': str(target)}, label='export_bare', on_output=on_output, check=check)

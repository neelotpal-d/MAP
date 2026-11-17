"""Remove short, travel-separated raster runs before height computation."""
from __future__ import annotations

from multiprocessing import get_context

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
import numpy as np

from adaptive_path_decimation import _load_tags, _write_tags, _load_break_indices, _write_long_jump_tags


def filter_infill_parts(parts, widths, config):
    """Apply the short-run policy before support walls choose their width.

    Every supplied run is infill following a support boundary. Splits at long
    jumps match the later file filter. Geometry is retained unchanged; the
    caller represents surviving parts with travel breaks as usual.
    """
    if not config.get('enabled',False):return parts,widths,dict(removed_runs=0,removed_points=0)
    minimum=float(config.get('min_length',2.))
    count=config.get('min_waypoints',5)
    count=None if count is None else int(count)
    jump=config.get('long_jump_threshold',2.)
    jump=None if jump is None else float(jump)
    if not np.isfinite(minimum) or minimum<=0 or (count is not None and count<2):
        raise ValueError('Fragment min_length must be positive; min_waypoints must be >=2 or null.')
    if jump is not None and (not np.isfinite(jump) or jump<=0):raise ValueError('Invalid fragment long-jump threshold.')
    kept=[];profiles=[];removed=points=0
    for part,width in zip(parts,widths):
        part=np.asarray(part);width=np.asarray(width)
        if len(part)!=len(width):raise ValueError('Infill width profile mismatch.')
        lengths=np.linalg.norm(np.diff(np.round(part,10),axis=0),axis=1)
        cuts=np.flatnonzero(lengths>jump)+1 if jump is not None else []
        bounds=[0,*cuts,len(part)]
        for start,end in zip(bounds[:-1],bounds[1:]):
            length=float(lengths[start:max(start,end-1)].sum())
            if length<minimum and (count is None or end-start<count):
                removed+=1;points+=end-start
            else:kept.append(part[start:end]);profiles.append(width[start:end])
    return kept,profiles,dict(removed_runs=removed,removed_points=points)


def _filter_layer_fragments(path_dir, layer, config):
    report = filter_short_fragments(path_dir, layer, config)
    return layer, report


def filter_layer_fragments(path_dir: Path, layers: list[dict], config: dict, *, workers: int = 1) -> list[dict]:
    """Run the unchanged file filter concurrently only for independent layers.

    Keep the original text round trips, numerical operations and layer order.
    Shared sidecars fall back to serial processing to preserve their semantics.
    """
    workers = min(max(1, int(workers)), len(layers))
    if workers > 1 and config.get('enabled', False):
        owned = set()
        for layer in layers:
            path = path_dir / layer['path_file']
            names = [layer.get(field) for field in
                     ('path_file', 'width_profile_file', 'path_tag_file', 'path_break_file')]
            names.extend((f'{path.stem}.breaks.txt', f'{path.stem}.jumps.txt'))
            files = {(path_dir / name).resolve() for name in names if name}
            if owned.intersection(files):
                workers = 1
                break
            owned.update(files)
    if workers <= 1 or not config.get('enabled', False):
        return [filter_short_fragments(path_dir, layer, config) for layer in layers]
    reports = []
    chunksize = max(1, len(layers) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        results = pool.map(_filter_layer_fragments, repeat(path_dir), layers, repeat(config), chunksize=chunksize)
        for layer, (updated, report) in zip(layers, results):
            layer.update(updated)
            reports.append(report)
    return reports


def filter_short_fragments(path_dir: Path, layer: dict, config: dict) -> dict:
    if not config.get('enabled', False):
        return dict(enabled=False)
    min_length = float(config.get('min_length', 2.0))
    min_points = config.get('min_waypoints', 5)
    min_points = None if min_points is None else int(min_points)
    jump = config.get('long_jump_threshold', 2.0)
    jump = None if jump is None else float(jump)
    if not np.isfinite(min_length) or min_length <= 0 or (min_points is not None and min_points < 2):
        raise ValueError('Fragment min_length must be positive; min_waypoints must be >=2 or null.')
    if jump is not None and (not np.isfinite(jump) or jump <= 0):
        raise ValueError('Fragment long_jump_threshold must be positive or null.')
    path = path_dir / layer['path_file']
    if path.stat().st_size == 0:
        return dict(enabled=True, removed_runs=0, removed_points=0)
    rows = np.atleast_2d(np.loadtxt(path))
    count = len(rows)
    tag_name = layer.get('path_tag_file')
    if not tag_name:
        raise ValueError(f'Fragment filtering requires role tags: {path}')
    tags = _load_tags(path_dir / tag_name, count)
    if tags is None or np.any(tags == ''):
        raise ValueError(f'Incomplete role tags: {path}')
    lengths = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1)
    break_name = layer.get('path_break_file')
    breaks = set(_load_break_indices(path_dir / break_name, count)) if break_name else set()
    if jump is not None:
        breaks.update((np.flatnonzero(lengths > jump) + 1).tolist())
    bounds = [0, *sorted(breaks), count]
    selected_parts = []
    removed_runs = 0
    removed_length = 0.0
    for start, end in zip(bounds[:-1], bounds[1:]):
        run_tags = set(map(str, tags[start:end]))
        eligible = all(('raster' in tag or 'infill' in tag) and 'boundary' not in tag for tag in run_tags)
        length = float(lengths[start:max(start, end - 1)].sum())
        too_few = min_points is None or end - start < min_points
        if start > 0 and eligible and length < min_length and too_few:
            removed_runs += 1
            removed_length += length
        else:
            selected_parts.append(np.arange(start, end))
    selected = np.concatenate(selected_parts)
    rewritten_breaks = np.cumsum([len(part) for part in selected_parts[:-1]]).tolist()
    width_name = layer.get('width_profile_file')
    widths = None
    if width_name:
        widths = np.atleast_1d(np.loadtxt(path_dir / width_name))
        if len(widths) != count:
            raise ValueError(f'Width profile mismatch: {width_name}')
        widths = widths[selected]
    np.savetxt(path, rows[selected], fmt='%.10f')
    _write_tags(path_dir / tag_name, tags[selected])
    if widths is not None:
        np.savetxt(path_dir / width_name, widths, fmt='%.10f')
        layer.update(width_profile_point_count=len(widths), width_profile_min=float(widths.min()),
                     width_profile_max=float(widths.max()))
    break_name = break_name or f'{path.stem}.breaks.txt'
    break_path = path_dir / break_name
    if rewritten_breaks:
        break_path.write_text(''.join(f'{i}\n' for i in rewritten_breaks), encoding='utf-8')
    elif break_path.exists():
        break_path.unlink()
    new_lengths = np.linalg.norm(np.diff(rows[selected, :3], axis=0), axis=1)
    jumps = [(i, float(new_lengths[i - 1])) for i in rewritten_breaks
             if jump is not None and new_lengths[i - 1] > jump]
    jump_path = path_dir / f'{path.stem}.jumps.txt'
    if jumps:
        _write_long_jump_tags(jump_path, jumps)
    elif jump_path.exists():
        jump_path.unlink()
    layer.update(path_point_count=len(selected), path_break_count=len(rewritten_breaks),
                 path_break_file=break_name if rewritten_breaks else None,
                 path_jump_file=jump_path.name if jumps else None, path_long_jump_count=len(jumps),
                 path_jump_format='destination_index length tag' if jumps else None,
                 path_tag_values=sorted(set(map(str, tags[selected]))),
                 path_tag_block_count=int(np.count_nonzero(np.r_[True, tags[selected][1:] != tags[selected][:-1]])))
    report = dict(enabled=True, input_points=count, output_points=len(selected), removed_runs=removed_runs,
                  removed_points=count - len(selected), removed_print_length=removed_length,
                  min_length=min_length, min_waypoints=min_points)
    layer['short_fragment_filter'] = report
    return report

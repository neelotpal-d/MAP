"""Copy independent height sidecars without repeating whole-manifest searches."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil

from batch_compute_heights import (
    copy_width_profile_for_layer, copy_path_tags_for_layer, copy_path_breaks_for_layer,
)

COPIERS = ((copy_width_profile_for_layer, 'width_profile_file'),
           (copy_path_tags_for_layer, 'path_tag_file'),
           (copy_path_breaks_for_layer, 'path_break_file'))


def write_height_metadata(path: Path, value):
    """Keep the complete JSON data using the fast compact encoder."""
    path.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')


def _copy_one(job, source_dir, output_dir):
    key, source, jump = job
    # The existing helpers retain all file/name/missing-file semantics. Their
    # lookup now visits just this layer instead of scanning the full manifest.
    metadata = {'layers': [source]}
    result = {field: copier(source_batch_metadata=metadata, key=key,
        source_dir=source_dir, output_dir=output_dir) for copier, field in COPIERS}
    if jump:
        shutil.copy2(source_dir / jump, output_dir / jump)
    return result


def copy_height_sidecars(metadata, keys, source_dir: Path, output_dir: Path, *, workers: int):
    """Return copied filenames in key order; serialize ambiguous destinations."""
    first, last = {}, {}
    for layer in metadata['layers']:
        identity = (int(layer.get('layer_main', -1)), int(layer.get('layer_sub', -1)))
        first.setdefault(identity, layer)
        last[identity] = layer
    jobs = [(key, first[(key.main, key.sub)], last[(key.main, key.sub)].get('path_jump_file'))
            for key in keys]
    workers = min(max(1, int(workers)), 8, len(jobs))
    destinations = set()
    for _, layer, jump in jobs:
        names = [layer.get(field) for _, field in COPIERS]
        targets = [output_dir / Path(name).name for name in names if isinstance(name, str) and name]
        if jump:
            # Preserve ordering for nested/absolute jump paths and their aliases.
            if Path(jump).parent != Path('.'):
                workers = 1
            targets.append(output_dir / jump)
        for target in targets:
            identity = os.path.normcase(os.path.abspath(target))
            if identity in destinations:
                workers = 1
            destinations.add(identity)
    # Pre-existing links can alias different output names. Fresh output folders
    # need no per-file filesystem probes here; inspect existing entries once.
    if workers > 1:
        try:
            with os.scandir(output_dir) as entries:
                for entry in entries:
                    if os.path.normcase(os.path.abspath(entry.path)) in destinations:
                        # Windows DirEntry.stat() can report st_nlink=0 even
                        # when os.stat(path) reports the real NTFS link count.
                        # Probe only existing intended destinations whose cached
                        # count is unknown; fresh output folders keep the fast path.
                        links=entry.stat().st_nlink
                        if links==0:
                            links=os.stat(entry.path).st_nlink
                        if entry.is_symlink() or links != 1:
                            workers = 1
                            break
        except OSError:
            # Listing/stat access is not a new requirement for the legacy copy.
            workers = 1
    if workers <= 1:
        return [_copy_one(job, source_dir, output_dir) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda job: _copy_one(job, source_dir, output_dir), jobs))

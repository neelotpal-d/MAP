"""Paths-only previews: no invented heights and no manufacturing qualification."""
from pathlib import Path
import json

METADATA = 'draft_paths.meta.json'
SCHEMA = 'multiaxis.draft-paths.v1'


def enabled(config):
    value = config.get('draft_paths_only', False)
    if not isinstance(value, bool):
        raise ValueError('draft_paths_only must be a boolean.')
    return value


def validate_outputs(config, base):
    """Reject stale completion artifacts before any output is changed."""
    draft = enabled(config)
    paths = {k: (Path(base) / str(v)).resolve() for k, v in config['paths'].items()}
    if draft:
        for name in ('toolpaths', 'heights'):
            folder = paths[name]
            if folder.exists() and any(folder.iterdir()):
                raise ValueError('Draft paths require fresh toolpath and height output folders; '
                                 'existing results are preserved: ' + str(folder))
    elif (paths['toolpaths'] / METADATA).exists():
        raise ValueError('This output contains draft paths. Run Paths and heights in a fresh output folder to finalize.')
    return draft


def write_draft(folder, metadata):
    """Keep six-column paths and original role/width/travel metadata intact."""
    folder = Path(folder)
    result = dict(metadata, draft_schema=SCHEMA, draft_paths_only=True,
                  experimental=True, machine_export_supported=False,
                  output_format='x y z nx ny nz', height_status='not_evaluated',
                  manufacturing_checks={name: 'not_evaluated' for name in (
                      'heights', 'contact_height_correction', 'support_backing',
                      'final_tool_clearance', 'machine_kinematics')})
    result.setdefault('tool_motion','fixed_axis')
    for name in ('batch_paths.meta.json', METADATA):
        (folder / name).write_text(json.dumps(result, indent=2), encoding='utf8')
    print('[draft] Paths only: heights, contact-height correction, backing and final tool clearance '
          'not evaluated. Machine export blocked.', flush=True)
    return result

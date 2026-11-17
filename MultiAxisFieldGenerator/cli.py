"""Run a portable model job with training, validation and preview actions."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
from .runtime.collision_settings import collision_enabled

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT/'runtime'
PATH_OPTIONS = {'mesh', 'volume_mesh', 'start_checkpoint', 'sdf_checkpoint', 'stress_file', 'replay_results', 'tool_config'}


def resolve(path, base):
    p = Path(path)
    return (p if p.is_absolute() else base/p).resolve()


def supported_options(script='run_model.py'):
    tree = ast.parse((RUNTIME/script).read_text())
    options, booleans = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith('--'):
                    key = arg.value[2:].replace('-', '_')
                    options.add(key)
                    if any(k.arg == 'action' and isinstance(k.value, ast.Constant) and k.value.value == 'store_true' for k in node.keywords):
                        booleans.add(key)
    return options, booleans


def load_job(path, output_override=None, action='train'):
    path = path.resolve()
    job = json.loads(path.read_text(encoding='utf-8'))
    if set(job)-{'output', 'model', 'training', 'tool', 'stress', 'builders'}:
        raise ValueError('Unknown top-level job keys')
    if not job.get('output') and output_override is None:
        raise ValueError('Job requires an output directory')
    values = {}
    for section in ['model', 'training']:
        obj = job.get(section, {})
        if not isinstance(obj, dict) or set(values) & set(obj):
            raise ValueError(f'Invalid/duplicated options in {section}')
        values.update(obj)
    output = Path(output_override).resolve() if output_override is not None else resolve(job['output'], path.parent)
    values.setdefault('start_checkpoint', str(output/'inputs'/'initializer.pt'))
    needs_sdf = collision_enabled(values.get('collision_weight', 10000), values.get('epochs', 50), values.get('collision_weight_schedule'))
    if needs_sdf or action in ('build-sdf','validate-sdf'):
        values.setdefault('sdf_checkpoint', str(output/'inputs'/'sdf.pt'))
    tools = job.get('tool', {})
    if not tools and (action.startswith('build-') or action == 'validate-sdf' or (not needs_sdf and action != 'check')):
        tools = dict(training=str(ROOT/'tools/abb_training.json'), validation=str(ROOT/'tools/abb_nominal.json'))
    if set(tools) != {'training', 'validation'}:
        raise ValueError('tool requires explicit training and validation preset paths')
    values['tool_config'] = tools['training']
    stress = dict(job.get('stress', {}))
    enabled = stress.pop('enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('stress.enabled must be boolean')
    allowed_stress = {'file', 'weight', 'samples', 'batch', 'neighbors', 'max_distance_mm',
                      'point_columns', 'direction_columns', 'coordinate_space',
                      'region', 'signed_field', 'threshold', 'sampling', 'top_percent'}
    if set(stress)-allowed_stress:
        raise ValueError('Unknown stress settings')
    if enabled:
        values.setdefault('objective', 'combined')
        if values['objective'] == 'support_free':
            raise ValueError('Enabled stress constraints require objective stress or combined')
        for key, value in stress.items():
            values['stress_'+key] = value
        if not values.get('stress_file'):
            raise ValueError('Enabled stress constraints require stress.file')
    options, _ = supported_options()
    if set(values)-options:
        raise ValueError(f'Unknown runner options: {sorted(set(values)-options)}')
    for key in PATH_OPTIONS & set(values):
        if values[key] is not None:
            values[key] = str(resolve(values[key], path.parent))
            builder_target = action.startswith('build-') and key in {'start_checkpoint', 'sdf_checkpoint'}
            builder_target |= action == 'validate-sdf' and key == 'start_checkpoint'
            unused_sdf = key == 'sdf_checkpoint' and action != 'validate-sdf' and (not needs_sdf or action != 'train')
            if not builder_target and not unused_sdf and not Path(values[key]).exists():
                raise FileNotFoundError(f'{key}: {values[key]}')
    for key in ['mesh', 'start_checkpoint']+(['sdf_checkpoint'] if needs_sdf else []):
        if not values.get(key):
            raise ValueError(f'model.{key} is required')
    values['output'] = str(output)
    validation_tool = resolve(tools['validation'], path.parent)
    if not validation_tool.is_file():
        raise FileNotFoundError(validation_tool)
    return values, validation_tool


def training_command(values):
    _, booleans = supported_options()
    command = [sys.executable, str(RUNTIME/'run_model.py')]
    for key, value in values.items():
        if value is None:
            continue
        flag = '--'+key.replace('_', '-')
        if key in booleans:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean')
            if value:
                command.append(flag)
        else:
            command.append(flag)
            command.extend(str(x) for x in value) if isinstance(value, list) else command.append(str(value))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--action', choices=['train', 'check', 'view', 'view-collisions', 'build-sdf', 'build-initializer', 'validate-sdf'], default='train')
    parser.add_argument('--output', type=Path, help='Override run directory (relative to current working directory)')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args(argv)
    try:
        values, tool = load_job(args.job, args.output, args.action)
        if args.action.startswith('build-'):
            kind = args.action[6:]
            raw = json.loads(args.job.read_text(encoding='utf-8'))
            settings = dict(raw.get('builders', {}).get(kind, {}))
            allowed, _ = supported_options('build_checkpoint.py')
            if set(settings)-allowed or set(settings) & {'kind', 'mesh', 'output'}:
                raise ValueError('Unknown or reserved builder settings')
            prefix = 'sdf' if kind == 'sdf' else 'field'
            settings.setdefault('width', values.get(prefix+'_width', 256 if kind == 'sdf' else 128))
            settings.setdefault('layers', values.get(prefix+'_layers', 5 if kind == 'sdf' else 10))
            settings.setdefault('frequency', values.get(prefix+'_frequency', 30 if kind == 'sdf' else 7))
            settings.setdefault('up_axis', values.get('up_axis', 'y'))
            for key in ['device', 'seed']:
                if key in values:
                    settings.setdefault(key, values[key])
            if kind == 'initializer' and settings.get('planar_mode') == 'stress-informed':
                # The builder resolves stress inputs and sampling settings from
                # the same job as the main optimizer, including its train split.
                settings['stress_job'] = str(args.job.resolve())
            target = values['sdf_checkpoint' if kind == 'sdf' else 'start_checkpoint']
            command = [sys.executable, str(RUNTIME/'build_checkpoint.py'), '--kind', kind, '--mesh', values['mesh'], '--output', target]
            for key, value in settings.items():
                command += ['--'+key.replace('_', '-'), str(value)]
        elif args.action == 'train':
            command = training_command(values)
        elif args.action == 'check':
            command = [sys.executable, str(RUNTIME/'validate_nominal_collision.py'), '--run', values['output'], '--tool-config', str(tool)]
            if values.get('volume_mesh'):
                command += ['--volume-mesh', values['volume_mesh']]
        elif args.action == 'validate-sdf':
            command = [sys.executable,str(RUNTIME/'validate_sdf.py'),'--mesh',values['mesh'],
                       '--checkpoint',values['sdf_checkpoint'],'--output',str(Path(values['output'])/'sdf_validation')]
        elif args.action == 'view':
            command = [sys.executable, str(RUNTIME/'visualize_fertility.py'), '--run', values['output']]
        else:
            command = [sys.executable, str(RUNTIME/'visualize_nominal_collision.py'), '--results', str(Path(values['output'])/'nominal_validation')]
        command += extra
        if args.dry_run:
            print(json.dumps(dict(action=args.action, resolved=values, validation_tool=str(tool), command=command), indent=2))
            return 0
        if args.action == 'train':
            out = Path(values['output'])
            if (out/'config.json').exists():
                raise FileExistsError(f'Run exists: {out}; choose a new output')
            out.mkdir(parents=True, exist_ok=True)
            (out/'resolved_job.json').write_text(json.dumps(values, indent=2))
        return subprocess.call(command, cwd=RUNTIME)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())

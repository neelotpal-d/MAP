"""Portable CLI; configuration JSON is the complete downstream run contract."""
from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

if __package__:
    from . import api
else:
    import api


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init-config', help='Create an editable config for either projected pipeline.')
    init.add_argument('--config', type=Path, required=True)
    init.add_argument('--pipeline', choices=['projected', 'support-model'], default='projected')
    init.add_argument('--input-surfaces', type=Path)
    init.add_argument('--export-root', type=Path)
    init.add_argument('--output-root', type=Path, required=True)
    init.add_argument('--model-pattern', choices=['hybrid', 'sparse', 'contours', 'raster'], default='hybrid')
    init.add_argument('--spacing', type=float, default=.5)
    init.add_argument('--resample-spacing', type=float, default=.5)
    init.add_argument('--boundary-count', type=int, default=3)
    init.add_argument('--density', type=float, default=.25)
    init.add_argument('--layer-workers', type=int, default=2)
    init.add_argument('--axis-mode', choices=['fixed', 'fixed_angle', 'alternating_angle', 'auto_tangent_alternating'], default='auto_tangent_alternating')
    init.add_argument('--axis', choices=['x', 'z'], default='x')
    init.add_argument('--hatch-angle-degrees', type=float, default=0.)
    init.add_argument('--clean-output-dirs', action='store_true')
    init.add_argument('--overwrite-config', action='store_true')
    init.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='Override any dotted config key. Repeat as needed.')
    run = sub.add_parser('run', help='Run a config with logs, config snapshot and JSON job report.')
    run.add_argument('config', type=Path)
    run.add_argument('--log-dir', type=Path)
    export = sub.add_parser('export-bare', help='Export bare paths; auto-detect support flow. Additional machine-export flags pass through.')
    export.add_argument('toolpath_dir', type=Path)
    export.add_argument('output_dir', type=Path)
    export.add_argument('--machine-name', default='bambuH2D')
    export.add_argument('--machine-dir', type=Path)
    export.add_argument('--start-main', type=int)
    export.add_argument('--end-main', type=int)
    export.add_argument('--log-dir', type=Path)
    preview = sub.add_parser('preview', help='Bundled mesh/path viewer; pass --layer, --show-points etc.')
    preview.add_argument('mesh_dir', type=Path)
    preview.add_argument('path_dir', type=Path)
    preview.add_argument('--log-dir', type=Path, default=Path('logs'))
    lift = sub.add_parser('lift', help='Optional causal/current-mesh nozzle-clearance copy; pass the original lift flags.')
    lift.add_argument('input_dir', type=Path)
    lift.add_argument('output_dir', type=Path)
    lift.add_argument('--log-dir', type=Path, default=Path('logs'))
    sub.add_parser('doctor', help='Read dependency versions and check bundled source hashes; installs nothing.')
    return root


def main():
    root = parser()
    args, extra = root.parse_known_args()
    if extra and args.command not in {'export-bare', 'preview', 'lift'}:
        root.error('Unrecognized arguments: ' + ' '.join(extra))
    emit = lambda line: print(line, end='', flush=True)
    if args.command == 'init-config':
        overrides = {}
        for spec in args.set:
            if '=' not in spec:
                root.error('--set requires KEY=JSON')
            key, value = spec.split('=', 1)
            try:
                overrides[key] = json.loads(value)
            except json.JSONDecodeError:
                overrides[key] = value
        values = vars(args).copy()
        values.pop('command')
        values.pop('set')
        values['config_path'] = values.pop('config')
        values['clean_output'] = values.pop('clean_output_dirs')
        values['overwrite'] = values.pop('overwrite_config')
        values['overrides'] = overrides
        print(api.create_config(**values))
        return
    if args.command == 'doctor':
        versions = {}
        for name in ('numpy', 'scipy', 'shapely', 'libigl', 'pymeshlab', 'pyvista', 'potpourri3d'):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        manifest = api._read(api.HERE / 'bundle_manifest.json')
        mismatches = []
        for name, digest in manifest['files'].items():
            path = api.HERE / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                mismatches.append(name)
        import_errors = {}
        for name, module in (('numpy', 'numpy'), ('scipy', 'scipy'), ('shapely', 'shapely'),
                             ('libigl', 'igl'), ('pymeshlab', 'pymeshlab'), ('pyvista', 'pyvista')):
            if versions[name] is not None:
                try:
                    importlib.import_module(module)
                except Exception as exc:
                    import_errors[name] = str(exc)
        print(json.dumps(dict(dependencies=versions, import_errors=import_errors, modified_or_missing_files=mismatches), indent=2))
        if mismatches or import_errors or any(versions[name] is None for name in ('numpy', 'scipy', 'shapely', 'libigl', 'pymeshlab', 'pyvista')):
            raise SystemExit(1)
        return
    try:
        if args.command == 'run':
            result = api.run_config(args.config, log_dir=args.log_dir, on_output=emit)
        elif args.command == 'export-bare':
            result = api.export_bare(args.toolpath_dir, args.output_dir, machine_name=args.machine_name,
                                     machine_dir=args.machine_dir, start_main=args.start_main, end_main=args.end_main,
                                     options=extra, log_dir=args.log_dir, on_output=emit)
        else:
            if args.command == 'preview':
                script = 'visualize_layers_and_paths.py'
                parameters = [str(args.mesh_dir.resolve()), str(args.path_dir.resolve())]
                outputs = {}
            else:
                script = 'run_causal_path_lift.py'
                parameters = [str(args.input_dir.resolve()), str(args.output_dir.resolve())]
                outputs = {'lifted_paths': str(args.output_dir.resolve())}
            result = api._execute([sys.executable, '-u', str(api.RUNTIME / script), *parameters, *extra],
                                  cwd=Path.cwd(), logs_dir=args.log_dir, outputs=outputs, label=args.command,
                                  on_output=emit, check=True)
    except api.PipelineError as exc:
        print(json.dumps(asdict(exc.result)), file=sys.stderr)
        raise SystemExit(exc.result.returncode)
    print(json.dumps(asdict(result)))


if __name__ == '__main__':
    main()

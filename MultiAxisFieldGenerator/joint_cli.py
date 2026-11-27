"""Portable entry point for MAP's joint layer/path research beta.

Author: Neelotpal Dutta
Additional author details: [to be added]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def resolve(path, base):
    value = Path(path)
    return (value if value.is_absolute() else base/value).resolve()


def load_job(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf8'))
    if set(config)-{'output', 'model', 'initialization', 'training', 'preview'}:
        raise ValueError('Unknown joint beta job section')
    model = dict(config['model'])
    if set(model)-{'mesh', 'volume_mesh', 'vector_field', 'association', 'signed_field'}:
        raise ValueError('Unknown joint beta model setting')
    for key in ('mesh', 'volume_mesh'):
        model[key] = str(resolve(model[key], path.parent))
        if not Path(model[key]).is_file():
            raise FileNotFoundError(model[key])
    if model.get('association', 'cell') not in ('cell', 'point') or not model.get('vector_field'):
        raise ValueError('Select an original point/cell principal-stress vector')
    initialization = dict(config.get('initialization', {}))
    if set(initialization)-{'mode', 'checkpoint', 'up_axis', 'width', 'layers', 'frequency', 'steps'}:
        raise ValueError('Unknown joint beta initialization setting')
    if initialization.get('mode', 'build') not in ('build', 'load') or initialization.get('up_axis', 'z') not in ('x', 'y', 'z'):
        raise ValueError('Invalid initializer source or build axis')
    if initialization.get('mode') == 'load':
        initialization['checkpoint'] = str(resolve(initialization['checkpoint'], path.parent))
        if not Path(initialization['checkpoint']).is_file():
            raise FileNotFoundError(initialization['checkpoint'])
    preview = dict(config.get('preview', {}))
    if set(preview)-{'layers', 'path_contours'}:
        raise ValueError('Unknown preview setting')
    for key, default, maximum in (('layers', 40, 500), ('path_contours', 40, 1000)):
        value = preview.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f'Preview {key} must be 1..{maximum}')
    return dict(config, model=model, initialization=initialization, preview=preview,
                output=str(resolve(config.get('output', 'run'), path.parent)), _job=str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--native-threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        config = load_job(args.job)
        if args.native_threads < 1:
            raise ValueError('Native threads must be positive')
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        os.environ['OMP_NUM_THREADS'] = str(args.native_threads)
        os.environ['MKL_NUM_THREADS'] = str(args.native_threads)
        from .runtime.joint_path_beta import Options, train, export_preview, write_json, sha
        options = Options(**config.get('training', {}))
        if args.dry_run:
            print(json.dumps(config, indent=2))
            return 0
        import torch
        torch.set_num_threads(args.native_threads)
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; choose CPU or Auto')
        out = Path(config['output'])
        # Never adopt or overwrite a prior/resumable run.
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'joint_config.json', config)
        source_hashes = {config['model'][key]: sha(config['model'][key]) for key in ('mesh', 'volume_mesh')}
        settings = config['initialization']
        if settings.get('mode', 'build') == 'load':
            initializer = Path(settings['checkpoint'])
            source_hashes[str(initializer)] = sha(initializer)
        else:
            (out/'inputs').mkdir()
            initializer = out/'inputs'/'initializer.pt'
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve().parent/'runtime'/'build_checkpoint.py'),
                       '--kind', 'initializer', '--mesh', config['model']['mesh'], '--output', str(initializer),
                       '--width', str(settings.get('width', 128)), '--layers', str(settings.get('layers', 10)),
                       '--frequency', str(settings.get('frequency', 7.)), '--steps', str(settings.get('steps', 1000)),
                       '--samples', '32768', '--batch', '2048', '--gradient-weight', '.1', '--learning-rate', '.0001',
                       '--up-axis', settings.get('up_axis', 'z'), '--planar-mode', 'build-axis', '--device', device, '--seed', str(options.seed)]
            subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, check=True)
        f, g, mesh, center, scale, report = train(config, initializer, device, lambda line: print(line, flush=True))
        outputs = export_preview(config, f, g, mesh, center, scale)
        for source, digest in source_hashes.items():
            if sha(source) != digest:
                raise RuntimeError('A source input changed during training: '+source)
        report['source_files_unchanged'] = True
        report['outputs'] = outputs
        write_json(out/'joint_report.json', report)
        print(json.dumps(dict(stage='completed', outputs=outputs)), flush=True)
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, TypeError) as error:
        print('Joint layer/path beta: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

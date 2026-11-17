"""Create a code-only portable ZIP with a SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    files = [p for p in root.rglob('*') if p.is_file() and not any(
        name in {'__pycache__', 'cache', 'runs', 'dist'} for name in p.relative_to(root).parts)
        and p.suffix not in {'.zip', '.pyc', '.pt', '.npz'}]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in files:
            archive.write(p, 'MultiAxisFieldGenerator/'+p.relative_to(root).as_posix())
        archive.writestr('MultiAxisFieldGenerator/MANIFEST.json', json.dumps(manifest, indent=2))
    print(f'Packed {len(files)} files: {args.output.resolve()}')


if __name__ == '__main__':
    main()

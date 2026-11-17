"""Direct SDF builder entry point; also available through --action build-sdf."""
from pathlib import Path
import subprocess
import sys
if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).parent/'runtime/build_checkpoint.py'),
                                     '--kind', 'sdf', '--width', '256', '--layers', '5', '--frequency', '30']+sys.argv[1:]))

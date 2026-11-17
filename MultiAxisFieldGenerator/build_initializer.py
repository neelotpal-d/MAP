"""Direct height-initializer builder entry point."""
from pathlib import Path
import subprocess
import sys
if __name__ == '__main__':
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).parent/'runtime/build_checkpoint.py'),
                                     '--kind', 'initializer']+sys.argv[1:]))

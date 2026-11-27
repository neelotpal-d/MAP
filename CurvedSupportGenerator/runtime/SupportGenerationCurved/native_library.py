"""Load native geometry libraries with dependencies from the selected Python."""
from contextlib import ExitStack
from pathlib import Path
import ctypes,os,sys


def load_library(path):
    with ExitStack() as stack:
        dependencies=[]
        if os.name=='nt' and hasattr(os,'add_dll_directory'):
            for folder in (Path(sys.prefix),Path(sys.prefix)/'Library/bin',Path(path).resolve().parent):
                if folder.is_dir():stack.enter_context(os.add_dll_directory(str(folder)))
            openmp=Path(sys.prefix)/'Library/bin/vcomp140.dll'
            if openmp.is_file():dependencies.append(ctypes.WinDLL(str(openmp)))
        elif sys.platform=='darwin':
            openmp=Path(sys.prefix)/'lib/libomp.dylib'
            if openmp.is_file():
                try:dependencies.append(ctypes.CDLL(str(openmp),mode=ctypes.RTLD_GLOBAL))
                except OSError:pass  # A verified serial build does not require libomp.
        return ctypes.CDLL(str(path))

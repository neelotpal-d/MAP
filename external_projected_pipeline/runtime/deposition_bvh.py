"""Native six-plane triangle/box audit, independent of support extraction."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent


def build():
    """Compile privately, then publish atomically; concurrent exports may race."""
    import tempfile
    source = ROOT / 'deposition_bvh.cpp'
    cache=Path(os.environ.get('PROJECTED_NATIVE_CACHE',str(Path(tempfile.gettempdir())/'multi_axis_printing_native')))
    folder=cache/(sys.platform+'_'+str(ctypes.sizeof(ctypes.c_void_p)*8)+'_layer_audit_'+hashlib.sha256(source.read_bytes()).hexdigest()[:12])
    folder.mkdir(parents=True,exist_ok=True)
    suffix='.dll' if sys.platform=='win32' else ('.dylib' if sys.platform=='darwin' else '.so')
    library=folder/('layer_triangle_audit'+suffix)
    if library.is_file():return library
    with tempfile.TemporaryDirectory(prefix='build_',dir=folder) as work:
        built=_compile(Path(work),source,suffix)
        if not library.exists():
            try:os.replace(built,library)
            except PermissionError:
                if not library.is_file():raise
    return library


def _compile(folder,source,suffix):
    library = folder / ('layer_triangle_audit' + suffix)
    if library.exists():
        return library
    if sys.platform == 'win32':
        vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
        install = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
        if not install:
            raise RuntimeError('MSVC build tools unavailable.')
        vcvars = Path(install) / 'VC/Auxiliary/Build/vcvars64.bat'
        output = subprocess.check_output(f'call "{vcvars}" >nul && set', shell=True, text=True)
        environment = {key.upper(): value for key, value in os.environ.items()}
        for line in output.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                if key:
                    environment[key.upper()] = value
        compiler = shutil.which('cl.exe', path=environment['PATH'])
        command = [compiler, '/nologo', '/O2', '/EHsc', '/openmp', '/LD', str(source),
            '/Fo' + str(folder / 'audit.obj'), '/link', '/OUT:' + str(library), '/IMPLIB:' + str(folder / 'audit.lib')]
        subprocess.run(command, env=environment, cwd=folder, check=True)
    else:
        compiler = shutil.which('c++')
        flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fopenmp']
        subprocess.run([compiler, '-O3', '-std=c++17', '-fPIC', *flags, str(source), '-o', str(library)], check=True)
    return library


class TriangleAudit:
    def __init__(self, vertices, faces, offsets, ranks, parts, threads=4, face_ranks=None):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        self.ranks = np.ascontiguousarray(ranks, dtype=np.int32)
        self.parts = np.ascontiguousarray(parts, dtype=np.int32)
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3 and np.isfinite(self.vertices).all()
        assert self.faces.ndim == 2 and self.faces.shape[1] == 3 and self.faces.min() >= 0 and self.faces.max() < len(self.vertices)
        assert len(self.offsets) == len(ranks) + 1 and self.offsets[0] == 0 and self.offsets[-1] == len(faces)
        assert (np.diff(self.offsets) > 0).all() and len(parts) == len(ranks) and len(ranks) > 0
        self.threads = threads
        self.library_path = build()
        self.lib = ctypes.CDLL(str(self.library_path))
        self.dp = ctypes.POINTER(ctypes.c_double)
        self.ip = ctypes.POINTER(ctypes.c_int)
        self.lp = ctypes.POINTER(ctypes.c_int64)
        self.lib.layer_audit_create.argtypes = [self.dp, self.lp, ctypes.c_int, self.lp, self.ip, self.ip, ctypes.c_int]
        self.lib.layer_audit_create.restype = ctypes.c_void_p
        self.lib.layer_audit_destroy.argtypes = [ctypes.c_void_p]
        self.lib.layer_audit_query.argtypes = [ctypes.c_void_p, ctypes.c_int64, self.dp, self.ip, ctypes.c_int,
            self.dp, self.dp, ctypes.c_double, ctypes.c_int, self.lp, self.dp]
        self.face_ranks=None if face_ranks is None else np.ascontiguousarray(face_ranks,dtype=np.int32)
        if self.face_ranks is not None:assert self.face_ranks.shape==(len(self.faces),)
        creator=self.lib.layer_audit_create
        extra=[]
        if self.face_ranks is not None:
            creator=self.lib.layer_audit_create_ordered
            creator.argtypes=[*self.lib.layer_audit_create.argtypes,self.ip];creator.restype=ctypes.c_void_p
            extra=[self.face_ranks.ctypes.data_as(self.ip)]
        self.handle = creator(self.vertices.ctypes.data_as(self.dp), self.faces.ctypes.data_as(self.lp),
            len(ranks), self.offsets.ctypes.data_as(self.lp), self.ranks.ctypes.data_as(self.ip), self.parts.ctypes.data_as(self.ip), threads,*extra)
        if not self.handle:
            raise RuntimeError('Native audit BVH initialization failed.')

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.layer_audit_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def query(self, points, ranks, part, lower, upper, inset=0., threads=None):
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        ranks = np.ascontiguousarray(ranks, dtype=np.int32)
        lower = np.ascontiguousarray(lower, dtype=np.float64)
        upper = np.ascontiguousarray(upper, dtype=np.float64)
        assert len(points) == len(ranks) and np.isfinite(points).all() and np.all(lower + inset < upper - inset)
        faces = np.full(len(points), -1, np.int64)
        witness = np.full((len(points), 3), np.nan)
        self.lib.layer_audit_query(self.handle, len(points), points.ctypes.data_as(self.dp), ranks.ctypes.data_as(self.ip), part,
            lower.ctypes.data_as(self.dp), upper.ctypes.data_as(self.dp), inset, threads or self.threads,
            faces.ctypes.data_as(self.lp), witness.ctypes.data_as(self.dp))
        return faces, witness


def sat_intersects(triangle, lower, upper):
    """Independent separating-axis oracle, including edge cross-product axes."""
    half = (upper - lower) / 2
    triangle = np.asarray(triangle) - (upper + lower) / 2
    edges = np.roll(triangle, -1, axis=0) - triangle
    axes = np.vstack((np.eye(3), np.cross(edges[0], edges[1])[None],
                      np.cross(edges[:, None], np.eye(3)[None]).reshape(-1, 3)))
    projection = triangle @ axes.T
    extent = np.abs(axes) @ half
    return not np.any((projection.min(axis=0) > extent) | (projection.max(axis=0) < -extent))


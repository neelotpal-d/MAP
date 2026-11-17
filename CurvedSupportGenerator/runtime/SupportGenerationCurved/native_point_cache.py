"""Optional exact point cache, preserving sorted missing-query batches."""
import ctypes
from pathlib import Path
import sys
import numpy as np

SUFFIX = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
LIBRARY = Path(__file__).with_name('native_build') / 'point_cache_v1' / ('exact_point_cache' + SUFFIX)

class NativePointCache:
    def __init__(self, geometry):
        self.geometry = geometry
        self.requested = self.evaluated = 0
        self.lib = ctypes.CDLL(str(LIBRARY))
        dp = ctypes.POINTER(ctypes.c_double); ip = ctypes.POINTER(ctypes.c_int64)
        self.lib.point_cache_create.restype = ctypes.c_void_p
        self.lib.point_cache_destroy.argtypes = [ctypes.c_void_p]
        self.lib.point_cache_prepare.argtypes = [ctypes.c_void_p,ctypes.c_int64,dp,dp,dp,ip,ip,ip]
        self.lib.point_cache_prepare.restype = ctypes.c_int
        self.lib.point_cache_store.argtypes = [ctypes.c_void_p,ctypes.c_int64,dp,dp]
        self.lib.point_cache_store.restype = ctypes.c_int
        self.handle = self.lib.point_cache_create()
        if not self.handle: raise MemoryError('Native point cache allocation failed.')

    def __getattr__(self, name): return getattr(self.geometry, name)

    def close(self):
        if self.__dict__.get('handle'):
            self.lib.point_cache_destroy(self.handle); self.handle = None

    def __del__(self): self.close()

    def evaluate(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1,3)
        self.requested += len(points)
        if not len(points): return np.empty(0)
        if not np.isfinite(points).all(): raise ValueError('Nonfinite point-cache coordinates.')
        unique = np.empty_like(points); values = np.empty(len(points))
        inverse = np.empty(len(points), np.int64); missing = np.empty(len(points), np.int64)
        counts = np.empty(2, np.int64)
        dp = ctypes.POINTER(ctypes.c_double); ip = ctypes.POINTER(ctypes.c_int64)
        status = self.lib.point_cache_prepare(self.handle,len(points),points.ctypes.data_as(dp),
            unique.ctypes.data_as(dp),values.ctypes.data_as(dp),inverse.ctypes.data_as(ip),
            missing.ctypes.data_as(ip),counts.ctypes.data_as(ip))
        if status: raise MemoryError('Native point cache query failed.')
        if counts[1]:
            slots = missing[:counts[1]]
            queries = np.ascontiguousarray(unique[slots])
            computed = np.ascontiguousarray(self.geometry.evaluate(queries), dtype=np.float64).reshape(-1)
            if len(computed) != len(queries): raise ValueError('Scalar cache result count differs.')
            self.evaluated += len(queries)
            values[slots] = computed
            status = self.lib.point_cache_store(self.handle,len(queries),queries.ctypes.data_as(dp),computed.ctypes.data_as(dp))
            if status: raise MemoryError('Native point cache insertion failed.')
        return values[inverse]

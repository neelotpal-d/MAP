"""Standalone exact accelerator for hard, rounded, equal-gap tree geometry.

The BVH changes only candidate search/evaluation, not sweep radii or topology.
The existing model distance, plate and ceiling constraints are retained.
Unsupported configurations fail explicitly; there is no approximate fallback.
"""
import ctypes
from pathlib import Path
import sys
import time
import numpy as np

class LexCachedLayerField:
    """Same exact memoization with numeric lexsort instead of record sorting."""
    def __init__(self,geometry):
        self.geometry=geometry;self.cache={};self.requested=self.evaluated=0
        factory=getattr(geometry,'sample_cache_factory',None)
        self._sample_cache=factory(geometry) if factory is not None else None

    def __getattr__(self,name):return getattr(self.geometry,name)

    def evaluate(self,points):
        if self._sample_cache is not None:
            values=self._sample_cache.evaluate(points)
            self.requested=self._sample_cache.requested;self.evaluated=self._sample_cache.evaluated
            return values
        points=np.asarray(points,dtype=float).reshape(-1,3);self.requested+=len(points)
        if not len(points):return np.empty(0)
        order=np.lexsort((points[:,2],points[:,1],points[:,0]))
        sorted_points=points[order]
        first=np.r_[True,np.any(sorted_points[1:]!=sorted_points[:-1],axis=1)]
        unique=np.ascontiguousarray(sorted_points[first])
        inverse=np.empty(len(points),dtype=np.intp);inverse[order]=np.cumsum(first)-1
        # Fixed-width void conversion creates the same 24-byte coordinate keys
        # in a compiled loop rather than millions of Python row.tobytes calls.
        keys=unique.view(np.dtype((np.void,24))).ravel().tolist()
        values=[self.cache.get(key) for key in keys]
        missing=np.array([i for i,value in enumerate(values) if value is None],dtype=int)
        if len(missing):
            computed=self.geometry.evaluate(unique[missing]);self.evaluated+=len(missing)
            for i,value in zip(missing,computed):values[i]=self.cache[keys[i]]=float(value)
        return np.asarray(values)[inverse]

class ExactSupportBVH:
    def __init__(self,shape,threads=2,library_path=None,cache_backend='auto'):
        if shape.geometry_mode!='hard' or shape.terminal_cap!='rounded' or shape.side_gap!=shape.contact_gap or not shape.prune_pairs or shape.neighbor_backend!='arrays':
            raise ValueError('Exact BVH trial requires hard rounded sweeps, equal gaps and pair pruning.')
        self.shape=shape;self.threads=threads;self.original=shape.evaluate
        suffix='.dll' if sys.platform=='win32' else '.dylib' if sys.platform=='darwin' else '.so'
        root=Path(__file__).with_name('native_build')
        ordered=root/'v3_ordered'/('support_union_bvh'+suffix)
        path=Path(library_path) if library_path is not None else ordered if ordered.is_file() else root/'v2'/('support_union_bvh'+suffix)
        self.library_path=path
        if cache_backend not in ('auto','native','python'):raise ValueError('Unknown point-cache backend.')
        self.cache_backend='python'
        shape.sample_cache_factory=None
        if cache_backend!='python':
            try:
                from .native_point_cache import NativePointCache,LIBRARY
            except ImportError:
                if cache_backend=='native':raise
            else:
                if LIBRARY.is_file():
                    shape.sample_cache_factory=NativePointCache
                    self.cache_backend='native'
                elif cache_backend=='native':raise FileNotFoundError('Build the point cache with build_point_cache.py first.')
        self.lib=ctypes.CDLL(str(path))
        p=ctypes.POINTER(ctypes.c_double)
        self.lib.support_bvh_create.argtypes=[ctypes.c_longlong,p,p,p];self.lib.support_bvh_create.restype=ctypes.c_void_p
        self.lib.support_bvh_destroy.argtypes=[ctypes.c_void_p]
        self.lib.support_bvh_evaluate.argtypes=[ctypes.c_void_p,ctypes.c_longlong,p,p,ctypes.c_int,ctypes.c_int,ctypes.c_double]
        self.lib.support_bvh_overlap.argtypes=[ctypes.c_void_p,ctypes.c_longlong,p,p,p,ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int]
        rows=np.ascontiguousarray(shape.segments,dtype=float);up=np.ascontiguousarray(shape.direction,dtype=float)
        buckets=np.empty(len(rows),dtype=float)
        for _,ids,radius in shape._primitive_buckets:buckets[ids]=radius
        tick=time.perf_counter();self.handle=self.lib.support_bvh_create(len(rows),rows.ctypes.data_as(p),up.ctypes.data_as(p),buckets.ctypes.data_as(p))
        if not self.handle:raise RuntimeError('Native BVH construction failed.')
        self.build_seconds=time.perf_counter()-tick

    def close(self):
        if getattr(self,'handle',None):self.lib.support_bvh_destroy(self.handle);self.handle=None

    def __del__(self):self.close()

    def evaluate(self,points,distances=None):
        if distances is not None:return self.original(points,distances=distances)
        points=np.ascontiguousarray(points,dtype=float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('Nonfinite query coordinates.')
        result=np.empty(len(points),dtype=float);p=ctypes.POINTER(ctypes.c_double)
        self.lib.support_bvh_evaluate(self.handle,len(points),points.ctypes.data_as(p),result.ctypes.data_as(p),self.threads,self.shape.query_batch_size,self.shape.search_radius)
        possible=result<=0.
        if possible.any():
            distance=np.asarray(self.shape.bounded_distance(points[possible],self.shape.side_gap),dtype=float)
            result[possible]=np.maximum(result[possible],self.shape.side_gap-distance)
        return self.shape.constrain_bounds(points,result)

    def near_sweep_faces(self,surface,geometry=None):
        triangles=np.asarray(surface.points)[surface.regular_faces]
        centers=np.ascontiguousarray(triangles.mean(axis=1))
        half=np.ascontiguousarray(np.max(abs(triangles-centers[:,None]),axis=1))
        radius=np.ascontiguousarray(np.linalg.norm(triangles-centers[:,None],axis=2).max(axis=1))
        result=np.empty(len(centers),np.uint8);p=ctypes.POINTER(ctypes.c_double)
        self.lib.support_bvh_overlap(self.handle,len(centers),centers.ctypes.data_as(p),half.ctypes.data_as(p),radius.ctypes.data_as(p),result.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),self.threads)
        return result.astype(bool)

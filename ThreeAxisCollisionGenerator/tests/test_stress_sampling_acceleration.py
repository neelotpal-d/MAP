"""Spatially pruned FPS preserves original row identity and tie order."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from stressSampling import farthest_indices,filtered_farthest_indices


def reference(points,count,seed):
    count=min(int(count),len(points))
    if count<1:raise ValueError('FPS needs at least one point')
    nearest=np.full(len(points),np.inf)
    selected=np.empty(count,dtype=np.int64)
    index=int(np.random.default_rng(seed).integers(len(points)))
    for i in range(count):
        selected[i]=index
        delta=points-points[index]
        nearest=np.minimum(nearest,np.einsum('ij,ij->i',delta,delta))
        nearest[selected[:i+1]]=-1
        index=int(np.argmax(nearest))
    return selected


class ExactFPSAcceleration(unittest.TestCase):
    def compare(self,points,count,seed=544):
        before=reference(points,count,seed)
        after=farthest_indices(points,count,seed)
        np.testing.assert_array_equal(after,before)
        self.assertEqual(len(np.unique(after)),len(after))

    def test_random_noncontiguous_and_translated_geometry(self):
        for seed in (1,7,544):
            for shift,scale in ((0.,1.),(1e10,.001),(1e90,1e80)):
                with self.subTest(seed=seed,shift=shift):
                    points=np.random.default_rng(seed).normal(size=(4096,6))[:,::2]*scale+shift
                    self.compare(points,300,seed)

    def test_ties_collinear_duplicates_and_all_rows(self):
        grid=np.indices((16,16,4)).reshape(3,-1).T.astype(float)
        fixtures=[grid,np.repeat(grid,2,axis=0),np.zeros((1024,3)),
                  np.column_stack((np.arange(1024),np.zeros((1024,2)))).astype(float)]
        for points in fixtures:
            with self.subTest(unique=len(np.unique(points,axis=0))):
                self.compare(points,len(points)+10)

    def test_fallback_types_small_pools_and_absent_optional_spatial_module(self):
        points=np.random.default_rng(4).normal(size=(1024,3))
        for values in (points.astype(np.float32),(points*100).astype(np.int64),points[:100],points[:,:2]):
            with self.subTest(dtype=str(values.dtype),shape=values.shape):self.compare(values,90)
        with patch.dict(sys.modules,{'scipy.spatial':None}):self.compare(points,200)

    def test_tensile_filter_order_and_no_replacement(self):
        points=np.random.default_rng(11).normal(size=(2048,3));active=np.arange(len(points))%3==0
        for order in ('before','after'):
            candidates=np.flatnonzero(active) if order=='before' else np.arange(len(points))
            expected=candidates[reference(points[candidates],400,544)]
            retained,selected=filtered_farthest_indices(points,active,400,544,order)
            np.testing.assert_array_equal(selected,expected)
            np.testing.assert_array_equal(retained,expected[active[expected]])

    def test_empty_selection_errors(self):
        for points,count in ((np.empty((0,3)),1),(np.ones((10,3)),0)):
            with self.assertRaises(ValueError):farthest_indices(points,count,1)


if __name__=='__main__':unittest.main(verbosity=2)

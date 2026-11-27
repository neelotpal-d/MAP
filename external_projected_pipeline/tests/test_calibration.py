"""Budget, stability and numerical-agreement gates for worker calibration."""
from copy import deepcopy
import ctypes
from pathlib import Path
import sys
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from external_projected_pipeline.calibration import candidate_plan,recommend,run_calibration


def trials(stage='layers',workers=1,threads=1,times=(10.,10.1),sig='a'*64):
    return [dict(stage=stage,workers=workers,threads=threads,wall_seconds=t,
                 status='passed',signature=sig) for t in times]


class CalibrationPolicyTests(unittest.TestCase):
    def test_reserved_cpu_and_memory_budgets(self):
        plan=candidate_plan(32,2,cpu_count=20,system='Windows',available_memory=3*1024**3)
        self.assertTrue(plan)
        self.assertTrue(all(r['workers']<=3 and r['workers']*r['threads']<=18 for r in plan))
        self.assertEqual([r['threads'] for r in plan if r['stage']=='distance'],[1])
        self.assertIn(dict(stage='layers',workers=1,threads=2),plan)
        # Windows VTK is serial; the support BVH still has an independent budget.

    def test_single_cpu_and_invalid_limits(self):
        self.assertTrue(all(r['workers']==r['threads']==1 for r in candidate_plan(8,2,cpu_count=1)))
        for maximum,reserve in [(0,2),(33,2),(True,2),(4,-1)]:
            with self.assertRaises(ValueError):candidate_plan(maximum,reserve)

    def test_prefers_fewer_workers_when_within_five_percent(self):
        rows=trials(times=(10.2,10.3))+trials(workers=4,times=(10.,10.))
        _,selected,_=recommend(rows)
        self.assertEqual(selected['layers']['workers'],1)

    def test_chooses_faster_when_difference_is_material(self):
        rows=trials(times=(15.,15.1))+trials(workers=4,times=(10.,10.1))
        self.assertEqual(recommend(rows)[1]['layers']['workers'],4)

    def test_rejects_unstable_failed_missing_or_disagreeing_outputs(self):
        cases=[trials(times=(10.,20.)),trials(times=(10.,)),trials(sig=None),
               trials()+trials(workers=2,sig='b'*64)]
        failed=trials();failed[1]['status']='failed';cases.append(failed)
        invalid=trials();invalid[1]['wall_seconds']=float('nan');cases.append(invalid)
        for rows in cases:
            with self.subTest(rows=rows):self.assertNotIn('layers',recommend(rows)[1])

    def test_repeat_signatures_must_match_too(self):
        rows=trials();rows[1]['signature']='b'*64
        self.assertNotIn('layers',recommend(rows)[1])

    def test_trial_timeout_terminates_the_owned_child(self):
        original=subprocess.Popen;children=[]
        def sleep_instead_of_prepare(command,**kwargs):
            child=original([sys.executable,'-c','import time;time.sleep(60)'],**kwargs)
            children.append(child);return child
        with TemporaryDirectory() as tmp, \
             patch('external_projected_pipeline.calibration.subprocess.Popen',side_effect=sleep_instead_of_prepare), \
             self.assertRaises(TimeoutError):
            run_calibration(Path(tmp)/'timed_out',maximum_processes=1,timeout=.3)
        self.assertEqual(len(children),1)
        self.assertIsNotNone(children[0].poll())


class NativeBufferTests(unittest.TestCase):
    def test_float32_contours_are_converted_to_double_buffers(self):
        import numpy as np
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'CurvedSupportGenerator/runtime'))
        from SupportGenerationCurved.exact_support_bvh import ExactSupportBVH
        surface=SimpleNamespace(points=np.array([[0,0,0],[2,0,0],[0,2,0]],dtype=np.float32),
                                regular_faces=np.array([[0,1,2]]))
        def overlap(handle,count,centers,half,radius,out,threads):
            np.testing.assert_allclose(np.ctypeslib.as_array(centers,shape=(count*3,)),[2/3,2/3,0],rtol=1e-6)
            np.testing.assert_allclose(np.ctypeslib.as_array(half,shape=(count*3,)),[4/3,4/3,0],rtol=1e-6)
            out[0]=1
        fast=ExactSupportBVH.__new__(ExactSupportBVH)
        fast.handle=1;fast.threads=1
        fast.lib=SimpleNamespace(support_bvh_overlap=overlap,support_bvh_destroy=lambda _:None)
        self.assertEqual(fast.near_sweep_faces(surface).tolist(),[True])
        fast.close()


if __name__=='__main__':unittest.main()

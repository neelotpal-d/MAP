"""Exercise actual field optimizers with CLI schedules, in both training loops."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch
import pyvista as pv

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'runtime'))
sys.path.insert(0,str(PACKAGE.parent))
from scalarNetUtils import initialize_scalar_net
from ThreeAxisCollisionGenerator.cli import _flatten_job,_command

TRACE = '''
import json,runpy,sys,torch
from pathlib import Path
original=torch.optim.lr_scheduler.StepLR
events=[]
class TraceStepLR(original):
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs)
        events.append(self.optimizer.param_groups[0]['lr'])
        return result
torch.optim.lr_scheduler.StepLR=TraceStepLR
record,script,*arguments=sys.argv[1:]
sys.path.insert(0,str(Path(script).parent))
sys.argv=[script,*arguments]
try:runpy.run_path(script,run_name='__main__')
finally:Path(record).write_text(json.dumps(events))
'''


class LearningRateScheduleTests(unittest.TestCase):
    def test_decay_and_disabled_decay_reach_both_actual_training_loops(self):
        cache=PACKAGE/'cache';cache.mkdir(exist_ok=True)
        for fused in (True,False):
            for interval in (1,0):
                with self.subTest(fused=fused,interval=interval),tempfile.TemporaryDirectory(dir=cache) as tmp:
                    root=Path(tmp)
                    pv.Box(bounds=(-5,5,-5,5,-5,5)).triangulate().save(root/'cube.obj')
                    init=root/'runs/example/initializers/scalar_init_example_w7_w7_l1.pt'
                    init.parent.mkdir(parents=True)
                    net=initialize_scalar_net('cpu',num_layers=1,w0=7,w0_initial=7)
                    torch.save(dict(model_state_dict=net.state_dict(),num_layers=1,dim_hidden=128,w0=7,w0_initial=7),init)
                    job=dict(output='runs',workflow='support-free',pipeline=dict(
                        model=dict(model_name='example',mesh_path='cube.obj',w0=7,num_layers=1,device='cpu'),
                        training=dict(epochs=2,learning_rate=1e-5,batch_size=64,input_grid_spacing=3.,max_input_points=64,
                            input_points_per_epoch=32,warmup_epochs=0,norm_pool_size=64,disable_norm_banded_control=True,
                            save_every=0,no_fuse_regular_losses=not fused,
                            learning_rate_decay_every_epochs=interval,learning_rate_decay_factor=.5),
                        collision=dict(disable_collision=True),
                        surface_quality=dict(surface_sample_density=.1,skip_surface_selection=True,
                            skip_automatic_selection_window=True,skip_constraint_sample_preview=True,base_height=1.),
                        execution=dict(yes=True,skip_initializer_verification=True,skip_sdf_verification=True)))
                    file=root/'job.json';file.write_text(json.dumps(job))
                    command=_command(_flatten_job(job,file,None))
                    trace=root/'rates.json'
                    environment=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    run=subprocess.run([sys.executable,'-B','-c',TRACE,str(trace),*command[1:]],cwd=root,
                        capture_output=True,text=True,env=environment,timeout=120)
                    self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
                    rates=json.loads(trace.read_text())
                    if interval:
                        self.assertEqual(len(rates),3)
                        for actual,expected in zip(rates,(1e-5,5e-6,2.5e-6)):self.assertAlmostEqual(actual,expected,places=12)
                    else:self.assertEqual(rates,[])


if __name__=='__main__':unittest.main(verbosity=2)

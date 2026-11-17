"""Verify collision pruning and skipped losses preserve objective derivatives."""
import os
from pathlib import Path
import sys
import unittest
import contextlib
import io
import json
import tempfile
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
sys.path.insert(0, str(PACKAGE.parent))
from collision import ToolSamples, collision_loss
from run_model import surface_losses
from fields import ScalarField


class SmoothField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(1.3, dtype=torch.float64))
        self.bend = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.call_sizes = []

    def forward(self, points, compute_grads=False):
        self.call_sizes.append(len(points))
        x, y, z = points.unbind(1)
        values = self.amplitude*(.8*x+self.bend*y*y+.7*z+.05*x*z)
        result = dict(scalars=values[:,None])
        if compute_grads:
            result['grads'] = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        return result


class CollisionPerformanceTests(unittest.TestCase):
    def test_fast_training_step_updates_parameters_and_records_timing_policy(self):
        import numpy as np
        import pyvista as pv
        import run_model
        from MultiAxisFieldGenerator.cli import load_job, training_command
        package = Path(__file__).resolve().parents[1]
        cache = package/'cache/collision_performance_tests'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            folder = Path(temporary)
            mesh = pv.Box(bounds=(-5,5,-5,5,-5,5)).triangulate()
            mesh.save(folder/'model.obj')
            mesh.delaunay_3d().save(folder/'volume.vtu')
            torch.manual_seed(12)
            field = ScalarField(width=12,layers=2,frequency=3.)
            torch.save(dict(model_state_dict=field.state_dict(),field_spec=field.spec),folder/'initializer.pt')
            gate = ScalarField(width=12,layers=2,frequency=3.)
            with torch.no_grad():
                for parameter in gate.parameters():
                    parameter.zero_()
                gate.last_layer.bias.fill_(-1.)
            torch.save(dict(model_state_dict=gate.state_dict(),field_spec=gate.spec),folder/'sdf.pt')
            # A synthetic tiny tool extending just behind the tip creates a
            # known violation regardless of the random SIREN's orientation.
            # This ensures step two actually exercises online pair revisits.
            tool_config = json.loads((package/'tools/abb_training.json').read_text())
            tool_config.update(cone_distances=[-.01], inner_radius=.02, outer_radius=.03,
                               inner_heights=[-.01,.01], outer_heights=[-.01,.01],
                               outer_height_jitter_mm=0.)
            (folder/'tool.json').write_text(json.dumps(tool_config))
            points = np.random.default_rng(4).uniform(-4,4,size=(32,3))
            np.savetxt(folder/'stress.txt',np.column_stack((points,np.tile([1.,0.,0.],(32,1)))))
            job = dict(output='run',model=dict(mesh='model.obj',volume_mesh='volume.vtu',
                start_checkpoint='initializer.pt',sdf_checkpoint='sdf.pt',up_axis='z',base_band_mm=1.),
                tool=dict(training='tool.json',validation=str(package/'tools/abb_nominal.json')),
                training=dict(objective='stress',epochs=1,epoch_mode='sampled-steps',steps_per_epoch=2,
                    device='cpu',interior_batch=8,surface_batch=8,base_batch=8,grid_spacing=3.,
                    base_weight=0,anchor_weight=0,collision_roots=8,tool_volume_samples=0,
                    mining_candidates=16,validation_count=8,validation_batch=8),
                stress=dict(enabled=True,file='stress.txt',sampling='fps',region='all',batch=8))
            (folder/'job.json').write_text(json.dumps(job))
            values,_ = load_job(folder/'job.json')
            with patch.object(sys,'argv',training_command(values)[1:]), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(run_model, 'tool_witness_loss', wraps=run_model.tool_witness_loss) as witness_calls:
                run_model.main()
            self.assertGreater(witness_calls.call_count, 0, 'Smoke test must revisit a discovered pair')
            config = json.loads((folder/'run/config.json').read_text())
            self.assertEqual(config['hessian_evaluation'],'batched')
            self.assertEqual(config['collision_query_policy'],'penalty_active_gradients')
            events = [json.loads(line) for line in (folder/'run/metrics.jsonl').read_text().splitlines()]
            training = next(event for event in events if event['event']=='train')
            self.assertGreaterEqual(training['epoch_wall_seconds'],training['seconds'])
            self.assertGreater(training['mining_seconds'],0)
            self.assertLessEqual(training['mean_collision_gradient_pairs'],training['mean_collision_active_pairs'])
            self.assertEqual(training['support'],0)
            self.assertEqual(training['base'],0)
            self.assertEqual(events[-1]['event'],'complete')
            self.assertIn('online_witness', training)
            self.assertIn('online_witness_pairs', training)
            saved_witnesses = torch.load(folder/'run/online_witnesses.pt', weights_only=True)
            self.assertIn('offsets_mm', saved_witnesses)
            self.assertEqual(saved_witnesses['epoch'], 1)
            checkpoint = torch.load(folder/'run/latest.pt',weights_only=True)
            changed = [not torch.equal(value,checkpoint['model_state_dict'][key]) for key,value in field.state_dict().items()]
            self.assertTrue(any(changed))

    def test_batched_hessian_preserves_values_and_third_order_parameter_gradients(self):
        torch.manual_seed(19)
        original = ScalarField(width=8,layers=2,frequency=3.).double()
        points = torch.randn(7,3,dtype=torch.float64)
        results = []
        for batched in (False,True):
            model = ScalarField(width=8,layers=2,frequency=3.).double()
            model.load_state_dict(original.state_dict())
            model.batched_hessian = batched
            output = model(points.clone().requires_grad_(True),compute_grads=True,compute_hess=True)
            loss = output['hessian'].square().mean()+output['grads'].square().mean()+output['scalars'].square().mean()
            loss.backward()
            results.append((output,torch.cat([p.grad.reshape(-1) for p in model.parameters()])))
        for key in ('hessian','grads','scalars'):
            torch.testing.assert_close(results[0][0][key],results[1][0][key],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(results[0][1],results[1][1],rtol=1e-10,atol=1e-12)

    def test_siren_pruning_matches_full_backprop_with_normalization_and_gating(self):
        torch.manual_seed(7)
        original = ScalarField(width=8,layers=2,frequency=3.).double()
        state = original.state_dict()
        roots = torch.tensor([[0.,.2,0.],[.1,-.4,.2]],dtype=torch.float64)
        offsets = torch.tensor([[0.,0.,-.2],[0.,0.,.4], [.1,0.,-.3],
                                [0.,0.,.3], [0.,.1,-.2],[0.,0.,.5]],dtype=torch.float64)
        tool = ToolSamples(offsets,torch.tensor([0,0,1,1,2,2]))

        class Gate(torch.nn.Module):
            def forward(self, points, **kwargs):
                return dict(scalars=points[:,1:2]-.25)

        outputs = []
        for pruning in (False,True):
            model = ScalarField(width=8,layers=2,frequency=3.).double()
            model.load_state_dict(state)
            result = collision_loss(model,roots,tool,1.,torch.tensor([[-2.]*3,[2.]*3]),Gate(),
                normalize_collision_by_gradient=True,margin=.03,prune_zero_penalty=pruning)
            result['loss'].backward()
            outputs.append((result,torch.cat([p.grad.reshape(-1) for p in model.parameters()])))
        torch.testing.assert_close(outputs[0][0]['loss'],outputs[1][0]['loss'],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(outputs[0][1],outputs[1][1],rtol=1e-10,atol=1e-12)
        torch.testing.assert_close(outputs[0][0]['collision_mask'],outputs[1][0]['collision_mask'])

    def test_pruning_preserves_loss_masks_and_parameter_and_root_derivatives(self):
        roots = torch.tensor([[0.,.2,0.],[.1,-.4,.2]],dtype=torch.float64)
        offsets = torch.tensor([[0.,0.,-.2],[0.,0.,.4], [.1,0.,-.3],
                                [0.,0.,.3], [0.,.1,-.2],[0.,0.,.5]],dtype=torch.float64)
        tool = ToolSamples(offsets, torch.tensor([0,0,1,1,2,2]))
        bbox = torch.tensor([[-2.]*3,[2.]*3],dtype=torch.float64)
        for normalized in (False,True):
            for margin in (0., .1):
                with self.subTest(normalized=normalized,margin=margin):
                    results = []
                    for pruning in (False,True):
                        field = SmoothField()
                        points = roots.clone().requires_grad_(True)
                        out = collision_loss(field, points, tool, 1., bbox, margin=margin,
                            normalize_collision_by_gradient=normalized, prune_zero_penalty=pruning,
                            chunk_size=3)
                        out['loss'].backward()
                        results.append((out, [p.grad for p in field.parameters()],points.grad))
                    reference, actual = results
                    self.assertGreater(float(reference[0]['loss'].detach()),0)
                    self.assertLess(actual[0]['gradient_pairs'], reference[0]['gradient_pairs'])
                    for key in ('loss','root_scores','component_scores','differences','collision_mask','inside_mask'):
                        torch.testing.assert_close(actual[0][key],reference[0][key],rtol=1e-10,atol=1e-12)
                    for a,b in zip(actual[1],reference[1]):
                        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
                    torch.testing.assert_close(actual[2],reference[2],rtol=1e-10,atol=1e-12)

    def test_zero_penalty_omits_tool_derivative_graph_and_keeps_zero_gradients(self):
        for offset in (0., .4):
            field = SmoothField()
            tool = ToolSamples(torch.tensor([[0.,0.,offset]]*3,dtype=torch.float64),torch.arange(3))
            out = collision_loss(field, torch.zeros(2,3,dtype=torch.float64), tool, 1.,
                                 torch.tensor([[-2.]*3,[2.]*3],dtype=torch.float64))
            self.assertEqual(float(out['loss'].detach()),0)
            self.assertEqual(out['gradient_pairs'],0 if offset else 6)
            out['loss'].backward()
            for parameter in field.parameters():
                self.assertTrue(torch.equal(parameter.grad,torch.zeros_like(parameter.grad)))

    def test_disabled_surface_and_base_skip_work_but_preserve_active_gradients_and_rng(self):
        data = dict(surface=torch.tensor([[.1,.2,0.],[.2,.3,0.],[.3,.4,0.],[.4,.5,0.]],dtype=torch.float64),
                    normals=torch.tensor([[-1.,0.,0.]]*4,dtype=torch.float64),
                    surface_weights=torch.full((4,),.25,dtype=torch.float64),
                    base=torch.zeros(2,3,dtype=torch.float64),up_vector=torch.tensor([0.,0.,1.],dtype=torch.float64))
        roots = torch.tensor([[.1,.2,0.],[.2,.3,.1]],dtype=torch.float64)
        for axis in (None,torch.tensor([0.,0.,1.],dtype=torch.float64)):
            with self.subTest(late_orientation=axis is not None):
                results = []
                for sparse in (False,True):
                    field = SmoothField()
                    generator = torch.Generator().manual_seed(42)
                    options = dict(compute_support=False,compute_base=False,compute_anchor=False) if sparse else {}
                    result = surface_losses(field,data,4,132.,generator,base_count=2,
                        collision_roots=roots,orientation_axis=axis,orientation_max_angle_deg=30.,**options)
                    loss = result[0]*0+result[1]*0+result[2]*0+result[3]['grads'].square().mean()+5*result[4]
                    loss.backward()
                    results.append((loss, [p.grad for p in field.parameters()],generator.get_state(),field.call_sizes))
                torch.testing.assert_close(results[0][0],results[1][0])
                for first,second in zip(results[0][1],results[1][1]):
                    torch.testing.assert_close(first,second)
                torch.testing.assert_close(results[0][2],results[1][2])
                self.assertEqual(results[0][3],[8])
                self.assertEqual(results[1][3],[6 if axis is not None else 2])


if __name__ == '__main__':
    unittest.main()

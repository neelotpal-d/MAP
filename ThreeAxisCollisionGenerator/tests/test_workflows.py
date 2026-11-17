import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import subprocess
import sys
import unittest
import numpy as np
import torch
import pyvista as pv

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
sys.path.insert(0, str(PACKAGE.parent))
from scalarNetUtils import initialize_scalar_net
from strengthConstraints import StrengthConstraints
from ThreeAxisCollisionGenerator.cli import _flatten_job, _command


class WorkflowTests(unittest.TestCase):
    def test_standalone_modes_and_combination(self):
        cache = PACKAGE/'cache'; cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate().save(folder/'cube.obj')
            points = np.random.default_rng(1).uniform(-4, 4, (60, 3))
            stress = np.column_stack((points, np.tile([1., 0, 0], (60, 1)), np.where(np.arange(60)%2, 10., -10.)))
            np.savetxt(folder/'stress.txt', stress)
            for workflow in ['surface-quality', 'support-free', 'strength', 'combined']:
                model_dir = folder/'runs'/workflow
                init = model_dir/'initializers'/f'scalar_init_{workflow}_w7_w7_l1.pt'
                init.parent.mkdir(parents=True)
                net = initialize_scalar_net('cpu', num_layers=1, w0=7, w0_initial=7)
                torch.save(dict(model_state_dict=net.state_dict(), num_layers=1, dim_hidden=128, w0=7, w0_initial=7), init)
                job = dict(output='runs', workflow=workflow, pipeline=dict(
                    model=dict(model_name=workflow, mesh_path='cube.obj', w0=7, num_layers=1, device='cpu'),
                    training=dict(epochs=1, batch_size=64, input_grid_spacing=3., max_input_points=64,
                                  input_points_per_epoch=32, collision_roots_per_step=8, warmup_epochs=0,
                                  norm_pool_size=64, disable_norm_banded_control=True, save_every=0),
                    collision=dict(disable_collision=True),
                    surface_quality=dict(surface_sample_density=.1, skip_surface_selection=True, skip_automatic_selection_window=True,
                                         skip_constraint_sample_preview=True, base_height=1.),
                    execution=dict(yes=True, skip_initializer_verification=True, skip_sdf_verification=True)),
                    strength=dict(enabled=workflow in ['strength', 'combined'], file='stress.txt', weight=2.,
                                  signed_column=6, tensile_only=True, neighbors=4, max_distance_mm=20.))
                # Exercise both implementations of the shared loss hook.
                if workflow == 'strength':
                    job['pipeline']['training']['no_fuse_regular_losses'] = True
                path = folder/f'{workflow}.json'; path.write_text(json.dumps(job))
                values = _flatten_job(job, path, None)
                self.assertTrue(values['skip_sdf'])
                env = os.environ.copy(); env.pop('PYTHONPATH', None)
                result = subprocess.run(_command(values), cwd=folder, env=env, capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, workflow+'\n'+result.stdout+'\n'+result.stderr)
                records = list((model_dir/'checkpoints').glob('*_run_config.json'))
                self.assertEqual(len(records), 1)
                record = json.loads(records[0].read_text())
                self.assertEqual(record['losses']['support_free']['enabled'], workflow in ['support-free', 'combined'])
                self.assertEqual(record['losses']['surface_direction']['enabled'], workflow in ['surface-quality', 'combined'])
                self.assertEqual(record['losses']['strength']['enabled'], workflow in ['strength', 'combined'])
                self.assertTrue(list((model_dir/'checkpoints').glob('*.pt')))
                if workflow in ['strength', 'combined']:
                    metrics = [json.loads(s) for s in (model_dir/'checkpoints/strength_metrics.jsonl').read_text().splitlines()]
                    self.assertEqual(metrics[-1]['epoch'], 1)
                    self.assertGreater(metrics[-1]['active_queries'], 0)

    def test_compression_is_zero_weighted_and_direction_sign_is_irrelevant(self):
        cache = PACKAGE/'cache'; cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate().save(folder/'cube.obj')
            points = np.random.default_rng(7).uniform(-4, 4, (60, 3))
            rows = np.column_stack((points, np.tile([1., 0, 0], (60, 1)), np.where(np.arange(60)%2, 10., -10.)))
            np.savetxt(folder/'stress.txt', rows)
            args = SimpleNamespace(mesh_path=folder/'cube.obj', strength_file=folder/'stress.txt',
                strength_point_columns=[0, 1, 2], strength_direction_columns=[3, 4, 5], strength_signed_column=6,
                strength_all_signs=False, strength_min_tensile_stress=0., strength_seed=42,
                strength_neighbors=1, strength_max_distance_mm=10., strength_weight=1.)
            provider = StrengthConstraints(args, folder/'a')
            trace = np.trace(provider.projectors, axis1=1, axis2=2)
            compressed, tensile = np.where(trace == 0)[0][0], np.where(trace > .5)[0][0]
            query = torch.tensor((provider.tree.data[[compressed, tensile]]-provider.center)/provider.scale, dtype=torch.float32)
            gradients = torch.tensor([[1., 1, 0], [1., 1, 0]], requires_grad=True)
            provider.begin_epoch(0)
            loss = provider(query, gradients)
            loss.backward()
            torch.testing.assert_close(gradients.grad[0], torch.zeros(3))
            self.assertGreater(float(gradients.grad[1].norm()), 0)
            rows[:, 3:6] *= -1; np.savetxt(folder/'stress.txt', rows)
            other = StrengthConstraints(args, folder/'b')
            np.testing.assert_allclose(provider.projectors, other.projectors)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

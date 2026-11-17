import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pyvista as pv
import torch
from ThreeAxisCollisionGenerator.cli import _flatten_job
from ThreeAxisCollisionGenerator.import_fea import principal_eigenpair

PACKAGE = Path(__file__).resolve().parents[1]


class PortableUpdateTests(unittest.TestCase):
    def test_dominant_absolute_principal_before_tensile_filter(self):
        tensors = np.array([np.diag([-10., 2., 1.]), np.diag([-2., 10., 1.]), np.diag([-10., 10., 1.])])
        values, directions = principal_eigenpair(tensors)
        np.testing.assert_array_equal(values, [-10., 10., 10.])
        np.testing.assert_allclose(np.abs(directions), [[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    def test_strength_preset_and_fixed_axis(self):
        job = json.loads((PACKAGE/'example_strength_job.json').read_text())
        self.assertEqual(job['strength']['weight'], 1.)
        self.assertEqual(job['strength']['fps_filter_order'], 'after')
        self.assertEqual(job['pipeline']['model']['num_layers'], 10)
        self.assertEqual(job['pipeline']['model']['w0'], 'auto')
        self.assertEqual(job['pipeline']['training']['input_curvature_sample_fraction'], 1.)
        self.assertTrue(job['pipeline']['training']['input_curvature_boundary_coverage'])
        self.assertEqual(job['pipeline']['surface_quality']['base_direction_weight'], 0.)
        job['pipeline']['orientation'] = dict(disable_orientation=False)
        with self.assertRaisesRegex(ValueError, 'fixed tool axis'):
            _flatten_job(job, PACKAGE/'example_strength_job.json', None)

    def test_isolated_package_training_check_and_initial_final_view(self):
        cache = PACKAGE/'cache'; cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            folder = Path(temp).resolve()
            assert folder.is_relative_to(cache.resolve())
            bundle = folder/'ThreeAxisCollisionGenerator'; bundle.mkdir()
            for name in ('__init__.py', '__main__.py', 'cli.py', 'job_options.py'):
                shutil.copy2(PACKAGE/name, bundle/name)
            shutil.copytree(PACKAGE/'runtime', bundle/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate().save(folder/'cube.obj')
            rng = np.random.default_rng(3)
            points = rng.uniform(-4, 4, (60, 3))
            rows = np.c_[points, np.tile([1., 0., 0.], (60, 1)), np.where(np.arange(60)%3, 10., -100.)]
            np.savetxt(folder/'stress.txt', rows)
            (folder/'planes.json').write_text(json.dumps([dict(x_min=-1., x_max=1., z_min=-1., z_max=1., y_value=.5, num_x=3, num_z=3)]))
            job = dict(output='runs', workflow='strength', pipeline=dict(
                model=dict(model_name='part', mesh_path='cube.obj', w0='auto', w0_initial='auto', num_layers=1, device='cpu'),
                initializer=dict(initializer_epochs=1, initializer_max_input_points=16, initializer_grid_spacing=4., initializer_batch_size=16),
                sdf=dict(sdf_epochs=1, sdf_grid_spacing=4., sdf_point_limit=16, sdf_surface_point_limit=16),
                training=dict(epochs=2, batch_size=7, input_grid_spacing=3., max_input_points=32, input_points_per_epoch=32,
                    collision_roots_per_step=8, warmup_epochs=0, save_every=0, norm_pool_size=32,
                    gradient_band_ratio_weight=10., gradient_band_floor_weight=1., enable_input_curvature=True,
                    input_curvature_mode='geometric', input_curvature_weight=.1, input_curvature_sample_fraction=1.,
                    input_curvature_boundary_coverage=True),
                surface_quality=dict(base_direction_weight=0., surface_sample_density=.1, skip_surface_selection=True,
                    skip_automatic_selection_window=True, skip_constraint_sample_preview=True),
                collision=dict(disable_collision=True, normalize_collision_by_gradient=True, tool_plane_config='planes.json',
                    collision_weight=1000., collision_validation_count=8, mining_candidate_count=32, scoring_root_batch_size=8),
                execution=dict(yes=True, skip_initializer_verification=True, skip_sdf_verification=True)),
                strength=dict(enabled=True, file='stress.txt', weight=1., sampling='fps', samples=13,
                    batch_size=5, fps_filter_order='after', tensile_only=True, signed_column=6))
            env = os.environ.copy(); env.pop('PYTHONPATH', None)
            log = cache/'portable_update_smoke.log'
            with log.open('w') as output:
                for collision in (False, True):
                    current = copy.deepcopy(job)
                    current['pipeline']['model']['model_name'] = 'collision_on' if collision else 'collision_off'
                    current['pipeline']['collision']['disable_collision'] = not collision
                    path = folder/'job.json'; path.write_text(json.dumps(current))
                    result = subprocess.run([sys.executable, '-m', 'ThreeAxisCollisionGenerator', str(path)],
                        cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=240)
                    output.flush()
                    self.assertEqual(result.returncode, 0, log.read_text())
                    run = folder/'runs'/current['pipeline']['model']['model_name']
                    config = json.loads(next((run/'checkpoints').glob('*run_config.json')).read_text())
                    saved = torch.load(next((run/'checkpoints').glob('*.pt')), map_location='cpu', weights_only=False)
                    self.assertEqual(saved['epoch'], 2)
                    self.assertTrue(all(torch.isfinite(t).all() for t in saved['model_state_dict'].values()))
                    self.assertEqual(config['arguments']['w0'], .5)
                    self.assertEqual(config['arguments']['w0_initial'], .5)
                    self.assertFalse(config['orientation_search']['enabled'])
                    self.assertEqual(config['losses']['base_direction'], dict(enabled=False, weight=0.))
                    self.assertEqual(config['losses']['gradient_ratio'], dict(enabled=True, weight=10.))
                    self.assertEqual(config['losses']['gradient_floor'], dict(enabled=True, weight=1.))
                    self.assertEqual(config['losses']['collision']['enabled'], collision)
                    self.assertEqual(bool(list((run/'sdf').glob('*.pt'))), collision)
                    coverage = [json.loads(s) for s in (run/'checkpoints/curvature_coverage.jsonl').read_text().splitlines()]
                    self.assertEqual(len(coverage), 2)
                    for row in coverage:
                        self.assertEqual(row['interior_unique_seen'], row['interior_pool'])
                        self.assertEqual(row['boundary_unique_seen'], 8)
                    with np.load(run/'checkpoints/strength_fps_sources.npz') as selected:
                        self.assertEqual(len(selected['source_ids_before_filter']), 13)
                        self.assertTrue((rows[selected['source_ids'], 6] > 0).all())
                        self.assertLess(len(selected['source_ids']), 13)
                result = subprocess.run([sys.executable, '-m', 'ThreeAxisCollisionGenerator', '--run-dir', str(run),
                    '--view', '--device', 'cpu', '--off-screen', '--screenshot', str(cache/'portable_update_layers.png'),
                    '--spacing', '3', '--layer-height', '2', '--max-layers', '6'],
                    cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
                output.flush()
                self.assertEqual(result.returncode, 0, log.read_text())
                self.assertTrue((cache/'portable_update_layers.png').is_file())
                tool = dict(build_direction=[0, 1, 0], in_plane_u=[1, 0, 0], tool_x_min=-1., tool_x_max=1.,
                    tool_y_min=.5, tool_y_max=3., tool_z_min=-1., tool_z_max=1.)
                (folder/'tool.json').write_text(json.dumps(tool))
                result = subprocess.run([sys.executable, '-m', 'ThreeAxisCollisionGenerator', '--run-dir', str(run),
                    '--collision-check', '--tool-config', str(folder/'tool.json'), '--device', 'cpu',
                    '--grid-spacing', '4', '--max-sampled-points', '32', '--origin-batch-size', '8'],
                    cwd=folder, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
                output.flush()
                self.assertEqual(result.returncode, 0, log.read_text())


if __name__ == '__main__':
    unittest.main()

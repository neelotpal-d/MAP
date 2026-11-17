import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
import pyvista as pv

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
sys.path.insert(0, str(PACKAGE.parent))
from fields import ScalarField
from stress import DirectionInterpolator, read_directions
from root_sampling import SurfaceRootDomain
from MultiAxisFieldGenerator.cli import load_job, training_command
from collision_settings import collision_enabled


class PortableTests(unittest.TestCase):
    def test_collision_activation_respects_schedule_and_run_length(self):
        self.assertFalse(collision_enabled(0, 100))
        self.assertFalse(collision_enabled(10000, 100, ['1:0']))
        self.assertFalse(collision_enabled(0, 100, ['101:10000']))
        self.assertTrue(collision_enabled(0, 100, ['26:10000']))

    def test_builders_and_training_without_sdf(self):
        cache = PACKAGE/'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate().save(folder/'cube.obj')
            job = dict(output='run', model=dict(mesh='cube.obj', up_axis='z', base_band_mm=1.),
                       training=dict(epochs=1, collision_weight=0, device='cpu', interior_batch=8192,
                                     surface_batch=8192, base_batch=8192, validation_count=8, validation_batch=8,
                                     grid_spacing=2., collision_roots=16, tool_volume_samples=0),
                       builders=dict(initializer=dict(steps=2, samples=64, validation_samples=16, batch=16, width=8, layers=2),
                                     sdf=dict(steps=2, samples=64, validation_samples=16, batch=16, width=8, layers=2)))
            path = folder/'job.json'; path.write_text(json.dumps(job))
            for action in ['build-initializer', 'build-sdf']:
                result = subprocess.run([sys.executable, '-m', 'MultiAxisFieldGenerator', str(path), '--action', action],
                                        cwd=PACKAGE.parent, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            self.assertTrue((folder/'run/inputs/sdf.pt').exists())
            sdf = torch.load(folder/'run/inputs/sdf.pt', weights_only=True)
            self.assertEqual(sdf['kind'], 'sdf')
            self.assertIn('sign_accuracy_off_surface', sdf['builder']['metrics'])
            # Leave an intentionally nonexistent SDF path in the job. Neither
            # the wrapper nor trainer may require/load it when weight is zero.
            job['model']['sdf_checkpoint'] = 'does_not_exist.pt'
            path.write_text(json.dumps(job))
            values, _ = load_job(path)
            result = subprocess.run(training_command(values), capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            output = folder/'run'
            rows = [json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
            self.assertFalse(any('benchmark' in r['event'] or r['event']=='sdf_root_check' for r in rows))
            final = next(r for r in reversed(rows) if r['event']=='validation')
            self.assertIsNone(final['root_collision_fraction'])
            self.assertEqual(final['gate'], 'not_checked')
            config = json.loads((output/'config.json').read_text())
            self.assertFalse(config['collision_training_enabled'])
            self.assertNotIn('sdf', config['input_sha256'])
            command = [sys.executable, str(PACKAGE/'runtime/visualize_fertility.py'), '--run', str(output),
                       '--layers', '3', '--off-screen', '--check-controls']
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            self.assertEqual(json.loads((output/'visualization/scene.json').read_text())['gate'], 'mesh')

    def test_stress_interpolation_is_sign_invariant(self):
        points = np.array([[0., 0, 0], [1., 0, 0]])
        directions = np.array([[1., 0, 0], [-1., 0, 0]])
        matrices, valid, distances = DirectionInterpolator(points, directions).query([[.5, 0, 0]])
        np.testing.assert_allclose(matrices[0], np.diag([1., 0, 0]))
        self.assertTrue(valid.all())
        _, valid, _ = DirectionInterpolator(points, directions, max_distance_mm=.1).query([[.5, 0, 0]])
        self.assertFalse(valid.any())

    def test_surface_only_training_stress_and_tet_training(self):
        cache = PACKAGE/'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            mesh = pv.Box(bounds=(-5, 5, -5, 5, -5, 5)).triangulate()
            mesh.save(folder/'cube.obj')
            mesh.delaunay_3d().save(folder/'cube.vtu')
            torch.manual_seed(6)
            field = ScalarField(width=12, layers=2, frequency=7)
            for name in ['initializer', 'sdf']:
                torch.save(dict(model_state_dict=field.state_dict(), field_spec=field.spec), folder/f'{name}.pt')
            rng = np.random.default_rng(1)
            points = rng.uniform(-4, 4, size=(50, 3))
            stress = np.column_stack((points, np.tile([1., 0, 0], (50, 1)), np.zeros((50, 3))))
            np.savetxt(folder/'stress.txt', stress)
            domain = SurfaceRootDomain(folder/'cube.obj', np.zeros(3), 5., 'cpu')
            samples = domain.sample(100, torch.Generator().manual_seed(4)).numpy()*5
            self.assertTrue((np.abs(samples) < 5).all())
            for mode in ['tet', 'surface_stress']:
                job = dict(output=f'run_{mode}', model=dict(mesh='cube.obj', start_checkpoint='initializer.pt',
                           sdf_checkpoint='sdf.pt', up_axis='z', base_band_mm=1.),
                           tool=dict(training=str(PACKAGE/'tools/abb_training.json'), validation=str(PACKAGE/'tools/abb_nominal.json')),
                           training=dict(epochs=1, device='cpu', interior_batch=8192, surface_batch=8192, base_batch=8192,
                                         grid_spacing=2., validation_count=8, validation_batch=8, collision_roots=16,
                                         tool_volume_samples=4, mining_candidates=32, validate_every=1), stress=dict(enabled=False))
                if mode == 'tet':
                    job['model']['volume_mesh'] = 'cube.vtu'
                else:
                    job['training']['objective'] = 'combined'
                    job['stress'] = dict(enabled=True, file='stress.txt', samples=16, batch=8192, neighbors=4)
                path = folder/f'{mode}.json'
                path.write_text(json.dumps(job))
                values, _ = load_job(path)
                command = training_command(values)
                result = subprocess.run(command, cwd=PACKAGE/'runtime', capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                output = folder/f'run_{mode}'
                rows = [json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
                self.assertTrue(any(x['event'] == 'complete' for x in rows))
                config = json.loads((output/'config.json').read_text())
                self.assertEqual(config['normalization']['up_axis'], 'z')
                self.assertEqual(config['sdf_spec']['width'], 12)
                if mode == 'surface_stress':
                    self.assertTrue(any(x['event'] == 'stress_resampling' for x in rows))
                    self.assertIn('stress_validation_mean_squared_dot', next(x for x in reversed(rows) if x['event']=='validation'))
                # Check each saved field using nominal analytical geometry,
                # including the surface-only obstacle sampling fallback.
                command = [sys.executable, str(PACKAGE/'runtime/validate_nominal_collision.py'), '--run', str(output),
                           '--volume-samples', '12', '--surface-samples', '12', '--max-roots', '8', '--device', 'cpu']
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                self.assertTrue((output/'nominal_validation/report.json').exists())
                # Preview the non-Y build axis and both tet and clipped-grid
                # contour backends, including initial/final slider controls.
                command = [sys.executable, str(PACKAGE/'runtime/visualize_fertility.py'), '--run', str(output),
                           '--compare', '--layers', '4', '--off-screen', '--check-controls']
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                self.assertTrue((output/'visualization_comparison/viewer.png').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

"""Stress-informed planar selection and real builder gradient-target coverage."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import igl
import numpy as np
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
from build_checkpoint import best_planar_normal, stress_informed_normal
from fields import load_field, first_order_output


def projectors(directions):
    directions = np.asarray(directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return directions[:, :, None]*directions[:, None, :]


class StressPlanarInitializerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = PACKAGE/'cache'/'stress_planar_tests'
        cls.cache.mkdir(parents=True, exist_ok=True)

    def test_minimizes_planar_alignment_and_is_sign_invariant(self):
        directions = [[1., 0., 0.], [0., 1., -1.], [-1., 0., 0.], [0., -1., 1.]]
        normal, report = best_planar_normal(projectors(directions), [0., 0., 1.])
        expected = np.array([0., 1., 1.])/np.sqrt(2.)
        np.testing.assert_allclose(normal, expected, atol=1e-8)
        np.testing.assert_allclose(projectors(directions) @ normal, 0., atol=1e-8)
        flipped = np.asarray(directions)*np.array([1., -1., -1., 1.])[:, None]
        other, _ = best_planar_normal(projectors(flipped), [0., 0., 1.])
        np.testing.assert_allclose(other, normal, atol=1e-8)
        self.assertAlmostEqual(report['planar_stress_loss'], 0.)
        self.assertEqual(report['minimizing_eigenspace_dimension'], 1)

    def test_ties_choose_the_projected_build_axis_deterministically(self):
        data = projectors([[1., 0., 0.], [-1., 0., 0.]])
        normal, report = best_planar_normal(data, [0., 0., 1.])
        np.testing.assert_allclose(normal, [0., 0., 1.])
        self.assertEqual(report['minimizing_eigenspace_dimension'], 2)
        normal, _ = best_planar_normal(data, [1., 0., 0.])
        np.testing.assert_allclose(normal, [0., 1., 0.])
        normal, _ = best_planar_normal(projectors(np.eye(3)), [1., 2., 3.])
        np.testing.assert_allclose(normal, np.array([1., 2., 3.])/np.sqrt(14.))

    def make_job(self, folder, region='all', up='z'):
        volume = pv.ImageData(dimensions=(3, 3, 3), spacing=(2.,)*3).cast_to_unstructured_grid().triangulate()
        points = volume.cell_centers().points
        volume.cell_data['signed'] = np.where(points[:, 0] > 2., 2., -3.)
        volume.save(folder/'volume.vtu')
        volume.extract_surface(algorithm='dataset_surface').triangulate().save(folder/'model.obj')
        directions = np.tile([0., 0., 1.], (len(points), 1))
        if region != 'all':
            directions[points[:, 0] <= 2.] = [0., 1., 0.]
        np.savetxt(folder/'stress.txt', np.column_stack((points, directions)))
        job = dict(output='run', model=dict(mesh='model.obj', volume_mesh='volume.vtu', up_axis=up,
                       base_band_mm=.2, field_width=16, field_layers=2, field_frequency=2.),
                   training=dict(objective='stress', collision_weight=0, grid_spacing=1., device='cpu'),
                   stress=dict(enabled=True, file='stress.txt', sampling='fps', region=region,
                               signed_field='signed', threshold=0.),
                   builders=dict(initializer=dict(planar_mode='stress-informed', steps=300, samples=512,
                       validation_samples=128, batch=128, learning_rate=.002, gradient_weight=1.)))
        (folder/'job.json').write_text(json.dumps(job))
        return folder/'job.json'

    def test_fps_selection_respects_tensile_region_and_training_split(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            job = self.make_job(folder, region='tensile', up='y')
            vertices, faces = igl.read_triangle_mesh(str(folder/'model.obj'))
            normal, report = stress_informed_normal(job, folder/'model.obj', vertices, faces,
                                                    np.array([2., 2., 2.]), 2., folder/'initial.pt')
            np.testing.assert_allclose(normal, [0., 1., 0.], atol=1e-8)
            np.testing.assert_allclose(report['mean_direction_projector'], np.diag([0., 0., 1.]))
            self.assertEqual(report['stress_region'], 'tensile')
            self.assertEqual(report['stress_source'], 'exact_fps_training_projectors')
            self.assertGreater(report['selected_training_samples'], 0)
            self.assertLess(report['selected_training_samples'], report['stress_metadata']['source_samples'])
            self.assertGreater(report['stress_metadata']['validation_source_samples'], 0)

    def test_real_public_builder_fits_an_arbitrary_planar_normal(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            job = self.make_job(folder)
            result = subprocess.run([sys.executable, '-m', 'MultiAxisFieldGenerator', str(job),
                                     '--action', 'build-initializer'], cwd=PACKAGE.parent,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            path = folder/'run'/'inputs'/'initializer.pt'
            field, saved = load_field(path, 'cpu')
            initialization = saved['initialization']
            np.testing.assert_allclose(initialization['normal_world'], [1., 0., 0.], atol=1e-7)
            self.assertEqual(initialization['planar_mode'], 'stress-informed')
            self.assertEqual(saved['normalization']['up_axis'], 'z')
            points = torch.tensor(np.random.default_rng(7).uniform(-.8, .8, (64, 3)), dtype=torch.float32)
            prediction = first_order_output(field, points)
            self.assertLess(float((prediction['scalars'].reshape(-1)-(points[:, 0]+1)).abs().mean()), .15)
            self.assertLess(float((prediction['grads']-torch.tensor([1., 0., 0.])).square().mean()), .1)
            report = json.loads(path.with_suffix('.build.json').read_text())
            self.assertEqual(report['initialization'], initialization)

    def test_resampled_mode_uses_filtered_normalized_training_sources(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            job = self.make_job(folder, region='tensile', up='y')
            settings = json.loads(job.read_text())
            settings['stress']['sampling'] = 'resampled'
            job.write_text(json.dumps(settings))
            rows = np.loadtxt(folder/'stress.txt')
            rows[:, 3:] *= np.random.default_rng(11).choice([-100., -.1, 2., 500.], len(rows))[:, None]
            np.savetxt(folder/'stress.txt', rows)
            vertices, faces = igl.read_triangle_mesh(str(folder/'model.obj'))
            normal, report = stress_informed_normal(job, folder/'model.obj', vertices, faces,
                                                    np.array([2., 2., 2.]), 2., folder/'initial.pt')
            np.testing.assert_allclose(normal, [0., 1., 0.], atol=1e-8)
            np.testing.assert_allclose(report['mean_direction_projector'], np.diag([0., 0., 1.]))
            self.assertEqual(report['stress_source'], 'training_source_projectors_for_resampled_stress')
            self.assertEqual(report['selected_training_samples'], report['stress_metadata']['training_source_samples'])


if __name__ == '__main__':
    unittest.main()

"""Percentile stress selection, initializer consistency and public worker export."""
import json
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import igl
import numpy as np
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
APP = PACKAGE.parent/'MultiAxisPrinting-dev'
sys.path.insert(0, str(PACKAGE/'runtime'))
from stress import StressRegion, StressConstraints
from build_checkpoint import stress_informed_normal
from fields import ScalarField


class TopStressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = PACKAGE/'cache'
        cls.cache.mkdir(exist_ok=True)
        torch.set_num_threads(2)

    def cell_fixture(self, folder, values):
        mesh = pv.ImageData(dimensions=(len(values)+1, 2, 2)).cast_to_unstructured_grid()
        mesh.cell_data['signed'] = np.asarray(values, dtype=float)
        path = folder/'volume.vtu'
        mesh.save(path)
        return path, mesh.cell_centers().points

    def test_signed_regions_cutoff_ties_outside_and_legacy_default(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            path, points = self.cell_fixture(folder, [1, 2, 3, 4, 5, -1000, np.nan, np.inf])
            selected = StressRegion(path, 'signed', 'tensile', top_percent=30)
            np.testing.assert_array_equal(selected.mask(points), [False, False, False, True, True, False, False, False])
            self.assertEqual(selected.metadata['eligible_cells'], 5)
            self.assertEqual(selected.metadata['selected_cells'], 2)
            self.assertEqual(selected.metadata['magnitude_cutoff'], 4)
            self.assertFalse(selected.mask([[100, 0, 0]])[0])
            self.assertEqual(len(selected.mask([])), 0)
            np.testing.assert_array_equal(StressRegion(path, 'signed', 'tensile').mask(points),
                                          [True, True, True, True, True, False, False, False])
            path, points = self.cell_fixture(folder, [-1, -2, -3, -4, -5, 1000])
            np.testing.assert_array_equal(StressRegion(path, 'signed', 'compressive', top_percent=30).mask(points),
                                          [False, False, False, True, True, False])
            np.testing.assert_array_equal(StressRegion(path, 'signed', 'all', top_percent=30).mask(points),
                                          [False, False, False, False, True, True])
            path, points = self.cell_fixture(folder, [1, 2, 3, 3])
            self.assertEqual(StressRegion(path, 'signed', 'tensile', top_percent=25).metadata['selected_cells'], 2)

    def test_threshold_precedes_ranking_and_invalid_percentages_fail(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            path, points = self.cell_fixture(folder, [1, 2, 3, 4, 5, 6])
            selection = StressRegion(path, 'signed', 'tensile', threshold=3, top_percent=50)
            self.assertEqual(selection.metadata['eligible_cells'], 3)
            np.testing.assert_array_equal(selection.mask(points), [False, False, False, False, True, True])
            for value in [0, -1, 101, np.nan, np.inf]:
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'percentage'):
                    StressRegion(path, 'signed', 'all', top_percent=value)
            with self.assertRaisesRegex(ValueError, 'No finite FEA cells'):
                StressRegion(path, 'signed', 'compressive', top_percent=30)

    def source_fixture(self, folder):
        volume = pv.ImageData(dimensions=(5, 4, 4)).cast_to_unstructured_grid().triangulate()
        points = volume.cell_centers().points
        magnitudes = np.empty(volume.n_cells)
        magnitudes[np.argsort(points[:, 0], kind='stable')] = np.arange(1, volume.n_cells+1)
        count = int(np.ceil(.3*volume.n_cells))
        high = magnitudes >= volume.n_cells-count+1
        directions = np.tile([1., 0., 0.], (volume.n_cells, 1))
        directions[high] = [0., 1., 0.]
        volume.cell_data['abs_max_principal_stress_signed'] = magnitudes
        volume.cell_data['abs_max_principal_stress_direction'] = directions
        volume.save(folder/'fea.vtu')
        surface = volume.extract_surface(algorithm='dataset_surface').triangulate()
        surface.clear_data()
        surface.save(folder/'model.obj')
        np.savetxt(folder/'stress.txt', np.column_stack((points, directions)))
        (folder/'fea_export.json').write_text(json.dumps(dict(association='cell', field='abs_max_principal_stress_direction')))
        return volume, points, count

    def test_invalid_percent_rejected_even_without_region_filter(self):
        for value in [0, -1, 101, np.nan, np.inf]:
            args = SimpleNamespace(stress_top_percent=value, stress_region='all')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'percentage'):
                StressConstraints(args, np.zeros(3), 1., 'cpu', None, None)

    def test_fps_and_resampled_targets_remain_in_full_fea_top_region(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            volume, points, count = self.source_fixture(folder)
            vertices, faces = igl.read_triangle_mesh(str(folder/'model.obj'))
            center, scale = np.array([2., 1.5, 1.5]), 2.
            world = points[np.random.default_rng(7).permutation(len(points))]
            class Domain:
                def sample(self, n, generator):
                    return torch.tensor((world[np.arange(n) % len(world)]-center)/scale, dtype=torch.float32)
            for sampling in ['fps', 'resampled']:
                args = SimpleNamespace(stress_file=folder/'stress.txt', stress_point_columns=[0, 1, 2],
                    stress_direction_columns=[3, 4, 5], stress_coordinate_space='world', stress_region='tensile',
                    stress_top_percent=30., volume_mesh=folder/'fea.vtu', stress_signed_field='abs_max_principal_stress_signed',
                    stress_threshold=0., stress_sampling=sampling, seed=42, stress_neighbors=4,
                    stress_max_distance_mm=None, stress_samples=256)
                stress = StressConstraints(args, center, scale, 'cpu', vertices, faces)
                self.assertGreater(stress.resample(Domain(), None), 0)
                self.assertEqual(stress.metadata['selection']['selected_cells'], count)
                self.assertEqual(stress.metadata['selection']['eligible_cells'], volume.n_cells)
                self.assertTrue(stress.region.mask(stress.points_epoch.numpy()*scale+center).all())
                self.assertTrue(stress.region.mask(stress.validation_points.numpy()*scale+center).all())
                np.testing.assert_allclose(stress.matrices_epoch[:, 1, 1], 1., atol=1e-6)
                # The selected-subset mean keeps its weight: it is not divided
                # again by the requested percentage of the original dataset.
                gradients = torch.tensor([[0., 5., 0.]]).repeat(len(stress.points_epoch), 1)
                self.assertAlmostEqual(float(stress.loss(gradients, torch.arange(len(gradients)))), 1., places=6)

    def job(self, folder, sampling):
        return dict(output='run_'+sampling, model=dict(mesh='model.obj', volume_mesh='fea.vtu',
            start_checkpoint='initializer.pt', up_axis='z', base_band_mm=.2),
            training=dict(objective='combined', collision_weight=0, device='cpu', epochs=1,
                grid_spacing=.5, interior_batch=8192, surface_batch=8192, base_batch=8192,
                validation_count=16, validation_batch=16),
            stress=dict(enabled=True, file='stress.txt', weight=1., region='tensile',
                signed_field='abs_max_principal_stress_signed', top_percent=30., sampling=sampling,
                samples=256, batch=8192),
            builders=dict(initializer=dict(planar_mode='stress-informed')))

    def test_stress_informed_initializer_uses_same_percentile_for_both_modes(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            self.source_fixture(folder)
            vertices, faces = igl.read_triangle_mesh(str(folder/'model.obj'))
            for sampling in ['fps', 'resampled']:
                job = self.job(folder, sampling)
                job['model']['up_axis'] = 'y'
                path = folder/'job.json'
                path.write_text(json.dumps(job))
                normal, report = stress_informed_normal(path, folder/'model.obj', vertices, faces,
                    np.array([2., 1.5, 1.5]), 2., folder/(sampling+'_initial.pt'))
                np.testing.assert_allclose(normal, [1., 0., 0.], atol=1e-8)
                self.assertEqual(report['stress_top_percent'], 30.)
                np.testing.assert_allclose(report['mean_direction_projector'], np.diag([0., 1., 0.]))

    def test_public_training_worker_exports_selected_cell_mask(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder = Path(tmp)
            _, _, expected = self.source_fixture(folder)
            field = ScalarField(width=8, layers=2, frequency=2.)
            torch.save(dict(kind='initializer', model_state_dict=field.state_dict(), field_spec=field.spec,
                normalization=dict(center_mm=[2., 1.5, 1.5], scale_mm=2., up_axis='z')), folder/'initializer.pt')
            for sampling in ['fps', 'resampled']:
                path = folder/'job.json'
                path.write_text(json.dumps(self.job(folder, sampling)))
                result = subprocess.run([sys.executable, str(APP/'scripts/multiaxis_field_job.py'),
                    '--bundle', str(PACKAGE), '--job', str(path), '--train', '--device', 'cpu'],
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                report = json.loads((folder/'field_report.json').read_text())
                self.assertEqual(report['stress_top_percent'], 30.)
                self.assertEqual(report['valid_alignment_samples'], expected)
                saved = pv.read(folder/'field.vtu')
                self.assertEqual(int(saved['stress_alignment_valid'].sum()), expected)


if __name__ == '__main__':
    unittest.main()

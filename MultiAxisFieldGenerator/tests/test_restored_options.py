"""Runtime coverage for downstream interface options restored after the update."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
sys.path.insert(0, str(PACKAGE.parent))
from fields import ScalarField
from stress import StressRegion, StressConstraints, farthest_indices
from MultiAxisFieldGenerator.cli import load_job, training_command
from MultiAxisFieldGenerator.runtime.run_model import (orientation_constraint, orientation_weight_at_epoch,
                                                        orientation_reference_refresh_due,
                                                        inward_surface_orientation_constraint,
                                                        capture_orientation_reference,
                                                        topology_regularization_scale_at_epoch,
                                                        gradient_regularization_blend_at_epoch,
                                                        unit_gradient_norm_loss, full_epoch_batches)


class RestoredOptionsTests(unittest.TestCase):
    def test_full_pass_covers_fixed_interior_and_stress_pools_including_tails(self):
        generator = torch.Generator().manual_seed(42)
        first = list(full_epoch_batches([11, 1, 1, 11], [4, 4, 4, 4], 'cpu', generator))
        self.assertEqual(len(first), 3)
        for stream in (0, 3):
            self.assertEqual([len(batch[stream]) for batch in first], [4, 4, 3])
            indices = torch.cat([batch[stream] for batch in first])
            self.assertEqual(sorted(indices.tolist()), list(range(11)))
        second = list(full_epoch_batches([11, 1, 1, 11], [4, 4, 4, 4], 'cpu', generator))
        self.assertFalse(torch.equal(torch.cat([b[0] for b in first]), torch.cat([b[0] for b in second])))

    def test_late_orientation_loss_is_one_sided_and_ramped(self):
        grads = torch.tensor([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
        loss, dots = orientation_constraint(grads, torch.tensor([0., 0., 1.]), 75.)
        self.assertAlmostEqual(float(dots[0]), 1.)
        self.assertAlmostEqual(float(dots[1]), -1.)
        self.assertGreater(float(loss), 0.)
        self.assertEqual(orientation_weight_at_epoch(1, 10, 2., .7, .2), 0.)
        self.assertEqual(orientation_weight_at_epoch(7, 10, 2., .7, .2), 1.)
        self.assertEqual(orientation_weight_at_epoch(8, 10, 2., .7, .2), 2.)
        self.assertTrue(orientation_reference_refresh_due(7, 7, 5))
        self.assertFalse(orientation_reference_refresh_due(11, 7, 5))
        self.assertTrue(orientation_reference_refresh_due(12, 7, 5))

    def test_topology_regularization_ramp_relaxes_only_early_epochs(self):
        self.assertEqual(topology_regularization_scale_at_epoch(1, 0, .05), 1.)
        self.assertEqual(topology_regularization_scale_at_epoch(1, 1, .05), 1.)
        self.assertEqual(topology_regularization_scale_at_epoch(1, 10, .05), .05)
        self.assertAlmostEqual(topology_regularization_scale_at_epoch(5, 10, .05), .05 + .95*4/9)
        self.assertEqual(topology_regularization_scale_at_epoch(10, 10, .05), 1.)
        self.assertEqual(topology_regularization_scale_at_epoch(20, 10, .05), 1.)

    def test_unit_gradient_warmup_and_transition(self):
        blend = gradient_regularization_blend_at_epoch
        self.assertEqual(blend(1, 0, 10), (1., 0.))
        self.assertEqual(blend(1, 10, 10), (0., 1.))
        self.assertEqual(blend(10, 10, 10), (0., 1.))
        self.assertEqual(blend(15, 10, 10), (.5, .5))
        self.assertEqual(blend(20, 10, 10), (1., 0.))
        self.assertEqual(blend(30, 10, 10), (1., 0.))
        self.assertEqual(blend(11, 10, 0), (1., 0.))

    def test_unit_gradient_loss_and_derivative_are_finite_near_critical_points(self):
        grads = torch.tensor([[1., 0., 0.], [0., 0., 0.], [2., 0., 0.]], requires_grad=True)
        loss = unit_gradient_norm_loss(grads)
        self.assertAlmostEqual(float(loss.detach()), 2/3, places=6)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(grads.grad).all()))
        self.assertGreater(float(grads.grad[2, 0]), 0.)
        self.assertEqual(float(unit_gradient_norm_loss(torch.tensor([[0., 1., 0.]]))), 0.)

    def test_late_orientation_surface_term_uses_only_normalized_inward_gradients(self):
        grads = torch.tensor([[0., 0., 20.], [0., 0., -1.], [1., 0., 0.]])
        outward_normals = torch.tensor([[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]])
        loss, count = inward_surface_orientation_constraint(
            grads, outward_normals, torch.tensor([0., 0., 1.]), 75.)
        self.assertEqual(count, 1)
        self.assertGreater(float(loss), 0.)
        # The training metric accumulator requires differentiable scalar
        # tensors; the count is converted before it enters that accumulator.
        self.assertIsInstance(loss*0 + count, torch.Tensor)

    def test_late_orientation_reference_uses_inward_surface_gradients_only(self):
        class LinearField:
            def __call__(self, points, compute_grads=False):
                return {'scalars': -points[:, 2:3]}

        surface = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        normals = torch.tensor([[0., 0., 1.]]*3)
        axis, source, samples, inward = capture_orientation_reference(
            LinearField(), surface, normals, torch.tensor([0., 0., 1.]), 16,
            torch.Generator().manual_seed(3))
        self.assertEqual(source, 'normalized_inward_surface_gradients')
        self.assertEqual(samples, 3)
        self.assertEqual(inward, 3)
        # The raw field gradient is -Z at the surface, and the inferred sign
        # is retained so Support Generation Orientation can rotate it to +Z.
        torch.testing.assert_close(axis, torch.tensor([0., 0., -1.]))

    def test_fps_is_unique_deterministic_and_caps_to_sources(self):
        points = np.array([[0., 0, 0], [0., 0, 0], [1., 0, 0], [3., 0, 0]])
        ids = farthest_indices(points, 20, 42)
        self.assertEqual(len(set(ids)), 4)
        np.testing.assert_array_equal(ids, farthest_indices(points, 20, 42))

    def test_region_sign_threshold_and_outside(self):
        PACKAGE.joinpath('cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PACKAGE/'cache') as tmp:
            grid = pv.ImageData(dimensions=(3, 2, 2)).cast_to_unstructured_grid().triangulate()
            grid.cell_data['signed'] = np.where(grid.cell_centers().points[:, 0] < 1, 2., -3.)
            path = Path(tmp)/'volume.vtu'
            grid.save(path)
            points = [[.4, .4, .4], [1.4, .4, .4], [3., .4, .4]]
            np.testing.assert_array_equal(StressRegion(path, 'signed', 'tensile', 1).mask(points), [True, False, False])
            np.testing.assert_array_equal(StressRegion(path, 'signed', 'compressive', 2).mask(points), [False, True, False])
            self.assertFalse(StressRegion(path, 'signed', 'tensile', 2).mask(points).any())

    def test_fps_uses_original_training_sources_in_selected_region(self):
        PACKAGE.joinpath('cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PACKAGE/'cache') as tmp:
            folder = Path(tmp)
            grid = pv.ImageData(dimensions=(3, 2, 2)).cast_to_unstructured_grid().triangulate()
            grid.cell_data['signed'] = np.where(grid.cell_centers().points[:, 0] < 1, 2., -3.)
            grid.save(folder/'volume.vtu')
            surface = grid.extract_surface(algorithm='dataset_surface').triangulate()
            points = np.random.default_rng(3).uniform([.01, .01, .01], [1.99, .99, .99], (100, 3))
            directions = np.random.default_rng(4).normal(size=(100, 3))
            np.savetxt(folder/'stress.txt', np.column_stack((points, directions)))
            args = SimpleNamespace(stress_file=folder/'stress.txt', stress_point_columns=[0, 1, 2],
                stress_direction_columns=[3, 4, 5], stress_coordinate_space='world', stress_region='compressive',
                volume_mesh=folder/'volume.vtu', stress_signed_field='signed', stress_threshold=1.,
                stress_sampling='fps', seed=42, stress_neighbors=4, stress_max_distance_mm=None, stress_samples=50)
            stress = StressConstraints(args, np.zeros(3), 1., 'cpu', np.asarray(surface.points, dtype=np.float64),
                                       np.asarray(surface.faces.reshape(-1, 4)[:, 1:], dtype=np.int64))
            # FPS never asks a domain to synthesize a target.
            count = stress.resample(None, None)
            self.assertGreater(count, 0)
            self.assertTrue(np.all(stress.points_epoch.numpy()[:, 0] > 1))
            self.assertTrue(np.all(stress.validation_points.numpy()[:, 0] > 1))
            for point, matrix in zip(stress.points_epoch.numpy(), stress.matrices_epoch.numpy()):
                ids = np.flatnonzero(np.linalg.norm(points-point, axis=1) < 1e-6)
                self.assertEqual(len(ids), 1)
                direction = directions[ids[0]] / np.linalg.norm(directions[ids[0]])
                np.testing.assert_allclose(matrix, np.outer(direction, direction), atol=1e-6)
                self.assertGreater(np.linalg.norm(stress.validation_points.numpy()-point, axis=1).min(), 1e-6)
            original = stress.points_epoch
            self.assertEqual(stress.resample(None, None), count)
            self.assertIs(stress.points_epoch, original)

    def test_training_regions_sampling_and_checkpoint_intervals(self):
        PACKAGE.joinpath('cache').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=PACKAGE/'cache') as tmp:
            folder = Path(tmp)
            volume = pv.ImageData(dimensions=(5, 5, 5), spacing=(2.5,)*3, origin=(-5.,)*3).cast_to_unstructured_grid().triangulate()
            volume.cell_data['signed'] = np.where(volume.cell_centers().points[:, 0] < 0, -3., 2.)
            volume.save(folder/'volume.vtu')
            volume.extract_surface(algorithm='dataset_surface').triangulate().save(folder/'model.obj')
            field = ScalarField(width=8, layers=2, frequency=2)
            torch.save(dict(model_state_dict=field.state_dict(), field_spec=field.spec), folder/'initializer.pt')
            points = np.random.default_rng(42).uniform(-4.9, 4.9, (200, 3))
            np.savetxt(folder/'stress.txt', np.column_stack((points, np.tile([1., 0, 0], (len(points), 1)))))
            for region, sampling in [('tensile', 'resampled'), ('compressive', 'fps')]:
                run = folder/region
                job = dict(output=str(run), model=dict(mesh='model.obj', volume_mesh='volume.vtu',
                    start_checkpoint='initializer.pt', up_axis='z', base_band_mm=1.),
                    training=dict(objective='stress', epochs=3, epoch_mode='sampled-steps', steps_per_epoch=1,
                        collision_weight=0, device='cpu', interior_batch=8, surface_batch=8, base_batch=8,
                        grid_spacing=2., validation_count=8, validation_batch=8, validate_every=2,
                        save_every=2, snapshot_every=2, normalize_collision_by_gradient=True,
                        gradient_norm_warmup_epochs=1, gradient_norm_transition_epochs=2,
                        gradient_norm_weight=.3,
                        orientation_constraint_weight=1., orientation_max_angle_deg=75.,
                        orientation_start_fraction=.34, orientation_ramp_fraction=.34,
                        orientation_reference_samples=16, orientation_reference_update_every=5),
                    stress=dict(enabled=True, file='stress.txt', region=region, signed_field='signed',
                        threshold=1., sampling=sampling, samples=64, batch=8))
                path = folder/'job.json'
                path.write_text(json.dumps(job))
                values, _ = load_job(path)
                result = subprocess.run(training_command(values), cwd=PACKAGE/'runtime', capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
                self.assertEqual(torch.load(run/'latest.pt', weights_only=True)['epoch'], 3)
                self.assertEqual(sorted(p.name for p in (run/'checkpoints').iterdir()), ['epoch_000002.pt', 'epoch_000003.pt'])
                config = json.loads((run/'config.json').read_text())
                self.assertTrue(config['normalize_collision_by_gradient'])
                self.assertEqual(config['stress_metadata']['region'], region)
                self.assertEqual(config['stress_metadata']['sampling'], sampling)
                rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in rows if r['event']=='validation'], [0, 2, 3])
                self.assertTrue(all(r['accepted'] > 0 for r in rows if r['event']=='stress_resampling'))
                train_rows = [r for r in rows if r['event'] == 'train']
                np.testing.assert_allclose([r['variation_weight_effective'] for r in train_rows], [0., .01, .02])
                np.testing.assert_allclose([r['small_gradient_weight_effective'] for r in train_rows], [0., .5, 1.])
                np.testing.assert_allclose([r['gradient_norm_weight_effective'] for r in train_rows], [.3, .15, 0.])
                np.testing.assert_allclose([r['curvature_weight_effective'] for r in train_rows], [.1, .1, .1])
                self.assertEqual(train_rows[0]['variation'], 0.)
                self.assertEqual(train_rows[0]['small_gradient'], 0.)
                self.assertTrue(all(np.isfinite(r['gradient_norm']) and np.isfinite(r['total']) for r in train_rows))
                reference = next(r for r in rows if r['event'] == 'orientation_reference')
                self.assertEqual(reference['epoch'], 2)
                self.assertEqual(reference['samples'], 16)
                final = next(r for r in reversed(rows) if r['event'] == 'validation')
                self.assertIn('orientation_violation_fraction', final)
                self.assertIn('upward_growth_fraction', final)
                config = json.loads((run/'config.json').read_text())
                self.assertEqual(config['orientation_constraint']['captured_epoch'], 2)
                self.assertEqual(len(config['orientation_constraint']['reference_axis']), 3)
                self.assertEqual(config['orientation_constraint']['reference_update_every_epochs'], 5)
                self.assertEqual(len(config['orientation_constraint']['reference_updates']), 1)


if __name__ == '__main__':
    unittest.main()

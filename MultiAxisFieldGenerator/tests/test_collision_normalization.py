import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import sys
import unittest
import torch

torch.set_num_threads(2)
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))

class Field(torch.nn.Module):
    def __init__(self, amplitude):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.tensor(float(amplitude), dtype=torch.float64))
        self.bend = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    def forward(self, x, compute_grads=False, **kwargs):
        values = self.amplitude*(x[:, :1]+self.bend*x[:, 1:2]**2+.7*x[:, 2:3])
        out = dict(scalars=values)
        if compute_grads:
            out['grads'] = torch.autograd.grad(values.sum(), x, create_graph=True)[0]
        return out

from collision import collision_loss, collision_diagnostics, ToolSamples

class NormalizationTests(unittest.TestCase):
    def evaluate(self, amplitude=2., normalized=True, cull=True, inspection=False):
        field = Field(amplitude)
        roots = torch.tensor([[0., .2, 0.], [.1, -.4, .2]], dtype=torch.float64)
        tool = ToolSamples(torch.tensor([[0.,0.,-.2],[0.,0.,-.4],[0.,0.,-8.]], dtype=torch.float64), torch.zeros(3, dtype=torch.long), ('tool',))
        bbox = torch.tensor([[-2.]*3,[2.]*3], dtype=torch.float64)
        function = collision_diagnostics if inspection else collision_loss
        out = function(field, roots, tool, 1., bbox, topk_fraction=.7,
                       normalize_collision_by_gradient=normalized, cull=cull, chunk_size=2)
        return field, out

    def test_scale_invariance_and_attached_denominator(self):
        field, a = self.evaluate()
        _, b = self.evaluate(7.)
        torch.testing.assert_close(a['loss'], b['loss'])
        a['loss'].backward()
        self.assertLess(abs(float(field.amplitude.grad)), 1e-10)
        self.assertTrue(torch.isfinite(field.bend.grad))
        _, raw_a = self.evaluate(normalized=False)
        _, raw_b = self.evaluate(7., normalized=False)
        torch.testing.assert_close(raw_b['loss'], raw_a['loss']*(7/2)**2)

    def test_culling_and_inspection_match(self):
        field, a = self.evaluate()
        dense_field, b = self.evaluate(cull=False)
        _, inspection = self.evaluate(inspection=True)
        torch.testing.assert_close(a['loss'], b['loss'])
        torch.testing.assert_close(a['loss'], inspection['loss'])
        self.assertFalse(inspection['loss'].requires_grad)
        self.assertLess(a['field_pairs'], b['field_pairs'])
        a['loss'].backward(); b['loss'].backward()
        torch.testing.assert_close(field.bend.grad, dense_field.bend.grad)
        self.assertTrue(torch.equal(a['collision_mask'], b['collision_mask']))

    def test_floor_is_finite_and_shape_gradient_matches_finite_difference(self):
        field, out = self.evaluate()
        out['loss'].backward()
        analytic = field.bend.grad.clone()
        # Evaluate the same geometry and root samples with perturbed shape.
        from unittest.mock import patch
        original = Field.forward
        def shifted(delta):
            def forward(model, *args, **kwargs):
                with torch.no_grad():
                    model.bend.fill_(.3+delta)
                return original(model, *args, **kwargs)
            with patch.object(Field, 'forward', forward):
                return self.evaluate()[1]['loss'].detach()
        torch.testing.assert_close(analytic, (shifted(1e-5)-shifted(-1e-5))/2e-5, rtol=1e-4, atol=1e-7)
        tiny_field, tiny = self.evaluate(amplitude=1e-9)
        tiny['loss'].backward()
        self.assertTrue(torch.isfinite(tiny['loss']))
        self.assertTrue(torch.isfinite(tiny_field.amplitude.grad))

    def test_siren_backpropagates_and_replay_uses_normalization(self):
        from fields import ScalarField
        field = ScalarField(width=8, layers=2).double()
        roots = torch.tensor([[0., .2, 0.], [.1, -.4, .2]], dtype=torch.float64, requires_grad=True)
        tool = ToolSamples(torch.tensor([[0.,0.,-.2],[0.,0.,.4]], dtype=torch.float64), torch.zeros(2,dtype=torch.long), ('tool',))
        out = collision_loss(field, roots, tool, 1., torch.tensor([[-2.]*3,[2.]*3]), normalize_collision_by_gradient=True)
        out['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in field.parameters()))
        from root_sampling import WitnessReplay
        replay = object.__new__(WitnessReplay)
        replay.scale = 1.
        shape = dict(z_min=-2., z_max=2., radius=2., half_angle_degrees=60.)
        replay.tool = {key:shape for key in ['cone','inner_cylinder','outer_cylinder']}
        def evaluate(amplitude):
            model = Field(amplitude)
            x = roots.detach().requires_grad_(True)
            return model, replay.loss(model, x, x.detach()-torch.tensor([.2,0.,0.]), model(x,compute_grads=True), normalize_collision_by_gradient=True)
        model, first = evaluate(2.)
        _, second = evaluate(7.)
        torch.testing.assert_close(first, second)
        first.backward()
        self.assertLess(abs(float(model.amplitude.grad)), 1e-10)

if __name__ == "__main__":
    unittest.main()

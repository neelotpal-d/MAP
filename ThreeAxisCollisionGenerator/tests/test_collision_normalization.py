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

from types import SimpleNamespace
from experiments.convex_hard_collision.hard_collision_sampling import install, HardCollisionConfig, compute_bbox_culled_collision_loss

class NormalizationTests(unittest.TestCase):
    def evaluate(self, amplitude=2., normalized=True, inspection=False):
        module = SimpleNamespace(compute_gated_translated_scalar_loss=compute_bbox_culled_collision_loss,
            build_hybrid_dataloaders=lambda **kw: {}, train_scalar_field=lambda **kw: None,
            NORMALIZE_COLLISION_BY_GRADIENT=normalized, COLLISION_GRADIENT_EPSILON=1e-6,
            COLLISION_TOOL_WEIGHTS=(1.,), COLLISION_TOPK_FRACTION=.7)
        manager = install(module, HardCollisionConfig(collision_roots_per_step=2, validation_count=0,
                          mining_candidate_count=2, scoring_root_batch_size=2))
        roots = torch.tensor([[0., .2, 0.], [.1, -.4, .2]], dtype=torch.float64)
        manager.set_root_pool(roots)
        field = Field(amplitude)
        tool = torch.tensor([[-.2,0.,0.],[-.4,0.,0.],[-.8,0.,0.]], dtype=torch.float64)
        with torch.no_grad() if inspection else torch.enable_grad():
            out = module.compute_gated_translated_scalar_loss(field, roots, tool)
        return field, out

    def test_scale_invariance_and_attached_denominator(self):
        field, a = self.evaluate()
        _, b = self.evaluate(7.)
        torch.testing.assert_close(a['loss'], b['loss'])
        a['loss'].backward()
        self.assertLess(abs(float(field.amplitude.grad)), 1e-10)
        self.assertTrue(torch.isfinite(field.bend.grad))
        self.assertGreater(abs(float(field.bend.grad)), 1e-5)
        _, raw_a = self.evaluate(normalized=False)
        _, raw_b = self.evaluate(7., normalized=False)
        torch.testing.assert_close(raw_b['loss'], raw_a['loss']*(7/2)**2)
        torch.testing.assert_close(a['scalar_differences'].square().mean(), a['loss'])

    def test_inspection_matches_training(self):
        _, a = self.evaluate()
        _, b = self.evaluate(inspection=True)
        torch.testing.assert_close(a['loss'], b['loss'])
        self.assertFalse(b['loss'].requires_grad)
        self.assertTrue(torch.isfinite(b['scalar_differences']).all())

    def test_siren_and_numerical_floor(self):
        from scalarNetUtils import initialize_scalar_net
        net = initialize_scalar_net('cpu', num_layers=1, w0=7, w0_initial=7).double()
        original = Field.forward
        from unittest.mock import patch
        with patch.object(Field, 'forward', lambda model, x, **kw: net(x, **kw)):
            _, out = self.evaluate()
            out['loss'].backward()
        self.assertTrue(any(p.grad is not None for p in net.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        tiny_field, tiny = self.evaluate(amplitude=1e-9)
        tiny['loss'].backward()
        self.assertTrue(torch.isfinite(tiny['loss']))
        self.assertTrue(torch.isfinite(tiny_field.amplitude.grad))

if __name__ == "__main__":
    unittest.main()

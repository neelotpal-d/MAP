"""Coverage bounds, perturbations, and persistent sampled collision witnesses."""
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
from collision import (ToolSamples, ToolWitnessBuffer, collision_loss, place_tool,
                       sample_abb_tool, tool_witness_loss)


class BentField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bend = torch.nn.Parameter(torch.tensor(-.6))

    def forward(self, points, compute_grads=False):
        values = points[:, 2]+self.bend*points[:, 0].square()
        out = dict(scalars=values[:, None])
        if compute_grads:
            out['grads'] = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
        return out


class CollisionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((PACKAGE/'tools/abb_training.json').read_text())
        self.bbox = torch.tensor([[-10.]*3, [10.]*3])

    def test_fixed_rings_fill_gap_and_preserve_bounds_and_anchors(self):
        tool = sample_abb_tool(self.config, volume_samples=0)
        for component, name, upper in [(1, 'inner', 57.5), (2, 'outer', 61.25)]:
            points = tool.offsets_mm[tool.component_ids == component]
            self.assertGreaterEqual(float(points[:, 2].min()), min(self.config[name+'_heights']))
            self.assertLessEqual(float(points[:, 2].max()), upper)
            side = points[torch.isclose(points[:, :2].norm(dim=1), torch.tensor(self.config[name+'_radius']))]
            heights = side[:, 2].unique(sorted=True)
            self.assertLessEqual(float(torch.diff(heights).max()), 3.75+1e-5)
            for anchor in self.config[name+'_heights']:
                self.assertTrue(bool(torch.isclose(heights, torch.tensor(anchor)).any()))
        inner = tool.offsets_mm[tool.component_ids == 1]
        self.assertTrue(bool(((inner[:, 2] > 11.25) & (inner[:, 2] < 33.75)).any()))

    def test_both_cylinders_perturb_reproducibly_without_losing_fixed_rings(self):
        tools = [sample_abb_tool(self.config, jitter=True, generator=torch.Generator().manual_seed(seed))
                 for seed in (17, 17, 18)]
        torch.testing.assert_close(tools[0].offsets_mm, tools[1].offsets_mm, rtol=0, atol=0)
        for component, name, upper in [(1, 'inner', 57.5), (2, 'outer', 61.25)]:
            first, second = [tool.offsets_mm[tool.component_ids == component] for tool in (tools[0], tools[2])]
            self.assertFalse(torch.equal(first[:, 2].sort().values, second[:, 2].sort().values))
            self.assertGreaterEqual(float(first[:, 2].min()), min(self.config[name+'_heights']))
            self.assertLessEqual(float(first[:, 2].max()), upper)
            for anchor in self.config[name+'_heights']:
                self.assertTrue(bool(torch.isclose(first[:, 2], torch.tensor(anchor)).any()))

    def test_paired_placement_matches_individual_shared_placements(self):
        roots = torch.tensor([[1., 2., 3.], [-1., 0., 2.]])
        grads = torch.tensor([[.3, .1, 1.], [.2, 1., .3]])
        offsets = torch.tensor([[1., 2., 4.], [2., -1., 5.]])
        paired = place_tool(roots, grads, offsets[:, None])[:, 0]
        expected = torch.cat([place_tool(roots[i:i+1], grads[i:i+1], offsets[i:i+1])[:, 0]
                              for i in range(len(roots))])
        torch.testing.assert_close(paired, expected)

    def collision_fixture(self, count=1):
        field = BentField()
        roots = torch.zeros(count, 3)
        roots[:, 1] = torch.arange(count)*.1
        tool = ToolSamples(torch.tensor([[0., 1., .2]]*3), torch.arange(3))
        result = collision_loss(field, roots, tool, 1., self.bbox)
        return field, roots, tool, result

    def test_exact_pair_survives_a_fresh_sampling_miss_and_retires_after_resolution(self):
        field, roots, tool, result = self.collision_fixture()
        memory = ToolWitnessBuffer(16)
        memory.observe(roots, tool, result)
        self.assertEqual(len(memory), 1)
        # A fresh draw with no detected collision must not erase a saved pair.
        miss = dict(differences=torch.ones_like(result['differences']),
                    inside_mask=torch.ones_like(result['inside_mask']))
        memory.observe(roots, tool, miss)
        self.assertEqual(len(memory), 1)
        ids, remembered_roots, offsets, normalizers = memory.draw(64)
        recheck = tool_witness_loss(field, remembered_roots, offsets, normalizers, 1., self.bbox)
        self.assertGreater(float(recheck['loss'].detach()), 0)
        recheck['loss'].backward()
        self.assertTrue(bool(torch.isfinite(field.bend.grad)))
        self.assertGreater(abs(float(field.bend.grad)), 0)
        memory.retire(ids, recheck['scores'])
        self.assertEqual(len(memory), 1)
        with torch.no_grad():
            field.bend.fill_(.6)
        ids, remembered_roots, offsets, normalizers = memory.draw(64)
        resolved = tool_witness_loss(field, remembered_roots, offsets, normalizers, 1., self.bbox)
        self.assertEqual(float(resolved['loss'].detach()), 0)
        memory.retire(ids, resolved['scores'])
        self.assertEqual(len(memory), 0)

    def test_pair_gate_and_component_denominator_match_sampled_condition(self):
        field, roots, tool, result = self.collision_fixture()
        memory = ToolWitnessBuffer(16)
        memory.observe(roots, tool, result)
        _, remembered_roots, offsets, normalizers = memory.draw(64)
        replayed = tool_witness_loss(field, remembered_roots, offsets, normalizers, 1., self.bbox)
        # Three identical components contribute equally: one retained pair has
        # one third of the original all-component collision loss.
        torch.testing.assert_close(replayed['loss'], result['loss']/3)
        class OutsideGate(torch.nn.Module):
            def forward(self, points, **kwargs):
                return dict(scalars=torch.ones(len(points), 1))
        gated = tool_witness_loss(field, remembered_roots, offsets, normalizers, 1., self.bbox, OutsideGate())
        self.assertEqual(float(gated['loss'].detach()), 0)

    def test_memory_is_bounded_deduplicated_and_cycles(self):
        _, roots, tool, result = self.collision_fixture(3)
        memory = ToolWitnessBuffer(2)
        memory.observe(roots, tool, result)
        memory.observe(roots, tool, result)
        self.assertEqual(len(memory), 2)
        seen = [float(memory.draw(1)[1][0, 1]) for _ in range(2)]
        self.assertEqual(len(set(seen)), 2)
        self.assertFalse(memory.roots.requires_grad)
        self.assertEqual(len(memory.state()['roots']), 2)

    def test_retained_pair_preserves_orientation_derivatives_on_cpu_and_cuda(self):
        devices = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for normalized in (False, True):
                with self.subTest(device=device, normalized=normalized):
                    field = BentField().to(device)
                    roots = torch.tensor([[.25, 0., 0.]], device=device)
                    tool = ToolSamples(torch.tensor([[1., 0., .2]]*3), torch.arange(3)).to(device)
                    bbox = self.bbox.to(device)
                    result = collision_loss(field, roots, tool, 1., bbox,
                                            normalize_collision_by_gradient=normalized)
                    memory = ToolWitnessBuffer(16)
                    memory.observe(roots, tool, result)
                    self.assertEqual(len(memory), 1)
                    _, remembered_roots, offsets, normalizers = memory.draw(64)
                    paired = tool_witness_loss(field, remembered_roots, offsets, normalizers, 1., bbox,
                                               normalize_collision_by_gradient=normalized)
                    reference_gradient, = torch.autograd.grad(result['loss']/3, field.bend)
                    paired_gradient, = torch.autograd.grad(paired['loss'], field.bend)
                    torch.testing.assert_close(paired['loss'], result['loss']/3)
                    torch.testing.assert_close(paired_gradient, reference_gradient)
                    self.assertTrue(bool(torch.isfinite(paired_gradient)))


if __name__ == '__main__':
    unittest.main()

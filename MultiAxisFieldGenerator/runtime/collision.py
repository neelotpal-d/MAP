"""Differentiable, culled collision checks for a layer-normal-aligned ABB tool.

Hard SDF membership is deliberately detached. Tool placement is NOT detached:
its derivative through the root gradient is part of the collision objective.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class ToolSamples:
    offsets_mm: torch.Tensor
    component_ids: torch.Tensor
    names: tuple = ("cone", "inner_cylinder", "outer_cylinder")

    def to(self, device, dtype=torch.float32):
        return ToolSamples(self.offsets_mm.to(device=device, dtype=dtype),
                           self.component_ids.to(device=device), self.names)

    @property
    def counts(self):
        return [int((self.component_ids == i).sum()) for i in range(len(self.names))]


def sample_abb_tool(config=None, *, dense=False, jitter=False, generator=None, volume_samples=0):
    """Sample in a local +Z tool frame, in mm, without repeated 0/2pi endpoints.

    Retain the legacy enclosure coverage; do not remove nested samples merely
    because their continuous enclosure is contained in another enclosure.
    Dense validation uses independent finer radial/angular/axial sampling.
    """
    if config is None:
        config = Path(__file__).resolve().parent.parent/'tools'/'abb_training.json'
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    angle = math.radians(config["cone_angle_degrees"])
    axial_spacing = float(config.get('cylinder_axial_spacing_mm', 3.75))
    if not math.isfinite(axial_spacing) or axial_spacing <= 0:
        raise ValueError('cylinder_axial_spacing_mm must be finite and positive')
    parts, groups = [], []

    def add(points, group):
        parts.append(points)
        groups.append(torch.full((len(points),), group, dtype=torch.long))

    def theta(n, phase=0.):
        return torch.arange(n, dtype=torch.float64) * (2 * math.pi / n) + phase

    if dense:
        polar = [0.] + [angle * i / 6 for i in range(1, 7)]
        directions = [torch.tensor([[0., 0., 1.]], dtype=torch.float64)]
        for alpha in polar[1:]:
            t = theta(36)
            directions.append(torch.stack((math.sin(alpha)*t.cos(),
                                            math.sin(alpha)*t.sin(),
                                            torch.full_like(t, math.cos(alpha))), -1))
        distances = torch.linspace(min(config["cone_distances"]),
                                   max(config["cone_distances"]), 10, dtype=torch.float64)
    else:
        directions = [torch.tensor([[0., 0., 1.]], dtype=torch.float64)]
        for alpha, n in [(angle/2, 5), (angle, 14)]:
            t = theta(n)
            directions.append(torch.stack((math.sin(alpha)*t.cos(),
                                            math.sin(alpha)*t.sin(),
                                            torch.full_like(t, math.cos(alpha))), -1))
        distances = torch.tensor(config["cone_distances"], dtype=torch.float64)
    directions = torch.cat(directions)
    add((directions[:, None, :] * distances[None, :, None]).reshape(-1, 3), 0)
    for group, name in [(1, "inner"), (2, "outer")]:
        heights = torch.tensor(config[name+"_heights"], dtype=torch.float64)
        lower, upper = float(heights.min()), float(heights.max())
        padding = config.get("outer_height_jitter_mm", 3.75) if name == "outer" else 0.
        upper += padding
        n = config["angular_samples"]
        if dense:
            # Include every original level as well as a finer uniform grid.
            grid = torch.linspace(lower, upper,
                                  math.ceil((upper-lower)/2.)+1,
                                  dtype=torch.float64)
            heights = torch.unique(torch.cat((heights, grid)))
            n = 36
        else:
            # Preset levels are anchors, not permission to leave long gaps.
            # Include the full enclosure extent, including the outer padding.
            anchors = torch.unique(torch.cat((heights, heights.new_tensor([upper]))), sorted=True)
            heights = torch.unique(torch.cat([
                torch.linspace(float(a), float(b), math.ceil(float(b-a)/axial_spacing)+1,
                               dtype=torch.float64)
                for a, b in zip(anchors[:-1], anchors[1:])
            ])) if len(anchors) > 1 else anchors
        phase = 0.
        if jitter:
            phase = float(torch.rand((), generator=generator))*2*math.pi/n
        t = theta(n, phase)
        radial = config[name+"_radius"] * torch.stack((t.cos(), t.sin()), -1)

        def rings(levels, angles_radial):
            xy = angles_radial[None].expand(len(levels), -1, -1)
            z = levels[:, None, None].expand(-1, n, 1)
            return torch.cat((xy, z), -1).reshape(-1, 3)

        add(rings(heights, radial), group)
        if not dense and config.get('cylinder_axial_jitter', True) and len(heights) > 1:
            # Keep the fixed rings and independently perturb an additional
            # ring inside every interval, on BOTH cylinders. Never extend
            # the enclosure beyond its declared upper bound.
            fraction = (torch.rand(len(heights)-1, generator=generator, dtype=torch.float64)
                        if jitter else torch.full((len(heights)-1,), .5, dtype=torch.float64))
            extra_heights = heights[:-1] + fraction*(heights[1:]-heights[:-1])
            t2 = theta(n, phase + math.pi/n)
            radial2 = config[name+"_radius"] * torch.stack((t2.cos(), t2.sin()), -1)
            add(rings(extra_heights, radial2), group)
        if config.get("sample_cylinder_caps", True):
            # Side rings alone leave the underside annulus and upper cap
            # uncovered. Include radial strata on both ends of each enclosure.
            radial_steps = 8 if dense else 4
            for height in [lower, upper]:
                add(torch.tensor([[0., 0., height]], dtype=torch.float64), group)
                for radial_index in range(1, radial_steps+1):
                    xy = radial*(radial_index/radial_steps)
                    add(torch.cat((xy, torch.full((n, 1), height, dtype=torch.float64)), -1), group)
    offsets, ids = torch.cat(parts), torch.cat(groups)
    # Exact duplicate removal only. Component boundaries keep their semantics.
    key = torch.cat((ids[:, None].double(), (offsets*1e8).round()), -1)
    _, inverse = torch.unique(key, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max())+1,), len(offsets), dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(len(offsets)), reduce="amin")
    keep = first.sort().values
    offsets, ids = offsets[keep].float(), ids[keep]
    if volume_samples < 0:
        raise ValueError('volume_samples cannot be negative')
    if volume_samples:
        # Keep all original strata, adding fresh volume points per component.
        # Cone volume fills the tip-to-stem interval; cylinders fill their
        # entire axial/radial interiors, not just the enclosing side rings.
        extra = []
        for component in range(3):
            u = torch.rand(volume_samples, 3, generator=generator)
            if component == 0:
                z = min(config['inner_heights'])*u[:, 0].pow(1/3)
                radial = torch.minimum(z*math.tan(angle), torch.full_like(z, config['inner_radius']))*u[:, 1].sqrt()
            else:
                name = 'inner' if component == 1 else 'outer'
                low, high = min(config[name+'_heights']), max(config[name+'_heights'])
                high += config.get('outer_height_jitter_mm', 3.75) if component == 2 else 0.
                z = low+(high-low)*u[:, 0]
                radial = config[name+'_radius']*u[:, 1].sqrt()
            theta = u[:, 2]*(2*math.pi)
            extra.append(torch.stack((radial*theta.cos(), radial*theta.sin(), z), 1))
        offsets = torch.cat([offsets]+extra)
        ids = torch.cat([ids]+[torch.full((volume_samples,), i, dtype=torch.long) for i in range(3)])
    return ToolSamples(offsets, ids)


def place_tool(roots, gradients, offsets_model):
    """Build one differentiable frame per root, robust near coordinate axes."""
    normal = F.normalize(gradients, dim=-1, eps=1e-10)
    # Use the least-aligned coordinate axis. Frame selection is piecewise;
    # all subsequent arithmetic retains the orientation derivative.
    helper = F.one_hot(normal.detach().abs().argmin(-1), 3).to(roots)
    u = F.normalize(torch.linalg.cross(normal, helper), dim=-1, eps=1e-10)
    v = torch.linalg.cross(normal, u)
    # Also accept per-root samples [N,K,3] for retained collision witnesses.
    offsets = offsets_model[None] if offsets_model.ndim == 2 else offsets_model
    q = roots[:, None, :] + (offsets[..., 0:1] * u[:, None, :]
                            + offsets[..., 1:2] * v[:, None, :]
                            + offsets[..., 2:3] * normal[:, None, :])
    return q


def scalar_values(field, points, chunk_size=16384):
    values = []
    for batch in points.split(chunk_size):
        out = field(batch, compute_grads=False)
        values.append((out["scalars"] if isinstance(out, dict) else out).reshape(-1))
    return torch.cat(values) if values else points.new_empty(0)


def reduce_violations(violations, tool, fraction=.3, weights=None):
    if not 0 < fraction <= 1:
        raise ValueError("top-k fraction must lie in (0, 1]")
    weights = [1.] * len(tool.names) if weights is None else weights
    if len(weights) != len(tool.names) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Supply one nonnegative weight per component, with positive total")
    components = []
    for i, name in enumerate(tool.names):
        value = violations[:, tool.component_ids == i]
        if value.shape[1] == 0:
            raise ValueError(f"Empty tool component: {name}")
        k = max(1, math.ceil(fraction * value.shape[1]))
        components.append(value.topk(k, dim=1).values.mean(1))
    matrix = torch.stack(components, -1)
    per_root = (matrix * matrix.new_tensor(weights)).sum(-1) / sum(weights)
    return per_root.mean(), per_root, matrix


def collision_loss(field, roots, tool, scale_mm, bbox, gate=None, *,
                   topk_fraction=.3, margin=0., gate_epsilon=0.,
                   root_output=None, chunk_size=16384, sdf_chunk_size=None, cull=True,
                   normalize_collision_by_gradient=False, collision_gradient_epsilon=1e-6,
                   prune_zero_penalty=True):
    """Return loss, root scores and sampled diagnostics.

    cull=False is a dense-computation reference using the SAME samples, masks,
    and reduction. A separately denser ToolSamples set checks spatial coverage.
    All-sample denominators are fixed, including zero violations outside part.
    Training screens scalar differences without a derivative graph, then builds
    that graph only where the ReLU penalty may be nonzero. Near-boundary samples
    are retained conservatively. Differences remain available for inspection;
    only penalized entries retain parameter/orientation derivatives. Disable
    pruning to obtain the complete derivative-graph reference.
    """
    if len(roots) == 0 or scale_mm <= 0 or margin < 0:
        raise ValueError("Need nonempty roots, positive scale and nonnegative margin")
    if root_output is None:
        if not roots.requires_grad:
            roots = roots.detach().requires_grad_(True)
        root_output = field(roots, compute_grads=True)
    root_values = root_output["scalars"].reshape(-1)
    points = place_tool(roots, root_output["grads"], tool.offsets_mm / scale_mm)
    flat = points.reshape(-1, 3)
    with torch.no_grad():
        bbox_mask = ((flat >= bbox[0]) & (flat <= bbox[1])).all(-1)
        candidate = bbox_mask.nonzero().flatten()
        active = candidate
        if gate is not None and len(candidate):
            gate_values = scalar_values(gate, flat[candidate], sdf_chunk_size or chunk_size)
            active = candidate[gate_values < gate_epsilon]
    n, k = points.shape[:2]
    # Zero anchor makes empty results differentiable without doing tool work.
    differences = (root_values * 0)[:, None].expand(n, k).reshape(-1).clone()
    gradient_pairs = 0
    prefilter_pairs = 0
    if cull:
        if len(active):
            query_ids = active
            if prune_zero_penalty and torch.is_grad_enabled():
                with torch.no_grad():
                    selected = scalar_values(field, flat[active], chunk_size)
                    root_selected = root_values[torch.div(active, k, rounding_mode="floor")]
                    delta = selected-root_selected
                    tolerance = 16*torch.finfo(delta.dtype).eps*(1+selected.abs()+root_selected.abs())
                    query_ids = active[delta < margin+tolerance]
                    prefilter_pairs = len(active)
                differences = differences.index_copy(0, active, delta)
            if len(query_ids):
                selected = scalar_values(field, flat[query_ids], chunk_size)
                delta = selected-root_values[torch.div(query_ids, k, rounding_mode="floor")]
                differences = differences.index_copy(0, query_ids, delta)
            gradient_pairs = len(query_ids) if torch.is_grad_enabled() else 0
        evaluated = len(active)
    else:
        differences = scalar_values(field, flat, chunk_size) - root_values.repeat_interleave(k)
        evaluated = len(flat)
        gradient_pairs = evaluated if torch.is_grad_enabled() else 0
    mask = torch.zeros(n*k, dtype=torch.bool, device=roots.device)
    mask[active] = True
    differences, mask = differences.reshape(n, k), mask.reshape(n, k)
    error = torch.relu(margin - differences)
    if normalize_collision_by_gradient:
        if not math.isfinite(collision_gradient_epsilon) or collision_gradient_epsilon <= 0:
            raise ValueError('collision_gradient_epsilon must be positive')
        # Keep the denominator attached: positive field rescaling then cancels
        # in both the value and its derivative (above the numerical floor).
        denominator = root_output['grads'].norm(dim=1).clamp_min(collision_gradient_epsilon)
        error = error / denominator[:, None]
    violations = error.square() * mask
    loss, scores, components = reduce_violations(violations, tool, topk_fraction)
    return {"loss": loss, "root_scores": scores, "component_scores": components,
            "differences": differences, "inside_mask": mask,
            "collision_mask": (differences < 0) & mask,
            "sample_points": points, "total_pairs": n*k,
            "bbox_pairs": len(candidate), "active_pairs": len(active),
            "field_pairs": evaluated, "gradient_pairs": gradient_pairs,
            "prefilter_pairs": prefilter_pairs,
            "field_evaluations": prefilter_pairs+gradient_pairs if prefilter_pairs else evaluated}


def collision_diagnostics(field, roots, *args, **kwargs):
    """Same collision values, but no parameter/higher derivative graph."""
    from fields import first_order_output
    root_output = first_order_output(field, roots)
    with torch.no_grad():
        out = collision_loss(field, roots.detach(), *args, root_output=root_output, **kwargs)
    out['root_gradient_norms'] = root_output['grads'].norm(dim=1)
    return out


class ToolWitnessBuffer:
    """Remember a root AND the local tool point which exposed its violation.

    Unlike offline geometric WitnessReplay, local offsets move with the current
    tool frame and are gated against the part again on every revisit. This is
    the same sampled collision condition, without depending on rediscovery by
    a new random tool draw. Memory is bounded and consumed round-robin.
    """
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Witness capacity must be positive')
        self.capacity, self.cursor = capacity, 0
        self.roots = self.offsets_mm = self.components = self.normalizers = None

    def __len__(self):
        return 0 if self.roots is None else len(self.roots)

    @torch.no_grad()
    def observe(self, roots, tool, result, *, topk_fraction=.3, margin=0.):
        # Keep one strongest pair per observed root; broad sampling is unchanged.
        counts = tool.offsets_mm.new_tensor([max(1, math.ceil(topk_fraction*c))*len(tool.names)
                                            for c in tool.counts])
        normalization = counts[tool.component_ids]
        scores = torch.relu(margin-result['differences'].detach()).square()*result['inside_mask']
        scores = scores/normalization[None]
        worst, selected = scores.max(1)
        positive = worst > 0
        if not positive.any():
            return
        indices = selected[positive]
        values = (roots[positive].detach(), tool.offsets_mm[indices].detach(),
                  tool.component_ids[indices].detach(), normalization[indices].detach())
        if len(self):
            values = tuple(torch.cat((old, new)) for old, new in zip(
                (self.roots, self.offsets_mm, self.components, self.normalizers), values))
        key = torch.cat((values[0], values[1], values[2][:, None].to(values[0])), 1)
        _, inverse = torch.unique(key, dim=0, return_inverse=True)
        # Keep the latest copy of an exact pair and the most recent bounded set.
        last = torch.full((int(inverse.max())+1,), -1, dtype=torch.long, device=roots.device)
        last.scatter_reduce_(0, inverse, torch.arange(len(key), device=roots.device), reduce='amax')
        keep = last.sort().values[-self.capacity:]
        self.roots, self.offsets_mm, self.components, self.normalizers = tuple(value[keep] for value in values)

    def draw(self, count):
        if not len(self):
            return None
        count = min(count, len(self))
        ids = (torch.arange(count, device=self.roots.device)+self.cursor) % len(self)
        self.cursor = (self.cursor+count) % len(self)
        return ids, self.roots[ids], self.offsets_mm[ids], self.normalizers[ids]

    @torch.no_grad()
    def retire(self, ids, scores):
        # A failed fresh random draw cannot erase a witness. Only checking its
        # actual saved tool point at the current pose can retire that pair.
        keep = torch.ones(len(self), dtype=torch.bool, device=self.roots.device)
        keep[ids] = scores.detach() > 0
        self.roots, self.offsets_mm, self.components, self.normalizers = tuple(
            value[keep] for value in (self.roots, self.offsets_mm, self.components, self.normalizers))

    def state(self):
        return {name: getattr(self, name).detach().cpu() if getattr(self, name) is not None else None
                for name in ('roots', 'offsets_mm', 'components', 'normalizers')}


def tool_witness_loss(field, roots, offsets_mm, normalizers, scale_mm, bbox, gate=None, *,
                      margin=0., normalize_collision_by_gradient=False, collision_gradient_epsilon=1e-6,
                      chunk_size=16384):
    """Recheck paired local tool samples with the ordinary hard membership gate.

    Preserve the source component/top-k denominator so one retained point does
    not acquire the weight of an entire sampled tool. Root averaging is over
    the targeted batch, intentionally concentrating attention on known failures.
    """
    roots = roots.detach().requires_grad_(True)
    output = field(roots, compute_grads=True)
    values = output['scalars'].reshape(-1)
    points = place_tool(roots, output['grads'], offsets_mm[:, None, :]/scale_mm)[:, 0]
    with torch.no_grad():
        active = ((points >= bbox[0]) & (points <= bbox[1])).all(1)
        ids = active.nonzero().flatten()
        if gate is not None and len(ids):
            active[ids] = scalar_values(gate, points[ids], chunk_size) < 0
        ids = active.nonzero().flatten()
    differences = values*0
    if len(ids):
        differences = differences.index_copy(0, ids, scalar_values(field, points[ids], chunk_size)-values[ids])
    error = torch.relu(margin-differences)
    if normalize_collision_by_gradient:
        error = error/output['grads'].norm(dim=1).clamp_min(collision_gradient_epsilon)
    scores = error.square()*active
    return dict(loss=(scores/normalizers).mean(), scores=scores, sample_points=points)

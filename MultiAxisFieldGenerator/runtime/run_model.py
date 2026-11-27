"""Configurable multi-axis support/stress field optimization. Run --help.

No legacy optimization module is imported, and no legacy output is overwritten.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn.functional as F

from collision import (collision_loss, collision_diagnostics, sample_abb_tool, scalar_values,
                       ToolWitnessBuffer, tool_witness_loss)
from fields import load_field, first_order_output
from root_sampling import IndexCycle, PartRootDomain, SurfaceRootDomain, WitnessReplay
from stress import StressConstraints
from collision_settings import collision_enabled
from surface_regions import surface_samples

HERE = Path(__file__).resolve().parent


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_data(mesh_path, spacing, seed, validation_count, device, output, up_axis='y', base_band_mm=2.5,
                 support_on_base=False):
    import igl
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    lo, hi = vertices.min(0), vertices.max(0)
    center, scale = (lo+hi)/2, float((hi-lo).max()/2)
    # Cache is geometry-specific, checked against the actual mesh arrays.
    import hashlib
    signature = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
    cache = output/'cache'/f'grid_{signature[:16]}_{spacing:g}.npz'
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        saved = np.load(cache)
        points = saved["points"] if str(saved["signature"]) == signature else None
    else:
        points = None
    if points is None:
        axes = [np.arange(a, b+spacing*.01, spacing) for a, b in zip(lo, hi)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        distances = igl.signed_distance(grid, vertices, faces)[0]
        points = grid[distances < 0]
        np.savez_compressed(cache, points=points, signature=signature)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    nval = min(validation_count, len(points)//4)
    if nval < 1:
        raise ValueError("Grid is too coarse to define a held-out root set")
    train, validation = points[order[nval:]], points[order[:nval]]
    _, centers, normals, area, base_mask = surface_samples(vertices, faces, up_axis, base_band_mm)
    axis = 'xyz'.index(up_axis)
    base = centers[base_mask]
    if base_band_mm > 0 and len(base) == 0:
        raise ValueError(f"No base faces within {base_band_mm:g} mm along +{up_axis.upper()}.")
    support_mask = np.ones(len(centers), dtype=bool) if support_on_base else ~base_mask
    surface, normals, area = centers[support_mask], normals[support_mask], area[support_mask]
    surface_order = rng.permutation(len(surface))
    nsval = min(4096, len(surface)//4)
    if nsval < 1:
        raise ValueError('At least four non-base surface triangles are required; reduce the base band or refine the mesh')
    vi, ti = surface_order[:nsval], surface_order[nsval:]

    def tensor(x, normalize=True):
        return torch.tensor((x-center)/scale if normalize else x, dtype=torch.float32, device=device)

    result = dict(train=tensor(train), validation=tensor(validation),
                  surface=tensor(surface[ti]), normals=tensor(normals[ti], False),
                  surface_weights=tensor(area[ti]/area[ti].sum(), False),
                  surface_validation=tensor(surface[vi]),
                  normals_validation=tensor(normals[vi], False),
                  validation_area=tensor(area[vi], False), base=tensor(base),
                  up_vector=tensor(np.eye(3)[axis], False),
                  bbox=tensor(np.stack((lo, hi))))
    metadata = dict(mesh=str(mesh_path), mesh_sha256=signature, center_mm=center.tolist(),
                    scale_mm=scale, up_axis=up_axis, spacing_mm=spacing,
                    train_roots=len(train), validation_roots=len(validation),
                    surface_train_faces=len(ti), surface_validation_faces=len(vi),
                    base_faces=len(base), base_band_mm=base_band_mm,
                    support_on_base=bool(support_on_base))
    np.savez_compressed(output/"validation_points.npz", roots_mm=validation,
                        surface_mm=surface[vi], normals=normals[vi], area=area[vi])
    return result, metadata


def draw_indices(total, count, device, generator=None):
    return torch.randperm(total, device=device, generator=generator)[:min(total, count)]


def full_epoch_batches(counts, batch_sizes, device, generator):
    """Cover every dataset without replacement, cycling shorter streams.

    Keep each final partial batch. Each new epoch/cycle gets a fresh permutation.
    The longest stream determines epoch length, as for zipped/cycled loaders.
    """
    if any(n < 1 for n in counts) or any(b < 1 for b in batch_sizes):
        raise ValueError("Dataset counts and batch sizes must be positive")
    if len(counts) != len(batch_sizes):
        raise ValueError("Supply one batch size per dataset")

    def shuffle(i):
        return torch.randperm(counts[i], device=device, generator=generator).split(batch_sizes[i])

    batches = [shuffle(i) for i in range(len(counts))]
    for step in range(max(len(stream) for stream in batches)):
        for i, stream in enumerate(batches):
            if step and step % len(stream) == 0:
                batches[i] = shuffle(i)
        yield tuple(stream[step % len(stream)] for stream in batches)


class RootSampler:
    def __init__(self, points, generator=None, cover_candidates=False):
        self.points = points
        self.generator = generator
        self.cover_candidates = cover_candidates
        self._candidate_batches = None
        self.hard = torch.empty(0, dtype=torch.long, device=points.device)
        self.positive = self.hard.clone()

    def candidate_ids(self, count):
        if not self.cover_candidates:
            return draw_indices(len(self.points), count, self.points.device, self.generator)
        candidates = next(self._candidate_batches, None) if self._candidate_batches is not None else None
        if candidates is None:
            order = torch.randperm(len(self.points), device=self.points.device, generator=self.generator)
            self._candidate_batches = iter(order.split(count))
            candidates = next(self._candidate_batches)
        return candidates

    def refresh(self, field, tool, scale, data, gate, args, full=False):
        device = self.points.device
        ids = (torch.arange(len(self.points), device=device) if full else
               torch.unique(torch.cat((self.candidate_ids(args.mining_candidates),
                                       self.hard))))
        scores = []
        for batch in ids.split(args.collision_roots):
            out = collision_diagnostics(field, self.points[batch], tool, scale, data["bbox"], gate,
                                 topk_fraction=args.topk, margin=args.margin,
                                 normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                                 collision_gradient_epsilon=args.collision_gradient_epsilon,
                                 chunk_size=args.field_query_batch, sdf_chunk_size=args.sdf_query_batch)
            scores.append(out["root_scores"].detach())
        scores = torch.cat(scores)
        positive = scores > 0
        positive_ids, positive_scores = ids[positive], scores[positive]
        count = min(args.hard_pool, len(positive_ids))
        self.hard = positive_ids[positive_scores.topk(count).indices]
        self.positive = positive_ids[~torch.isin(positive_ids, self.hard)]
        return dict(candidates=len(ids), positives=len(positive_ids), hard=len(self.hard))

    def draw(self, count):
        device = self.points.device
        hard = self.hard[draw_indices(len(self.hard), count//2, device, self.generator)]
        positive = self.positive[draw_indices(len(self.positive), count//4, device, self.generator)]
        chosen = torch.cat((hard, positive))
        uniform = torch.randperm(len(self.points), device=device, generator=self.generator)
        uniform = uniform[~torch.isin(uniform, chosen)][:count-len(chosen)]
        return self.points[torch.cat((chosen, uniform))]


class CoverageRootSampler(RootSampler):
    """Half persistent coverage, quarter hard roots, quarter fresh geometry."""
    def __init__(self, points, domain, generator, discovered_capacity=1024):
        super().__init__(points, generator, cover_candidates=True)
        self.domain = domain
        self.coverage = IndexCycle(len(points), points.device, generator)
        self.discovered_capacity = discovered_capacity
        self.discovered = points[:0].detach()
        self._observation = None

    def draw(self, count):
        if count < 4:
            raise ValueError('Coverage mode needs at least four roots per step')
        fresh_count, hard_count = count//4, count//4
        ids = self.coverage.take(count-fresh_count-hard_count)
        available = self.hard[~torch.isin(self.hard, ids)]
        dynamic_count = min(len(self.discovered), hard_count//2)
        hard = available[draw_indices(len(available), hard_count-dynamic_count, self.points.device, self.generator)]
        dynamic_count = min(len(self.discovered), hard_count-len(hard))
        dynamic_ids = draw_indices(len(self.discovered), dynamic_count, self.points.device, self.generator)
        # Unfilled hard slots also advance the coverage stream.
        missing = hard_count-len(hard)-len(dynamic_ids)
        if missing:
            ids = torch.cat((ids, self.coverage.take(missing)))
        fresh = self.domain.sample(fresh_count, self.generator)
        self._observation = (len(ids)+len(hard), dynamic_ids, count-fresh_count, fresh.detach())
        return torch.cat((self.points[ids], self.points[hard], self.discovered[dynamic_ids], fresh))

    @torch.no_grad()
    def observe(self, scores):
        """Retain fresh failures found by training itself; retire resolved revisits."""
        if self._observation is None:
            raise RuntimeError('Draw roots before observing their collision scores')
        start, dynamic_ids, fresh_start, fresh = self._observation
        if len(scores) != fresh_start+len(fresh):
            raise ValueError('Collision scores do not match the latest root batch')
        keep = torch.ones(len(self.discovered), dtype=torch.bool, device=self.points.device)
        keep[dynamic_ids] = scores[start:start+len(dynamic_ids)].detach() > 0
        new = fresh[scores[fresh_start:].detach() > 0]
        self.discovered = torch.unique(torch.cat((self.discovered[keep], new)), dim=0).detach()
        if len(self.discovered) > self.discovered_capacity:
            ids = draw_indices(len(self.discovered), self.discovered_capacity, self.points.device, self.generator)
            self.discovered = self.discovered[ids]
        self._observation = None


def regularization(field, roots, scale, curvature_radius_mm=35., *, orientation_axis=None,
                   orientation_max_angle_deg=75., gradient_regularizers=True,
                   compute_variation=True, compute_small=True, compute_curvature=True,
                   compute_unit_norm=True):
    compute_variation = compute_variation and gradient_regularizers
    compute_small = compute_small and gradient_regularizers
    if not any((compute_variation, compute_small, compute_curvature, compute_unit_norm, orientation_axis is not None)):
        zero = next(field.parameters()).sum()*0
        return zero, zero, zero, zero, zero
    roots = roots.detach().requires_grad_(True)
    out = field(roots, compute_grads=True, compute_hess=compute_variation or compute_curvature)
    grads, hessian = out["grads"], out.get("hessian")
    zero = grads.sum()*0
    unit_norm = unit_gradient_norm_loss(grads) if compute_unit_norm else zero
    norm = grads.norm(dim=1).clamp_min(1e-8)
    unit = grads/norm[:, None]
    if compute_variation:
        norm_gradient = torch.einsum("bij,bj->bi", hessian, unit)
        variation = (norm_gradient.square().sum(1)/norm).mean()
    else:
        variation = zero
    small = torch.exp(-100*norm.square()).mean() if compute_small else zero
    if compute_curvature:
        projection = torch.eye(3, device=roots.device)[None]-unit[:, :, None]*unit[:, None, :]
        shape = projection @ hessian @ projection / (norm[:, None, None]*scale)
        # Symmetric eigenvalues: two principal curvatures and a zero normal mode.
        curvatures = torch.linalg.eigvalsh((shape+shape.transpose(1, 2))/2)
        curvature = (30*torch.relu(curvatures.abs()-1/curvature_radius_mm)).square().sum(1).mean()/2
    else:
        curvature = zero
    orientation = (orientation_constraint(grads, orientation_axis, orientation_max_angle_deg)[0]
                   if orientation_axis is not None else curvature*0)
    return variation, small, curvature, orientation, unit_norm


def unit_gradient_norm_loss(grads):
    """Soft Eikonal penalty in the field's normalized training coordinates."""
    return (grads.norm(dim=1)-1.).square().mean()


def orientation_constraint(grads, reference_axis, max_angle_deg):
    """Penalize scalar gradients that leave a one-sided build-up cone.

    The scalar-gradient sign is meaningful here: support branches descend along
    ``-grad(field)``.  After this reference is aligned to the physical +Z build
    direction, keeping ``grad(field)`` inside the cone rules out an initially
    upward branch wherever the constraint is satisfied.
    """
    if reference_axis.shape != (3,):
        raise ValueError("Orientation reference axis must be a three-vector")
    if not 0 < max_angle_deg < 90:
        raise ValueError("Orientation maximum angle must be between 0 and 90 degrees")
    direction = F.normalize(grads, dim=1)
    target = F.normalize(reference_axis, dim=0)
    minimum_dot = math.cos(math.radians(max_angle_deg))
    violation = F.relu(minimum_dot - direction @ target)
    return violation.square().mean(), direction @ target


def capture_orientation_reference(field, surface_points, surface_normals, up_vector, count, generator=None):
    """Estimate a signed build-up direction from normalized inward surface gradients."""
    if count < 1:
        raise ValueError("Orientation reference sample count must be positive")
    ids = draw_indices(len(surface_points), min(count, len(surface_points)), surface_points.device, generator)
    directions = F.normalize(first_order_output(field, surface_points[ids])['grads'], dim=1).detach()
    inward = (directions*surface_normals[ids]).sum(1) < 0
    inward_surface_count = int(inward.sum())
    mean = directions[inward].mean(0) if inward_surface_count else torch.zeros(3, device=surface_points.device)
    # A near-zero resultant means there is no stable learned average.  The
    # configured build axis is a deterministic physical fallback, not an
    # arbitrary neural-field direction.
    if float(mean.norm()) < 1e-6:
        reference = up_vector.detach().clone()
        source = 'configured_up_axis_fallback_no_inward_surface_gradient'
    else:
        reference = F.normalize(mean, dim=0)
        source = 'normalized_inward_surface_gradients'
    return reference.detach(), source, len(ids), inward_surface_count


def inward_surface_orientation_constraint(grads, surface_normals, reference_axis, max_angle_deg):
    """Apply the one-sided cone only to scalar gradients entering the material boundary."""
    directions = F.normalize(grads, dim=1)
    inward = (directions*surface_normals).sum(1) < 0
    if not bool(inward.any()):
        return grads.sum()*0, 0
    return orientation_constraint(grads[inward], reference_axis, max_angle_deg)[0], int(inward.sum())


def orientation_weight_at_epoch(epoch, epochs, base_weight, start_fraction, ramp_fraction):
    """Late-stage linear ramp, zero before the requested fraction of training."""
    if base_weight == 0:
        return 0.
    start_epoch = max(1, math.ceil(epochs*start_fraction))
    if epoch < start_epoch:
        return 0.
    ramp_epochs = math.ceil(epochs*ramp_fraction)
    if ramp_epochs <= 0:
        return float(base_weight)
    return float(base_weight)*min(1., (epoch-start_epoch+1)/ramp_epochs)


def topology_regularization_scale_at_epoch(epoch, ramp_epochs, start_factor):
    """Linearly restore topology-suppressing geometric losses after initialization.

    Critical points are required for a scalar field to split or merge level-set
    topology.  The variation, small-gradient, and curvature losses all resist
    those points, so a short early ramp lets the stress objective establish the
    required structure before those regularizers return at full strength.
    """
    if ramp_epochs <= 0:
        return 1.
    if ramp_epochs == 1:
        return 1.
    progress = min(1., max(0., (epoch-1)/(ramp_epochs-1)))
    return float(start_factor) + (1.-float(start_factor))*progress


def gradient_regularization_blend_at_epoch(epoch, warmup_epochs, transition_epochs):
    """Return (original-gradient-loss factor, unit-norm factor).

    A zero warmup preserves the original losses. Otherwise the variation and
    small-gradient penalties are off for the complete warmup, then cross-fade
    with the unit-norm penalty over the requested transition. Curvature is
    controlled separately by the existing topology regularization ramp.
    """
    if warmup_epochs <= 0:
        return 1., 0.
    if epoch <= warmup_epochs:
        return 0., 1.
    blend = min(1., (epoch-warmup_epochs)/max(1, transition_epochs))
    return blend, 1.-blend


def orientation_reference_refresh_due(epoch, onset_epoch, update_every):
    """The onset captures the first target; later updates occur at fixed epochs."""
    return epoch >= onset_epoch and (epoch-onset_epoch) % update_every == 0


def surface_losses(field, data, count, angle, generator=None, *, surface_indices=None,
                   base_indices=None, base_count=256, collision_roots=None, grouped=True,
                   orientation_axis=None, orientation_max_angle_deg=75.,
                   compute_support=True, compute_base=True, compute_anchor=True):
    idx = surface_indices
    if idx is None:
        idx = torch.multinomial(data["surface_weights"], min(count, len(data["surface"])), replacement=False,
                                generator=generator)
    points = data["surface"][idx]
    if base_indices is None:
        base_indices = draw_indices(len(data["base"]), base_count, points.device, generator)
    base = data["base"][base_indices] if len(data["base"]) else data["base"]
    compute_base = compute_base and len(base) > 0
    compute_anchor = compute_anchor and len(base) > 0
    # Preserve random draws so skipping unused network evaluations does not
    # change subsequent batches or the stochastic training sequence.
    need_surface = compute_support or orientation_axis is not None
    labels, batches = [], []
    for label, batch, enabled in (('surface', points, need_surface),
                                  ('base', base, compute_base or compute_anchor),
                                  ('roots', collision_roots, collision_roots is not None)):
        if enabled:
            labels.append(label)
            batches.append(batch)
    if grouped and batches:
        x = torch.cat(batches).detach().requires_grad_(True)
        combined = field(x, compute_grads=True)
        split = {key: value.split([len(batch) for batch in batches]) for key, value in combined.items()}
        outputs = [{key: value[i] for key, value in split.items()} for i in range(len(batches))]
    else:
        outputs = [field(batch.detach().requires_grad_(True), compute_grads=True) for batch in batches]
    outputs = dict(zip(labels, outputs))
    zero = points.new_zeros(())
    support = base_loss = anchor = zero
    if compute_support:
        normals = F.normalize(outputs['surface']["grads"], dim=1)
        dot = (normals*data["normals"][idx]).sum(1)
        errors = (10*torch.relu(math.cos(math.radians(angle))-dot)).square()
    if compute_support and surface_indices is not None:
        # Uniform shuffled faces need explicit area weights to preserve the
        # area objective formerly approximated by weighted random sampling.
        errors = errors * (len(data["surface"])*data["surface_weights"][idx])
    if compute_support:
        support = errors.mean()
    if compute_base:
        direction = F.normalize(outputs['base']["grads"], dim=1)
        base_loss = (direction-data['up_vector']).square().mean()
    # Gauge anchor + planar base consistency, weakly weighted separately.
    if compute_anchor:
        anchor = outputs['base']["scalars"].square().mean()
    surface_orientation, inward_surface_count = (
        inward_surface_orientation_constraint(outputs['surface']['grads'], data['normals'][idx],
                                              orientation_axis, orientation_max_angle_deg)
        if orientation_axis is not None else (zero, 0))
    return (support, base_loss, anchor, outputs.get('roots'),
            surface_orientation, inward_surface_count)


def evaluate(field, tool, data, scale, gate, args, *, exact_mesh=None, orientation_axis=None):
    start = time.perf_counter()
    total, active, colliding_pairs, colliding_roots = 0, 0, 0, 0
    loss_sum, max_violation = 0., 0.
    root_norms = []
    if exact_mesh is not None:
        import igl
        vertices, faces, center = exact_mesh

        def evaluation_gate(points, compute_grads=False):
            world = points.detach().cpu().numpy()*scale+center
            values = igl.signed_distance(world, vertices, faces)[0]
            return torch.tensor(values, dtype=points.dtype, device=points.device)
    else:
        evaluation_gate = gate
    for roots in data["validation"].split(args.validation_batch):
        if gate is None and exact_mesh is None:
            root_norms.append(first_order_output(field, roots)['grads'].norm(dim=1))
            continue
        out = collision_diagnostics(field, roots, tool, scale, data["bbox"], evaluation_gate,
                             topk_fraction=args.topk, margin=args.margin,
                             normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                             collision_gradient_epsilon=args.collision_gradient_epsilon,
                             chunk_size=args.field_query_batch, sdf_chunk_size=args.sdf_query_batch)
        root_norms.append(out['root_gradient_norms'])
        mask = out["collision_mask"]
        colliding_roots += int(mask.any(1).sum())
        colliding_pairs += int(mask.sum())
        total += out["total_pairs"]
        active += out["active_pairs"]
        loss_sum += float(out["root_scores"].detach().sum())
        if mask.any():
            max_violation = max(max_violation, float((-out["differences"][mask]).max().detach()))
    dots, surface_directions = [], []
    for points, normals in zip(data["surface_validation"].split(512), data["normals_validation"].split(512)):
        direction = F.normalize(first_order_output(field, points)["grads"], dim=1)
        dots.append((direction*normals).sum(1).detach())
        surface_directions.append(direction.detach())
    dot = torch.cat(dots)
    surface_direction = torch.cat(surface_directions)
    areas = data["validation_area"]
    result = dict(root_collision_fraction=colliding_roots/len(data["validation"]),
                  colliding_pairs=colliding_pairs, active_pairs=active, total_pairs=total,
                  collision_score=loss_sum/len(data["validation"]),
                  max_field_violation=max_violation,
                  support_failure_area_132=float((areas*(dot < math.cos(math.radians(132)))).sum()/areas.sum()),
                  support_failure_area_135=float((areas*(dot < math.cos(math.radians(135)))).sum()/areas.sum()),
                  gate="exact_mesh" if exact_mesh is not None else "learned_sdf")
    if orientation_axis is not None:
        directions = []
        for points in data['validation'].split(args.validation_batch):
            directions.append(F.normalize(first_order_output(field, points)['grads'], dim=1).detach())
        dots_to_axis = torch.cat(directions) @ orientation_axis
        safe_dots = dots_to_axis.clamp(-1., 1.)
        minimum_dot = math.cos(math.radians(args.orientation_max_angle_deg))
        result.update(orientation_reference_axis=orientation_axis.detach().cpu().tolist(),
                      orientation_max_angle_deg=float(args.orientation_max_angle_deg),
                      orientation_mean_dot=float(dots_to_axis.mean()),
                      orientation_min_dot=float(dots_to_axis.min()),
                      orientation_max_angle_observed_deg=float(torch.rad2deg(torch.acos(safe_dots)).max()),
                      orientation_violation_fraction=float((dots_to_axis < minimum_dot).float().mean()),
                      upward_growth_fraction=float((dots_to_axis <= 0).float().mean()))
        inward_surface = dot < 0
        if bool(inward_surface.any()):
            surface_dots_to_axis = surface_direction[inward_surface] @ orientation_axis
            result.update(orientation_inward_surface_samples=int(inward_surface.sum()),
                          orientation_surface_violation_fraction=float((surface_dots_to_axis < minimum_dot).float().mean()),
                          orientation_surface_upward_growth_fraction=float((surface_dots_to_axis <= 0).float().mean()))
        else:
            result.update(orientation_inward_surface_samples=0,
                          orientation_surface_violation_fraction=None,
                          orientation_surface_upward_growth_fraction=None)
    if gate is None and exact_mesh is None:
        for key in ['root_collision_fraction', 'colliding_pairs', 'active_pairs', 'total_pairs', 'collision_score', 'max_field_violation']:
            result[key] = None
        result['gate'] = 'not_checked'
    norms = torch.cat(root_norms)
    result["min_root_gradient_norm"] = float(norms.min())
    result["median_root_gradient_norm"] = float(norms.median())
    result["degenerate_root_fraction"] = float((norms < 1e-6).float().mean())
    with torch.no_grad():
        values = scalar_values(field, data["validation"])
        result["validation_scalar_range"] = float(values.max()-values.min())
    sync(data["train"].device)
    result["seconds"] = time.perf_counter()-start
    return result


def benchmark(field, tool, data, scale, gate, args):
    roots = data["validation"][:min(args.collision_roots, 64)]
    outputs, gradients, times = {}, {}, {}
    for cull in [True, False]:
        label = "culled" if cull else "unculled"
        for iteration in range(4):
            field.zero_grad(set_to_none=True)
            sync(roots.device)
            start = time.perf_counter()
            out = collision_loss(field, roots, tool, scale, data["bbox"], gate,
                                 topk_fraction=args.topk, margin=args.margin,
                                 normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                                 prune_zero_penalty=not getattr(args, 'disable_collision_pruning', False),
                                 collision_gradient_epsilon=args.collision_gradient_epsilon, cull=cull,
                                 chunk_size=getattr(args, 'field_query_batch', 16384),
                                 sdf_chunk_size=getattr(args, 'sdf_query_batch', 32768))
            out["loss"].backward()
            sync(roots.device)
            elapsed = time.perf_counter()-start
            if iteration > 0:
                times.setdefault(label, []).append(elapsed)
        outputs[label] = out
        gradients[label] = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                                     for p in field.parameters()]).detach().clone()
    torch.testing.assert_close(outputs["culled"]["loss"], outputs["unculled"]["loss"], rtol=2e-4, atol=1e-8)
    torch.testing.assert_close(gradients["culled"], gradients["unculled"], rtol=3e-3, atol=3e-6)
    field.zero_grad(set_to_none=True)
    return dict(roots=len(roots), tool_points=len(tool.offsets_mm),
                total_pairs=outputs["culled"]["total_pairs"],
                active_pairs=outputs["culled"]["active_pairs"],
                culled_seconds=float(np.median(times["culled"])),
                unculled_seconds=float(np.median(times["unculled"])),
                speedup=float(np.median(times["unculled"])/np.median(times["culled"])),
                loss=float(outputs["culled"]["loss"].detach()),
                max_parameter_gradient_error=float((gradients["culled"]-gradients["unculled"]).abs().max()))


def parse_collision_schedule(entries):
    schedule = []
    for item in entries or []:
        try:
            start, weight = item.split(':')
            start, weight = int(start), float(weight)
        except (ValueError, AttributeError):
            raise ValueError('Schedule entries must be EPOCH:WEIGHT, e.g. 26:20000') from None
        if start < 1 or not math.isfinite(weight) or weight < 0:
            raise ValueError('Schedule epochs must be positive and weights finite and nonnegative')
        if schedule and start <= schedule[-1][0]:
            raise ValueError('Schedule epochs must be strictly increasing')
        schedule.append((start, weight))
    return schedule


def collision_weight_at_epoch(epoch, base_weight, schedule):
    weight = base_weight
    for start, value in schedule or []:
        if start > epoch:
            break
        weight = value
    return weight


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=HERE.parent/'runs'/'model')
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--start-checkpoint", type=Path, required=True)
    p.add_argument("--sdf-checkpoint", type=Path, help='Required only when a collision weight is positive during this run')
    p.add_argument("--tool-config", type=Path, default=HERE.parent/'tools'/'abb_training.json')
    p.add_argument('--up-axis', choices=['x', 'y', 'z'], default='y')
    p.add_argument('--base-band-mm', type=float, default=2.5, help='Base region thickness in mm; zero or negative disables the region and its constraints.')
    p.add_argument('--support-on-base', action='store_true',
                   help='Include the bottom base band in the support-angle objective; use for experiments without a prescribed platform base.')
    p.add_argument('--field-width', type=int, default=128)
    p.add_argument('--field-layers', type=int, default=10)
    p.add_argument('--field-frequency', type=float, default=7.)
    p.add_argument('--sdf-width', type=int, default=256)
    p.add_argument('--sdf-layers', type=int, default=5)
    p.add_argument('--sdf-frequency', type=float, default=30.)
    p.add_argument('--objective', choices=['support_free', 'stress', 'combined'], default='support_free')
    p.add_argument('--stress-file', type=Path)
    p.add_argument('--stress-weight', type=float, default=1.)
    p.add_argument('--stress-samples', type=int, default=4096)
    p.add_argument('--stress-batch', type=int, default=256)
    p.add_argument('--stress-neighbors', type=int, default=8)
    p.add_argument('--stress-max-distance-mm', type=float)
    p.add_argument('--stress-point-columns', type=int, nargs=3, default=[0, 1, 2])
    p.add_argument('--stress-direction-columns', type=int, nargs=3, default=[3, 4, 5])
    p.add_argument('--stress-coordinate-space', choices=['world', 'model'], default='world')
    p.add_argument('--stress-region', choices=['all', 'tensile', 'compressive'], default='all')
    p.add_argument('--stress-signed-field')
    p.add_argument('--stress-threshold', type=float, default=0.)
    p.add_argument('--stress-top-percent', type=float, default=100.,
                   help='Align the highest-magnitude C percent of eligible FEA cells; 100 keeps the whole selected stress region. Cutoff ties are included.')
    p.add_argument('--stress-sampling', choices=['resampled', 'fps'], default='resampled')
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--epoch-mode", choices=["full-pass", "sampled-steps"], default="full-pass",
                   help="Default covers all interior/surface/base points; sampled-steps reproduces the earlier fixed-budget mode")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="Required only with --epoch-mode sampled-steps; full-pass derives steps from dataset and batch sizes")
    p.add_argument("--interior-batch", type=int, default=256)
    p.add_argument('--separate-hessian-passes', action='store_true',
                   help='Use three separate Hessian derivative calls instead of the batched vector products (reference implementation)')
    p.add_argument("--surface-batch", type=int, default=1024)
    p.add_argument("--base-batch", type=int, default=256)
    p.add_argument("--collision-roots", type=int, default=256)
    p.add_argument('--collision-root-mode', choices=['coverage', 'legacy'], default='coverage')
    p.add_argument('--root-cell-mm', type=float, default=3.)
    p.add_argument('--root-surface-inset-mm', type=float, default=.25)
    p.add_argument('--volume-mesh', type=Path, help='Default: surface mesh path with .ele suffix')
    p.add_argument('--tool-volume-samples', type=int, default=128, help='Additional samples per tool component per step')
    p.add_argument('--online-witness-batch', type=int, default=64,
                   help='Previously detected root/tool pairs revisited per step; zero disables online retention')
    p.add_argument('--online-witness-capacity', type=int, default=4096,
                   help='Maximum retained root/tool pairs, separate from the hard-root pool')
    p.add_argument('--replay-results', type=Path, help='Saved nominal validation directory to use for targeted refinement')
    p.add_argument('--replay-batch', type=int, default=64)
    p.add_argument('--replay-softness-mm', type=float, default=.25)
    p.add_argument('--replay-weight', type=float, default=1., help='Multiplier relative to collision weight')
    p.add_argument("--field-query-batch", type=int, default=32768)
    p.add_argument("--sdf-query-batch", type=int, default=32768)
    p.add_argument("--learning-rate", type=float, default=4e-6)
    p.add_argument('--learning-rate-decay-every-epochs', type=int, default=150,
                   help='Main optimizer decay interval; zero keeps the learning rate constant')
    p.add_argument('--learning-rate-decay-factor', type=float, default=.4,
                   help='Multiplier applied to the main optimizer learning rate at each decay')
    p.add_argument("--collision-weight", type=float, default=1e4)
    p.add_argument('--disable-collision-pruning', action='store_true',
                   help='Retain derivative graphs for all active collision queries instead of only potential nonzero penalties')
    p.add_argument('--collision-weight-schedule', nargs='+', metavar='EPOCH:WEIGHT',
                   help='Piecewise constant weights, e.g. 1:10000 26:20000 51:30000 76:50000')
    p.add_argument("--support-weight", type=float, default=5.)
    p.add_argument('--variation-weight', type=float, default=.02)
    p.add_argument('--small-gradient-weight', type=float, default=1.)
    p.add_argument('--curvature-weight', type=float, default=.1)
    p.add_argument('--curvature-radius-mm', type=float, default=35.)
    p.add_argument('--topology-regularization-ramp-epochs', type=int, default=0,
                   help='Early epochs that ramp variation, small-gradient, and curvature losses to full strength; zero disables it')
    p.add_argument('--topology-regularization-start-factor', type=float, default=.05,
                   help='Initial fraction of topology-suppressing geometric regularizers during the early ramp')
    p.add_argument('--gradient-norm-warmup-epochs', type=int, default=0,
                   help='Initial epochs using only the unit-norm gradient penalty in place of variation and small-gradient penalties; zero disables substitution')
    p.add_argument('--gradient-norm-transition-epochs', type=int, default=10,
                   help='Epochs to restore variation/small-gradient penalties while fading out the unit-norm penalty after warmup; zero switches immediately')
    p.add_argument('--gradient-norm-weight', type=float, default=1.,
                   help='Weight of mean((norm(gradient)-1)^2) during warmup, fading out during transition')
    p.add_argument('--orientation-constraint-weight', type=float, default=0.,
                   help='Late-stage penalty for scalar gradients outside the selected build-up cone; zero disables it')
    p.add_argument('--orientation-max-angle-deg', type=float, default=75.,
                   help='Maximum scalar-gradient angle from the inferred late-stage reference direction')
    p.add_argument('--orientation-start-fraction', type=float, default=.70,
                   help='Training fraction at which the late orientation constraint starts')
    p.add_argument('--orientation-ramp-fraction', type=float, default=.15,
                   help='Fraction of all epochs over which the orientation weight rises to its requested value')
    p.add_argument('--orientation-reference-samples', type=int, default=4096,
                   help='Boundary gradients sampled at each late-stage build-up reference update')
    p.add_argument('--orientation-reference-update-every', type=int, default=5,
                   help='Refresh the inferred late-stage reference every N epochs')
    p.add_argument('--base-weight', type=float, default=5.)
    p.add_argument('--anchor-weight', type=float, default=.001)
    p.add_argument("--support-angle", type=float, default=132.)
    p.add_argument("--normalize-collision-by-gradient", action="store_true")
    p.add_argument("--collision-gradient-epsilon", type=float, default=1e-6)
    p.add_argument("--topk", type=float, default=.3)
    p.add_argument("--margin", type=float, default=0., help="Scalar-field margin, not a distance in mm")
    p.add_argument("--grid-spacing", type=float, default=3.)
    p.add_argument("--validation-count", type=int, default=1024)
    p.add_argument("--validation-batch", type=int, default=16)
    p.add_argument("--validate-every", type=int, default=10)
    p.add_argument("--refresh-every", type=int, default=5)
    p.add_argument("--mining-candidates", type=int, default=4096)
    p.add_argument("--hard-pool", type=int, default=1024)
    p.add_argument("--exact-validation", action="store_true", help="Also gate final dense samples with mesh signed distance")
    p.add_argument("--evaluate-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.objective != 'support_free' and args.stress_file is None:
        raise ValueError('Stress and combined objectives require --stress-file')
    if args.objective == 'support_free' and args.stress_file is not None:
        raise ValueError('Choose stress or combined objective when supplying stress data')
    if args.objective == 'stress':
        args.support_weight = 0.
    if not math.isfinite(args.base_band_mm):
        raise ValueError('Base band must be finite')
    if args.base_band_mm <= 0:
        args.base_band_mm = 0.
        args.base_weight = args.anchor_weight = 0.
        args.support_on_base = True
    if min(args.stress_samples, args.stress_batch, args.stress_neighbors) <= 0 or args.stress_weight < 0:
        raise ValueError('Invalid base/stress parameters')
    if args.stress_max_distance_mm is not None and args.stress_max_distance_mm <= 0:
        raise ValueError('Stress distance must be positive')
    if not math.isfinite(args.stress_threshold) or args.stress_threshold < 0:
        raise ValueError('Stress threshold must be finite and nonnegative')
    if not math.isfinite(args.stress_top_percent) or not 0 < args.stress_top_percent <= 100:
        raise ValueError('Top stress percentage must be greater than zero and at most 100')
    if (args.stress_region != 'all' or args.stress_top_percent < 100) and (not args.volume_mesh or not args.stress_signed_field):
        raise ValueError('Stress region/percentage selection requires a volume mesh and signed cell stress field')
    if args.save_every < 1 or args.snapshot_every < 0:
        raise ValueError('save-every must be positive and snapshot-every nonnegative')
    if args.curvature_radius_mm <= 0 or min(args.variation_weight, args.small_gradient_weight,
        args.curvature_weight, args.base_weight, args.anchor_weight, args.support_weight,
        args.orientation_constraint_weight) < 0:
        raise ValueError('Regularization radius must be positive and loss weights nonnegative')
    if args.learning_rate_decay_every_epochs < 0 or not 0 < args.learning_rate_decay_factor <= 1:
        raise ValueError('Learning-rate decay interval must be nonnegative and factor must be in (0, 1]')
    if args.topology_regularization_ramp_epochs < 0 or not 0 <= args.topology_regularization_start_factor <= 1:
        raise ValueError('Topology regularization ramp must be nonnegative and its start factor must be in [0, 1]')
    if min(args.gradient_norm_warmup_epochs, args.gradient_norm_transition_epochs) < 0 or not math.isfinite(args.gradient_norm_weight) or args.gradient_norm_weight < 0:
        raise ValueError('Gradient norm warmup and transition epochs, and its finite weight, must be nonnegative')
    if not 0 < args.orientation_max_angle_deg < 90:
        raise ValueError('Orientation maximum angle must be strictly between 0 and 90 degrees')
    if not 0 <= args.orientation_start_fraction <= 1 or not 0 <= args.orientation_ramp_fraction <= 1:
        raise ValueError('Orientation start and ramp fractions must be within [0, 1]')
    if args.orientation_start_fraction + args.orientation_ramp_fraction > 1:
        raise ValueError('Orientation start and ramp fractions must finish by the final epoch')
    if min(args.orientation_reference_samples, args.orientation_reference_update_every) < 1:
        raise ValueError('Orientation reference sample count and update interval must be positive')
    args.collision_weight_schedule = parse_collision_schedule(args.collision_weight_schedule)
    needs_sdf = collision_enabled(args.collision_weight, args.epochs, args.collision_weight_schedule)
    if needs_sdf and args.sdf_checkpoint is None:
        raise ValueError('Positive collision training weight requires --sdf-checkpoint; build one or set collision weight to zero')
    if not math.isfinite(args.collision_weight) or args.collision_weight < 0:
        raise ValueError('Collision weight must be finite and nonnegative')
    for name in ["interior_batch", "surface_batch", "base_batch", "collision_roots", "validation_count",
                "validation_batch", "validate_every", "refresh_every", "mining_candidates", "hard_pool",
                "field_query_batch", "sdf_query_batch"]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.tool_volume_samples < 0 or args.replay_batch < 1 or args.replay_weight < 0:
        raise ValueError('Invalid tool volume sample count or replay settings')
    if args.online_witness_batch < 0 or args.online_witness_capacity < 1:
        raise ValueError('Online witness batch must be nonnegative and capacity positive')
    if min(args.root_cell_mm, args.root_surface_inset_mm, args.replay_softness_mm) <= 0:
        raise ValueError('Spatial cell, inset and replay softness must be positive')
    if args.collision_root_mode == 'coverage' and args.collision_roots < 4:
        raise ValueError('Coverage mode requires at least four collision roots')
    if args.epoch_mode == "sampled-steps":
        if args.steps_per_epoch is None or args.steps_per_epoch < 1:
            raise ValueError("sampled-steps requires a positive --steps-per-epoch")
    elif args.steps_per_epoch is not None:
        raise ValueError("For complete coverage, omit --steps-per-epoch. For the old fixed-budget mode, add --epoch-mode sampled-steps")
    if not np.isfinite(args.collision_gradient_epsilon) or args.collision_gradient_epsilon <= 0:
        raise ValueError("collision-gradient-epsilon must be finite and positive")
    if args.epochs < 0 or args.grid_spacing <= 0 or not 0 < args.topk <= 1:
        raise ValueError("Invalid epochs, grid spacing or top-k fraction")
    if args.output.exists() and (args.output/"config.json").exists():
        raise FileExistsError(f"Run already exists: {args.output}; choose a fresh --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Oversubscription dominates these small batches on CPUs.
    torch.set_num_threads(int(os.environ.get('MULTIAXIS_NATIVE_THREADS', 0)) or min(8, os.cpu_count() or 1))
    field, checkpoint = load_field(args.start_checkpoint, device, width=args.field_width, layers=args.field_layers, frequency=args.field_frequency)
    field.batched_hessian = not args.separate_hessian_passes
    gate = None
    if needs_sdf:
        gate, _ = load_field(args.sdf_checkpoint, device, width=args.sdf_width, layers=args.sdf_layers, frequency=args.sdf_frequency)
        gate.requires_grad_(False).eval()
    data, metadata = prepare_data(args.mesh, args.grid_spacing, args.seed, args.validation_count, device, args.output,
                                  up_axis=args.up_axis, base_band_mm=args.base_band_mm,
                                  support_on_base=args.support_on_base)
    dataset_counts = [len(data[name]) for name in ["train", "surface", "base"]]
    if args.support_weight == 0:
        dataset_counts[1] = 1
    if args.base_weight == 0 and args.anchor_weight == 0:
        dataset_counts[2] = 1
    dataset_batch_sizes = [args.interior_batch, args.surface_batch, args.base_batch]
    if args.epoch_mode == "full-pass":
        args.steps_per_epoch = max(math.ceil(n/b) for n, b in zip(dataset_counts, dataset_batch_sizes))
    scale = metadata["scale_mm"]
    if "normalization" in checkpoint:
        previous = checkpoint["normalization"]
        if not np.allclose(previous["center_mm"], metadata["center_mm"]) or not np.isclose(previous["scale_mm"], scale):
            raise ValueError("Checkpoint and mesh normalization do not match")
    tool_spec = json.loads(args.tool_config.read_text())
    tool = sample_abb_tool(tool_spec, volume_samples=args.tool_volume_samples,
                           generator=torch.Generator().manual_seed(args.seed+101)).to(device)
    dense_tool = sample_abb_tool(tool_spec, dense=True).to(device)
    domain = None
    collision_points = data['train']
    if args.collision_root_mode == 'coverage' and (needs_sdf or args.stress_file is not None):
        volume_path = args.volume_mesh or args.mesh.with_suffix('.ele')
        if args.volume_mesh and not volume_path.exists():
            raise FileNotFoundError(volume_path)
        domain = PartRootDomain(volume_path,
                                metadata['center_mm'], scale, device,
                                cell_mm=args.root_cell_mm, inset_mm=args.root_surface_inset_mm) if volume_path.exists() else SurfaceRootDomain(
                                    args.mesh, metadata['center_mm'], scale, device, inset_mm=args.root_surface_inset_mm)
        if not np.allclose((domain.bounds_mm-np.array(metadata['center_mm']))/scale,
                           data['bbox'].cpu().numpy(), atol=1e-4):
            raise ValueError('Collision tetrahedral mesh bounds differ from surface mesh')
        extra = torch.tensor((domain.pool_mm-np.array(metadata['center_mm']))/scale,
                             dtype=torch.float32, device=device)
        collision_points = torch.unique(torch.cat((data['train'], extra)), dim=0)
    replay = None
    if args.replay_results:
        replay = WitnessReplay(args.replay_results, np.array(metadata['center_mm']), scale, device,
                                torch.Generator(device=device).manual_seed(args.seed+400))
    stress = None
    stress_generator = torch.Generator(device=device).manual_seed(args.seed+500)
    if args.stress_file is not None:
        if domain is None:
            raise ValueError('Stress resampling requires coverage root mode')
        import igl
        vertices, faces = igl.read_triangle_mesh(str(args.mesh))
        stress = StressConstraints(args, np.array(metadata['center_mm']), scale, device, vertices, faces)
        if args.stress_sampling == 'fps':
            stress.count = len(data['train'])
        stress.resample(domain, stress_generator)
        dataset_counts.append(len(stress.points_epoch))
        dataset_batch_sizes.append(args.stress_batch)
        if args.epoch_mode == 'full-pass':
            args.steps_per_epoch = max(math.ceil(n/b) for n, b in zip(dataset_counts, dataset_batch_sizes))
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(normalization=metadata, field_spec=field.spec,
                  initialization=checkpoint.get('initialization', checkpoint.get('config', {}).get('initialization')),
                  collision_training_enabled=needs_sdf,
                  collision_query_policy=('full_gradients' if args.disable_collision_pruning else 'penalty_active_gradients') if needs_sdf else 'disabled',
                  inactive_surface_losses_skipped=True,
                  hessian_evaluation='batched' if field.batched_hessian else 'separate',
                  collision_pool_points=len(collision_points),
                  root_coverage=domain.metadata if domain else None,
                  replay_witnesses=len(replay.roots) if replay else 0,
                  stress_metadata=stress.metadata if stress else None,
                  collision_mining_mode="shuffled_sweeps" if domain or args.epoch_mode == "full-pass" else "random_candidates",
                  tool=tool_spec,
                  tool_counts=dict(zip(tool.names, tool.counts)), dense_tool_counts=dense_tool.counts,
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                  torch_version=str(torch.__version__))
    import hashlib
    config["source_sha256"] = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ["run_model.py", "collision.py", "fields.py", "root_sampling.py", "stress.py"]}
    config['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in [('mesh', args.mesh), ('initializer', args.start_checkpoint),
                                                 ('tool', args.tool_config)]}
    if gate is not None:
        config['input_sha256']['sdf'] = hashlib.sha256(args.sdf_checkpoint.read_bytes()).hexdigest()
    if args.stress_file:
        config['input_sha256']['stress'] = hashlib.sha256(args.stress_file.read_bytes()).hexdigest()
    config['sdf_spec'] = gate.spec if gate is not None else None
    if args.replay_results:
        config['replay_sha256'] = {name: hashlib.sha256((args.replay_results/name).read_bytes()).hexdigest()
                                  for name in ['report.json', 'witnesses.npz']}
    orientation_axis = None
    orientation_generator = torch.Generator(device=device).manual_seed(args.seed+600)
    orientation_start_epoch = max(1, math.ceil(args.epochs*args.orientation_start_fraction))
    config['orientation_constraint'] = dict(enabled=bool(args.orientation_constraint_weight > 0),
                                            requested_weight=args.orientation_constraint_weight,
                                            max_angle_deg=args.orientation_max_angle_deg,
                                            start_fraction=args.orientation_start_fraction,
                                            ramp_fraction=args.orientation_ramp_fraction,
                                            reference_samples=args.orientation_reference_samples,
                                            reference_update_every_epochs=args.orientation_reference_update_every,
                                            sample_domain=('normalized inward complete-boundary gradients' if args.support_on_base
                                                           else 'normalized inward non-base surface gradients'),
                                            onset_epoch=orientation_start_epoch,
                                            reference_axis=None, reference_updates=[])
    (args.output/"config.json").write_text(json.dumps(config, indent=2))

    def record(event):
        print(json.dumps(event), flush=True)
        with (args.output/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(event)+"\n")

    print(f"{tool_spec.get('name', 'Tool')} counts: {tool.counts}; dense: {dense_tool.counts}; roots: {len(data['train'])}", flush=True)
    print(f"Collision pool: {len(collision_points):,}; root mode: {args.collision_root_mode}; "
          f"replay witnesses: {len(replay.roots) if replay else 0}", flush=True)
    print(f"Epoch mode: {args.epoch_mode}; {args.steps_per_epoch} steps/epoch; "
          f"interior/surface/base{'/stress' if stress else ''} stream counts: {dataset_counts}", flush=True)
    if stress and stress.sampling == 'fps':
        print(f"Fixed FPS stress pool: {len(stress.points_epoch)} points; "
              f"interior target: {len(data['train'])}; selected once at setup", flush=True)
    if gate is not None:
        record(dict(event="benchmark", **benchmark(field, tool, data, scale, gate, args)))
        with torch.no_grad():
            sdf_at_roots = scalar_values(gate, data["validation"])
        record(dict(event="sdf_root_check", false_outside_fraction=float((sdf_at_roots >= 0).float().mean())))
    initial = evaluate(field, dense_tool, data, scale, gate, args)
    if stress:
        initial['stress_validation_mean_squared_dot'] = stress.evaluate(field)
    record(dict(event="validation", epoch=0, **initial))
    optimizer = torch.optim.Adam(field.parameters(), lr=args.learning_rate)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_rate_decay_every_epochs,
                                                  gamma=args.learning_rate_decay_factor)
                 if args.learning_rate_decay_every_epochs else None)
    train_generator = torch.Generator(device=device).manual_seed(args.seed+200)
    collision_generator = torch.Generator(device=device).manual_seed(args.seed+300)
    sampler = (CoverageRootSampler(collision_points, domain, collision_generator, args.hard_pool) if domain else
               RootSampler(collision_points, collision_generator, cover_candidates=args.epoch_mode == "full-pass"))
    online_witnesses = ToolWitnessBuffer(args.online_witness_capacity) if needs_sdf and args.online_witness_batch else None
    generator = torch.Generator().manual_seed(args.seed+100)
    start = time.perf_counter()
    mining_started = False
    previous_weight = None
    if not args.evaluate_only:
        for epoch in range(1, args.epochs+1):
            learning_rate_used=float(optimizer.param_groups[0]['lr'])
            epoch_wall_start = time.perf_counter()
            mining_seconds = 0.
            if stress and stress.sampling != 'fps':
                dataset_counts[-1] = stress.resample(domain, stress_generator)
                record(dict(event='stress_resampling', epoch=epoch, accepted=dataset_counts[-1], rejected=stress.rejected,
                            max_nearest_source_distance_mm=stress.max_query_distance_mm))
                if args.epoch_mode == 'full-pass':
                    args.steps_per_epoch = max(math.ceil(n/b) for n, b in zip(dataset_counts, dataset_batch_sizes))
            epoch_weight = collision_weight_at_epoch(epoch, args.collision_weight, args.collision_weight_schedule)
            orientation_weight = orientation_weight_at_epoch(
                epoch, args.epochs, args.orientation_constraint_weight,
                args.orientation_start_fraction, args.orientation_ramp_fraction)
            topology_regularization_scale = topology_regularization_scale_at_epoch(
                epoch, args.topology_regularization_ramp_epochs,
                args.topology_regularization_start_factor)
            gradient_regularization_blend, gradient_norm_blend = gradient_regularization_blend_at_epoch(
                epoch, args.gradient_norm_warmup_epochs, args.gradient_norm_transition_epochs)
            gradient_regularization_scale = topology_regularization_scale*gradient_regularization_blend
            gradient_norm_weight = args.gradient_norm_weight*gradient_norm_blend
            if orientation_weight > 0 and orientation_reference_refresh_due(
                    epoch, orientation_start_epoch, args.orientation_reference_update_every):
                orientation_axis, reference_source, reference_count, inward_surface_count = capture_orientation_reference(
                    field, data['surface'], data['normals'], data['up_vector'],
                    args.orientation_reference_samples, orientation_generator)
                update = dict(epoch=epoch, reference_axis=orientation_axis.cpu().tolist(),
                              reference_source=reference_source, samples=reference_count,
                              inward_surface_samples=inward_surface_count)
                state = config['orientation_constraint']
                state.update(reference_axis=update['reference_axis'], reference_source=reference_source,
                             reference_sample_count=reference_count, last_updated_epoch=epoch)
                state['captured_epoch'] = state.get('captured_epoch', epoch)
                state['reference_updates'].append(update)
                (args.output/"config.json").write_text(json.dumps(config, indent=2))
                record(dict(event='orientation_reference', **update,
                            max_angle_deg=args.orientation_max_angle_deg))
            if epoch_weight != previous_weight:
                record(dict(event='collision_weight_change', epoch=epoch, collision_weight=epoch_weight))
                previous_weight = epoch_weight
            if epoch_weight > 0 and (not mining_started or (epoch-1)%args.refresh_every == 0):
                sync(device)
                mining_start = time.perf_counter()
                mining_result = sampler.refresh(field, tool, scale, data, gate, args, full=not mining_started)
                sync(device)
                mining_seconds = time.perf_counter()-mining_start
                record(dict(event="mining", epoch=epoch,
                            seconds=mining_seconds, **mining_result))
                mining_started = True
            sums = {}
            collision_active_pairs = collision_gradient_pairs = 0
            epoch_start = time.perf_counter()
            batches = (full_epoch_batches(dataset_counts, dataset_batch_sizes, device, train_generator)
                       if args.epoch_mode == "full-pass" else
                       (tuple(None for _ in dataset_counts) for _ in range(args.steps_per_epoch)))
            for step, batch_ids in enumerate(batches):
                ids, surface_ids, base_ids = batch_ids[:3]
                if args.support_weight == 0:
                    # The one-item placeholder must not restrict late
                    # orientation training to the first surface triangle.
                    # Surface constraints retain their random area sampling;
                    # fixed interior/stress datasets use the epoch batches.
                    surface_ids = None
                stress_ids = batch_ids[3] if stress else None
                if stress is not None and stress_ids is None:
                    stress_ids = draw_indices(len(stress.points_epoch), args.stress_batch, device, stress_generator)
                optimizer.zero_grad(set_to_none=True)
                if ids is None:
                    ids = draw_indices(len(data["train"]), args.interior_batch, device, train_generator)
                variation, small, curvature, orientation, unit_norm = regularization(
                    field, data["train"][ids], scale, args.curvature_radius_mm,
                    orientation_axis=None,
                    orientation_max_angle_deg=args.orientation_max_angle_deg,
                    gradient_regularizers=gradient_regularization_scale > 0,
                    compute_variation=args.variation_weight > 0,
                    compute_small=args.small_gradient_weight > 0,
                    compute_curvature=topology_regularization_scale*args.curvature_weight > 0,
                    compute_unit_norm=gradient_norm_weight > 0)
                collision_roots = sampler.draw(args.collision_roots) if epoch_weight > 0 else None
                replay_roots = replay_obstacles = None
                all_collision_roots = collision_roots
                if replay is not None and collision_roots is not None:
                    replay_roots, replay_obstacles = replay.draw(args.replay_batch)
                    all_collision_roots = torch.cat((collision_roots, replay_roots))
                if stress:
                    sp = stress.points_epoch[stress_ids]
                    all_collision_roots = sp if all_collision_roots is None else torch.cat((all_collision_roots, sp))
                support, base, anchor, collision_output, surface_orientation, inward_surface_count = surface_losses(
                    field, data, args.surface_batch, args.support_angle, train_generator,
                    surface_indices=surface_ids, base_indices=base_ids,
                    base_count=args.base_batch, collision_roots=all_collision_roots,
                    orientation_axis=orientation_axis,
                    orientation_max_angle_deg=args.orientation_max_angle_deg,
                    compute_support=args.support_weight > 0,
                    compute_base=args.base_weight > 0,
                    compute_anchor=args.anchor_weight > 0)
                if inward_surface_count:
                    orientation = surface_orientation
                topology_loss = (
                    gradient_regularization_scale*(args.variation_weight*variation + args.small_gradient_weight*small) +
                    topology_regularization_scale*args.curvature_weight*curvature +
                    gradient_norm_weight*unit_norm)
                loss = topology_loss + args.support_weight*support + args.base_weight*base + args.anchor_weight*anchor + orientation_weight*orientation
                stress_loss = loss*0
                if stress:
                    stress_loss = stress.loss(collision_output['grads'][-len(stress_ids):], stress_ids)
                    loss = loss+args.stress_weight*stress_loss
                col_loss = loss*0
                replay_loss = loss*0
                online_witness_loss = loss*0
                if epoch_weight > 0:
                    jittered = sample_abb_tool(tool_spec, jitter=True, generator=generator,
                                               volume_samples=args.tool_volume_samples).to(device)
                    col = collision_loss(field, collision_roots, jittered, scale,
                                         data["bbox"], gate, topk_fraction=args.topk, margin=args.margin,
                                         normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                                         prune_zero_penalty=not args.disable_collision_pruning,
                                         collision_gradient_epsilon=args.collision_gradient_epsilon,
                                         root_output={key: value[:len(collision_roots)] for key, value in collision_output.items()},
                                         chunk_size=args.field_query_batch,
                                         sdf_chunk_size=args.sdf_query_batch)
                    col_loss = col["loss"]
                    collision_active_pairs += col['active_pairs']
                    collision_gradient_pairs += col['gradient_pairs']
                    if isinstance(sampler, CoverageRootSampler):
                        sampler.observe(col['root_scores'])
                    loss = loss + epoch_weight*col_loss
                    if online_witnesses is not None:
                        previous = online_witnesses.draw(args.online_witness_batch)
                        if previous is not None:
                            witness_ids, witness_roots, witness_offsets, normalizers = previous
                            retained = tool_witness_loss(field, witness_roots, witness_offsets, normalizers,
                                scale, data['bbox'], gate, margin=args.margin,
                                normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                                collision_gradient_epsilon=args.collision_gradient_epsilon,
                                chunk_size=args.field_query_batch)
                            online_witness_loss = retained['loss']
                            loss = loss+epoch_weight*online_witness_loss
                            online_witnesses.retire(witness_ids, retained['scores'])
                        online_witnesses.observe(collision_roots, jittered, col,
                            topk_fraction=args.topk, margin=args.margin)
                    if replay_roots is not None:
                        replay_loss = replay.loss(field, replay_roots, replay_obstacles,
                            {key: value[len(collision_roots):len(collision_roots)+len(replay_roots)] for key, value in collision_output.items()},
                            softness_mm=args.replay_softness_mm,
                            normalize_collision_by_gradient=args.normalize_collision_by_gradient,
                            collision_gradient_epsilon=args.collision_gradient_epsilon)
                        loss = loss+epoch_weight*args.replay_weight*replay_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
                loss.backward()
                finite_gradients = torch.stack([torch.isfinite(p.grad).all() for p in field.parameters()
                                                if p.grad is not None]).all()
                if not finite_gradients:
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}, step {step}")
                optimizer.step()
                for name, value in dict(total=loss, collision=col_loss, support=support,
                                        stress=stress_loss,
                                        witness_replay=replay_loss,
                                        online_witness=online_witness_loss,
                                        base=base, anchor=anchor, variation=variation, small_gradient=small,
                                        gradient_norm=unit_norm,
                                        curvature=curvature, orientation=orientation,
                                        orientation_surface=surface_orientation,
                                        orientation_inward_surface_samples=surface_orientation*0 + inward_surface_count).items():
                    sums[name] = sums.get(name, 0.)+value.detach()/args.steps_per_epoch
            if scheduler is not None:
                scheduler.step()
            sync(device)
            means={name:float(value) for name,value in sums.items()}
            weights=dict(stress=args.stress_weight if stress else 0.,support=args.support_weight,
                base=args.base_weight,anchor=args.anchor_weight,variation=gradient_regularization_scale*args.variation_weight,
                small_gradient=gradient_regularization_scale*args.small_gradient_weight,
                gradient_norm=gradient_norm_weight,curvature=topology_regularization_scale*args.curvature_weight,
                orientation=orientation_weight,collision=epoch_weight,
                online_witness=epoch_weight if online_witnesses is not None else 0.,
                witness_replay=epoch_weight*args.replay_weight if replay is not None else 0.)
            record(dict(event="train", epoch=epoch, seconds=time.perf_counter()-epoch_start,
                        schema='map.losses.v1',learning_rate_used=learning_rate_used,phase='main',
                        raw={name:means[name] for name in weights},weights=weights,
                        weighted={name:means[name]*weight for name,weight in weights.items()},
                        epoch_wall_seconds=time.perf_counter()-epoch_wall_start,
                        mining_seconds=mining_seconds,
                        mean_collision_active_pairs=collision_active_pairs/args.steps_per_epoch,
                        mean_collision_gradient_pairs=collision_gradient_pairs/args.steps_per_epoch,
                        collision_weight=epoch_weight,
                        orientation_weight=orientation_weight,
                        topology_regularization_scale=topology_regularization_scale,
                        gradient_regularization_scale=gradient_regularization_scale,
                        gradient_norm_weight_effective=gradient_norm_weight,
                        variation_weight_effective=gradient_regularization_scale*args.variation_weight,
                        small_gradient_weight_effective=gradient_regularization_scale*args.small_gradient_weight,
                        curvature_weight_effective=topology_regularization_scale*args.curvature_weight,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        coverage_cycles_completed=(sampler.coverage.cycles-1+int(sampler.coverage.position == len(sampler.points)))
                            if isinstance(sampler, CoverageRootSampler) else None,
                        coverage_position=sampler.coverage.position if isinstance(sampler, CoverageRootSampler) else None,
                        discovered_hard_roots=len(sampler.discovered) if isinstance(sampler, CoverageRootSampler) else 0,
                        online_witness_pairs=len(online_witnesses) if online_witnesses is not None else 0,
                        steps=args.steps_per_epoch,
                        stress_points=len(stress.points_epoch) if stress else 0,
                        **means))
            # Checkpoints are self-describing; optimizer state retained for future continuation.
            save_latest = epoch % args.save_every == 0 or epoch == args.epochs
            save_snapshot = args.snapshot_every > 0 and (epoch % args.snapshot_every == 0 or epoch == args.epochs)
            if save_latest or save_snapshot:
                state = dict(model_state_dict=field.state_dict(), field_spec=field.spec,
                             normalization=metadata, epoch=epoch, optimizer_state_dict=optimizer.state_dict(), config=config)
                if save_latest:
                    torch.save(state, args.output/"latest.pt")
                    if online_witnesses is not None:
                        torch.save(dict(**online_witnesses.state(), normalization=metadata,
                                        input_sha256=config['input_sha256'], epoch=epoch),
                                   args.output/'online_witnesses.pt')
                if save_snapshot:
                    (args.output/'checkpoints').mkdir(exist_ok=True)
                    torch.save(state, args.output/'checkpoints'/f'epoch_{epoch:06d}.pt')
            if epoch % args.validate_every == 0 or epoch == args.epochs:
                validation = evaluate(field, dense_tool, data, scale, gate, args,
                                      orientation_axis=orientation_axis)
                if stress:
                    validation['stress_validation_mean_squared_dot'] = stress.evaluate(field)
                record(dict(event="validation", epoch=epoch, **validation))
    if args.exact_validation:
        import igl
        vertices, faces = igl.read_triangle_mesh(str(args.mesh))
        record(dict(event="exact_validation", **evaluate(field, dense_tool, data, scale, gate, args,
                    exact_mesh=(vertices, faces, np.array(metadata["center_mm"])))))
    if not args.evaluate_only and args.epochs > 0 and gate is not None:
        record(dict(event="final_benchmark", **benchmark(field, tool, data, scale, gate, args)))
    record(dict(event="complete", training_and_final_validation_seconds=time.perf_counter()-start))


if __name__ == "__main__":
    main()

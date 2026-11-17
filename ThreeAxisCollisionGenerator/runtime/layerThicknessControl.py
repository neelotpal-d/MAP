"""Pointwise relative layer-spacing constraints with shared band references.

Normal spacing is locally scale * scalar_step / ||grad f||. Each scalar slab
gets an adaptive log-gradient interval of width log(R), fitted over the entire
available control set. Mini-batches share that interval rather than estimating
their own acceptable range. These remain sampled, first-order constraints.
"""

from __future__ import annotations
import math
import torch


def scalar_band_indices(scalars, band_count, scalar_range=None, shifted=False):
    """Assign detached scalars to a common grid or its half-width shift."""
    if band_count < 1:
        raise ValueError("band_count must be positive.")
    values = scalars.detach().reshape(-1)
    if scalar_range is None:
        scalar_range = torch.stack((values.min(), values.max()))
    lower, upper = scalar_range.detach().to(values).unbind()
    span = upper - lower
    if span <= 0:
        return torch.zeros_like(values, dtype=torch.long)
    position = ((values - lower) / span).clamp(0., 1.) * band_count
    return torch.floor(position + (0.5 if shifted else 0.)).long().clamp_(0, band_count if shifted else band_count - 1)


def _log_squared_norms(gradients):
    q = gradients.square().sum(dim=1)
    return q.clamp_min(torch.finfo(q.dtype).tiny).log()


def fit_interval_center(values, half_width):
    """Minimize ALL pointwise hinge errors over a fixed-width interval.

    Its derivative in the center is monotone. Min/max only bracket bisection;
    they do not define the loss. The detached optimum is shared across batches.
    """
    values = values.detach()
    lower, upper = values.min() - half_width, values.max() + half_width
    for _ in range(40):
        center = (lower + upper) * 0.5
        derivative = (torch.relu(center - half_width - values)
                      - torch.relu(values - center - half_width)).sum()
        lower, upper = torch.where(derivative < 0, center, lower), torch.where(derivative > 0, center, upper)
    return ((lower + upper) * 0.5).detach()


def _fit_band_centers(log_norms, memberships, band_count, half_width):
    centers = log_norms.new_zeros((2, band_count + 1))
    for grid, ids in enumerate(memberships):
        for band in torch.unique(ids):
            centers[grid, band] = fit_interval_center(log_norms[ids == band], half_width)
    return centers


def _interval_loss(log_norms, memberships, centers, half_width):
    losses, log_ranges = [], []
    for grid, ids in enumerate(memberships):
        errors = torch.relu(torch.abs(log_norms - centers[grid, ids]) - half_width).square()
        for band in torch.unique(ids):
            mask = ids == band
            # Match the previous scale for two symmetric outliers.
            losses.append(4. * errors[mask].mean())
            values = log_norms[mask].detach()
            log_ranges.append(values.max() - values.min())
    return torch.stack(losses).mean(), torch.exp(0.5 * torch.stack(log_ranges).max())


def sampled_gradient_ratio_loss(scalars, gradients, band_count, max_height_ratio, scalar_range=None):
    """Standalone pointwise loss for a supplied complete set.

    Training uses BandGradientReference instead of fitting unrelated batches.
    """
    if not math.isfinite(max_height_ratio) or max_height_ratio < 1:
        raise ValueError("max_height_ratio must be finite and at least one.")
    if not gradients.shape[0]:
        raise ValueError("Gradient ratio requires at least one sample.")
    log_norms = _log_squared_norms(gradients)
    memberships = torch.stack([scalar_band_indices(scalars, band_count, scalar_range, shifted)
                               for shifted in (False, True)])
    half_width = math.log(max_height_ratio)  # log(q) width = 2 log(R).
    centers = _fit_band_centers(log_norms, memberships, band_count, half_width)
    return _interval_loss(log_norms, memberships, centers, half_width)


class BandGradientReference:
    """Full-set epoch reference plus shuffled per-band coverage queues."""

    def __init__(self, field, points, band_count, max_height_ratio,
                 probe_batch_size, points_per_band, steps_per_epoch):
        if not math.isfinite(max_height_ratio) or max_height_ratio < 1:
            raise ValueError("max_height_ratio must be finite and at least one.")
        if min(probe_batch_size, points_per_band, steps_per_epoch) < 1 or not len(points):
            raise ValueError("Reference points, batch budgets and step count must be positive.")
        self.points = points.detach()
        self.band_count = band_count
        self.half_width = math.log(max_height_ratio)
        scalars, log_norms = [], []
        # Scan ALL available control points in bounded chunks.
        for start in range(0, len(points), probe_batch_size):
            candidates = self.points[start:start + probe_batch_size].detach().requires_grad_(True)
            with torch.enable_grad():
                values = field(candidates, compute_grads=False)["scalars"]
                gradients = torch.autograd.grad(values, candidates, torch.ones_like(values), create_graph=False)[0]
            scalars.append(values.detach().reshape(-1))
            log_norms.append(_log_squared_norms(gradients.detach()))
        scalars, log_norms = torch.cat(scalars), torch.cat(log_norms)
        self.scalar_range = torch.stack((scalars.min(), scalars.max()))
        self.memberships = torch.stack([
            scalar_band_indices(scalars, band_count, self.scalar_range, shifted)
            for shifted in (False, True)
        ])
        self.centers = _fit_band_centers(log_norms, self.memberships, band_count, self.half_width)
        _, self.probe_max_ratio = _interval_loss(log_norms, self.memberships, self.centers, self.half_width)
        self.seen = torch.zeros(len(points), device=points.device, dtype=torch.bool)
        self.queues, self.cursors, self.quotas = [], [], []
        for band in torch.unique(self.memberships[0]):
            indices = torch.nonzero(self.memberships[0] == band, as_tuple=False).reshape(-1)
            indices = indices[torch.randperm(len(indices), device=indices.device)]
            self.queues.append(indices)
            self.cursors.append(0)
            # Increase quota when necessary to cover every point this epoch.
            self.quotas.append(min(len(indices), max(points_per_band, math.ceil(len(indices) / steps_per_epoch))))

    def sample(self):
        """Visit each band's queue without replacement before reshuffling it."""
        parts = []
        for band, quota in enumerate(self.quotas):
            remaining = quota
            while remaining:
                queue, cursor = self.queues[band], self.cursors[band]
                take = min(remaining, len(queue) - cursor)
                parts.append(queue[cursor:cursor + take])
                remaining -= take
                cursor += take
                if cursor == len(queue):
                    self.queues[band] = queue[torch.randperm(len(queue), device=queue.device)]
                    cursor = 0
                self.cursors[band] = cursor
        indices = torch.cat(parts)
        self.seen[indices] = True
        return self.points[indices].detach().requires_grad_(True), self.memberships[:, indices]

    def loss(self, scalars, gradients, memberships=None):
        if memberships is None:
            memberships = torch.stack([
                scalar_band_indices(scalars, self.band_count, self.scalar_range, shifted)
                for shifted in (False, True)
            ])
        return _interval_loss(_log_squared_norms(gradients), memberships, self.centers, self.half_width)

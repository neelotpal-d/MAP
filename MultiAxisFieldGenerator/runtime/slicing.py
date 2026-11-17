"""Preview iso-surfaces with uniform or gradient-adapted scalar levels."""
import numpy as np
import torch
import pyvista as pv
from fields import first_order_output


def build_layers(volume, field, center, scale, device, *, mode='uniform', count=20,
                 height_mm=1., max_layers=500, gradient_batch=4096):
    values = np.asarray(volume.point_data['field'])
    low, high = float(values.min()), float(values.max())
    if not np.isfinite(values).all() or high <= low:
        raise ValueError('A finite, nonconstant scalar field is required for slicing')
    if min(scale, height_mm, max_layers, gradient_batch) <= 0 or count < 2:
        raise ValueError('Invalid slicing scale, spacing, count or batch size')
    layers, records = pv.MultiBlock(), []
    truncated = False
    if mode == 'uniform':
        levels = np.linspace(np.quantile(values, .02), np.quantile(values, .98), count)
        for level in levels:
            surface = volume.contour([level], scalars='field').triangulate()
            if surface.n_cells:
                layers.append(surface)
                records.append(dict(scalar=float(level)))
    elif mode == 'gradient':
        epsilon = max((high-low)*1e-4, 1e-8)
        level, stop = low+epsilon, high-epsilon
        attempts = 0
        while level <= stop:
            if len(layers) >= max_layers:
                truncated = True
                break
            surface = volume.contour([level], scalars='field').triangulate()
            if not surface.n_cells:
                # Guard disconnected/empty intervals without an endless loop.
                level += max((high-low)*1e-3, epsilon)
                attempts += 1
                if attempts > 10000:
                    raise RuntimeError('Too many empty scalar levels')
                continue
            norms = []
            for points in np.array_split(surface.points, max(1, int(np.ceil(surface.n_points/gradient_batch)))):
                x = torch.tensor((points-center)/scale, dtype=torch.float32, device=device)
                norms.append(first_order_output(field, x)['grads'].norm(dim=1).cpu().numpy()/scale)
            norms = np.concatenate(norms)
            mean = float(norms.mean())
            step = height_mm*mean
            if not np.isfinite(step) or step <= max(1e-12, (high-low)*1e-9):
                raise ValueError('Gradient too small/nonfinite to advance slicing reliably')
            surface.point_data['gradient_norm_per_mm'] = norms
            predicted = step/np.maximum(norms, 1e-15)
            surface.point_data['estimated_next_spacing_mm'] = predicted
            layers.append(surface)
            records.append(dict(scalar=float(level), next_scalar_step=float(step),
                                mean_gradient_per_mm=mean,
                                estimated_next_spacing_mm_quantiles=np.quantile(predicted, [0, .5, 1]).tolist()))
            level += step
    else:
        raise ValueError(f'Unknown slicing mode: {mode}')
    if not len(layers):
        raise ValueError('No layer surfaces could be extracted')
    return layers, dict(mode=mode, target_height_mm=height_mm if mode == 'gradient' else None,
                        layers=len(layers), truncated=truncated, scalar_bounds=[low, high], records=records,
                        method='Mean neural gradient norm at current contour vertices; model gradient divided by normalization scale.',
                        limitation='First-order spacing estimate on linearly interpolated tet contours, not uniform physical spacing or production toolpaths.')

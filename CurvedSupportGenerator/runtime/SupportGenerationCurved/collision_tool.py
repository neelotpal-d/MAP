"""Explicit tool motion and nominal-volume membership for grid collision checks.

Boxes use local (u, axis, u-cross-axis). Five-axis boxes project saved U onto
the gradient's tangent plane; saved V is the deterministic parallel fallback.
Axisymmetric tools have no roll dependence. No rotation is inferred from data.
"""
import numpy as np

MODES = ('fixed_axis', 'field_gradient')
COMPONENTS = ('cone', 'inner_cylinder', 'outer_cylinder')


def mode_from_metadata(meta):
    mode = meta.get('collision_mode', 'fixed_axis')  # archived jobs were fixed
    if mode not in MODES:
        raise ValueError('collision_mode must be fixed_axis or field_gradient.')
    return mode


def validate_axisymmetric(tool):
    if tool.get('units') != 'mm':
        raise ValueError('Axisymmetric nominal tool dimensions must be in mm.')
    for name in COMPONENTS:
        part = tool[name]
        radius = part['radius_max'] if name == 'cone' else part['radius']
        lo, hi = part['z_min'], part['z_max']
        if not np.isfinite([lo, hi, radius]).all() or lo < 0 or hi <= lo or radius <= 0:
            raise ValueError('Invalid nominal tool component: '+name)
    angle = tool['cone']['half_angle_degrees']
    if not np.isfinite(angle) or not 0 < angle < 90:
        raise ValueError('Cone half angle must be between 0 and 90 degrees.')


def bounds(tool):
    if all(name in tool for name in COMPONENTS):
        validate_axisymmetric(tool)
        radius = max(tool[n].get('radius', tool[n].get('radius_max')) for n in COMPONENTS)
        return np.array([-radius, min(tool[n]['z_min'] for n in COMPONENTS), -radius]), np.array([radius, max(tool[n]['z_max'] for n in COMPONENTS), radius])
    lo = np.array([tool['tool_'+a+'_min'] for a in 'xyz'], float)
    hi = np.array([tool['tool_'+a+'_max'] for a in 'xyz'], float)
    if not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError('Tool box bounds must be finite and increasing.')
    return lo, hi


def frames(gradients, tool):
    g = np.asarray(gradients, float)
    lengths = np.linalg.norm(g, axis=1)
    if not np.isfinite(g).all() or np.any(lengths <= 1e-10):
        raise ValueError('Five-axis collision pose has a zero/nonfinite gradient; repair the field before checking clearance.')
    n = g / lengths[:, None]
    u0 = np.asarray(tool.get('in_plane_u', [1., 0., 0.]), float)
    w0 = np.asarray(tool.get('build_direction', [0., 1., 0.]), float)
    u = u0 - (n @ u0)[:, None]*n
    fallback = np.linalg.norm(u, axis=1) < 1e-8
    if fallback.any():
        v0 = np.cross(u0, w0)
        u[fallback] = v0 - (n[fallback] @ v0)[:, None]*n[fallback]
    lengths = np.linalg.norm(u, axis=1)
    if not np.isfinite(u).all() or np.any(lengths < 1e-10):
        raise ValueError('Invalid reference U/build frame for five-axis box.')
    u /= lengths[:, None]
    return np.stack((u, n, np.cross(u, n)), axis=2)


def contains(local, tool):
    """Exact nominal component membership; no enclosing-box substitution."""
    if all(n in tool for n in COMPONENTS):
        axial = local[..., 1]
        radial2 = local[..., 0]**2 + local[..., 2]**2
        hit = np.zeros(axial.shape, bool)
        for name in COMPONENTS:
            part = tool[name]
            radius = part.get('radius', part.get('radius_max'))
            mask = (axial >= part['z_min']) & (axial <= part['z_max']) & (radial2 < radius**2)
            if name == 'cone':
                limit = axial*np.tan(np.deg2rad(part['half_angle_degrees']))
                mask &= (limit > 0) & (radial2 < np.maximum(limit, 0)**2)
            hit |= mask
        return hit
    lo, hi = bounds(tool)
    return np.all(local >= lo-1e-9, axis=-1) & np.all(local <= hi+1e-9, axis=-1)


def pair_membership(points, gradients, root_ids, target_ids, tool):
    """Torch hard membership refreshed in each closure, outside autograd.

Optimization is an active-set scalar-order solve, not a smooth derivative of
tool geometry. Mining discovers new pairs between solves; stale pairs are gated.
"""
    import torch
    with torch.no_grad():
        g = gradients[root_ids]
        length = torch.linalg.vector_norm(g, dim=1)
        if not bool(torch.isfinite(g).all()) or bool((length <= 1e-10).any()):
            raise ValueError('Five-axis collision pose has a zero/nonfinite gradient during optimization.')
        n = g/length[:, None]
        u0 = g.new_tensor(tool.get('in_plane_u', [1., 0., 0.]))
        w0 = g.new_tensor(tool.get('build_direction', [0., 1., 0.]))
        u = u0-(n*u0).sum(1)[:, None]*n
        fallback = torch.linalg.vector_norm(u, dim=1) < 1e-8
        v0 = torch.linalg.cross(u0, w0)
        alternate = v0-(n*v0).sum(1)[:, None]*n
        u = torch.where(fallback[:, None], alternate, u)
        u = u/torch.linalg.vector_norm(u, dim=1)[:, None]
        v = torch.linalg.cross(u, n)
        relative = points[target_ids]-points[root_ids]
        local = torch.stack(((relative*u).sum(1), (relative*n).sum(1), (relative*v).sum(1)), dim=1)
        if all(k in tool for k in COMPONENTS):
            axial = local[:, 1]; radial2 = local[:, 0].square()+local[:, 2].square()
            hit = torch.zeros_like(axial, dtype=torch.bool)
            for name in COMPONENTS:
                part = tool[name]; radius = part.get('radius', part.get('radius_max'))
                mask = (axial >= part['z_min']) & (axial <= part['z_max']) & (radial2 < radius**2)
                if name == 'cone':
                    limit = axial*np.tan(np.deg2rad(part['half_angle_degrees']))
                    mask &= (limit > 0) & (radial2 < limit.clamp_min(0).square())
                hit |= mask
            return hit
        lo, hi = bounds(tool)
        return ((local >= local.new_tensor(lo)-1e-9) & (local <= local.new_tensor(hi)+1e-9)).all(1)

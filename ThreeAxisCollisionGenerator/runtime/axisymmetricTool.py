"""Cone/cylinder geometry in a constant tool frame for three-axis slicing.

Reuse the five-axis sampling recipe, but map its (U,V,H) offsets into the
three-axis runtime's (U,H,V) convention. No field-gradient rotation is used.
"""
import math
from pathlib import Path
import sys


def sample_tool(config, device, seed=0):
    import torch
    # The two generators ship together; script entrypoints initially only have
    # their own runtime directory on sys.path. Resolve from code, never cwd.
    product = str(Path(__file__).resolve().parents[2])
    if product not in sys.path:
        sys.path.insert(0, product)
    from MultiAxisFieldGenerator.runtime.collision import sample_abb_tool
    samples = sample_abb_tool(config, jitter=True,
                              generator=torch.Generator().manual_seed(int(seed)))
    local = samples.offsets_mm[:, [0, 2, 1]].to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(local).all()) or bool((local[:, 1] < 0).any()):
        raise ValueError('Axisymmetric training samples must be finite and above the tip.')
    result = [local[samples.component_ids.to(device) == i] for i in range(3)]
    if any(len(part) == 0 for part in result):
        raise ValueError('Each axisymmetric training component needs samples.')
    return result


def validate_nominal(tool):
    if tool.get('units') != 'mm':
        raise ValueError('Axisymmetric tool dimensions must be in mm.')
    for name in ('cone', 'inner_cylinder', 'outer_cylinder'):
        part = tool[name]
        lo, hi = part['z_min'], part['z_max']
        radius = part['radius_max' if name == 'cone' else 'radius']
        if not all(math.isfinite(x) for x in (lo, hi, radius)) or lo < 0 or hi <= lo or radius <= 0:
            raise ValueError('Invalid nominal tool component: '+name)
    angle = tool['cone']['half_angle_degrees']
    if not math.isfinite(angle) or not 0 < angle < 90:
        raise ValueError('Cone half-angle must be between 0 and 90 degrees.')


def tool_contains(u, v, h, bounds):
    """Exact union membership at candidate points; bounds may be normalized."""
    if 'axisymmetric_tool' not in bounds:
        return ((u >= bounds['x_min']) & (u <= bounds['x_max']) &
                (v >= bounds['z_min']) & (v <= bounds['z_max']) &
                (h >= bounds['y_min']) & (h <= bounds['y_max']))
    tool = bounds['axisymmetric_tool']
    scale = float(bounds.get('scale_mm', 1.))
    radial2 = (u*scale).square()+(v*scale).square()
    axial = h*scale
    hit = axial < -float('inf')
    for name in ('cone', 'inner_cylinder', 'outer_cylinder'):
        part = tool[name]
        radius = part.get('radius', part.get('radius_max'))
        component = (axial >= part['z_min']) & (axial <= part['z_max']) & (radial2 <= radius**2)
        if name == 'cone':
            limit = axial*math.tan(math.radians(part['half_angle_degrees']))
            component &= (limit >= 0) & (radial2 <= limit.square())
        hit |= component
    return hit


def nominal_mesh(tool, origin, frame):
    """Display the actual components, including the cone's radial clamp."""
    import numpy as np
    import pyvista as pv
    validate_nominal(tool)
    u, v, n = [x.detach().cpu().numpy() for x in frame]
    theta = np.arange(96)*2*np.pi/96
    meshes = []
    for name in ('cone', 'inner_cylinder', 'outer_cylinder'):
        part = tool[name]
        radius = part.get('radius', part.get('radius_max'))
        heights = [part['z_min'], part['z_max']]
        if name == 'cone':
            tangent = math.tan(math.radians(part['half_angle_degrees']))
            corner = radius/tangent
            if heights[0] < corner < heights[-1]:
                heights.insert(1, corner)
        rings = []
        for h in heights:
            r = min(radius, h*tangent) if name == 'cone' else radius
            rings.append(np.asarray(origin)+h*n+r*np.cos(theta)[:, None]*u+r*np.sin(theta)[:, None]*v)
        points = np.concatenate(rings)
        faces = []
        for j in range(len(rings)-1):
            for k in range(96):
                nxt = (k+1)%96
                faces.extend([4,j*96+k,j*96+nxt,(j+1)*96+nxt,(j+1)*96+k])
        faces.extend([96, *range(95, -1, -1)])
        faces.extend([96, *range((len(rings)-1)*96, len(rings)*96)])
        meshes.append(pv.PolyData(points, faces).clean().triangulate())
    return pv.merge(meshes).extract_surface(algorithm='dataset_surface')

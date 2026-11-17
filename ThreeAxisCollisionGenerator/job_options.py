"""Resolve portable field-network options before constructing the runtime CLI."""
import math


def resolve_frequencies(values):
    """Resolve auto on the actual training mesh, in its current world-mm pose."""
    value = values['w0']
    if isinstance(value, str) and value.lower() == 'auto':
        import igl
        import numpy as np
        vertices, _ = igl.read_triangle_mesh(values['mesh_path'])
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Automatic frequency requires finite mesh vertices')
        value = float(np.ptp(vertices, axis=0).max()/20.)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('w0 must be positive or auto')
    values['w0'] = value
    first = values.get('w0_initial')
    if isinstance(first, str) and first.lower() == 'auto':
        first = value
    if first is not None:
        first = float(first)
        if not math.isfinite(first) or first <= 0:
            raise ValueError('w0_initial must be positive or auto')
        values['w0_initial'] = first
    return values

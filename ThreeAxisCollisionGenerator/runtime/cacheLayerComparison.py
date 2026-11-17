"""Cache gradient-height display layers using one shared preclipped volume."""
import json
import numpy as np
import pyvista as pv
from scalarNetUtils import load_scalar_net_from_checkpoint
from exportIsoSurfaces import (compute_normalization, evaluate_scalar_values,
    evaluate_average_grad_norm, build_padded_voxel_grid)


def cache_layers(mesh, mesh_path, checkpoints, titles, output, note, layer_height=2.):
    spacing = 1.
    extent = np.ptp(mesh.points, axis=0)
    while np.prod(np.ceil(extent/spacing)+4) > 800000:
        spacing *= 1.15
    print('Preparing shared clipped volume at spacing', spacing, flush=True)
    volume = build_padded_voxel_grid(mesh, spacing, spacing).clip_surface(mesh, invert=True)
    scene = pv.MultiBlock()
    center, scale = compute_normalization(mesh.points, 'cuda')
    for checkpoint, title in zip(checkpoints, titles):
        # New checkpoints carry the exact SIREN specification; the arguments
        # here supply the historical fallback for older frequency15 fields.
        field = load_scalar_net_from_checkpoint(str(checkpoint), 'cuda', num_layers=5, w0=15., w0_initial=15.)
        field.requires_grad_(False)
        volume['scalars'] = evaluate_scalar_values(field, volume.points, center, scale, 'cuda', 16384)
        boundary = evaluate_scalar_values(field, mesh.points, center, scale, 'cuda', 16384)
        current, stop = float(boundary.min()+1e-5), float(boundary.max()-1e-5)
        layers = pv.MultiBlock()
        attempts = 0
        while current < stop and len(layers) < 400 and attempts < 3200:
            surface = volume.contour([current], scalars='scalars')
            if surface.n_points:
                norm = evaluate_average_grad_norm(field, surface.points, center, scale, 'cuda', 8192)
                surface['field_value'] = np.full(surface.n_points, current)
                layers.append(surface)
                step = norm*layer_height/float(scale[0])
            else:
                step = layer_height/float(scale[0])
            current += max(step, float(np.ptp(boundary))*1e-5, 1e-7)
            attempts += 1
        scene.append(layers, name=f'field_{len(scene)}')
        print('Cached', len(layers), 'gradient-height layers:', title, flush=True)
    scene.save(str(output/'layers.vtm'))
    (output/'scene.json').write_text(json.dumps(dict(mesh=str(mesh_path), titles=titles,
        checkpoints=[str(p) for p in checkpoints], layers=str(output/'layers.vtm'),
        layer_height_mm=layer_height, grid_spacing_mm=spacing, note=note), indent=2))

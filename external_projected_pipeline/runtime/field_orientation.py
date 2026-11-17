"""Layer orientation from retained slicing gradients, with geometry identity checks."""
from pathlib import Path
import hashlib
import numpy as np


def load_gradients(mesh_path, triangle_count):
    mesh_path = Path(mesh_path)
    path = mesh_path.with_suffix('.field_gradients.npz')
    if not path.is_file():
        raise ValueError(f'{mesh_path.name}: no saved slicing-field gradients. Re-export the generated layers, or explicitly choose Mesh winding / Reverse winding for imported geometry.')
    with np.load(path, allow_pickle=False) as data:
        digest = str(data['mesh_sha256'].item())
        gradients = np.asarray(data['gradients'], dtype=float)
        source = str(data['source'].item())
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != digest:
        raise ValueError(f'{mesh_path.name}: field-gradient sidecar does not match the mesh. Re-export the layers.')
    if gradients.shape != (triangle_count, 3) or not np.isfinite(gradients).all():
        raise ValueError(f'{mesh_path.name}: invalid per-triangle slicing-field gradients.')
    return gradients, source


def save_gradients(mesh_path, gradients, source):
    mesh_path = Path(mesh_path)
    np.savez_compressed(mesh_path.with_suffix('.field_gradients.npz'), gradients=np.asarray(gradients, dtype=float),
        mesh_sha256=np.asarray(hashlib.sha256(mesh_path.read_bytes()).hexdigest()), source=np.asarray(source))


def orient_from_gradients(vertices, faces, gradients):
    gradients = np.asarray(gradients, dtype=float)
    if gradients.shape != (len(faces), 3) or not np.isfinite(gradients).all():
        raise ValueError('Invalid per-triangle slicing-field gradients.')
    vectors = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]])
    lengths = np.linalg.norm(gradients, axis=1)
    area = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(lengths).all() or not np.isfinite(area).all() or np.any(lengths <= 1e-15) or np.any(area <= 1e-14):
        raise ValueError('Zero or degenerate slicing-gradient / face normal; explicit geometry repair is required.')
    alignment = np.einsum('ij,ij->i', vectors/area[:, None], gradients/lengths[:, None])
    if np.any(np.abs(alignment) < 1e-6):
        raise ValueError('Slicing gradients are tangential to some layer faces; cannot determine their normal sign.')
    flip = alignment < 0
    result = faces.copy()
    result[flip] = result[flip][:, [0, 2, 1]]
    edges = np.concatenate((result[:, [0, 1]], result[:, [1, 2]], result[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    directions = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1., -1.))
    if np.any(counts > 2) or np.any((counts == 2) & (directions != 0)):
        raise ValueError('Field-guided normals are inconsistent across shared edges. Check mesh topology and field data.')
    return vertices, result, dict(mode='field_gradient', flipped_faces=int(flip.sum()),
        minimum_absolute_gradient_alignment=float(np.abs(alignment).min()),
        normal_definition='Layer-surface normals oriented toward increasing saved slicing field; gradients resolve sign. No preceding-layer correspondence.')

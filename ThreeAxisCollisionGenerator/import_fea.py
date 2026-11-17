"""Export a tetrahedral FEA VTU and FPS tensile directions for orientation.

Run ``python -m ThreeAxisCollisionGenerator.import_fea --help``.
By default select the largest absolute principal stress, preserve its sign,
then retain only tensile sources for FPS. The old algebraic rule is explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def farthest_point_indices(points, count, seed=42):
    """Deterministic seeded FPS in world coordinates, with unique source IDs."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('FPS requires finite Nx3 points')
    if not isinstance(count, (int, np.integer)) or count < 1 or not len(points):
        raise ValueError('FPS needs a positive count and nonempty points')
    count = min(count, len(points))
    nearest = np.full(len(points), np.inf)
    selected = np.empty(count, dtype=np.int64)
    used = np.zeros(len(points), dtype=bool)
    current = int(np.random.default_rng(seed).integers(len(points)))
    for i in range(count):
        selected[i] = current
        used[current] = True
        delta = points-points[current]
        np.minimum(nearest, np.einsum('ij,ij->i', delta, delta), out=nearest)
        current = int(np.argmax(np.where(used, -1., nearest)))
    return selected, float(np.sqrt(nearest.max()))


def principal_eigenpair(tensors, selection='dominant-absolute'):
    tensors = np.asarray(tensors, dtype=np.float64)
    if tensors.ndim == 2 and tensors.shape[1] == 9:
        tensors = tensors.reshape(-1, 3, 3)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3) or not np.isfinite(tensors).all():
        raise ValueError('Need finite Nx9 or Nx3x3 stress tensors')
    if not np.allclose(tensors, tensors.transpose(0, 2, 1), rtol=1e-6, atol=1e-12):
        raise ValueError('Stress tensors must be symmetric')
    values, vectors = np.linalg.eigh((tensors+tensors.transpose(0, 2, 1))/2)
    if selection == 'dominant-absolute':
        # Match IntegratedCode: prefer the larger algebraic value in an exact tie.
        indices = 2-np.argmax(np.abs(values[:, ::-1]), axis=1)
    elif selection == 'largest-algebraic':
        indices = np.full(len(values), 2, dtype=int)
    else:
        raise ValueError(f'Unknown principal selection: {selection}')
    rows = np.arange(len(values))
    directions = vectors[rows, :, indices].copy()
    # Stable sign for saved data; alignment objectives themselves are sign invariant.
    dominant = directions[np.arange(len(directions)), np.abs(directions).argmax(1)]
    directions *= np.where(dominant < 0, -1., 1.)[:, None]
    return values[rows, indices], directions


def maximum_principal(tensors):
    """Legacy largest-algebraic eigenpair; kept for auditing historical runs."""
    return principal_eigenpair(tensors, 'largest-algebraic')


def import_vtu(source, output, fps_count=4096, seed=42, min_tensile_stress=0., tensor_name='stress_tensor',
               principal_selection='dominant-absolute'):
    import pyvista as pv
    import igl
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Import output exists; choose a fresh directory: {output}')
    if not np.isfinite(min_tensile_stress) or min_tensile_stress < 0:
        raise ValueError('Tensile threshold must be finite and nonnegative')
    grid = pv.read(source)
    if not isinstance(grid, pv.UnstructuredGrid) or set(np.unique(grid.celltypes)) != {int(pv.CellType.TETRA)}:
        raise ValueError('Expected a pure linear tetrahedral VTU')
    vertices = np.asarray(grid.points, dtype=np.float64)
    ids = grid.cells_dict[int(pv.CellType.TETRA)]
    if not np.isfinite(vertices).all() or tensor_name not in grid.cell_data:
        raise ValueError('Need finite mesh points and the requested cell stress tensor')
    tensors = np.asarray(grid.cell_data[tensor_name])
    if len(tensors) != len(ids):
        raise ValueError('Expected one stress tensor per tetrahedron')
    sigma, directions = principal_eigenpair(tensors, principal_selection)
    tetra = vertices[ids]
    volumes = np.abs(np.linalg.det(tetra[:, 1:]-tetra[:, :1]))/6
    if np.any(volumes <= 1e-12):
        raise ValueError('Degenerate tetrahedra require mesh repair before export')
    centers = tetra.mean(1)
    tensile_ids = np.flatnonzero(sigma > min_tensile_stress)
    if not len(tensile_ids):
        raise ValueError('No selected principal stresses above the tensile threshold')
    local_ids, coverage_radius = farthest_point_indices(centers[tensile_ids], fps_count, seed)
    selected_ids = tensile_ids[local_ids]
    surface = grid.extract_surface(algorithm='dataset_surface').triangulate().clean()
    if surface.n_open_edges or not surface.is_manifold:
        raise ValueError('VTU boundary must be a closed manifold surface')
    surface = surface.compute_normals(auto_orient_normals=True, consistent_normals=True, split_vertices=False)
    faces = surface.faces.reshape(-1, 4)[:, 1:]
    output.mkdir(parents=True)
    igl.write_triangle_mesh(str(output/'mesh.obj'), np.asarray(surface.points), faces)
    # Preserve all volume vertices and linear tetrahedra, using explicit TetGen IDs.
    node_rows = np.column_stack((np.arange(1, len(vertices)+1), vertices))
    ele_rows = np.column_stack((np.arange(1, len(ids)+1), ids+1))
    np.savetxt(output/'mesh.node', node_rows, fmt=['%d', '%.17g', '%.17g', '%.17g'],
               header=f'{len(vertices)} 3 0 0', comments='')
    np.savetxt(output/'mesh.ele', ele_rows, fmt='%d', header=f'{len(ids)} 4 0', comments='')
    rows = np.column_stack((centers, directions, sigma))
    header = 'x y z principal_dir_x principal_dir_y principal_dir_z signed_selected_principal_stress'
    # Keep non-tensile sources as zero-weight neighbors in downstream interpolation.
    np.savetxt(output/'stress_all_cells.txt', rows, fmt='%.17g', header=header)
    np.savetxt(output/'stress_tensile_fps.txt', rows[selected_ids], fmt='%.17g', header=header)
    np.savez_compressed(output/'fps_selection.npz', source_cell_indices=selected_ids,
                        source_cell_centers_mm=centers[selected_ids])
    report = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        stress_association='cell', tensor_name=tensor_name,
        principal_selection_rule=principal_selection,
        principal_selection=('largest absolute eigenvalue; preserve sign; retain positive stresses for FPS'
                             if principal_selection == 'dominant-absolute' else
                             'largest algebraic eigenvalue; retain positive stresses for FPS'),
        absolute_tie_policy='prefer larger algebraic value',
        geometry='complete VTU tetrahedral solid boundary; no contour of stored scalar fields',
        units='input coordinates assumed millimetres; stress units preserved',
        volume_vertices=len(vertices), tetrahedra=len(ids), surface_vertices=surface.n_points,
        surface_faces=len(faces), tensile_cells=len(tensile_ids), min_tensile_stress=min_tensile_stress,
        fps_requested=int(fps_count), fps_selected=len(selected_ids), fps_seed=seed,
        fps_max_distance_to_tensile_cell_center_mm=coverage_radius,
        minimum_cell_volume_mm3=float(volumes.min()),
        files=dict(mesh='mesh.obj', node='mesh.node', ele='mesh.ele',
                   orientation_stress='stress_tensile_fps.txt', strength_stress='stress_all_cells.txt'))
    (output/'import_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps-count', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-tensile-stress', type=float, default=0.)
    parser.add_argument('--tensor-name', default='stress_tensor')
    parser.add_argument('--principal-selection', choices=['dominant-absolute', 'largest-algebraic'],
                        default='dominant-absolute',
                        help='Select an eigenpair before applying the tensile sign filter')
    args = parser.parse_args(argv)
    report = import_vtu(args.source, args.output, args.fps_count, args.seed,
                        args.min_tensile_stress, args.tensor_name, args.principal_selection)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

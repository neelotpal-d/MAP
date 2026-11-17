"""Optional audited isotropic remeshing of saved model-layer surfaces."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import igl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def topology(vertices, faces):
    if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Remeshing requires a finite, nonempty triangle mesh.')
    if faces.ndim != 2 or faces.shape[1] != 3 or np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError('Invalid triangle indices.')
    lengths = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                                     vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError('Degenerate input/output triangles; repair the extraction before remeshing.')
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Non-manifold edge; repair the extraction before remeshing.')
    used = np.unique(faces)
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
    components = connected_components(graph[used][:, used], directed=False, return_labels=False)
    boundary = edges[counts == 1]
    loops = 0
    if len(boundary):
        ids, degree = np.unique(boundary, return_counts=True)
        if np.any(degree != 2):
            raise ValueError('Boundary is not a set of closed manifold loops.')
        bg = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=graph.shape).tocsr()
        loops = connected_components(bg[ids][:, ids], directed=False, return_labels=False)
    return dict(components=int(components), boundary_loops=int(loops),
                euler_characteristic=int(len(used)-len(edges)+len(faces))), boundary


def surface_distance(points, vertices, faces):
    tree = igl.AABB()
    tree.init(vertices, faces)
    squared, ids, _ = tree.squared_distance(vertices, faces, points)
    return np.sqrt(np.maximum(np.asarray(squared).reshape(-1), 0)), np.asarray(ids).reshape(-1)


def boundary_within(points, vertices, edges, tolerance):
    """Exact point-to-segment threshold query; spatial index only prunes candidates."""
    if not len(points): return 0.
    if not len(edges): raise ValueError('Boundary disappeared during remeshing.')
    a = vertices[edges[:, 0]]
    vectors = vertices[edges[:, 1]]-a
    lengths2 = np.einsum('ij,ij->i', vectors, vectors)
    tree = cKDTree(a+vectors*.5)
    radius = float(np.sqrt(lengths2).max())*.5+tolerance+1e-10
    maximum = 0.
    for begin in range(0, len(points), 256):
        for point, candidates in zip(points[begin:begin+256], tree.query_ball_point(points[begin:begin+256], radius)):
            if not candidates:
                raise ValueError('Remeshed boundary exceeds the allowed displacement.')
            ids = np.asarray(candidates)
            t = np.clip(np.einsum('ij,ij->i', point-a[ids], vectors[ids])/lengths2[ids], 0., 1.)
            distance = float(np.linalg.norm(point-a[ids]-t[:, None]*vectors[ids], axis=1).min())
            if distance > tolerance+1e-10:
                raise ValueError('Remeshed boundary exceeds the allowed displacement.')
            maximum = max(maximum, distance)
    return maximum


def audit(original_v, original_f, remeshed_v, remeshed_f, tolerance):
    before, old_edges = topology(original_v, original_f)
    after, new_edges = topology(remeshed_v, remeshed_f)
    if before != after:
        raise ValueError(f'Remeshing changed component / boundary / topology counts: {before} -> {after}')
    old_samples = np.vstack((original_v, original_v[original_f].mean(axis=1)))
    new_samples = np.vstack((remeshed_v, remeshed_v[remeshed_f].mean(axis=1)))
    forward, nearest = surface_distance(new_samples, original_v, original_f)
    backward, _ = surface_distance(old_samples, remeshed_v, remeshed_f)
    maximum = float(max(forward.max(), backward.max()))
    if maximum > tolerance+1e-10:
        raise ValueError(f'Remeshed surface deviation {maximum:.6g} mm exceeds {tolerance:.6g} mm.')
    boundary_max = 0.
    for v, edges, target_v, target_edges in ((original_v, old_edges, remeshed_v, new_edges),
                                             (remeshed_v, new_edges, original_v, old_edges)):
        if len(edges):
            samples = np.vstack((v[np.unique(edges)], v[edges].mean(axis=1)))
            boundary_max = max(boundary_max, boundary_within(samples, target_v, target_edges, tolerance))
    old_normal = np.cross(original_v[original_f[:, 1]]-original_v[original_f[:, 0]], original_v[original_f[:, 2]]-original_v[original_f[:, 0]])
    new_normal = np.cross(remeshed_v[remeshed_f[:, 1]]-remeshed_v[remeshed_f[:, 0]], remeshed_v[remeshed_f[:, 2]]-remeshed_v[remeshed_f[:, 0]])
    dots = np.einsum('ij,ij->i', new_normal, old_normal[nearest[len(remeshed_v):]])
    if np.any(dots <= 0):
        raise ValueError('Remeshing reversed normals relative to nearest original triangles.')
    return dict(topology=after, maximum_sampled_surface_deviation_mm=maximum,
                maximum_sampled_boundary_deviation_mm=boundary_max,
                sampling='Both directions: all vertices and triangle centroids; boundary vertices and edge midpoints. Not a continuous Hausdorff or neural-isovalue proof.')


def remesh_layer(job):
    from mesh_distance_workflow import load_obj_mesh
    from batch_meshlab_isotropic_remesh import remesh_file
    started = time.perf_counter()
    source = Path(job['source']); output = Path(job['output']); settings = job['settings']
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    vertices, faces = load_obj_mesh(str(source))
    topology(vertices, faces)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    tolerance = float(settings['maximum_deviation_mm'])
    record = dict(source=str(source), output=str(output), source_sha256=original_hash,
                  input_vertices=len(vertices), input_triangles=len(faces), settings=settings, passed=False)
    try:
        remesh_file(source, output, iterations=settings['iterations'], adaptive=False, selectedonly=False,
            targetlen_pct=.5, targetlen_abs=settings['target_edge_mm'], featuredeg=settings['feature_angle_deg'],
            checksurfdist=True, max_surfdist_pct=100*tolerance/diagonal, splitflag=True, collapseflag=True,
            swapflag=True, smoothflag=True, reprojectflag=True, save_vertex_normal=False)
        v, f = load_obj_mesh(str(output))
        record.update(audit(vertices, faces, v, f, tolerance))
        if source.with_suffix('.field_gradients.npz').is_file():
            from field_orientation import load_gradients, save_gradients
            from field_orientation import orient_from_gradients
            gradients, origin = load_gradients(source, len(faces))
            _, nearest = surface_distance(v[f].mean(axis=1), vertices, faces)
            transferred = gradients[nearest]
            orient_from_gradients(v, f, transferred)  # Validate coherence before publishing the sidecar.
            save_gradients(output, transferred, origin+'; transferred from nearest original triangle on the same audited layer')
            record['field_gradient_transfer'] = 'Nearest triangle on same original layer; no previous-layer lookup or neural re-evaluation.'
        if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
            raise RuntimeError('Source layer changed during remeshing.')
        record.update(output_vertices=len(v), output_triangles=len(f), passed=True)
    except Exception as error:
        record['error'] = str(error)
        raise
    finally:
        record['seconds'] = time.perf_counter()-started
        output.with_suffix('.remesh.json').write_text(json.dumps(record, indent=2), encoding='utf8')
    return record


def prepare_layers(meshes, folder, config, parallel_map, workers):
    settings = dict(target_edge_mm=float(config.get('target_edge_mm', .5)),
                    iterations=int(config.get('iterations', 20)),
                    maximum_deviation_mm=float(config.get('maximum_deviation_mm', .02)),
                    feature_angle_deg=float(config.get('feature_angle_deg', 30.)))
    if not np.isfinite(list(settings.values())).all() or settings['target_edge_mm'] <= 0 or settings['maximum_deviation_mm'] <= 0 or settings['iterations'] < 1 or not 0 <= settings['feature_angle_deg'] <= 180:
        raise ValueError('Invalid surface remeshing settings.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    jobs = [dict(source=str(mesh), output=str(folder/mesh.name), settings=settings) for mesh in meshes]
    started = time.perf_counter()
    records = parallel_map(remesh_layer, jobs, workers)
    report = dict(enabled=True, seconds=time.perf_counter()-started, settings=settings, layers=records,
                  original_layers_retained=True, height_reference_geometry='Same audited remeshed layers used for planning.')
    (folder/'remesh_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    return [folder/mesh.name for mesh in meshes], report

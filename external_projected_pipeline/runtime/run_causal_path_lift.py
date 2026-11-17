"""Create a lifted copy of processed paths using earlier nominal bead geometry."""
from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree
from shapely import points as shapely_points
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from extrusion_height import apply_axis_lift
from mesh_distance_workflow import load_obj_mesh


XZ = np.asarray((0, 2), dtype=np.int64)


@dataclass(slots=True)
class SurfaceEnvelope:
    """Upper Y envelope of one mesh set in the nozzle-perpendicular XZ plane."""

    vertices: np.ndarray
    faces: np.ndarray
    face_indices: np.ndarray
    projected_triangles: np.ndarray
    tree: STRtree

    @classmethod
    def build(
        cls,
        mesh_paths: tuple[Path, ...],
        *,
        model_lift: float,
        model_lift_axis: str,
    ) -> "SurfaceEnvelope":
        vertices_parts: list[np.ndarray] = []
        faces_parts: list[np.ndarray] = []
        vertex_offset = 0
        for mesh_path in mesh_paths:
            vertices, faces = load_obj_mesh(str(mesh_path))
            vertices = np.asarray(vertices, dtype=np.float64)
            if abs(model_lift) > 0.0:
                vertices = apply_axis_lift(vertices, axis=model_lift_axis, amount=model_lift)
            vertices_parts.append(vertices)
            faces_parts.append(np.asarray(faces, dtype=np.int64) + vertex_offset)
            vertex_offset += len(vertices)
        if not vertices_parts:
            raise ValueError("A surface-clearance check requires at least one mesh.")
        vertices = np.vstack(vertices_parts)
        faces = np.vstack(faces_parts)
        projected = vertices[:, XZ]
        all_triangles = projected[faces]
        twice_areas = np.cross(
            all_triangles[:, 1] - all_triangles[:, 0],
            all_triangles[:, 2] - all_triangles[:, 0],
        )
        valid = np.abs(twice_areas) > 1e-12
        if not np.any(valid):
            raise ValueError("All prior mesh triangles are degenerate in the XZ projection.")
        face_indices = np.flatnonzero(valid)
        projected_triangles = all_triangles[valid]
        polygons = np.asarray([Polygon(triangle) for triangle in projected_triangles], dtype=object)
        return cls(vertices, faces, face_indices, projected_triangles, STRtree(polygons))

    def upper_y(self, points_xz: np.ndarray) -> np.ndarray:
        """Return the highest triangle-interpolated Y at each projected query point."""
        points = np.asarray(points_xz, dtype=np.float64)
        upper = np.full(len(points), np.nan, dtype=np.float64)
        if not len(points):
            return upper
        pairs = np.asarray(self.tree.query(shapely_points(points[:, 0], points[:, 1])), dtype=np.int64)
        if not pairs.size:
            return upper
        point_indices, local_indices = pairs
        triangles = self.projected_triangles[local_indices]
        edge_u = triangles[:, 1] - triangles[:, 0]
        edge_v = triangles[:, 2] - triangles[:, 0]
        offsets = points[point_indices] - triangles[:, 0]
        determinants = edge_u[:, 0] * edge_v[:, 1] - edge_u[:, 1] * edge_v[:, 0]
        nondegenerate = np.abs(determinants) > 1e-16
        beta = np.zeros(len(point_indices), dtype=np.float64)
        gamma = np.zeros(len(point_indices), dtype=np.float64)
        beta[nondegenerate] = (
            offsets[nondegenerate, 0] * edge_v[nondegenerate, 1]
            - offsets[nondegenerate, 1] * edge_v[nondegenerate, 0]
        ) / determinants[nondegenerate]
        gamma[nondegenerate] = (
            edge_u[nondegenerate, 0] * offsets[nondegenerate, 1]
            - edge_u[nondegenerate, 1] * offsets[nondegenerate, 0]
        ) / determinants[nondegenerate]
        barycentric = np.column_stack((1.0 - beta - gamma, beta, gamma))
        valid = nondegenerate & (np.min(barycentric, axis=1) >= -1e-7)
        if not np.any(valid):
            return upper
        valid_points = point_indices[valid]
        valid_faces = self.faces[self.face_indices[local_indices[valid]]]
        y_values = np.einsum("ij,ij->i", barycentric[valid], self.vertices[valid_faces, 1])
        upper[valid_points] = -np.inf
        np.maximum.at(upper, valid_points, y_values)
        upper[~np.isfinite(upper)] = np.nan
        return upper


def disk_offsets(radius: float, *, directions: int, rings: int) -> np.ndarray:
    """Center plus evenly distributed samples over a horizontal nozzle-tip disk."""
    offsets = [np.zeros(2, dtype=np.float64)]
    angles = np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False)
    for ring in range(1, rings + 1):
        ring_radius = radius * ring / rings
        offsets.extend(np.column_stack((ring_radius * np.cos(angles), ring_radius * np.sin(angles))))
    return np.asarray(offsets, dtype=np.float64)


def surface_requirements(
    samples: np.ndarray,
    envelope: SurfaceEnvelope,
    offsets: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute lift needed to keep every sampled tip disk above a nominal mesh."""
    required = np.zeros(len(samples), dtype=np.float64)
    gaps = np.full(len(samples), np.nan, dtype=np.float64)
    uncovered = 0
    for start in range(0, len(samples), 2048):
        end = min(start + 2048, len(samples))
        projected = samples[start:end, None, :][:, :, XZ] + offsets[None, :, :]
        upper = envelope.upper_y(projected.reshape(-1, 2)).reshape(end - start, len(offsets))
        has_surface = np.any(np.isfinite(upper), axis=1)
        # A finite disk touches the highest previous-surface point under any disk sample.
        highest = np.max(np.where(np.isfinite(upper), upper, -np.inf), axis=1)
        local_gaps = samples[start:end, 1] - highest
        local_gaps[~has_surface] = np.nan
        gaps[start:end] = local_gaps
        required[start:end] = np.where(
            has_surface & (local_gaps < -tolerance),
            -local_gaps,
            0.0,
        )
        uncovered += int(np.count_nonzero(~has_surface))
    return required, gaps, uncovered


def edge_samples(points: np.ndarray, breaks: set[int], spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample printable path edges and retain their source edge/fraction indices."""
    if spacing <= 0.0:
        raise ValueError("Edge sample spacing must be positive.")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    edge_ids = np.flatnonzero(~np.isin(np.arange(len(lengths)) + 1, list(breaks)))
    sample_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    fraction_parts: list[np.ndarray] = []
    for count in np.unique(counts[edge_ids]):
        group = edge_ids[counts[edge_ids] == count]
        fractions = np.linspace(0.0, 1.0, count + 1)
        sample_parts.append(
            (points[group, None, :] + fractions[None, :, None] * (points[group + 1] - points[group])[:, None, :]).reshape(-1, 3)
        )
        edge_parts.append(np.repeat(group, len(fractions)))
        fraction_parts.append(np.tile(fractions, len(group)))
    if not sample_parts:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    return np.concatenate(sample_parts), np.concatenate(edge_parts), np.concatenate(fraction_parts)


def causal_heights(points, widths, breaks, samples, edges, fractions, radius):
    """Highest nominal bead top intersecting a disk, excluding future material.

    A bead is a capsule with linear centreline height. Its radius is the maximum
    of its endpoint half widths. Disk/capsule overlap is solved analytically.
    """
    ids = np.arange(len(points) - 1)
    valid = ~np.isin(ids + 1, list(breaks))
    valid &= np.maximum(widths[:-1], widths[1:]) > 0
    ids = ids[valid]
    a, b = points[ids], points[ids + 1]
    delta = b[:, [0, 2]] - a[:, [0, 2]]
    length = np.linalg.norm(delta, axis=1)
    bead_radius = .5 * np.maximum(widths[ids], widths[ids + 1])
    short = np.flatnonzero(length <= 1.)
    long = np.flatnonzero(length > 1.)
    tree = cKDTree(.5 * (a[short][:, [0, 2]] + b[short][:, [0, 2]])) if len(short) else None
    search_radius = radius + float(bead_radius.max(initial=0)) + .5
    result = np.full(len(samples), -np.inf)
    for start in range(0, len(samples), 2048):
        end = min(start + 2048, len(samples))
        q = samples[start:end, :][:, [0, 2]]
        neighbors = tree.query_ball_point(q, search_radius) if tree is not None else [[] for _ in q]
        counts = np.array([len(n) + len(long) for n in neighbors])
        query_ids = np.repeat(np.arange(len(q)), counts)
        if not len(query_ids):
            continue
        bead_ids = np.concatenate([np.r_[short[np.asarray(n, dtype=int)], long] for n in neighbors])
        current_edges = edges[start:end][query_ids]
        earlier = ids[bead_ids] <= current_edges
        query_ids, bead_ids = query_ids[earlier], bead_ids[earlier]
        if not len(query_ids):
            continue
        d = delta[bead_ids]
        relative = q[query_ids] - a[bead_ids][:, [0, 2]]
        length2 = np.sum(d*d, axis=1)
        dot = np.sum(relative*d, axis=1)
        rr = radius + bead_radius[bead_ids]
        discriminant = dot*dot - length2 * (np.sum(relative*relative, axis=1) - rr*rr)
        nonzero = length2 > 1e-20
        root = np.sqrt(np.maximum(discriminant, 0))
        low = np.where(nonzero, (dot-root)/np.maximum(length2, 1e-20), 0.)
        high = np.where(nonzero, (dot+root)/np.maximum(length2, 1e-20), 1.)
        same_edge = ids[bead_ids] == edges[start:end][query_ids]
        # At fraction zero, none of this edge has been deposited yet.
        high = np.minimum(high, np.where(same_edge, fractions[start:end][query_ids], 1.))
        low = np.maximum(low, 0.)
        high = np.minimum(high, 1.)
        overlaps = (discriminant >= 0) & (high >= low)
        overlaps &= ~same_edge | (fractions[start:end][query_ids] > 0)
        overlaps &= nonzero | (np.sum(relative*relative, axis=1) <= rr*rr)
        dy = b[bead_ids, 1] - a[bead_ids, 1]
        t = np.where(dy >= 0, high, low)
        tops = a[bead_ids, 1] + t*dy
        tops[~overlaps] = -np.inf
        np.maximum.at(result[start:end], query_ids, tops)
    return result


def process_layer(
    input_dir,
    output_dir,
    layer,
    diameter,
    step,
    tolerance,
    max_lift,
    jump_threshold,
    immediate_surface_meshes=(),
    model_lift=0.0,
    model_lift_axis="y",
    immediate_surface_sample_spacing=0.25,
    immediate_surface_directions=16,
    immediate_surface_rings=1,
    current_surface_meshes=(),
    current_surface_sample_spacing=0.25,
    current_surface_directions=16,
    current_surface_rings=1,
):
    started = time.perf_counter()
    name = layer['output_path_file']
    rows = np.atleast_2d(np.loadtxt(input_dir / name))
    points = rows[:, :3]
    widths = np.atleast_1d(np.loadtxt(input_dir / layer['width_profile_file']))
    if len(widths) != len(rows) or rows.shape[1] != 7:
        raise ValueError(f'Invalid rows or width count: {name}')
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    break_name = layer.get('path_break_file')
    breaks = set(np.atleast_1d(np.loadtxt(input_dir / break_name, dtype=int))) if break_name else set()
    if jump_threshold is not None:
        breaks.update(np.flatnonzero(lengths > jump_threshold) + 1)
    samples, edges, fractions = edge_samples(points, breaks, step)
    lifts = np.zeros(len(rows))
    nominal = np.empty(0)
    residual = np.empty(0)
    if len(samples):
        tops = causal_heights(points, widths, breaks, samples, edges, fractions, diameter/2)
        nominal = samples[:, 1]-tops
        required = np.where(nominal < -tolerance, -nominal, 0.)
        np.maximum.at(lifts, edges, required)
        np.maximum.at(lifts, edges+1, required)

    immediate_surface_stats = None
    surface_samples = surface_edges = surface_fractions = surface_required = surface_gaps = None
    if immediate_surface_meshes:
        if immediate_surface_directions < 3 or immediate_surface_rings < 1:
            raise ValueError("Immediate-surface tip sampling requires at least 3 directions and 1 ring.")
        surface_samples, surface_edges, surface_fractions = edge_samples(
            points,
            breaks,
            float(immediate_surface_sample_spacing),
        )
        envelope = SurfaceEnvelope.build(
            tuple(Path(path) for path in immediate_surface_meshes),
            model_lift=float(model_lift),
            model_lift_axis=str(model_lift_axis),
        )
        surface_required, surface_gaps, uncovered = surface_requirements(
            surface_samples,
            envelope,
            disk_offsets(diameter / 2.0, directions=int(immediate_surface_directions), rings=int(immediate_surface_rings)),
            tolerance=tolerance,
        )
        previous_lifts = lifts.copy()
        np.maximum.at(lifts, surface_edges, surface_required)
        np.maximum.at(lifts, surface_edges + 1, surface_required)
        immediate_surface_stats = dict(
            reference_meshes=[Path(path).name for path in immediate_surface_meshes],
            sample_spacing=float(immediate_surface_sample_spacing),
            tip_directions=int(immediate_surface_directions),
            tip_rings=int(immediate_surface_rings),
            evaluated_samples=int(len(surface_samples)),
            uncovered_samples=int(uncovered),
            nominal_collision_samples=int(np.count_nonzero(surface_required > 0.0)),
            added_lift_points=int(np.count_nonzero(lifts > previous_lifts + 1e-12)),
            max_required_lift=float(surface_required.max(initial=0.0)),
            minimum_nominal_gap=float(np.nanmin(surface_gaps)) if np.any(np.isfinite(surface_gaps)) else None,
        )

    # The causal bead model only considers already traversed paths.  This optional
    # test also clears high portions of the current curved mesh under the flat tip.
    current_surface_stats = None
    current_samples = current_edges = current_fractions = current_required = current_gaps = None
    if current_surface_meshes:
        if current_surface_directions < 3 or current_surface_rings < 1:
            raise ValueError("Current-surface tip sampling requires at least 3 directions and 1 ring.")
        current_samples, current_edges, current_fractions = edge_samples(
            points,
            breaks,
            float(current_surface_sample_spacing),
        )
        current_envelope = SurfaceEnvelope.build(
            tuple(Path(path) for path in current_surface_meshes),
            model_lift=float(model_lift),
            model_lift_axis=str(model_lift_axis),
        )
        current_required, current_gaps, uncovered = surface_requirements(
            current_samples,
            current_envelope,
            disk_offsets(
                diameter / 2.0,
                directions=int(current_surface_directions),
                rings=int(current_surface_rings),
            ),
            tolerance=tolerance,
        )
        previous_lifts = lifts.copy()
        np.maximum.at(lifts, current_edges, current_required)
        np.maximum.at(lifts, current_edges + 1, current_required)
        current_surface_stats = dict(
            reference_meshes=[Path(path).name for path in current_surface_meshes],
            sample_spacing=float(current_surface_sample_spacing),
            tip_directions=int(current_surface_directions),
            tip_rings=int(current_surface_rings),
            evaluated_samples=int(len(current_samples)),
            uncovered_samples=int(uncovered),
            nominal_collision_samples=int(np.count_nonzero(current_required > 0.0)),
            added_lift_points=int(np.count_nonzero(lifts > previous_lifts + 1e-12)),
            max_required_lift=float(current_required.max(initial=0.0)),
            minimum_nominal_gap=float(np.nanmin(current_gaps)) if np.any(np.isfinite(current_gaps)) else None,
        )

    lifts = np.minimum(lifts, max_lift)
    if len(samples):
        residual = nominal + (1-fractions)*lifts[edges] + fractions*lifts[edges+1]
    if immediate_surface_stats is not None:
        surface_residual = (
            surface_gaps
            + (1.0-surface_fractions)*lifts[surface_edges]
            + surface_fractions*lifts[surface_edges+1]
        )
        immediate_surface_stats["residual_collision_samples"] = int(
            np.count_nonzero(np.isfinite(surface_residual) & (surface_residual < -tolerance - 1e-9))
        )
    if current_surface_stats is not None:
        current_residual = (
            current_gaps
            + (1.0-current_fractions)*lifts[current_edges]
            + current_fractions*lifts[current_edges+1]
        )
        current_surface_stats["residual_collision_samples"] = int(
            np.count_nonzero(np.isfinite(current_residual) & (current_residual < -tolerance - 1e-9))
        )
    rows[:, 1] += lifts
    np.savetxt(output_dir / name, rows, fmt='%.10f')
    stem = Path(name).stem
    lift_name = f'{stem}.clearance_lifts.txt'
    length_name = f'{stem}.nominal_lengths.txt'
    np.savetxt(output_dir / lift_name, lifts, fmt='%.10f')
    np.savetxt(output_dir / length_name, lengths, fmt='%.10f')
    if breaks:
        break_name = f'{stem}.breaks.txt'
        np.savetxt(output_dir / break_name, sorted(breaks), fmt='%d')
    stats = dict(lifted_points=int(np.count_nonzero(lifts)), max_lift=float(lifts.max(initial=0)), nominal_collision_samples=int(np.count_nonzero(nominal < -tolerance)), residual_collision_samples=int(np.count_nonzero(residual < -tolerance-1e-9)), lift_limit_points=int(np.count_nonzero(lifts >= max_lift-1e-9)), seconds=time.perf_counter()-started)
    if immediate_surface_stats is not None:
        stats["immediate_previous_surface_clearance"] = immediate_surface_stats
    if current_surface_stats is not None:
        stats["current_surface_clearance"] = current_surface_stats
    return dict(layer, nozzle_clearance_lift=stats, clearance_lift_file=lift_name, nominal_segment_length_file=length_name, extrusion_length_mode='nominal_before_clearance_lift', path_break_file=break_name, path_break_count=len(breaks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--tip-outer-diameter', type=float, required=True, help='Flat tip outside diameter, not the extrusion bore size.')
    parser.add_argument('--diameter-measured', action='store_true', help='Record the diameter as measured by the user.')
    parser.add_argument('--edge-sample-spacing', type=float, default=.1)
    parser.add_argument('--contact-tolerance', type=float, default=.01)
    parser.add_argument('--max-lift', type=float, default=.15)
    parser.add_argument('--long-jump-threshold', type=float, default=2.)
    parser.add_argument('--layer-workers', type=int, default=4)
    parser.add_argument('--immediate-previous-surface-clearance', action='store_true', help='Also clear only the nominal mesh from the immediately preceding main layer with the flat tip footprint.')
    parser.add_argument('--immediate-previous-surface-dir', type=Path, default=None, help='Optional folder of <main>_<sub>.obj meshes. Defaults to input_mesh_dir from batch metadata.')
    parser.add_argument('--immediate-previous-surface-sample-spacing', type=float, default=.25, help='Path sample spacing for the optional immediate-surface check.')
    parser.add_argument('--immediate-previous-surface-directions', type=int, default=16, help='Circumference samples for the optional flat-tip disk.')
    parser.add_argument('--immediate-previous-surface-rings', type=int, default=1, help='Radial rings for the optional flat-tip disk, excluding the centre.')
    parser.add_argument('--current-surface-clearance', action='store_true', help='Also clear the current layer nominal mesh with the flat tip footprint; the intended centre contact remains allowed.')
    parser.add_argument('--current-surface-dir', type=Path, default=None, help='Optional folder of current <main>_<sub>.obj meshes. Defaults to input_mesh_dir from batch metadata.')
    parser.add_argument('--current-surface-sample-spacing', type=float, default=.25, help='Path sample spacing for the optional current-surface check.')
    parser.add_argument('--current-surface-directions', type=int, default=16, help='Circumference samples for the optional current-surface flat-tip disk.')
    parser.add_argument('--current-surface-rings', type=int, default=1, help='Radial rings for the optional current-surface flat-tip disk, excluding the centre.')
    parser.add_argument('--layer', default=None, help='Optional single-layer smoke test.')
    parser.add_argument('--retry-limited', action='store_true', help='Recompute only layers with residual collisions in an existing completed output, using the original input.')
    args = parser.parse_args()
    if (
        min(
            args.tip_outer_diameter,
            args.edge_sample_spacing,
            args.max_lift,
            args.layer_workers,
            args.long_jump_threshold,
            args.immediate_previous_surface_sample_spacing,
            args.current_surface_sample_spacing,
        ) <= 0
        or args.contact_tolerance < 0
        or args.immediate_previous_surface_directions < 3
        or args.immediate_previous_surface_rings < 1
        or args.current_surface_directions < 3
        or args.current_surface_rings < 1
    ):
        raise ValueError('Invalid diameter, sampling, tolerance, or worker setting.')
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    if destination.exists() and not args.retry_limited:
        raise FileExistsError('Choose a new output directory; existing folders are preserved.')
    metadata = json.loads((source / 'batch_3axis.meta.json').read_text())
    if metadata.get('causal_nozzle_clearance'):
        raise ValueError('Input already contains causal lifts; use the original processed folder.')
    layers = metadata['layers']
    immediate_surface_by_layer = {}
    current_surface_by_layer = {}
    if args.immediate_previous_surface_clearance or args.current_surface_clearance:
        mesh_dir_raw = (
            args.immediate_previous_surface_dir
            if args.immediate_previous_surface_clearance and args.immediate_previous_surface_dir is not None
            else args.current_surface_dir
            if args.current_surface_clearance and args.current_surface_dir is not None
            else metadata.get('input_mesh_dir')
        )
        if mesh_dir_raw is None:
            raise ValueError('Surface clearance needs a mesh directory flag or input_mesh_dir in batch metadata.')
        mesh_dir = Path(mesh_dir_raw).resolve()
        mesh_files = []
        for candidate in mesh_dir.glob('*.obj'):
            try:
                main, sub = map(int, candidate.stem.split('_', maxsplit=1))
            except ValueError:
                continue
            mesh_files.append((main, sub, candidate))
        if not mesh_files:
            raise FileNotFoundError(f'No <main>_<sub>.obj meshes found in {mesh_dir}')
        mesh_files.sort(key=lambda item: (item[0], item[1]))
        meshes_by_main = {}
        mesh_by_key = {}
        for main, sub, candidate in mesh_files:
            meshes_by_main.setdefault(main, []).append(candidate)
            mesh_by_key[(main, sub)] = candidate
        available_mains = sorted(meshes_by_main)
        model_lift_axis = str(metadata.get('model_lift_axis', 'y')).strip().lower()
        if model_lift_axis != 'y' and abs(float(metadata.get('model_lift', 0.0))) > 0.0:
            raise ValueError('Immediate previous-surface clearance currently supports only a y-axis model lift.')
        for layer in layers:
            main, sub = int(layer['layer_main']), int(layer['layer_sub'])
            key = (main, sub)
            if args.immediate_previous_surface_clearance:
                prior_mains = [candidate for candidate in available_mains if candidate < main]
                immediate_surface_by_layer[key] = tuple(meshes_by_main[prior_mains[-1]]) if prior_mains else ()
            if args.current_surface_clearance:
                source_mesh = layer.get('source_mesh', f'{main}_{sub}.obj')
                current_mesh = mesh_by_key.get(key)
                if current_mesh is None or current_mesh.name != source_mesh:
                    expected = mesh_dir / source_mesh
                    if not expected.is_file():
                        raise FileNotFoundError(f'Current-surface mesh is missing for {main}_{sub}: {expected}')
                    current_mesh = expected
                current_surface_by_layer[key] = (current_mesh,)
    retained_results = []
    if args.retry_limited:
        if args.layer or (destination / '.causal_lift_in_progress').exists():
            raise ValueError('Retry requires a completed batch and no --layer selection.')
        previous = json.loads((destination / 'batch_3axis.meta.json').read_text())
        previous_settings = previous['causal_nozzle_clearance']
        if previous_settings['input_dir'] != str(source) or previous_settings['tip_outer_diameter'] != args.tip_outer_diameter:
            raise ValueError('Retry source or outer tip diameter differs from the completed batch.')
        retry_keys = {(l['layer_main'], l['layer_sub']) for l in previous['layers'] if l['nozzle_clearance_lift']['residual_collision_samples'] > 0}
        retained_results = [l for l in previous['layers'] if (l['layer_main'], l['layer_sub']) not in retry_keys]
        layers = [l for l in layers if (l['layer_main'], l['layer_sub']) in retry_keys]
    if args.layer:
        layers = [layer for layer in layers if f"{layer['layer_main']}_{layer['layer_sub']}" == args.layer]
        if not layers:
            raise ValueError('Requested layer is missing.')
    if not args.retry_limited:
        shutil.copytree(source, destination)
    progress_marker = destination / '.causal_lift_in_progress'
    progress_marker.write_text('Clearance correction is incomplete; do not export this folder.\n')
    if args.layer:
        # Metadata exposes only the corrected subset to exporters.
        metadata['layers'] = layers
        for path in destination.glob('*_3axis.txt'):
            if path.name not in {l['output_path_file'] for l in layers}:
                path.unlink()
    print(f'[causal lift] {len(layers)} layers, outer tip diameter={args.tip_outer_diameter}, workers={args.layer_workers}', flush=True)
    if args.immediate_previous_surface_clearance:
        print('[causal lift] Immediate previous nominal-surface footprint clearance enabled', flush=True)
    if args.current_surface_clearance:
        print('[causal lift] Current nominal-surface flat-tip clearance enabled', flush=True)
    results = retained_results.copy()
    completed_count = 0
    def record_result(result):
        nonlocal completed_count
        results.append(result)
        completed_count += 1
        print(f"[causal lift] {completed_count}/{len(layers)}: {result['output_path_file']} {result['nozzle_clearance_lift']}", flush=True)
    if args.layer_workers == 1:
        for layer in layers:
            record_result(process_layer(
                source, destination, layer, args.tip_outer_diameter, args.edge_sample_spacing,
                args.contact_tolerance, args.max_lift, args.long_jump_threshold,
                immediate_surface_by_layer.get((int(layer['layer_main']), int(layer['layer_sub'])), ()),
                float(metadata.get('model_lift', 0.0)), str(metadata.get('model_lift_axis', 'y')),
                args.immediate_previous_surface_sample_spacing,
                args.immediate_previous_surface_directions,
                args.immediate_previous_surface_rings,
                current_surface_by_layer.get((int(layer['layer_main']), int(layer['layer_sub'])), ()),
                args.current_surface_sample_spacing,
                args.current_surface_directions,
                args.current_surface_rings,
            ))
    else:
        with ProcessPoolExecutor(max_workers=args.layer_workers, mp_context=get_context('spawn')) as executor:
            jobs = [executor.submit(
                process_layer,
                source, destination, layer, args.tip_outer_diameter, args.edge_sample_spacing,
                args.contact_tolerance, args.max_lift, args.long_jump_threshold,
                immediate_surface_by_layer.get((int(layer['layer_main']), int(layer['layer_sub'])), ()),
                float(metadata.get('model_lift', 0.0)), str(metadata.get('model_lift_axis', 'y')),
                args.immediate_previous_surface_sample_spacing,
                args.immediate_previous_surface_directions,
                args.immediate_previous_surface_rings,
                current_surface_by_layer.get((int(layer['layer_main']), int(layer['layer_sub'])), ()),
                args.current_surface_sample_spacing,
                args.current_surface_directions,
                args.current_surface_rings,
            ) for layer in layers]
            for future in as_completed(jobs):
                record_result(future.result())
    metadata['layers'] = sorted(results, key=lambda l: (l['layer_main'], l['layer_sub']))
    metadata['output_dir'] = str(destination)
    metadata['causal_nozzle_clearance'] = dict(tip_outer_diameter=args.tip_outer_diameter, diameter_verified=args.diameter_measured, diameter_source='user measurement' if args.diameter_measured else 'trial assumption', bead_top_model='nominal', previous_layer_check=False, immediate_previous_surface_clearance=bool(args.immediate_previous_surface_clearance), immediate_previous_surface_dir=str(args.immediate_previous_surface_dir or metadata.get('input_mesh_dir')) if args.immediate_previous_surface_clearance else None, immediate_previous_surface_scope='immediately preceding main layer only' if args.immediate_previous_surface_clearance else None, immediate_previous_surface_sample_spacing=args.immediate_previous_surface_sample_spacing if args.immediate_previous_surface_clearance else None, immediate_previous_surface_directions=args.immediate_previous_surface_directions if args.immediate_previous_surface_clearance else None, immediate_previous_surface_rings=args.immediate_previous_surface_rings if args.immediate_previous_surface_clearance else None, current_surface_clearance=bool(args.current_surface_clearance), current_surface_dir=str(args.current_surface_dir or metadata.get('input_mesh_dir')) if args.current_surface_clearance else None, current_surface_scope='matching current <main>_<sub> mesh only' if args.current_surface_clearance else None, current_surface_sample_spacing=args.current_surface_sample_spacing if args.current_surface_clearance else None, current_surface_directions=args.current_surface_directions if args.current_surface_clearance else None, current_surface_rings=args.current_surface_rings if args.current_surface_clearance else None, max_lift=args.max_lift, contact_tolerance=args.contact_tolerance, edge_sample_spacing=args.edge_sample_spacing, long_jump_threshold=args.long_jump_threshold, input_dir=str(source), heights_and_widths='preserved', extrusion_lengths='nominal_before_lift')
    (destination / 'batch_3axis.meta.json').write_text(json.dumps(metadata, indent=2))
    progress_marker.unlink()
    print(f'[causal lift] Saved {destination}', flush=True)


if __name__ == '__main__':
    main()

"""Plan disconnected 3D surface components before any projected polygon union.

Keep the original <main>_<sub> identity and merge each patch's complete path
with explicit travel, normals, widths and role tags. Patch numbers are internal.
"""
from functools import wraps
from inspect import signature
from pathlib import Path
import os

import numpy as np
import pyvista as pv

from adaptive_path_decimation import _load_tags, _write_tags
from mesh_distance_workflow import load_obj_mesh
from heat_method_core import validate_triangle_mesh


def planner_mesh(mesh_path, mesh_arrays=None):
    """Reuse validated arrays; generated patch OBJs store these doubles losslessly."""
    if mesh_arrays is None:
        return load_obj_mesh(str(mesh_path))
    vertices = np.asarray(mesh_arrays[0], dtype=np.float64)
    faces = np.asarray(mesh_arrays[1], dtype=np.int64)
    validate_triangle_mesh(vertices, faces)
    return vertices, faces


def surface_components(vertices, faces):
    cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    mesh = pv.PolyData(vertices, cells)
    for body in mesh.split_bodies(label=False):
        patch = body.extract_surface(algorithm="dataset_surface").triangulate()
        yield patch.points.copy(), patch.faces.reshape(-1, 4)[:, 1:].copy()


def _write_obj(path, vertices, faces):
    with path.open('w', encoding='utf-8') as stream:
        np.savetxt(stream, vertices, fmt='v %.17g %.17g %.17g')
        np.savetxt(stream, faces + 1, fmt='f %d %d %d')


def plan_connected_patches(planner):
    parameters = signature(planner).parameters
    supports_arrays = '_mesh_arrays' in parameters and '_path_arrays' in parameters
    supports_memory=supports_arrays and '_write_artifacts' in parameters

    @wraps(planner)
    def run(mesh_path, output_dir, **options):
        mesh_path, output_dir = Path(mesh_path), Path(output_dir)
        vertices, faces = load_obj_mesh(str(mesh_path))
        source_bounds = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
        patches = list(surface_components(vertices, faces))
        if len(patches) == 1:
            prepared = dict(_mesh_arrays=(vertices, faces)) if supports_arrays else {}
            path, metadata = planner(mesh_path, output_dir, **options, **prepared)
            metadata['connected_patch_count'] = 1
            metadata['source_mesh_bounds'] = source_bounds
            if metadata.get('projected_degenerate_artifact'):
                print(f'[projection] {mesh_path.name}: omitted small projection-degenerate patch; '
                      f"diagonal={metadata['projected_degenerate_artifact']['diagonal_mm']:.6g} mm", flush=True)
            return path, metadata

        rows, widths, tags, breaks, reports = [], [], [], [], []
        retain=not supports_memory or os.environ.get('PROJECTED_KEEP_PATCH_ARTIFACTS','1')!='0'
        offset = 0
        print(f'[patches] {mesh_path.name}: planning {len(patches)} disconnected surface patches independently', flush=True)
        for index, (patch_vertices, patch_faces) in enumerate(patches):
            # Retain the original stem so layer parity, roles and overrides agree.
            work = output_dir / 'patches' / mesh_path.stem / f'{index:04d}'
            source, target = work / 'input', work / 'output'
            if retain:
                source.mkdir(parents=True, exist_ok=True)
                target.mkdir(parents=True, exist_ok=True)
                _write_obj(source / mesh_path.name, patch_vertices, patch_faces)
            captured = {} if supports_arrays else None
            prepared = dict(_mesh_arrays=(patch_vertices, patch_faces), _path_arrays=captured) if supports_arrays else {}
            if supports_memory:prepared['_write_artifacts']=retain
            try:
                path, metadata = planner(source / mesh_path.name, target, **options, **prepared)
            except Exception as exc:
                if not retain:
                    source.mkdir(parents=True,exist_ok=True)
                    _write_obj(source/mesh_path.name,patch_vertices,patch_faces)
                diagonal=float(np.linalg.norm(np.ptp(patch_vertices,axis=0)))
                raise RuntimeError(f'{mesh_path.name}, connected patch {index}/{len(patches)-1} '
                    f'({len(patch_vertices)} vertices, {len(patch_faces)} faces, diagonal {diagonal:g} mm): {exc}') from exc
            if metadata.get('projected_degenerate_artifact'):
                print(f'[projection] {mesh_path.name} patch {index}: omitted small projection-degenerate patch; '
                      f"diagonal={metadata['projected_degenerate_artifact']['diagonal_mm']:.6g} mm", flush=True)
            reports.append(dict(patch_index=index, source_vertices=len(patch_vertices),
                                source_faces=len(patch_faces), generated_points=metadata['path_point_count'],
                                input_obj=str(source / mesh_path.name) if retain else None, metadata=metadata))
            if captured is not None:
                points = captured['rows']
            else:
                if not path.stat().st_size:
                    continue
                points = np.loadtxt(path, ndmin=2)
            if not len(points):
                continue
            if captured is not None:
                profile, roles = captured['widths'], captured['tags']
            else:
                profile = np.loadtxt(target / metadata['width_profile_file'], ndmin=1).reshape(-1)
                roles = _load_tags(target / metadata['path_tag_file'], len(points))
            if len(profile) != len(points) or roles is None or np.any(roles == ''):
                raise ValueError(f'Incomplete patch widths or role tags: {path}')
            if offset:
                # Even overlapping projections must never connect separate patches.
                breaks.append(offset)
            if metadata.get('path_break_file'):
                if captured is not None:
                    breaks.extend(offset + value for value in captured['breaks'])
                else:
                    text = (target / metadata['path_break_file']).read_text(encoding='utf-8')
                    breaks.extend(offset + int(value) for value in text.split())
            rows.append(points)
            widths.append(profile)
            tags.append(roles)
            offset += len(points)

        points = np.vstack(rows) if rows else np.empty((0, 6))
        profile = np.concatenate(widths) if widths else np.empty(0)
        roles = np.concatenate(tags) if tags else np.empty(0, dtype=object)
        output = output_dir / f'{mesh_path.stem}.txt'
        width_file = output_dir / f'{mesh_path.stem}.widths.txt'
        tag_file = output_dir / f'{mesh_path.stem}.tags.txt'
        break_file = output_dir / f'{mesh_path.stem}.breaks.txt'
        np.savetxt(output, points, fmt='%.10f')
        np.savetxt(width_file, profile, fmt='%.10f')
        _write_tags(tag_file, roles)
        breaks = sorted(set(breaks))
        # Always overwrite the sidecar, including an empty result on a rerun.
        break_file.write_text(''.join(f'{value}\n' for value in breaks), encoding='utf-8')

        metadata = dict(reports[0]['metadata'])
        # Layer-level diagnostics refer to all patches, not just the first one.
        metadata.pop('projected_degenerate_artifact', None)
        artifacts = [dict(report['metadata']['projected_degenerate_artifact'], patch_index=report['patch_index'])
                     for report in reports if report['metadata'].get('projected_degenerate_artifact')]
        if artifacts:
            metadata['projected_degenerate_artifacts'] = artifacts
        counts = ('boundary_loop_count', 'parallel_level_count', 'parallel_polyline_count',
                  'planar_infill_polyline_count', 'ironing_polyline_count', 'chained_boundary_part_count',
                  'chained_parallel_part_count', 'projected_ambiguous_lift_point_count',
                  'projected_snapped_lift_point_count')
        for key in counts:
            if key in metadata:
                metadata[key] = sum(report['metadata'].get(key, 0) for report in reports)
        if 'support_components' in metadata:
            metadata['support_components'] = [component for report in reports
                                              for component in report['metadata']['support_components']]
        reductions=[r['metadata']['point_reduction'] for r in reports if r['metadata'].get('point_reduction',{}).get('enabled')]
        if reductions:
            metadata['point_reduction']=dict(reductions[0],input_points=sum(r['input_points'] for r in reductions),
                output_points=sum(r['output_points'] for r in reductions),
                removed_points=sum(r.get('removed_points',0) for r in reductions),seconds=sum(r['seconds'] for r in reductions))
        repairs=[r['metadata']['support_hole_repair'] for r in reports if 'support_hole_repair' in r['metadata']]
        metadata['support_reference_additions']=[p for r in reports for p in r['metadata'].get('support_reference_additions',[])]
        if repairs:
            clearance=[r['minimum_model_distance_mm'] for r in repairs if r.get('minimum_model_distance_mm') is not None]
            metadata['support_hole_repair']=dict(mask_count=sum(r['mask_count'] for r in repairs),
                field_lift_points=sum(r['field_lift_points'] for r in repairs),
                maximum_physical_field_error_mm=max(r['maximum_physical_field_error_mm'] for r in repairs),
                minimum_model_distance_mm=min(clearance) if clearance else None)
        metadata.update(path_file=output.name, path_point_count=len(points),
                        source_mesh_bounds=source_bounds,
                        width_profile_file=width_file.name if len(profile) else None,
                        width_profile_point_count=len(profile),
                        width_profile_min=float(profile.min()) if len(profile) else None,
                        width_profile_max=float(profile.max()) if len(profile) else None,
                        path_tag_file=tag_file.name if len(roles) else None,
                        path_tag_values=sorted(set(map(str, roles))),
                        path_tag_block_count=int(np.count_nonzero(np.r_[True, roles[1:] != roles[:-1]])) if len(roles) else 0,
                        path_break_file=break_file.name if breaks else None, path_break_count=len(breaks),
                        connected_patch_count=len(patches), connected_patches=reports,
                        patch_order='PyVista connected-region order; complete each patch before travel to next')
        if not retain:metadata['connected_patch_artifacts']='memory_only; original layer retained; failed patch inputs saved on error'
        return output, metadata
    return run

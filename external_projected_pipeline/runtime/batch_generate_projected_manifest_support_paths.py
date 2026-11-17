"""Projected support paths selected by platform/contact manifest roles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from batch_generate_projected_hybrid_paths import (
    ProjectedSurfaceLifter, _iter_polygons, _planar_region_from_projected_triangles,
    _planar_zigzag_infill_items, _ring_items, _path_break_indices, _covered_inset,
)
from batch_generate_projected_support_model_paths import support_resample_spacing
from extrusion_height import apply_axis_lift
from test_hybrid_contours import (
    chain_boundary_family, chain_parallel_family, concatenate_paths,
    concatenate_width_profiles, concatenate_path_tags, encode_path_tag_blocks,
)
from toolpath_export import compute_path_normals, save_oriented_path, save_numeric_array
from connected_patch_planner import plan_connected_patches, planner_mesh
from path_fragment_filter import filter_infill_parts


def _boundary_family(component,count,width,step,lifter,min_points,min_step):
    items=[]
    for level in range(count):
        offset=(level+.5)*width
        inset=_covered_inset(component,offset)
        if not inset.is_empty:
            rings,ambiguous,_=_ring_items(inset,offset,step,lifter,
                min_points_per_contour=min_points,min_resample_spacing=min_step)
            if ambiguous:raise ValueError('Support boundary has multiple Y values in its XZ projection.')
            for item in rings:item.effective_width=width
            items.extend(rings)
    return chain_boundary_family(items,tolerance=1e-8)


def support_profile(entry: dict, config: dict, default_width: float) -> dict:
    if entry['is_platform_layer']:
        role = 'platform'
    elif entry['is_contact_layer']:
        role = 'contact'
    else:
        role = 'regular'
    values = dict(config.get(role, {}))
    width = float(values.get('width', config.get('width', default_width)))
    count = values.get('contour_count', 2)
    if isinstance(count,bool) or not isinstance(count,int) or count<1:
        raise ValueError(f'{role}.contour_count must be a positive integer.')
    density = float(values.get('density', 0.5 if role == 'contact' else 1.0))
    if not np.isfinite([width,density]).all() or width<=0 or not 0<density<=1:
        raise ValueError(f'{role} support width must be positive and density must be in (0, 1].')
    pitch = float(values.get('line_spacing', width / density))
    flow = float(values.get('flow_ratio', 1.0))
    cutoff = float(values.get('min_fill_area', 0.0))
    wall_overlap = float(values.get('infill_wall_overlap', 0.1 * width))
    filled_wall_width=float(values.get('filled_wall_width',width))
    if not np.isfinite(filled_wall_width) or not 0<filled_wall_width<=width:
        raise ValueError('filled_wall_width must be positive and no larger than the nominal support width.')
    if not np.isfinite([width, density, pitch, flow, cutoff, wall_overlap]).all():
        raise ValueError('Support settings must be finite.')
    if width <= 0 or count < 1 or not 0 < density <= 1 or pitch < width or not 0 < flow <= 1 or cutoff < 0:
        raise ValueError(f'Invalid {role} support settings: {values}')
    if not 0.0 <= wall_overlap <= 0.5 * width:
        raise ValueError('Support infill_wall_overlap must be between zero and half the bead width.')
    if role=='contact' and wall_overlap>filled_wall_width:
        raise ValueError('Support overlap must not exceed the filled wall width; otherwise the hatch bead extends outside the patch.')
    if role == 'platform':
        pitch, flow = width, 1.0
    for key in ('angle_degrees','alternate_angle_degrees'):
        if key in values:
            if key=='alternate_angle_degrees' and values[key] is None:continue
            if values[key] is None or not np.isfinite(float(values[key])):
                raise ValueError(f'{role}.{key} must be finite'+(' or null.' if key=='alternate_angle_degrees' else '.'))
    return dict(values, role=role, width=width, contour_count=count, line_spacing=pitch,
                filled_wall_width=filled_wall_width,
                flow_ratio=flow, min_fill_area=cutoff, infill_wall_overlap=wall_overlap,
                reduce_contour_for_infill=bool(values.get('reduce_contour_for_infill', False)))


def support_sampling(config,hybrid):
    base=float(config.get('resample_spacing',hybrid.get('resample_spacing') or hybrid['spacing']))
    minimum=float(config.get('min_resample_spacing',min(base,.1)))
    count=config.get('min_points_per_contour',16)
    if isinstance(count,bool) or not isinstance(count,int):
        raise ValueError('support_paths.min_points_per_contour must be an integer >= 3.')
    support_resample_spacing(0.,base_spacing=base,min_spacing=minimum,min_points_per_contour=count)
    return base,minimum,count


@plan_connected_patches
def process_manifest_support_layer(mesh_path: Path, output_dir: Path, *, entry: dict,
                                   support_config: dict, hybrid_config: dict,
                                   repair_context: str | None = None,
                                   _mesh_arrays: tuple[np.ndarray, np.ndarray] | None = None,
                                   _path_arrays: dict | None = None,_write_artifacts: bool=True) -> tuple[Path, dict]:
    profile = support_profile(entry, support_config, float(hybrid_config['spacing']))
    role = profile['role']
    width = profile['width']
    count = profile['contour_count']
    vertices, faces = planner_mesh(mesh_path, _mesh_arrays)
    lift = float(hybrid_config.get('model_lift', 0.0))
    if not np.isfinite(lift):raise ValueError('model_lift must be finite.')
    if lift:
        vertices = apply_axis_lift(vertices, axis=str(hybrid_config.get('model_lift_axis', 'y')), amount=lift)
    lifter = ProjectedSurfaceLifter.build(vertices, faces)
    if repair_context:
        from support_hole_repair import bind_repair
        lifter,region=bind_repair(lifter,repair_context)
    else:region = _planar_region_from_projected_triangles(lifter)
    reference_addition=None
    if repair_context:
        from support_hole_repair import save_reference_addition
        reference_addition=save_reference_addition(lifter,output_dir,mesh_path.stem)
    components = _iter_polygons(region)
    base_step,min_step,min_points=support_sampling(support_config,dict(hybrid_config,spacing=width))
    parts, widths, roles = [], [], []
    component_stats = []
    boundary_roles = f'support_{role}_boundary'
    fill_role = f'support_{role}_infill'
    pitch = profile['line_spacing']
    main = int(entry['layer_main'])
    angle = float(profile.get('angle_degrees', 0.0))
    alternate = profile.get('alternate_angle_degrees', 90.0)
    if alternate is not None and main % 2:
        angle = float(alternate)
    for component in components:
        step, sample_meta = support_resample_spacing(float(component.area), base_spacing=base_step,
                                                    min_spacing=min_step, min_points_per_contour=min_points)
        fill = role == 'platform' or (role == 'contact' and component.area >= profile['min_fill_area'])
        candidate_count = (
            max(1, count - 1)
            if fill and role == 'contact' and profile['reduce_contour_for_infill']
            else count
        )
        # The infill helper adds half a bead-width inset. Account for it here
        # so the raster bead reaches the inner wall, independently of its pitch.
        candidate_wall_width=profile['filled_wall_width'] if fill and role=='contact' else width
        interior = _covered_inset(component, candidate_count * candidate_wall_width)
        infill_items = []
        if fill and not interior.is_empty:
            infill_items, ambiguous, _ = _planar_zigzag_infill_items(
                interior, flow_spacing=width, density=width / pitch,
                angle_degrees=angle, origin_xz=(0.0, 0.0), overlap=profile['infill_wall_overlap'],
                max_link_length=1.5 * pitch, max_boundary_link_length=2.0 * pitch,
                min_segment_length=min(width * 0.5, step), resample_spacing=step, lifter=lifter,
            )
            if ambiguous:raise ValueError('Support infill has multiple Y values in its XZ projection.')
        # Retain the original walls if a narrow patch or sparse scan produces
        # no printable infill. Only real contact zig-zags replace the inner wall.
        effective_count = candidate_count if infill_items else count
        wall_width=candidate_wall_width if infill_items else width
        bp,bw,current=_boundary_family(component,effective_count,wall_width,step,lifter,min_points,min_step)
        ip, iw, _ = chain_parallel_family(infill_items, current, tolerance=1e-8)
        prefilter=dict(removed_runs=0,removed_points=0)
        if role=='contact':
            ip,iw,prefilter=filter_infill_parts(ip,iw,support_config.get('short_fragment_filter',{}))
        restored=role=='contact' and not ip and (wall_width!=width or effective_count!=count)
        if restored:
            wall_width=width;effective_count=count
            bp,bw,current=_boundary_family(component,count,width,step,lifter,min_points,min_step)
        parts.extend(bp)
        widths.extend(bw)
        roles.extend([boundary_roles] * len(bp))
        parts.extend(ip)
        widths.extend(iw)
        roles.extend([fill_role] * len(ip))
        component_stats.append(dict(
            sample_meta, fill_enabled=bool(ip), infill_polyline_count=len(ip),
            configured_contour_count=count, effective_contour_count=effective_count,
            outer_contour_center_offset=0.5*wall_width,wall_bead_width=wall_width,infill_bead_width=width,
            contour_reduced_for_infill=effective_count < count,
            hollow_wall_restored_after_filter=restored,
            contact_infill_prefilter=prefilter,
            infill_nominal_boundary_inset=(effective_count*wall_width+0.5*width-profile['infill_wall_overlap']) if ip else None,
        ))

    # Preserve one ordered path file; only short, in-region boundary links print.
    break_before = set()
    minimum_wall_width=min((r['wall_bead_width'] for r in component_stats),default=width)
    printable = _covered_inset(region,0.5*minimum_wall_width).buffer(1e-8)
    for index in range(1, len(parts)):
        connector = LineString([parts[index - 1][-1][[0, 2]], parts[index][0][[0, 2]]])
        if ('infill' in roles[index] or roles[index] != roles[index - 1]
                or connector.length > 2.0 * width or not printable.covers(connector)):
            break_before.add(index)
    points = concatenate_paths(parts)
    profile_widths = concatenate_width_profiles(parts, widths)
    tags = concatenate_path_tags(parts, roles)
    breaks = _path_break_indices(parts, break_before)
    normals = compute_path_normals(vertices, faces, points) if len(points) else np.empty((0, 3))
    if repair_context:normals=lifter.normals(points,normals)
    from early_path_reduction import reduce_generated
    reduction_settings=support_config.get('point_reduction',hybrid_config.get('point_reduction'))
    if isinstance(reduction_settings,dict):reduction_settings=dict(reduction_settings,fragment_filter=support_config.get('short_fragment_filter',{}))
    points,normals,profile_widths,tags,breaks,reduction=reduce_generated(
        points,normals,profile_widths,tags,breaks,parts=parts,part_roles=roles,target_distance=base_step,
        settings=reduction_settings,region=region,
        minimum_boundary_points=min_points)
    output = output_dir / f'{mesh_path.stem}.txt'
    serialized_rows = save_oriented_path(str(output), points, normals, return_serialized=_path_arrays is not None,write_file=_write_artifacts)
    width_path = output_dir / f'{mesh_path.stem}.widths.txt'
    tag_path = output_dir / f'{mesh_path.stem}.tags.txt'
    break_path = output_dir / f'{mesh_path.stem}.breaks.txt'
    serialized_widths = save_numeric_array(width_path, profile_widths, return_serialized=_path_arrays is not None,write_file=_write_artifacts)
    blocks = encode_path_tag_blocks(tags)
    if _write_artifacts:tag_path.write_text(''.join(f'{start} {end} {tag}\n' for start, end, tag in blocks), encoding='utf-8')
    if breaks and _write_artifacts:
        break_path.write_text(''.join(f'{index}\n' for index in breaks), encoding='utf-8')
    elif _write_artifacts and break_path.exists():
        break_path.unlink()
    if _path_arrays is not None:
        _path_arrays.update(rows=serialized_rows, widths=serialized_widths, tags=tags, breaks=breaks)
    return output, dict(
        layer_role=f'support_{role}', planner_backend='xz_projected_2d',
        point_reduction=reduction,
        path_file=output.name, path_point_count=len(points), width_profile_file=width_path.name,
        width_profile_point_count=len(profile_widths), width_profile_min=float(profile_widths.min()) if len(profile_widths) else width,
        width_profile_max=float(profile_widths.max()) if len(profile_widths) else width,
        path_tag_file=tag_path.name, path_tag_values=sorted(set(roles)),
        path_tag_format='half_open_index_blocks', path_tag_block_count=len(blocks),
        path_break_file=break_path.name if breaks else None, path_break_count=len(breaks),
        path_break_semantics='segment_before_index_is_non_printing_travel',
        support_profile=profile, support_components=component_stats,
        extrusion_role_scales={fill_role: profile['flow_ratio']} if role == 'contact' else {},
        configured_boundary_level_count=count,
        boundary_level_count=max((component['effective_contour_count'] for component in component_stats), default=0),
        parallel_disabled=not any('infill' in tag for tag in roles),
        contact_pitch_used_for_extrusion=False,
        support_reference_additions=[reference_addition] if reference_addition else [],
        **({'support_hole_repair':lifter.report()} if repair_context else {}),
        **({'projected_degenerate_artifact': lifter.degenerate_artifact} if lifter.degenerate_artifact else {}),
    )

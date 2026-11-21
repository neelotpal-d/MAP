"""Paired-layer adapter for the existing experimental surface planner.

Author: Neelotpal Dutta
Affiliation / contact / additional authors: [add details]
Numerical solvers live in surface_field_pipeline; routing is not repeated here.
"""
from pathlib import Path
import json
import sys
import numpy as np

from compatible_support_schedule import load_schedule, build_combined_input, resolve
from mesh_distance_workflow import load_obj_mesh
from connected_patch_planner import surface_components
from batch_generate_projected_manifest_support_paths import support_profile, support_sampling
from batch_generate_projected_support_model_paths import support_resample_spacing


def prepare_inputs(config, base, folder, normal_mode):
    """Copy the established manifest schedule, resolving gradient signs once."""
    if normal_mode not in {'field_gradient', 'mesh_winding', 'reverse_winding'}:
        raise ValueError('Paired normal orientation requires saved field gradients or an explicit winding choice.')
    if config['support_paths'].get('hole_repair', {}).get('enabled'):
        raise ValueError('Projected support-hole repair is unavailable for experimental surface paths. Disable support_paths.hole_repair.enabled; original surfaces are retained for inspection.')
    entries, metadata, _ = load_schedule(base, config['compatible_inputs'])
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Experimental combined inputs require a fresh output folder.')
    folder.mkdir(parents=True, exist_ok=True)
    build_combined_input(entries, folder,
        sublayer_order=config['compatible_inputs'].get('sublayer_order', 'index'))
    meshes = [folder/e['combined_mesh'] for e in entries]
    if normal_mode != 'field_gradient':
        return meshes, entries, metadata
    from field_orientation import save_gradients, load_gradients
    root = resolve(base, config['compatible_inputs']['export_root'])
    rebuild_path = resolve(root, metadata.get('rebuild_pipeline', 'rebuild_pipeline.json'))
    rebuild = json.loads(rebuild_path.read_text(encoding='utf-8-sig')) if rebuild_path.is_file() else {}
    inputs = rebuild.get('input', {})
    model = support = None
    for entry, mesh in zip(entries, meshes):
        source = Path(entry['source_dir'])/entry['source_mesh']
        sidecar = source.with_suffix('.field_gradients.npz')
        V, F = load_obj_mesh(str(mesh))
        if sidecar.is_file():
            # Validate against the original OBJ before copying its signed field
            # data to the byte-identical, renamed execution mesh.
            gradients, origin = load_gradients(source, len(F))
        elif entry.get('is_planar_base_layer') or metadata.get('field_mode') == 'analytic_y':
            gradients = np.tile([0., 1., 0.], (len(F), 1))
            origin = 'Physical planar platform base / declared analytic-Y field'
        else:
            # Reuse Neelotpal Dutta's bundled support-field evaluators. They use
            # the saved tetrahedral/trilinear fields, without retraining or NN queries.
            runtime = Path(__file__).resolve().parents[2]/'CurvedSupportGenerator/runtime'
            if str(runtime) not in sys.path:
                sys.path.append(str(runtime))
            centers = V[F].mean(axis=1)
            if entry['part'] == 'model':
                if model is None:
                    from SupportGenerationVoxel.model_field import TetrahedralVertexField
                    value = inputs.get('tetrahedral_volume') or metadata.get('model_volume_source')
                    if not value or not resolve(root, value).is_file():
                        raise ValueError('Saved tetrahedral model field is unavailable. Supply its original paired layer job or choose explicit winding orientation.')
                    model = TetrahedralVertexField(resolve(root, value), inputs.get('tetrahedral_scalar', 'scalar'))
                _, gradients = model.evaluate_scalars_and_gradients(centers)
                origin = str(model.path)
            else:
                if support is None:
                    from SupportGenerationCurved.field_grid import GridField
                    value = inputs.get('support_field_source') or metadata.get('support_field_source')
                    if not value:
                        raise ValueError('Saved support field is unavailable. Supply its original paired layer job or choose explicit winding orientation.')
                    support = GridField.load(resolve(root, value))
                _, gradients = support.evaluate_scalars_and_gradients(centers)
                origin = 'Saved optimized trilinear support field'
        save_gradients(mesh, gradients, origin)
    return meshes, entries, metadata


def support_paths(vertices, faces, settings, entry, config):
    """Apply the established role/width policy using the model-only solver."""
    from surface_field_pipeline import surface_paths
    profile = support_profile(entry, config, settings['spacing'])
    role = profile['role']; width = profile['width']
    base_step, minimum, min_points = support_sampling(config, settings)
    rows_all=[]; widths_all=[]; tags_all=[]; breaks_all=[]; diagnostics=[]; count=0
    angle=float(profile.get('angle_degrees', 0.))
    alternate=profile.get('alternate_angle_degrees', 90.)
    if alternate is not None and entry['original_main'] % 2:
        angle=float(alternate)
    for V, F in surface_components(vertices, faces):
        triangle=V[F]
        area=float(.5*np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0], triangle[:,2]-triangle[:,0]), axis=1).sum())
        step, sampling=support_resample_spacing(area, base_spacing=base_step,
            min_spacing=minimum, min_points_per_contour=min_points)
        fill=role=='platform' or ((role=='contact' or
            (role=='regular' and profile['infill_enabled'])) and area>=profile['min_fill_area'])
        contours=profile['contour_count']
        candidate=max(1, contours-1) if fill and role=='contact' and profile['reduce_contour_for_infill'] else contours
        wall=profile['filled_wall_width'] if fill and role=='contact' else width
        options=dict(settings, spacing=width, wall_width=wall, inset=True,
            resample_spacing=step, boundary_count=candidate, pattern='sparse' if fill else 'contours',
            sparse=fill, density=width/profile['line_spacing'], pitch=profile['line_spacing'],
            overlap=profile['infill_wall_overlap'], angle=angle)
        if options.get('waypoint_sampling') is not None:
            # Retain the support profile's finer contour sampling budget.
            options['waypoint_sampling']=dict(options['waypoint_sampling'],maximum_spacing_mm=min(
                float(options['waypoint_sampling']['maximum_spacing_mm']),2*step))
        rows, widths, tags, breaks, stats=surface_paths(V,F,options)
        filled=bool(np.any(tags!='boundary_loop'))
        restored=fill and not filled and (wall!=width or candidate!=contours)
        if restored:
            options.update(wall_width=width,boundary_count=contours,pattern='contours',sparse=False)
            rows,widths,tags,breaks,stats=surface_paths(V,F,options)
            wall=width;candidate=contours
        tags=np.where(tags=='boundary_loop',f'support_{role}_boundary',f'support_{role}_infill')
        if len(rows):
            if count:breaks_all.append(count)
            breaks_all.extend(count+int(i) for i in breaks)
            rows_all.append(rows);widths_all.append(widths);tags_all.append(tags);count+=len(rows)
        diagnostics.append(dict(area_mm2=area,area_definition='triangle surface area',
            fill_enabled=filled,wall_bead_width=wall,infill_bead_width=width,
            effective_contour_count=candidate,configured_contour_count=contours,
            hollow_wall_restored=restored,outer_contour_center_offset=.5*wall,
            infill_nominal_boundary_inset=(candidate*wall+.5*width-profile['infill_wall_overlap']) if filled else None,
            sampling=sampling,field_diagnostics=stats))
    rows=np.vstack(rows_all) if rows_all else np.empty((0,6))
    widths=np.concatenate(widths_all) if widths_all else np.empty(0)
    tags=np.concatenate(tags_all) if tags_all else np.empty(0,dtype=str)
    return rows,widths,tags,breaks_all,diagnostics,profile


def height_references(layer, layers, mesh_lookup, mains, ranks, window):
    """Only already scheduled material, including intermediate support levels.

    A main index can hold several earlier support scalars and a final model
    scalar. Excluding the entire current main would lose their correspondence;
    including the entire main would query geometry that has not been printed.
    """
    main=layer['layer_main']; allowed=set(mains[max(0,ranks[main]-window):ranks[main]+1])
    return [str(mesh_lookup[(r['layer_main'],r['layer_sub'])]) for r in layers
        if r['layer_main'] in allowed and r['print_order']<layer['print_order'] and r['path_point_count']>0]

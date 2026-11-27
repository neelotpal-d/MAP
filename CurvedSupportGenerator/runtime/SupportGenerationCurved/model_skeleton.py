"""Model/field adapter for the supplied progressive skeleton growth algorithm."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from vtkmodules.vtkCommonCore import reference
from vtkmodules.vtkCommonDataModel import vtkCellLocator, vtkGenericCell

from .contact_direction_guide import ContactDirectionGuide
from .landing_checks import ModelLandingCheck
from .output import prepare_output
from .progressive_skeleton import grow


class CachedSurfaceQuery:
    """PyVista-equivalent closest-cell queries with one immutable locator."""
    def __init__(self,mesh):
        self.locator=vtkCellLocator()
        self.locator.SetDataSet(mesh)
        self.locator.BuildLocator()

    def __call__(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        closest=np.empty_like(points);ids=np.empty(len(points),dtype=np.int64)
        cell=vtkGenericCell();cell_id=reference(0);sub_id=reference(0);dist2=reference(0.)
        for i,point in enumerate(points):
            foot=[0.,0.,0.]
            self.locator.FindClosestPoint(point,foot,cell,cell_id,sub_id,dist2)
            closest[i]=foot;ids[i]=int(cell_id)
        return closest,ids


class BoundaryTaylorField:
    """Explicit routing-only first-order continuation of a model FEM field.

    Native FEM evaluation remains on the original boundary. This approximation
    supplies directions/near-boundary descent checks, not support layer scalars.
    """
    def __init__(self, native, mesh, query=None):
        self.native = native
        self.mesh = mesh
        self.query=CachedSurfaceQuery(mesh) if query is None else query

    def evaluate_scalars_and_gradients(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        closest = self.query(points)[0]
        values, gradients = self.native.evaluate_scalars_and_gradients(closest)
        return values + np.einsum('ij,ij->i', gradients, points-closest), gradients


def main(pipeline, output, argv=None, clean=False, field_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=float, default=.6)
    parser.add_argument('--angle', type=float, default=45)
    parser.add_argument('--merge-distance', type=float, default=6)
    parser.add_argument('--platform-downward-bias', type=float, default=.5)
    parser.add_argument('--prefer-platform', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--contact-gradient-length', type=float, default=3)
    parser.add_argument('--contact-transition-length', type=float, default=3)
    parser.add_argument('--max-steps', type=int, default=450)
    parser.add_argument('--clearance', type=float, default=.3)
    parser.add_argument('--contact-spacing', type=float, default=2)
    parser.add_argument('--maximum-contacts', type=int, default=5000)
    parser.add_argument('--overhang-angle', type=float, default=60)
    parser.add_argument('--minimum-support-height', type=float, default=.6)
    parser.add_argument('--near-band', type=float, default=5)
    parser.add_argument('--vertical-distance', type=float, default=7)
    parser.add_argument('--landing-min-up-dot', type=float, default=.15)
    parser.add_argument('--root-contact-radius', type=float, default=2.5)
    parser.add_argument('--root-min-contact-radius', type=float, default=.6)
    parser.add_argument('--plate', type=float, default=0)
    parser.add_argument('--clearance-escape-band', type=float, default=1.5)
    parser.add_argument('--clearance-escape-weight', type=float, default=.45)
    parser.add_argument('--reroute-stalled', action='store_true')
    parser.add_argument('--reroute-search', choices=('local','wide'), default='local')
    parser.add_argument('--reroute-radius', type=float, default=1.5)
    parser.add_argument('--reroute-time-budget', type=float, default=5)
    parser.add_argument('--reroute-source', type=Path, help='Recover a saved direct-model graph into a separate output directory.')
    args = parser.parse_args(argv)
    if not all(np.isfinite(v) and v >= 0 for v in (args.contact_gradient_length, args.contact_transition_length)):
        parser.error('Contact stage lengths must be finite and nonnegative.')
    numerical = np.array([value for value in vars(args).values() if isinstance(value, (int,float))], dtype=float)
    if not np.isfinite(numerical).all():
        parser.error('All planner settings must be finite.')
    if min(args.step, args.merge_distance, args.contact_spacing, args.maximum_contacts, args.max_steps) <= 0:
        parser.error('Step, merge distance, contact spacing and limits must be positive.')
    if args.clearance < 0 or args.minimum_support_height < 0 or not 0 < args.angle < 90 or not 0 < args.overhang_angle < 90 or not np.isfinite(args.platform_downward_bias) or not 0 <= args.platform_downward_bias <= 1:
        parser.error('Invalid clearance, minimum height or angle.')
    if args.clearance_escape_band < 0 or not 0 <= args.clearance_escape_weight <= 1:
        parser.error('Escape band must be nonnegative and weight between zero and one.')
    if min(args.reroute_radius, args.reroute_time_budget) <= 0:
        parser.error('Recovery radius and time budget must be positive.')
    inputs = pipeline['input']
    output = Path(output).resolve()
    for name in ('mesh', 'tetrahedral_volume', 'checkpoint', 'normalization_mesh'):
        if inputs.get(name):
            source = Path(inputs[name]).resolve()
            if source == output or output in source.parents:
                raise ValueError('Planner output must not contain an input file.')
    from SupportGenerationVoxel.model_field import TetrahedralVertexField, TrustedModelField, load_closed_mesh
    from .config import NETWORK_DEFAULTS
    started = time.perf_counter()
    mesh = load_closed_mesh(inputs['mesh'])
    surface_query=CachedSurfaceQuery(mesh)
    if mesh.n_open_edges or not np.isfinite(mesh.points).all():
        raise ValueError('Skeleton planning requires a finite closed model surface.')
    if mesh.bounds[2] < args.plate-1e-6:
        raise ValueError('The model extends below the specified platform; choose the correct platform height.')
    mode = inputs.get('field_mode', 'checkpoint')
    if mode == 'tetrahedral':
        native = TetrahedralVertexField(inputs['tetrahedral_volume'], inputs.get('tetrahedral_scalar', 'scalar'))
        # Reuse the established matching-surface/volume checks.
        from .compatible_export import model_volume
        model_volume(inputs['mesh'], mesh, inputs['tetrahedral_volume'])
        routing_native = BoundaryTaylorField(native, mesh,query=surface_query)
    elif mode == 'checkpoint':
        normal = load_closed_mesh(inputs.get('normalization_mesh') or inputs['mesh'])
        native = TrustedModelField(inputs['checkpoint'], normal.points, {**NETWORK_DEFAULTS, **pipeline.get('network', {})})
        routing_native = native
    else:
        raise ValueError('Direct skeleton planning supports tetrahedral or checkpoint fields.')
    implicit = vtkImplicitPolyDataDistance()
    implicit.SetInput(mesh)
    from .tree_geometry import sample_mesh_distance, configure_geometry_threads
    configure_geometry_threads(1)
    sampler = vtkSampleImplicitFunctionFilter()
    sampler.SetImplicitFunction(implicit)
    sampler.SetComputeGradients(False)
    distance_input = pv.PolyData()
    def distance(points):
        return sample_mesh_distance(implicit, sampler, points, distance_input).astype(float)
    def query(points):
        return surface_query(points)
    field = (field_factory or ContactDirectionGuide)(routing_native, distance, lambda p: query(p)[0],
                                 [0,1,0], args.near_band, args.vertical_distance,
                                 # A nearest-boundary Taylor continuation is
                                 # discontinuous across FEM cells. Use its
                                 # directions, not scalar jumps, for routing.
                                 scalar_band=0.)
    faces = np.asarray(mesh.regular_faces)
    triangles = np.asarray(mesh.points)[faces]
    centers = triangles.mean(axis=1)
    normals = np.asarray(mesh.cell_normals)
    _, gradients = native.evaluate_scalars_and_gradients(centers)
    norms = np.linalg.norm(gradients, axis=1)
    dot = np.einsum('ij,ij->i', normals, gradients) / np.maximum(norms, 1e-15)
    candidate_ids = np.flatnonzero((norms > 1e-8) & np.isfinite(dot) &
        (dot < -np.cos(np.deg2rad(args.overhang_angle))) &
        (centers[:,1] > args.plate + args.minimum_support_height + args.clearance))
    area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)
    landing = ModelLandingCheck(query, normals, field, distance, gap=args.clearance,
        min_up_dot=args.landing_min_up_dot, contact_radius=args.root_contact_radius,
        minimum_contact_radius=args.root_min_contact_radius)
    if getattr(field,'has_scalar_grid',False):
        from .platform_router.routing_field import make_landing
        landing=make_landing(field,distance,vars(args))
    def protected_validator(seeds, contact_gradients):
        end = args.contact_gradient_length + args.contact_transition_length
        if end <= 0:
            return np.ones(len(seeds), dtype=bool)
        trial, _ = grow(seeds, field, distance, [0,1,0], args.plate, gap=args.clearance,
            step=args.step, angle_degrees=args.angle, merge_distance=args.merge_distance,
            max_steps=int(np.ceil(end/args.step))+2,
            platform_downward_bias=args.platform_downward_bias, prefer_platform=args.prefer_platform,
            contact_gradient_length=args.contact_gradient_length,
            contact_transition_length=args.contact_transition_length, contact_gradients=contact_gradients,
            clearance_escape_band=args.clearance_escape_band,
            clearance_escape_weight=args.clearance_escape_weight, landing_check=landing)
        parents = {int(a):int(b) for a,b in trial.lines.reshape(-1,3)[:,1:]}
        valid = []; rooted = []
        for seed in range(len(seeds)):
            node = seed
            while node in parents:
                node = parents[node]
            rooted.append(bool(trial['is_root'][node]))
            valid.append(bool(trial['is_root'][node] or trial['contact_path_length'][node] >= end-1e-8))
        return np.asarray(valid), np.asarray(rooted)

    # Select a feasible protected neck before growing. Every occupied
    # overhang bucket remains represented, including stalled candidates for inspection.
    from .contact_selection import select_contacts
    chosen, contact_points, contact_gradients, contact_selection = select_contacts(
        candidate_ids, centers, normals, gradients, area, field, distance,
        spacing=args.contact_spacing, gap=args.clearance,
        fixed_length=args.contact_gradient_length, angle=args.angle,
        plate=args.plate, maximum_contacts=args.maximum_contacts,
        triangles=triangles,
        gradient_evaluator=lambda p: native.evaluate_scalars_and_gradients(p)[1],
        overhang_angle=args.overhang_angle, protected_validator=protected_validator, retain_stalled=True)
    seeds = contact_points + (args.clearance+1e-5) * normals[chosen]
    accepted = np.ones(len(seeds), dtype=bool)
    if not len(seeds):
        raise ValueError('No usable overhang contacts above the platform. Check the scalar direction, overhang angle and minimum support height.')
    print(f'Growing {len(seeds)} contacts directly from model; no preserved tree required.', flush=True)
    if args.reroute_source is None:
        print(f'Contact stages: fixed contact-gradient {args.contact_gradient_length:g} mm; transition {args.contact_transition_length:g} mm; '
              f'merging after {args.contact_gradient_length+args.contact_transition_length:g} mm of branch travel.',flush=True)
    if args.reroute_source is None:
        skeleton, report = grow(seeds, field, distance, [0,1,0], args.plate, gap=args.clearance,
            step=args.step, angle_degrees=args.angle, merge_distance=args.merge_distance, max_steps=args.max_steps,
            platform_downward_bias=args.platform_downward_bias,
            prefer_platform=args.prefer_platform,
            contact_gradient_length=args.contact_gradient_length, contact_transition_length=args.contact_transition_length,
            contact_gradients=contact_gradients,
            clearance_escape_band=args.clearance_escape_band, clearance_escape_weight=args.clearance_escape_weight,
            landing_check=landing, progress=lambda status: print(json.dumps({'stage':'growth_before_recovery', **status}), flush=True))
    else:
        source = args.reroute_source.resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('Recovery output must be separate from the original saved graph.')
        previous = json.loads((source/'summary.json').read_text(encoding='utf-8'))
        if Path(previous['mesh']).resolve() != Path(inputs['mesh']).resolve():
            raise ValueError('Recovery graph belongs to a different model.')
        report = json.loads((source/'module_report.json').read_text(encoding='utf-8'))
        if report.get('routing_mode') != 'near-model' or abs(report['clearance_mm']-args.clearance)>1e-8:
            raise ValueError('Recovery graph must use near-model routing and the same clearance.')
        for key in ('near_band','vertical_distance','plate'):
            saved = previous.get('arguments', {}).get(key)
            if saved is not None and not np.isclose(saved, getattr(args,key), rtol=0, atol=1e-8):
                raise ValueError(f'Recovery must retain the saved {key} setting.')
        skeleton = pv.read(source/'support_centerlines.vtp')
        report['recovered_from'] = str(source)
        args.angle = float(report['local_cone_angle_degrees'])
        args.contact_gradient_length=float(report.get('contact_gradient_length_mm',0.))
        args.contact_transition_length=float(report.get('contact_transition_length_mm',0.))
        args.reroute_stalled = True
        # Warm up the field outside the bounded incremental recovery budget.
        field(skeleton.points)
    if args.reroute_stalled and report['unrooted_contacts']:
        from .stalled_recovery import recover_stalled
        skeleton, report = recover_stalled(skeleton, report, field, distance, landing, [0,1,0], args.plate,
            radius=args.reroute_radius, time_budget=args.reroute_time_budget, search_mode=args.reroute_search,
            prefer_platform=args.prefer_platform)
    report['recovery_enabled'] = bool(args.reroute_stalled)
    report['model_lift_review_recommended'] = bool(report['unrooted_contacts'])
    warnings = [item for item in report.get('warnings', []) if item.get('code') != 'unrooted_contacts_require_model_lift_review']
    if report['unrooted_contacts']:
        warnings.append(dict(code='unrooted_contacts_require_model_lift_review', contacts=report['unrooted_contacts'],
            message='Required contacts remain unrooted after the requested growth/recovery.',
            action='Review failed patches and recovery diagnostics before accepting the skeleton.'))
    report['warnings'] = warnings
    if report.get('stalled_in_contact_stage'):
        warnings.append(dict(code='blocked_contact_stage',nodes=report['stalled_in_contact_stage'],
            message='Growth stopped inside the initial fixed-gradient or transition section. Review clearance, descent, local-cone compatibility and the growth-step limit. Recovery will not bypass the protected contact section.'))
        print(f'WARNING: {len(report["stalled_in_contact_stage"])} fronts stopped inside the protected contact stages; '
              'inspect clearance, descent, field compatibility and the growth-step limit. The prescribed neck direction was retained.',flush=True)
    radii = np.asarray(skeleton.point_data.get('root_contact_radius', np.zeros(skeleton.n_points))).copy()
    for index, point in enumerate(skeleton.points):
        radii[index] = landing.contact_radii.get(np.asarray(point, dtype=float).tobytes(), radii[index])
    skeleton.point_data['root_contact_radius'] = radii
    report.update(routing_mode='near-model', field_mode=mode, source='model_surface_contacts',
        near_band_mm=args.near_band, vertical_distance_mm=args.vertical_distance,
        native_descent_band_mm=field.scalar_band,
        exterior_descent_policy='sampled_build_height_and_local_direction_cone',
        selected_overhang_faces=len(candidate_ids), contact_bucket_count=len(chosen),
        contact_selection=contact_selection,
        rejected_seed_clearance_count=int((~accepted).sum()), contact_spacing_mm=args.contact_spacing,
        model_query_backend='vtk', unique_field_points=field.evaluated, total_seconds=time.perf_counter()-started,
        native_direction_source='Boundary Taylor routing approximation' if mode=='tetrahedral' else 'Neural near-model queries')
    report['limitations'] += ['Face-centre contact sampling does not certify complete overhang coverage.',
        'Platform routes are preferred where feasible; accepted model attachments remain available when downward routing is blocked.',
        'Tetrahedral routing uses explicit boundary Taylor directions; final support layers need reconstructed-field validation.']
    output = prepare_output(output, clean=clean)
    skeleton.save(output/'support_centerlines.vtp')
    pv.PolyData(seeds).save(output/'contacts.vtp')
    (output/'module_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    arguments = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    summary = dict(mesh=str(Path(inputs['mesh']).resolve()), field_mode=mode, input=inputs,
        envelope={'build_direction':[0,1,0]}, arguments=arguments, skeleton_experiment=report)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output/'model_input.json').write_text(json.dumps(pipeline, indent=2), encoding='utf-8')
    print(f'FINAL SKELETON: {report["rooted_contacts"]}/{report["seed_count"]} contacts rooted; '
          f'stalled_fronts={report["stalled_fronts"]}; unrooted_contacts={report["unrooted_contacts"]}. '
          f'Saved: {output}', flush=True)
    return report

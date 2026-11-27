"""Bounded recovery of suspicious extrusion heights, without changing paths.

The expensive search is gated on gap-corrected bead height. A platform fallback
is also checked against its *uncapped* distance so the fallback cap cannot hide
an unsupported floating region. Scalar increments are never physical heights.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import time

import igl
import numpy as np

from contact_support_height import _triangle_ray_distances
from extrusion_height import apply_axis_lift
from mesh_distance_workflow import load_obj_mesh
from reference_geometry_cache import CostAwareLRU,content_digest


def resolve_height_settings(config):
    """Keep old configs compatible; new configs explicitly supply physical h."""
    result = dict(config)
    validation = dict(result.get('support_validation', {}))
    nominal = result.get('nominal_layer_height')
    if nominal is not None:
        nominal = float(nominal)
        if not np.isfinite(nominal) or nominal <= 0:
            raise ValueError('nominal_layer_height must be a positive physical thickness in mm.')
        for name, default in (('platform_height_factor', 1.5), ('nearest_direct_factor', 1.5)):
            factor = float(result.get(name, default))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f'{name} must be finite and positive.')
            result[name] = factor
        if result.get('max_height') is None:
            result['max_height'] = nominal * result['platform_height_factor']
        if result.get('nearest_direct_height') is None:
            result['nearest_direct_height'] = nominal * result['nearest_direct_factor']
    if validation.get('enabled', False):
        # Component exports can exceed the old 128-entry ceiling while using
        # only a fraction of the retained-geometry byte budget. Explicit saved
        # budgets still win; this is not a process-RSS or whole-stage cap.
        validation.setdefault('reference_cache_max_entries',512)
        if 'reference_cache_max_bytes' not in validation:
            from height_cache_budget import available_memory_bytes, reference_budget
            available = available_memory_bytes()
            workers = int(result.get('layer_workers', 1))
            validation['reference_cache_max_bytes'] = reference_budget(workers, available)
            validation['reference_cache_budget_policy'] = dict(
                mode='available_memory', workers=workers, available_memory_bytes=available,
                aggregate_available_fraction=.25, per_worker_cap_bytes=4096*1024*1024,
                scope='Estimated retained geometry; not process RSS.')
        for name in ('reference_cache_max_entries','reference_cache_max_bytes'):
            value=validation[name]
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(f'support_validation.{name} must be a positive integer.')
        if nominal is None:
            raise ValueError('Support height validation requires nominal_layer_height in physical mm.')
        for name, default in (('check_height_factor', 1.75), ('max_bead_height_factor', 1.5)):
            factor = float(validation.get(name, default))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f'support_validation.{name} must be finite and positive.')
            validation[name] = factor
        # Retire the old angular rejection gate, including saved configs that
        # still contain it. The actual/cap angle below is an independent policy.
        validation.pop('max_support_angle_degrees', None)
        miss_angle = float(validation.get('normal_miss_tolerance_degrees', 3.))
        if not np.isfinite(miss_angle) or not 0 <= miss_angle <= 90:
            raise ValueError('normal_miss_tolerance_degrees must lie between 0 and 90.')
        validation.update(nominal_layer_height=nominal,
            check_height=nominal * validation['check_height_factor'],
            max_bead_height=nominal * validation['max_bead_height_factor'],
            normal_miss_tolerance_degrees=miss_angle)
        validation['limit_interface_reach'] = bool(validation.get('limit_interface_reach', True))
        support_limit=float(validation.get('max_verified_support_height',.5))
        if not np.isfinite(support_limit) or support_limit<=0:
            raise ValueError('support_validation.max_verified_support_height must be finite and positive.')
        validation['max_verified_support_height']=support_limit
        outlier_policy = validation.get('height_outlier_policy', 'require_refinement')
        if outlier_policy not in ('clamp_warn', 'require_refinement'):
            raise ValueError('support_validation.height_outlier_policy must be clamp_warn or require_refinement.')
        validation['height_outlier_policy'] = outlier_policy
        if validation['limit_interface_reach']:
            contact = dict(result.get('contact_support_height', {}))
            contact['maximum_bead_height'] = validation['max_bead_height']
            result['contact_support_height'] = contact
        policy=validation.get('ambiguous_height_policy','threshold')
        if policy not in ('threshold','error','nominal_warn','threshold_warn'):
            raise ValueError('support_validation.ambiguous_height_policy must be threshold, error, nominal_warn or threshold_warn.')
        validation['ambiguous_height_policy']=policy
        if policy in ('nominal_warn','threshold_warn'):
            validation['ambiguous_height_acceptance']='bounded_fallback_v1'
            validation['ambiguous_fallback_height_mm']=nominal if policy=='nominal_warn' else validation['max_bead_height']
    result['support_validation'] = validation
    for name in ('max_height', 'nearest_direct_height'):
        value = result.get(name)
        if value is not None and (not np.isfinite(float(value)) or float(value) < 0 or (name == 'max_height' and float(value) == 0)):
            raise ValueError(f'{name} must be finite and positive, or null for automatic scaling.')
    return result


def assign_ambiguous_heights(rows, geometric_resolved, threshold, *, policy='threshold'):
    """Assign an extrusion fallback without asserting geometric support proof."""
    if policy not in ('threshold','error','nominal_warn','threshold_warn') or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Invalid ambiguous-height policy or threshold.')
    invalid_input=(~np.isfinite(rows[:,:6]).all(axis=1))|(np.linalg.norm(rows[:,3:6],axis=1)<=1e-12)
    missing_height=(~np.isfinite(rows[:,6]))|(rows[:,6]<=0)
    if policy=='threshold':
        fallback=((~geometric_resolved)|missing_height)&(~invalid_input)
    elif policy in ('nominal_warn','threshold_warn'):
        fallback=((~geometric_resolved)|missing_height)&(~invalid_input)
    else:
        fallback=np.zeros(len(rows),bool)
    rows[fallback,6]=threshold
    invalid=invalid_input|(~np.isfinite(rows[:,6]))|(rows[:,6]<=0)
    return (geometric_resolved|fallback)&~invalid,fallback,invalid


def build_reference_catalog(meshes, paths, layers, *, lift=0., lift_axis='y', bounds_mesh_dir=None):
    """One parallel bounds scan per stage; workers receive only compact records."""
    lookup = {(int(r['layer_main']), int(r['layer_sub'])): r for r in layers}
    bounds_parent = Path(bounds_mesh_dir).absolute() if bounds_mesh_dir else None
    def prepare(item):
        key, path = item
        layer = lookup.get((key.main, key.sub), {})
        metadata = {name: layer[name] for name in ('part', 'scalar', 'is_contact_layer', 'is_platform_layer',
            'is_planar_base_layer', 'is_model_base_layer', 'upstream_metadata') if name in layer}
        if key not in paths or paths[key].stat().st_size == 0 or layer.get('path_point_count') == 0:
            return None
        stored = layer.get('source_mesh_bounds')
        if stored is not None and bounds_parent == path.parent.absolute():
            geometry = np.asarray(stored, dtype=float).reshape(2, 3)
        else:
            # Scan face records in C instead of iterating millions of irrelevant
            # face lines in Python. Accept optional OBJ vertex colours/weights.
            geometry = b' '.join(re.findall(rb'(?m)^v[ \t]+(\S+[ \t]+\S+[ \t]+\S+)', path.read_bytes()))
        return key, path, metadata, geometry
    items = sorted(meshes.items(), key=lambda item: (item[0].main, item[0].sub))
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = []
        # Bound read-ahead memory and convert in the calling thread. Concurrent
        # NumPy ASCII conversion on Windows can be far slower than serial work.
        for start in range(0, len(items), 16):
            for row in pool.map(prepare, items[start:start+16]):
                if row is None: continue
                key, path, metadata, geometry = row
                vertices = (np.fromstring(geometry.decode('ascii'), sep=' ').reshape(-1, 3)
                            if isinstance(geometry, bytes) else geometry)
                if not len(vertices) or not np.isfinite(vertices).all():
                    raise ValueError(f'Invalid reference bounds: {path}')
                vertices = apply_axis_lift(vertices, axis=lift_axis, amount=lift)
                result.append(dict(main=key.main, sub=key.sub, path=str(path), metadata=metadata,
                    lower=vertices.min(axis=0).tolist(), upper=vertices.max(axis=0).tolist()))
            if (start+16) % 256 == 0:
                print(f'[height bounds] {min(start+16,len(items))}/{len(meshes)} surfaces indexed', flush=True)
        return result


class MaterialSolid:
    """Cached original-solid containment and segment-boundary acceleration."""
    def __init__(self, path, *, lift=0., lift_axis='y'):
        import pyvista as pv
        mesh = pv.read(path)
        surface = (mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface(algorithm='dataset_surface')).triangulate()
        surface = surface.clean()
        if not surface.n_cells or surface.n_open_edges:
            raise ValueError(f'Material-continuity input must be a closed solid: {path}')
        self.vertices = apply_axis_lift(np.asarray(surface.points, dtype=np.float64), axis=lift_axis, amount=lift)
        self.faces = np.asarray(surface.faces.reshape(-1, 4)[:, 1:], dtype=np.int64)
        self.tree = igl.AABB(); self.tree.init(self.vertices, self.faces)
        surface.points = self.vertices
        self.surface = surface

    def _contains(self, points):
        """Deterministic containment, independent of query order and density.

        VTK's per-point enclosed-surface helper uses randomized rays and can
        disagree on an identical interior point after other queries. Winding
        numbers remove that state; exact evaluation resolves the approximate
        method's classification band. Absolute values allow reversed winding.
        Boundary segments still require the explicit triangle proof below.
        """
        winding = np.abs(igl.fast_winding_number(self.vertices, self.faces, points))
        uncertain = (winding >= .4) & (winding <= .6)
        if np.any(uncertain):
            winding[uncertain] = np.abs(igl.winding_number(self.vertices, self.faces, points[uncertain]))
        return winding > .5 + 1e-9

    def _same_surface_triangle(self, starts, ends, tolerance):
        """Prove a boundary segment using one convex source triangle.

        Enclosed-point classification is ambiguous on the surface. A segment
        whose two endpoints project inside the same original triangle stays
        on that triangle, within the existing numerical distance tolerance.
        Merely being near the surface at the midpoint is not sufficient.
        """
        accepted=np.zeros(len(starts),dtype=bool)
        if not len(starts):return accepted
        squared,faces,_=self.tree.squared_distance(self.vertices,self.faces,(starts+ends)*.5)
        faces=np.asarray(faces).reshape(-1)
        ids=np.flatnonzero((np.asarray(squared).reshape(-1)<=tolerance*tolerance)&
            (faces>=0)&(faces<len(self.faces)))
        if not len(ids):return accepted
        triangles=self.vertices[self.faces[faces[ids]]]
        a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a
        normal=np.cross(u,v);norm=np.linalg.norm(normal,axis=1)
        uu=np.einsum('ij,ij->i',u,u);uv=np.einsum('ij,ij->i',u,v);vv=np.einsum('ij,ij->i',v,v)
        determinant=uu*vv-uv*uv
        valid=(norm>0)&(determinant>0)
        for points in (starts[ids],ends[ids]):
            delta=points-a;du=np.einsum('ij,ij->i',delta,u);dv=np.einsum('ij,ij->i',delta,v)
            b=np.divide(vv*du-uv*dv,determinant,out=np.zeros(len(ids)),where=determinant>0)
            c=np.divide(uu*dv-uv*du,determinant,out=np.zeros(len(ids)),where=determinant>0)
            plane=np.abs(np.einsum('ij,ij->i',delta,normal))
            valid&=(plane<=tolerance*norm)&(b>=-1e-9)&(c>=-1e-9)&(b+c<=1+1e-9)
        accepted[ids]=valid
        return accepted

    def continuous(self, starts, ends, tolerance):
        vectors = ends - starts
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(lengths[:, None], 1e-30)
        epsilon = np.minimum(tolerance, lengths * .1)
        origins = starts + epsilon[:, None] * directions
        hits, _, _ = self.tree.intersect_ray_first(self.vertices, self.faces, origins, directions)
        hits = np.asarray(hits).reshape(-1)
        crossed = np.zeros(len(starts), dtype=bool)
        selected = np.flatnonzero((hits >= 0) & (hits < len(self.faces)))
        if len(selected):
            distances = _triangle_ray_distances(origins[selected], directions[selected], self.vertices[self.faces[hits[selected]]])
            crossed[selected] = distances < lengths[selected] - 2 * epsilon[selected]
        middle = (starts + ends) * .5
        contained = self._contains(middle)
        accepted=~crossed & contained
        rejected=np.flatnonzero(~accepted)
        if len(rejected):
            accepted[rejected]=self._same_surface_triangle(starts[rejected],ends[rejected],tolerance)
        return accepted


class HeightValidator:
    def __init__(self, config, catalog, *, contact_config=None, platform_value=0.,
                 max_platform_height=None, lift=0., lift_axis='y'):
        self.config = config; self.catalog = catalog
        self.contact = contact_config or {}; self.platform = platform_value
        self.platform_cap = max_platform_height
        self.lift, self.lift_axis = lift, lift_axis
        self.cache = CostAwareLRU(config.get('reference_cache_max_entries',512),
                                  config.get('reference_cache_max_bytes',256*1024*1024))
        self.solid = None
        self.cache_stats = dict(recent_reference_reuses=0, older_reference_cache_hits=0,
            older_reference_mesh_loads=0, older_reference_cache_evictions=0,
            older_reference_load_seconds=0., older_reference_tree_build_seconds=0.,
            older_reference_content_hash_seconds=0.,older_reference_cache_peak_estimated_bytes=0)
        self.lower = np.asarray([r['lower'] for r in catalog]).reshape(-1, 3)
        self.upper = np.asarray([r['upper'] for r in catalog]).reshape(-1, 3)
        self.main = np.asarray([r['main'] for r in catalog]); self.sub = np.asarray([r['sub'] for r in catalog])
        self.tolerance = max(1e-6, config['nominal_layer_height'] * 1e-4)

    def _gap(self, source, reference):
        if not self.contact.get('enabled', False):
            return 0.
        model_on_support = source.get('part', 'model') == 'model' and reference.get('part') == 'support' and reference.get('is_contact_layer')
        support_on_model = (source.get('part') == 'support' and source.get('is_model_base_layer')
            and not source.get('is_platform_layer') and reference.get('part', 'model') == 'model')
        if not (model_on_support or support_on_model):
            return 0.
        owner = source if support_on_model else reference
        field = self.contact.get('gap_metadata_field')
        value = owner.get('upstream_metadata', {}).get(field, self.contact.get('gap', 0.)) if field else self.contact.get('gap', 0.)
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError('Contact gap must be finite and nonnegative.')
        return value

    def _reference(self, index, recent):
        record = self.catalog[index]; identity = (record['main'], record['sub'])
        if identity in recent:
            self.cache_stats['recent_reference_reuses'] += 1
            return recent[identity]
        began=time.perf_counter()
        version=content_digest(record['path'])
        self.cache_stats['older_reference_content_hash_seconds']+=time.perf_counter()-began
        before_hits,before_evictions=self.cache.hits,self.cache.evictions
        def load():
            began=time.perf_counter()
            vertices, faces = load_obj_mesh(record['path'])
            vertices = apply_axis_lift(vertices, axis=self.lift_axis, amount=self.lift)
            self.cache_stats['older_reference_mesh_loads'] += 1
            self.cache_stats['older_reference_load_seconds'] += time.perf_counter()-began
            began=time.perf_counter()
            tree = igl.AABB(); tree.init(vertices, faces)
            self.cache_stats['older_reference_tree_build_seconds'] += time.perf_counter()-began
            vertices.flags.writeable=False;faces.flags.writeable=False
            cost=int(vertices.nbytes+faces.nbytes+128*len(faces))
            return (vertices,faces,dict(record['metadata']),tree),cost
        key=(str(Path(record['path']).resolve()),self.lift_axis,float(self.lift),index,
             json.dumps(record['metadata'],sort_keys=True))
        result=self.cache.get(key,version,load)
        self.cache_stats['older_reference_cache_hits']+=self.cache.hits-before_hits
        self.cache_stats['older_reference_cache_evictions']+=self.cache.evictions-before_evictions
        self.cache_stats['older_reference_cache_peak_estimated_bytes']=self.cache.peak_bytes
        return result

    def validate(self, rows, *, source, key, recent, platform_mask, output_path):
        began = time.perf_counter(); config = self.config
        before = dict(self.cache_stats)
        report = dict(enabled=True, checked_points=0, recovered_points=0, unresolved_points=0,
            threshold_fallback_points=0,unverified_support_points=0,invalid_height_points=0,
            geometrically_verified_points=0,ambiguous_height_policy=config.get('ambiguous_height_policy','threshold'),
            normal_miss_fallback_points=0, clamped_height_points=0,
            normal_miss_actual_points=0, normal_miss_cap_points=0,
            uncapped_model_normal_points=0, above_cap_model_normal_points=0,
            verified_support_normal_points=0,support_normal_over_limit_points=0,
            support_normal_over_limit_clamped_points=0,
            height_outlier_policy=config.get('height_outlier_policy','require_refinement'),
            normal_miss_tolerance_degrees=config.get('normal_miss_tolerance_degrees', 3.),
            continuity_available=bool(config.get('model_solid')), reference_mesh_queries=0,
            check_height_mm=config['check_height'], max_bead_height_mm=config['max_bead_height'])
        report.update(nominal_layer_height=config['nominal_layer_height'],
            ambiguous_height_acceptance=config.get('ambiguous_height_acceptance',''),
            ambiguous_fallback_height_mm=config.get('ambiguous_fallback_height_mm',config['max_bead_height']),
            ambiguous_fallback_accepted_points=0)
        report.update({name:0 for name in self.cache_stats})
        if source.get('is_planar_base_layer') or not len(rows):
            report['seconds'] = time.perf_counter() - began
            return report
        raw_platform = np.maximum(0., rows[:, 1] - self.platform)
        # A general cap needs a normal/model classification even between the
        # cap and the previous expensive-check trigger; otherwise that exception
        # would depend on whether the old estimate happened to exceed 1.75h.
        effective_check = min(config['check_height'], config['max_bead_height'])
        report['effective_check_height_mm'] = effective_check
        suspect = (~np.isfinite(rows[:, 6])) | (rows[:,6]<=0) | (rows[:, 6] > effective_check + self.tolerance)
        suspect |= (~np.isfinite(rows[:,:6]).all(axis=1)) | (np.linalg.norm(rows[:,3:6],axis=1)<=1e-12)
        # Never use the capped .3 mm surrogate as proof of bed contact.
        suspect |= platform_mask & (raw_platform > (self.platform_cap or config['max_bead_height']) + self.tolerance)
        ids = np.flatnonzero(suspect); report['checked_points'] = len(ids)
        if not len(ids):
            report['seconds'] = time.perf_counter() - began
            return report
        points = rows[ids, :3]; normals = rows[ids, 3:6].copy()
        norm = np.linalg.norm(normals, axis=1)
        valid_normals = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (norm > 1e-12)
        normals /= np.maximum(norm[:, None], 1e-30); normals[normals[:, 1] < 0] *= -1
        directions = -normals
        gaps = np.array([self._gap(source, r['metadata']) for r in self.catalog])
        # Bead-height threshold is now a final extrusion cap, not a geometric
        # rejection distance. Keep the existing absolute contact-query bound.
        query_distance = float(self.contact.get('max_query_distance', 2.))
        reaches = np.full(len(self.catalog), query_distance)
        if config.get('limit_interface_reach', True):
            interface = gaps > 0
            reaches[interface] = np.minimum(reaches[interface], gaps[interface] + config['max_bead_height'])
        earlier = (self.main < key.main) | ((self.main == key.main) & (self.sub < key.sub))
        if source.get('part') == 'model':
            earlier &= np.array([not (r['main'] == key.main and r['metadata'].get('part') == 'model')
                                 for r in self.catalog], dtype=bool)
        if source.get('part') == 'support' and source.get('scalar') is not None:
            earlier &= np.array([not (r['metadata'].get('part') == 'support' and
                abs(float(source['scalar']) - float(r['metadata'].get('scalar', np.inf))) <= 1e-9) for r in self.catalog], dtype=bool)
        search_reach = max(reaches.max(initial=config['max_bead_height']), config['max_bead_height'])
        overlap=np.zeros(len(self.catalog),bool)
        if valid_normals.any():
            lo,hi=points[valid_normals].min(axis=0),points[valid_normals].max(axis=0)
            overlap = np.all(self.upper >= lo - search_reach, axis=1) & np.all(self.lower <= hi + search_reach, axis=1)
        recent_model = np.array([(r['main'],r['sub']) in recent and r['metadata'].get('part','model')=='model'
            for r in self.catalog],dtype=bool)
        recent_model &= gaps == 0
        candidates = np.flatnonzero(earlier & (overlap | recent_model))
        best = np.full(len(ids), np.inf); chosen = np.full(len(ids), -1, dtype=int)
        endpoint = np.zeros_like(points); candidate_rows = []; first_ray = np.full(len(ids), np.inf)
        chosen_fallback = np.zeros(len(ids), dtype=bool)
        chosen_cap = np.zeros(len(ids), dtype=bool)
        chosen_model_normal = np.zeros(len(ids), dtype=bool)
        chosen_support_normal = np.zeros(len(ids), dtype=bool)
        first_same_part = np.full(len(ids), np.inf)
        for index in candidates[::-1]:
            delta = np.maximum(np.maximum(self.lower[index] - points, points - self.upper[index]), 0.)
            selected = np.flatnonzero((np.linalg.norm(delta, axis=1) <= search_reach + self.tolerance) & valid_normals)
            if recent_model[index]:
                # Preserve a genuine normal hit on a recent model layer even
                # when its actual thickness exceeds the fallback search radius.
                selected = np.flatnonzero(valid_normals)
            if not len(selected):
                continue
            vertices, faces, metadata, tree = self._reference(index, recent)
            report['reference_mesh_queries'] += 1
            squared, nearest_faces, closest = tree.squared_distance(vertices, faces, points[selected])
            distance = np.sqrt(np.maximum(np.asarray(squared).reshape(-1), 0.))
            closest = np.asarray(closest)
            along = np.einsum('ij,ij->i', closest - points[selected], directions[selected])
            # A reference must lie on the supporting side of the local layer.
            # There is no 45-degree rejection cone for height candidates.
            valid = (distance > self.tolerance) & (along > self.tolerance) & (distance <= reaches[index] + self.tolerance)
            candidate_rows.append((index, selected[valid], distance[valid], closest[valid], False))
            hits, _, _ = tree.intersect_ray_first(vertices, faces, points[selected] + self.tolerance * directions[selected], directions[selected])
            hits = np.asarray(hits).reshape(-1)
            usable = (hits >= 0) & (hits < len(faces))
            ray_ids = selected[usable]
            ray_distance = _triangle_ray_distances(points[ray_ids], directions[ray_ids], vertices[faces[hits[usable]]])
            # Match the existing contact routine's exact shared-edge recovery.
            missing = selected[~usable]
            if len(missing):
                face_ids = np.asarray(nearest_faces).reshape(-1)[~usable]
                good_faces = (face_ids >= 0) & (face_ids < len(faces))
                missing = missing[good_faces]
                extra = _triangle_ray_distances(points[missing], directions[missing], vertices[faces[face_ids[good_faces]]])
                ray_ids = np.concatenate((ray_ids, missing))
                ray_distance = np.concatenate((ray_distance, extra))
            first_ray[ray_ids] = np.minimum(first_ray[ray_ids], ray_distance)
            if metadata.get('part', 'model') == source.get('part', 'model'):
                first_same_part[ray_ids] = np.minimum(first_same_part[ray_ids], ray_distance)
            good = np.isfinite(ray_distance) & ((ray_distance <= reaches[index] + self.tolerance) | recent_model[index])
            candidate_rows.append((index, ray_ids[good], ray_distance[good],
                points[ray_ids[good]] + ray_distance[good, None] * directions[ray_ids[good]], True))
        normal_mode = self.contact.get('direction_mode', 'model_normal') == 'model_normal'
        miss_angle = config.get('normal_miss_tolerance_degrees', 3.)
        miss_cosine = np.cos(np.deg2rad(miss_angle))
        # Resolve true normal hits first, then ordinary nearest support. The
        # missed-normal fallback cannot replace an already valid result. Its
        # angular threshold selects actual height versus the fallback cap;
        # it is not an additional rejection cone.
        candidate_rows.sort(key=lambda item: 0 if item[4] else
            (2 if gaps[item[0]] > 0 and normal_mode else 1))
        for index, selected, distance, ends, is_ray in candidate_rows:
            gap = gaps[index]; bead = distance - gap
            metadata = self.catalog[index]['metadata']
            model_normal = (is_ray and metadata.get('part','model')=='model'
                and source.get('part','model')=='model' and gap == 0)
            support_normal=(is_ray and metadata.get('part')=='support' and source.get('part')=='support' and gap==0)
            valid = bead > self.tolerance
            valid &= distance < best[selected] - self.tolerance
            if not is_ray:
                valid &= ~(chosen_model_normal[selected] | chosen_support_normal[selected])
            is_fallback = gap > 0 and not is_ray and normal_mode
            use_cap = np.zeros(len(selected), dtype=bool)
            if is_fallback:
                along = np.einsum('ij,ij->i', ends - points[selected], directions[selected])
                valid &= miss_angle > 0
                valid &= ~np.isfinite(best[selected]) | chosen_fallback[selected]
                valid &= distance <= first_ray[selected] + self.tolerance
                use_cap = along < miss_cosine * distance - 1e-12
            if is_ray:
                valid &= distance <= first_ray[selected] + self.tolerance
            if gap > 0:
                # A contact label cannot punch through an intervening or tied
                # same-material layer and invent an intentional air gap there.
                valid &= distance < first_same_part[selected] - self.tolerance
            # A declared air gap is intentional; do not demand model material
            # through that interface. Model/model segments must stay in solid.
            if (gap == 0 and source.get('part', 'model') == 'model' and metadata.get('part', 'model') == 'model'
                    and config.get('model_solid') and np.any(valid)):
                if self.solid is None:
                    self.solid = MaterialSolid(config['model_solid'], lift=self.lift, lift_axis=self.lift_axis)
                usable = np.flatnonzero(valid)
                valid[usable] &= self.solid.continuous(points[selected[usable]], ends[usable], self.tolerance)
            replace = valid & (distance < best[selected] - self.tolerance)
            take = selected[replace]
            best[take] = distance[replace]; chosen[take] = index; endpoint[take] = ends[replace]
            chosen_fallback[take] = is_fallback
            chosen_cap[take] = use_cap[replace]
            chosen_model_normal[take] = model_normal
            chosen_support_normal[take] = support_normal
        recovered = chosen >= 0
        old = rows[ids, 6].copy()
        # Cap estimates now; after geometric recovery, assign the threshold to
        # remaining ambiguous points while preserving the lack of support proof.
        rows[ids, 6] = np.minimum(rows[ids, 6], config['max_bead_height'])
        measured = best[recovered] - gaps[chosen[recovered]]
        support_limit=config.get('max_verified_support_height',.5)
        measured_limits=np.where(chosen_support_normal[recovered],support_limit,config['max_bead_height'])
        rows[ids[recovered], 6] = np.where(chosen_cap[recovered], config['max_bead_height'],
            np.where(chosen_model_normal[recovered], measured, np.minimum(measured, measured_limits)))
        support_over_limit=np.zeros(len(ids),dtype=bool)
        support_over_limit[recovered]=chosen_support_normal[recovered] & (measured>support_limit+self.tolerance)
        report['verified_support_normal_points']=int(chosen_support_normal[recovered].sum())
        report['support_normal_over_limit_points']=int(support_over_limit.sum())
        if config.get('height_outlier_policy') == 'clamp_warn':
            report['support_normal_over_limit_clamped_points'] = int(support_over_limit.sum())
        report['max_verified_support_height_mm']=support_limit
        report['normal_miss_fallback_points'] = int(chosen_fallback[recovered].sum())
        report['normal_miss_actual_points'] = int((chosen_fallback & ~chosen_cap)[recovered].sum())
        report['normal_miss_cap_points'] = int(chosen_cap[recovered].sum())
        report['uncapped_model_normal_points'] = int(chosen_model_normal[recovered].sum())
        report['above_cap_model_normal_points'] = int((chosen_model_normal[recovered] & (measured > config['max_bead_height'])).sum())
        was_clamped = ~chosen_model_normal[recovered] & (chosen_cap[recovered] | (measured > measured_limits))
        height_clamped = old > config['max_bead_height']
        height_clamped[recovered] = was_clamped
        report['clamped_height_points'] = int(height_clamped.sum())
        candidate_height = np.full(len(ids), np.nan)
        candidate_height[recovered] = measured
        # A real short distance to the bed is valid. A distant fallback is not.
        bed = ~recovered & (raw_platform[ids] > self.tolerance) & (raw_platform[ids] <= (self.platform_cap or config['max_bead_height']) + self.tolerance)
        rows[ids[bed], 6] = raw_platform[ids[bed]]
        recovered |= bed
        support_verified=recovered.copy()
        selected_rows=rows[ids]
        resolved,fallback,invalid=assign_ambiguous_heights(selected_rows,recovered,
            config.get('ambiguous_fallback_height_mm',config['max_bead_height']),
            policy=config.get('ambiguous_height_policy','threshold'))
        rows[ids]=selected_rows
        height_clamped[fallback]=True
        report.update(recovered_points=int(resolved.sum()), unresolved_points=int((~resolved).sum()),
            threshold_fallback_points=int(fallback.sum()),unverified_support_points=int((~support_verified).sum()),
            invalid_height_points=int(invalid.sum()),geometrically_verified_points=int(support_verified.sum()),
            ambiguous_fallback_accepted_points=int((fallback & ~support_verified & ~invalid).sum())
                if config.get('ambiguous_height_policy') in ('nominal_warn','threshold_warn') else 0,
            clamped_height_points=int(height_clamped.sum()),
            unresolved_classification='unsupported_or_bridge' if config.get('ambiguous_height_policy')=='error' else 'invalid_height',
            older_reference_layers=sorted({f'{self.catalog[i]["main"]}_{self.catalog[i]["sub"]}' for i in chosen if i >= 0}),
            seconds=time.perf_counter() - began)
        report.update({name:value-before[name] for name,value in self.cache_stats.items()})
        diagnostic = output_path.with_name(output_path.stem + '.height_validation.npz')
        np.savez_compressed(diagnostic, row_indices=ids, points=points, original_height=old,
            corrected_height=rows[ids, 6], resolved=resolved,support_verified=support_verified,threshold_fallback=fallback,
            reference_main=np.array([self.catalog[i]['main'] if i >= 0 else -1 for i in chosen]),
            reference_sub=np.array([self.catalog[i]['sub'] if i >= 0 else -1 for i in chosen]), support_points=endpoint,
            platform_fallback=platform_mask[ids], uncapped_platform_distance=raw_platform[ids],
            normal_miss_fallback=chosen_fallback, normal_miss_cap=chosen_cap, height_clamped=height_clamped,
            candidate_bead_height=candidate_height, uncapped_model_normal=chosen_model_normal,
            verified_support_normal=chosen_support_normal,support_normal_over_limit=support_over_limit)
        report['diagnostic_file'] = diagnostic.name
        return report

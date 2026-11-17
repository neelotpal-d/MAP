"""Preserved-tree geometry and optional experimental model-rooted forest planning."""
from pathlib import Path
import json
import sys
import time
import multiprocessing
import os
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "SupportGenerationCurved"
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance, vtkClipPolyData
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
from vtkmodules.vtkCommonCore import vtkSMPTools
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from .array_uniqueness import unique_rows_inverse

_BATCH_LAYER_ID='__support_batch_layer_id'


@contextmanager
def worker_blas_environment(threads=1):
    """Bound inherited numerical-library threads before spawning layer workers.

    VTK's STDThread backend has its own explicit thread budget. Changing these
    variables here does not reconfigure the parent's already-loaded libraries.
    """
    keys = ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if threads is not None:
            for key in keys:
                os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def effective_geometry_threads(threads):
    """Keep VTK filters serial on Windows; parallelize independent layers instead.

    Threaded contour/clean export was not repeatable in the Windows build.
    Keep independent layer processes parallel while their VTK filters run
    serially. This policy does not cure every native lifetime problem and does
    not change the independent native BVH or libigl thread budgets.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("Geometry thread count must be a positive integer.")
    return 1 if sys.platform == 'win32' else threads


def configure_geometry_threads(threads=8):
    """Select a safe VTK backend and report both requested and actual budgets."""
    effective = effective_geometry_threads(threads)
    backend = "Sequential" if effective == 1 else "STDThread"
    accepted = vtkSMPTools.SetBackend(backend)
    if not accepted:
        vtkSMPTools.SetBackend("Sequential")
    vtkSMPTools.Initialize(effective)
    return {"backend": vtkSMPTools.GetBackend(), "requested_threads": threads,
            "effective_threads": effective,
            "windows_serial_workaround": sys.platform == 'win32' and threads != effective,
            "estimated_threads": vtkSMPTools.GetEstimatedNumberOfThreads()}


def sample_mesh_distance(function, sampler, points, input_mesh=None):
    """Keep large queries in VTK batches; avoid filter setup for tiny batches."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) <= 32:
        # Match vtkSampleImplicitFunctionFilter's float32 distance values.
        return np.asarray([function.EvaluateFunction(point) for point in points], dtype=np.float32)
    if input_mesh is None:
        input_mesh = pv.PolyData()
    input_mesh.points = points
    sampler.SetInputData(input_mesh)
    sampler.Update()
    return np.asarray(pv.wrap(sampler.GetOutput())["Implicit scalars"]).copy()


def smooth_maximum(a, b, width):
    """Compact quadratic rounding above max(a,b), hence conservative for cuts."""
    if width == 0:
        return np.maximum(a, b)
    overlap = np.maximum(width - np.abs(np.asarray(a) - b), 0)
    return np.maximum(a, b) + overlap * overlap / (4 * width)


def radial_union(values, rows, size, width):
    """Order-independent compact blend of radial primitive fields.

    Solve sum(max(0, F - value_i + width)**2) = width**2 per query.
    One active primitive gives its exact field. Overlaps expand the union,
    bounded by width in field units, independently of primitive count.
    """
    minimum = np.full(size, np.inf)
    np.minimum.at(minimum, rows, values)
    if width == 0:
        return minimum
    offset = values - minimum[rows]
    active = offset < width
    rows, offset = rows[active], offset[active]
    present = np.isfinite(minimum)
    root = np.full(size, width)
    for _ in range(20):
        weight = np.maximum(root[rows] - offset, 0)
        residual = np.bincount(rows, weights=weight * weight, minlength=size) - width * width
        derivative = 2 * np.bincount(rows, weights=weight, minlength=size)
        update = np.divide(residual, derivative, out=np.zeros(size), where=derivative > 0)
        root -= update
        if np.max(np.abs(update), initial=0) < 1e-10:
            break
    minimum[present] += root[present] - width
    return minimum


class TaperedTree:
    # Diagnostic switch for matched comparisons with the earlier list search.
    neighbor_backend = "arrays"
    distance_backend='exact'

    def __init__(self, segments, mesh, direction, plate, side_gap, contact_gap,
                 roof_clearance_mode="plateau", terminal_normals=None, terminal_cap="flat_disc",
                 geometry_mode="hard", branch_blend=0.3, clearance_blend=0.1, rim_blend=0.1, distance_source=None,maximum_projection=None,side_mesh=None,
                 query_batch_size=2000,prune_pairs=False,distance_backend=None):
        if distance_backend is not None:self.distance_backend=distance_backend
        if self.distance_backend not in ('exact','certified'):raise ValueError('Unknown model distance backend.')
        if isinstance(query_batch_size,bool) or not isinstance(query_batch_size,(int,np.integer)) or query_batch_size<1:
            raise ValueError('Geometry query batch size must be a positive integer.')
        self.query_batch_size=int(query_batch_size)
        self.prune_pairs=bool(prune_pairs)
        self.pair_candidates=self.pair_evaluations=0
        self.segments = np.asarray(segments, dtype=float)  # a,b,ra,rb,terminal
        self.a, self.b = self.segments[:, :3], self.segments[:, 3:6]
        self.ra, self.rb = self.segments[:, 6], self.segments[:, 7]
        self.segment_delta = self.b - self.a
        self.segment_length_squared = np.sum(self.segment_delta * self.segment_delta, axis=1)
        self.radius_delta = self.rb - self.ra
        self.terminal = (self.segments[:,8]==1)|(self.segments[:,8]==2)
        self.model_bridge = self.segments[:,8]==2
        self.platform_base=self.segments[:,8]==3
        self.mesh, self.direction, self.plate = mesh, np.asarray(direction), plate
        self.maximum_projection = (getattr(distance_source,"maximum_projection",None)
            if maximum_projection is None else maximum_projection)
        if self.maximum_projection is not None and (not np.isfinite(self.maximum_projection) or self.maximum_projection<plate):
            raise ValueError("Support ceiling must be finite and at/above the plate.")
        if distance_source is None:
            self.model_distance_function = vtkImplicitPolyDataDistance()
            self.model_distance_function.SetInput(mesh)
            self.distance_sampler = vtkSampleImplicitFunctionFilter()
            self.distance_sampler.SetImplicitFunction(self.model_distance_function)
            self.distance_sampler.SetComputeGradients(False)
            self.distance_input = pv.PolyData()
        else:
            self.model_distance_function = distance_source.model_distance_function
            self.distance_sampler = distance_source.distance_sampler
            self.distance_input = distance_source.distance_input
        self.side_gap, self.contact_gap = side_gap, contact_gap
        self.side_mesh=side_mesh if side_mesh is not None else getattr(distance_source,'side_mesh',None)
        self.lateral_clearance=None
        # With equal gaps, signed model distance already enforces clearance to
        # every face. Plane-restricted side distance cannot be smaller than
        # that distance, so its additional constraint is redundant.
        if side_gap != contact_gap and self.side_mesh is not None and self.side_mesh.n_cells:
            from .side_clearance import LateralClearance
            self.lateral_clearance=LateralClearance(self.side_mesh,self.direction,self.side_gap)
        if roof_clearance_mode not in ("plateau", "legacy_blend"):
            raise ValueError("Unknown roof clearance mode.")
        self.roof_clearance_mode = roof_clearance_mode
        if terminal_cap not in ("flat_disc", "rounded"):
            raise ValueError("Unknown terminal cap mode.")
        self.terminal_cap = terminal_cap
        if geometry_mode not in ("hard", "smooth"):
            raise ValueError("Unknown geometry mode.")
        widths = np.asarray([branch_blend, clearance_blend, rim_blend], dtype=float)
        if not np.all(np.isfinite(widths)) or np.any(widths < 0):
            raise ValueError("Blend widths must be finite and nonnegative.")
        self.geometry_mode = geometry_mode
        self.branch_blend, self.clearance_blend, self.rim_blend = widths
        delta = self.a - self.b
        lengths = np.linalg.norm(delta, axis=1)
        self.edge_direction = np.divide(-delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 1e-12)
        if np.any(lengths[self.terminal] <= 1e-12):
            raise ValueError("Terminal edges must have positive length.")
        self.tip_normals = np.divide(delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 1e-12) if terminal_normals is None else np.asarray(terminal_normals, dtype=float)
        self.neck_height = np.einsum("ni,ni->n", delta, self.tip_normals)
        if np.any(self.neck_height[self.terminal] <= 1e-9):
            raise ValueError("Terminal starts must lie outside their contact planes.")
        # Plateau is a slab parallel to the contact patch, covering its whole
        # footprint. The blend finishes below it, not merely at the endpoint.
        self.roof_depth = np.minimum(self.rb, np.maximum(self.neck_height, 0) * 0.5)
        self.clearance_span = self.neck_height - self.roof_depth
        self._unique_primitive_ids = None
        self.midpoints = cKDTree((self.a + self.b) / 2)
        self.blend_extent = self.branch_blend if geometry_mode == "smooth" else 0
        search_extent = float(widths.max()) if geometry_mode == "smooth" else 0
        self.primitive_search_radius=np.linalg.norm(self.b-self.a,axis=1)/2+np.maximum(self.ra,self.rb)+search_extent
        self.primitive_half_extent=np.abs(self.b-self.a)/2+(np.maximum(self.ra,self.rb)+search_extent)[:,None]
        self.search_radius = float(np.max(np.linalg.norm(self.b - self.a, axis=1) / 2 + np.maximum(self.ra, self.rb))) + search_extent
        # One global radius is determined by the longest trunk and makes a
        # dense tree's short branches appear as candidates far from their
        # actual bounds.  Bucket primitives by their individual enclosing-ball
        # radius; every bucket still uses a conservative maximum radius.
        minimum=max(float(np.min(self.primitive_search_radius)),1e-12)
        bucket_ids=np.floor(np.log2(np.maximum(self.primitive_search_radius/minimum,1.))).astype(int)
        self._primitive_buckets=[]
        for bucket in np.unique(bucket_ids):
            ids=np.flatnonzero(bucket_ids==bucket)
            self._primitive_buckets.append((cKDTree(self.midpoints.data[ids]),ids,
                                            float(np.max(self.primitive_search_radius[ids]))))

    def _candidate_pairs(self, query):
        """Return conservative point/primitive pairs using size-bucketed trees."""
        query_tree=cKDTree(query)
        rows=[];ids=[]
        for tree,source_ids,radius in self._primitive_buckets:
            pairs=query_tree.sparse_distance_matrix(tree,radius+1e-9,output_type="ndarray")
            if len(pairs):
                rows.append(pairs['i'])
                ids.append(source_ids[pairs['j']])
        if not rows:
            return np.empty(0,dtype=int),np.empty(0,dtype=int)
        rows=np.concatenate(rows);ids=np.concatenate(ids)
        order=np.argsort(rows,kind='stable')
        return rows[order],ids[order]

    def distance(self, points):
        return sample_mesh_distance(self.model_distance_function, self.distance_sampler, points, self.distance_input)

    def bounded_distance(self,points,band):
        if self.distance_backend=='exact':return self.distance(points)
        if not hasattr(self,'distance_certificate'):
            from .distance_certificate import DistanceCertificate
            self.distance_certificate=DistanceCertificate(self.mesh,self.distance)
        return self.distance_certificate.evaluate(points,band)

    @property
    def unique_primitive_ids(self):
        if self._unique_primitive_ids is None:
            keys = np.column_stack((self.segments, self.tip_normals))
            _, first = np.unique(keys, axis=0, return_index=True)
            self._unique_primitive_ids = np.sort(first)
        return self._unique_primitive_ids

    def clearance(self, points, ids, t):
        gap = np.full(len(ids), self.side_gap, dtype=float)
        if self.side_gap == self.contact_gap:
            return gap
        selected = self.terminal[ids]
        if not np.any(selected):
            return gap
        ids = ids[selected]
        if self.roof_clearance_mode == "legacy_blend":
            s = t[selected]
        else:
            height = np.einsum("ni,ni->n", np.asarray(points)[selected] - self.b[ids], self.tip_normals[ids])
            span = self.clearance_span[ids]
            s = np.clip(np.divide(self.neck_height[ids] - height, span, out=np.zeros_like(height), where=span > 1e-12), 0, 1)
        weight = s * s * (3 - 2 * s)
        gap[selected] = self.side_gap + weight * (self.contact_gap - self.side_gap)
        bridge=self.model_bridge[ids]
        if bridge.any():
            depth=np.minimum(t[selected][bridge],1-t[selected][bridge])*np.sqrt(self.segment_length_squared[ids[bridge]])
            s=np.clip(depth/3.,0,1)
            gap[np.flatnonzero(selected)[bridge]]=self.contact_gap+(self.side_gap-self.contact_gap)*s*s*(3-2*s)
        return gap

    def primitive(self, points, ids, t):
        """Swept taper, optionally intersected with its terminal end half-space.

        flat_disc cuts terminal tips, and both ends of model bridges, with
        planes perpendicular to the branch. rounded retains the swept-sphere
        ends. Model clearance and platform-plane clipping apply to both modes.
        """
        points = np.asarray(points)
        delta = self.segment_delta[ids]
        radius = self.ra[ids] + t * self.radius_delta[ids]
        bridge=self.model_bridge[ids]
        if bridge.any():
            depth=np.minimum(t[bridge],1-t[bridge])*np.sqrt(self.segment_length_squared[ids[bridge]])
            s=np.clip(depth/3.,0,1);s=s*s*(3-2*s)
            radius[bridge]=self.rb[ids[bridge]]+(self.ra[ids[bridge]]-self.rb[ids[bridge]])*s
        field = np.linalg.norm(points - self.a[ids] - t[:, None] * delta, axis=1) - radius
        base=self.platform_base[ids]
        if base.any():
            # Physical feet have flat horizontal end planes regardless of the
            # rounded cap policy used by branch tips.
            lower=(self.a[ids[base]]-points[base])@self.direction
            upper=(points[base]-self.b[ids[base]])@self.direction
            field[base]=np.maximum(field[base],np.maximum(lower,upper))
        if self.terminal_cap == "flat_disc":
            selected = self.terminal[ids]
            terminal_ids = ids[selected]
            plane = np.einsum("ni,ni->n", points[selected] - self.b[terminal_ids], self.edge_direction[terminal_ids])
            field[selected] = smooth_maximum(field[selected], plane, self.rim_blend) if self.geometry_mode == "smooth" else np.maximum(field[selected], plane)
        if self.terminal_cap == "flat_disc" and bridge.any():
            lower=np.einsum('ni,ni->n',self.a[ids[bridge]]-points[bridge],self.edge_direction[ids[bridge]])
            field[bridge]=np.maximum(field[bridge],lower)
        return field

    def evaluate(self, points, distances=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        sample_distances=distances is None
        distances = np.full(len(points),np.inf) if sample_distances else np.asarray(distances)
        result = np.full(len(points), np.inf)
        # Radius search includes every primitive capable of containing the
        # point, unlike a fixed nearest-segment count. Positive values need
        # only certify outside membership, not be an exact signed distance.
        for start in range(0, len(points), self.query_batch_size):
            query = points[start:start + self.query_batch_size]
            if self.neighbor_backend == "arrays" and self.geometry_mode == "hard" and len(query) >= 64:
                # Compiled pair arrays avoid Python neighbor lists/integers.
                # Hard minima are order-independent; soft blends retain the
                # existing per-point primitive order and its rounding behavior.
                if self.prune_pairs:
                    rows, ids = self._candidate_pairs(query)
                    self.pair_candidates+=len(ids)
                    # A bucket's radius is deliberately conservative. Recover
                    # the former primitive-specific enclosing-sphere screen
                    # before the AABB test so pruning semantics stay identical.
                    within_sphere=np.sum((query[rows]-self.midpoints.data[ids])**2,axis=1)<=self.primitive_search_radius[ids]**2+1e-9
                    rows,ids=rows[within_sphere],ids[within_sphere]
                    # Both bounds enclose the entire swept primitive. Discarded
                    # primitives cannot contribute an inside/zero field value.
                    inside_bounds=np.all(np.abs(query[rows]-self.midpoints.data[ids])<=
                        self.primitive_half_extent[ids]+1e-9,axis=1)
                    rows,ids=rows[inside_bounds],ids[inside_bounds]
                else:
                    pairs = cKDTree(query).sparse_distance_matrix(self.midpoints, self.search_radius + 1e-9, output_type="ndarray")
                    self.pair_candidates+=len(pairs)
                    order=np.argsort(pairs['i'],kind='stable')
                    rows,ids=pairs['i'][order],pairs['j'][order]
                counts = np.bincount(rows, minlength=len(query))
            else:
                groups = self.midpoints.query_ball_point(query, self.search_radius + 1e-9)
                counts = np.asarray([len(ids) for ids in groups])
                rows = np.repeat(np.arange(len(query)), counts)
                ids = np.concatenate([ids for ids in groups if len(ids)]).astype(int) if counts.sum() else np.empty(0, dtype=int)
                self.pair_candidates+=len(ids)
            self.pair_evaluations+=len(ids)
            if not len(ids):
                continue
            delta = self.segment_delta[ids]
            lengths = self.segment_length_squared[ids]
            t = np.clip(np.divide(np.sum((query[rows] - self.a[ids]) * delta, axis=1), lengths, out=np.zeros_like(lengths), where=lengths > 1e-16), 0, 1)
            capsule = self.primitive(query[rows], ids, t)
            if sample_distances:
                possible=np.zeros(len(query),dtype=bool)
                np.logical_or.at(possible,rows,capsule<=self.blend_extent)
                sampler=self.bounded_distance if self.geometry_mode=='hard' else lambda p,band:self.distance(p)
                distances[start+np.flatnonzero(possible)]=sampler(query[possible],max(self.side_gap,self.contact_gap))
            gap = self.clearance(query[rows], ids, t)
            if self.geometry_mode == "hard":
                # Neighbors are contiguous per query point. Grouped reduction
                # avoids one scattered write for every point/primitive pair.
                present = np.flatnonzero(counts)
                offsets = np.cumsum(counts) - counts
                values = np.maximum(capsule, gap - distances[start + rows])
                result[start + present] = np.minimum.reduceat(values, offsets[present])
            else:
                union = radial_union(capsule, rows, len(query), self.branch_blend)
                # Contact allowance is confined to terminal geometry plus its
                # compact union blend band. Product union keeps permission in
                # [0,1], smooth at overlapping designated contact neighborhoods.
                band = max(self.branch_blend, self.clearance_blend, self.rim_blend, 1e-12)
                proximity = np.clip(1 - np.maximum(capsule, 0) / band, 0, 1)
                proximity = proximity * proximity * (3 - 2 * proximity)
                allowance = (self.side_gap - gap) / max(self.side_gap - self.contact_gap, 1e-12)
                permission = np.clip(allowance * proximity, 0, 1)
                complement = np.ones(len(query))
                np.multiply.at(complement, rows, 1 - permission)
                required = self.contact_gap + (self.side_gap - self.contact_gap) * complement
                result[start:start + len(query)] = smooth_maximum(union, required - distances[start:start + len(query)], self.clearance_blend)
        return self.constrain_bounds(points,result)

    def constrain_bounds(self,points,values):
        projection=np.asarray(points)@self.direction
        values=np.maximum(values,self.plate-projection)
        if self.maximum_projection is not None:
            values=np.maximum(values,projection-self.maximum_projection)
        if self.lateral_clearance is not None:
            possible=values<=0
            if possible.any():
                distance=self.lateral_clearance.distance(np.asarray(points)[possible])
                values[possible]=np.maximum(values[possible],self.side_gap-distance)
        return values


def load_preserved_tree(source, mesh, side_gap, contact_gap, tip_radius, roof_clearance_mode="plateau", terminal_cap="flat_disc",
                        geometry_mode="hard", branch_blend=0.3, clearance_blend=0.1, rim_blend=0.1,
                        branch_sizing="preserved", branch_diameter=5., branch_diameter_angle=5., minimum_support_height=None, build_plate_projection=None,support_ceiling="none",tip_alignment="legacy",tip_length=3.,field_evaluator=None):
    source = Path(source)
    cfg = json.loads((source / "resolved_config.json").read_text())
    if cfg["input"]["field_mode"] not in ('analytic_y','checkpoint'):
        raise ValueError('Unsupported saved tree source field.')
    summary = json.loads((source / "summary.json").read_text())
    tree = pv.read(source / "tree.vtp")
    paths = json.loads((source / "routing_paths.json").read_text())
    bridge_path=source/'model_bridges.json'
    bridge_data=json.loads(bridge_path.read_text()) if bridge_path.exists() else {'bridges':[]}
    bridges=bridge_data['bridges']
    bridge_rows=[[*r['lower'],*r['upper'],branch_diameter/2 if branch_sizing=='bambu' else .6,tip_radius,2] for r in bridges]
    if not paths and bridges:
        direction=np.asarray(cfg['domain']['build_direction'],dtype=float);direction/=np.linalg.norm(direction)
        plate=float(np.min(mesh.points@direction)) if build_plate_projection is None else build_plate_projection
        from .side_clearance import split_contact_surfaces
        import hashlib
        with np.load(source/'contact_regions.npz') as data:
            if str(data['faces_hash'][0])!=hashlib.sha256(np.asarray(mesh.faces).tobytes()).hexdigest():
                raise ValueError('Contact labels do not match model.')
            side,side_report=split_contact_surfaces(mesh,[],[],contact_face_ids=data['face_ids'])
        geometry=TaperedTree(bridge_rows,mesh,direction,plate,side_gap,contact_gap,roof_clearance_mode,
            [r['normal'] for r in bridges],terminal_cap,geometry_mode,branch_blend,clearance_blend,rim_blend,side_mesh=side)
        geometry.platform_roots=[];geometry.source_terminal_map={};geometry.source_terminal_range=range(len(bridges))
        return geometry,dict(source=str(source.resolve()),source_routed_contacts=0,retained_contacts=len(bridges),
            model_bridge_count=len(bridges),no_room_gap_count=len(bridge_data.get('no_room',[])),
            minimum_support_height_mm=minimum_support_height or 0.,build_plate_projection_mm=float(plate),
            source_routing_failures=summary.get('routing_failure_count'),side_clearance=side_gap,contact_gap=contact_gap,
            independent_side_clearance=side_report,new_tree_planning=False,
            model_contact_planning='gradient-aligned double-contact struts',
            branch_sizing={'mode':branch_sizing,'branch_diameter_mm':branch_diameter},
            tip_alignment='native-gradient ray with two gap intersections',terminal_cap=terminal_cap,
            layer_compression='zlib',segment_count=len(bridges))
    contacts = np.atleast_1d(np.genfromtxt(source / "contacts.csv", delimiter=",", names=True))
    with np.load(source / "support_field.npz") as data:
        origin, source_spacing = data["origin"], float(data["spacing"][0])
        shape, tree_mask, loads = data["shape"], data["tree_mask"], data["tree_load"]
        indices = np.rint((tree.points - origin) / source_spacing).astype(int)
        valid = np.all((indices >= 0) & (indices < shape), axis=1)
        radii = np.full(tree.n_points, float(cfg["interface"]["collar_radius"]))
        for point_id in np.flatnonzero(valid):
            index = tuple(indices[point_id])
            if tree_mask[index]:
                load = max(1, int(loads[index]))
                radii[point_id] = min(cfg["routing"]["maximum_branch_radius"], cfg["routing"]["branch_radius"] * load ** (1 / cfg["routing"]["radius_power"]))
    lines = []
    cursor = 0
    while cursor < len(tree.lines):
        count = int(tree.lines[cursor])
        lines.append(tree.lines[cursor + 1:cursor + 1 + count])
        cursor += count + 1
    if len(lines) != 2 * len(paths):
        raise ValueError("Expected preserved routed lines followed by one terminal continuation per route.")
    direction = np.asarray(cfg["domain"]["build_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    plate = float(np.min(np.asarray(mesh.points) @ direction))
    if build_plate_projection is not None:
        plate=float(build_plate_projection)
    from .support_region import height_mask,retained_routes,resolve_minimum_height
    minimum_support_height=resolve_minimum_height(minimum_support_height,mesh.points,direction,plate)
    source_count=len(paths)
    original_body_segments=sum(len(line)-1 for line in lines[:source_count])
    centers=np.array([[contacts[p["contact_index"]][k] for k in ("x","y","z")] for p in paths])
    contact_normals=np.array([[contacts[p["contact_index"]][k] for k in ("nx","ny","nz")] for p in paths])
    centers-=summary["effective_terminal_offset"]*contact_normals
    keep=height_mask(centers,direction,plate,minimum_support_height)
    if not np.any(keep):
        raise ValueError(f"No preserved support contacts at/above {minimum_support_height:g} mm above the plate.")
    source_ids=np.flatnonzero(keep)
    if minimum_support_height>0 and not np.all(keep):
        body=retained_routes(tree.points,lines[:source_count],keep)
        terminals=[lines[source_count+i] for i in source_ids]
        paths=[paths[i] for i in source_ids]
        lines=body+terminals
    sizing_report = {"mode": branch_sizing}
    platform_roots = []
    if branch_sizing == "bambu":
        from .bambu_sizing import route_radii, UPSTREAM_COMMIT
        contact_heights = np.array([sum(float(contacts[p["contact_index"]][key])*direction[j]
            for j, key in enumerate(("x", "y", "z"))) for p in paths])
        radii, distances = route_radii(tree.points, lines[:len(paths)], contact_heights,
                                       direction, branch_diameter, branch_diameter_angle)
        roots = {}
        for line, height in zip(lines[:len(paths)], contact_heights):
            root_id = int(line[-1])
            if tree.points[root_id] @ direction <= plate + source_spacing * 1.1:
                point = tree.points[root_id]
                key = tuple(point[[0, 2]])
                previous = roots.get(key, [*key, 0., float(radii[root_id])])
                previous[2] = max(previous[2], float(height - plate))
                roots[key] = previous
        platform_roots = list(roots.values())
        sizing_report.update(upstream_commit=UPSTREAM_COMMIT, branch_diameter_mm=branch_diameter,
            branch_diameter_angle_degrees=branch_diameter_angle,
            body_radius_min_mm=float(radii[np.isfinite(distances)].min()),
            body_radius_max_mm=float(radii[np.isfinite(distances)].max()),
            platform_root_count=len(platform_roots),
            adaptation="Preserved routes; greatest contributing contact height at shared nodes; no upstream collision-radius quantization.")
    elif branch_sizing != "preserved":
        raise ValueError("Unknown branch sizing policy.")
    if tip_alignment not in ('legacy','gradient'):
        raise ValueError('Unknown tip alignment mode.')
    necks=neck_radii=None
    tip_preparation_seconds=0.
    if tip_alignment=='gradient':
        tip_started=time.perf_counter()
        from .tip_geometry import gradient_approaches
        ends=centers[keep]+contact_gap*contact_normals[keep]
        if field_evaluator is None:
            tip_gradients=np.tile([0.,1.,0.],(len(paths),1))
        else:
            from .nonplanar import field_values
            _,tip_gradients=field_values(field_evaluator,ends)
        body,necks,neck_radii=gradient_approaches(tree.points,lines[:len(paths)],ends,
            tip_gradients,radii,tip_length,direction,plate)
        lines=body+lines[len(paths):]
        tip_preparation_seconds=time.perf_counter()-tip_started
    segments = []
    normals = []
    seen_edges=set()
    for line in lines[:len(paths)]:
        for first, second in zip(line[:-1], line[1:]):
            edge=(int(first),int(second))
            if (minimum_support_height>0 or tip_alignment=='gradient') and edge in seen_edges:
                continue
            seen_edges.add(edge)
            segments.append([*tree.points[first], *tree.points[second], radii[first], radii[second], 0])
            normals.append([0, 0, 0])
    new_body_segments=len(segments)
    terminal_map={}
    approach_lengths=[]
    for number,(path, line) in enumerate(zip(paths, lines[len(paths):])):
        contact = contacts[path["contact_index"]]
        normal = np.array([contact[k] for k in ("nx", "ny", "nz")])
        point = np.array([contact[k] for k in ("x", "y", "z")]) - summary["effective_terminal_offset"] * normal
        end = point + contact_gap * normal
        start = np.asarray(tree.points[line[0]])
        start_radius=radii[line[0]]
        if tip_alignment=='gradient':
            anchor=int(lines[number][0])
            start=np.asarray(tree.points[anchor])
            start_radius=neck_radii[number]
            neck=necks[number]
            if np.linalg.norm(neck-start)>1e-9:
                segments.append([*start,*neck,radii[anchor],start_radius,0])
                normals.append([0,0,0])
            start=neck
            approach_lengths.append(float(np.linalg.norm(end-start)))
        terminal_map[original_body_segments+int(source_ids[number])]=len(segments)
        segments.append([*start, *end, start_radius, tip_radius, 1])
        normals.append(normal)
    segments.extend(bridge_rows)
    normals.extend(r['normal'] for r in bridges)
    if support_ceiling not in ("none","contact_max"):
        raise ValueError("Unknown support ceiling mode.")
    maximum_projection=float(np.max(centers[keep]@direction))-contact_gap if support_ceiling=="contact_max" else None
    from .side_clearance import split_contact_surfaces
    contact_faces=None
    region_file=source/'contact_regions.npz'
    if region_file.exists():
        import hashlib
        with np.load(region_file) as data:
            if str(data['faces_hash'][0])!=hashlib.sha256(np.asarray(mesh.faces).tobytes()).hexdigest():
                raise ValueError('Saved contact face labels do not match the current model mesh.')
            contact_faces=data['face_ids']
    side_mesh,side_report=split_contact_surfaces(mesh,centers[keep],contact_normals[keep],contact_face_ids=contact_faces)
    geometry = TaperedTree(segments, mesh, direction, plate, side_gap, contact_gap, roof_clearance_mode, normals, terminal_cap,
                           geometry_mode, branch_blend, clearance_blend, rim_blend,maximum_projection=maximum_projection,side_mesh=side_mesh)
    geometry.platform_roots = platform_roots
    geometry.source_terminal_map=terminal_map
    geometry.source_terminal_range=range(original_body_segments,original_body_segments+source_count)
    return geometry, {"source": str(source.resolve()), "source_routed_contacts": source_count,
        "retained_contacts":len(paths),"excluded_contacts_below_height":source_count-len(paths),
        "minimum_support_height_mm":minimum_support_height,
        "build_plate_projection_mm":plate,
        "support_ceiling_mode":support_ceiling,"maximum_support_projection_mm":maximum_projection,
        "tip_alignment":tip_alignment,"requested_tip_length_mm":tip_length,
        "tip_preparation_seconds":tip_preparation_seconds,
        "aligned_tip_length_min_mm":min(approach_lengths) if approach_lengths else None,
        "aligned_tip_length_max_mm":max(approach_lengths) if approach_lengths else None,
        "source_routing_failures": summary.get("routing_failure_count"), "segment_count": len(segments),
        "new_tree_planning": False, "side_clearance": side_gap, "contact_gap": contact_gap,
        "independent_side_clearance":side_report,
        "tip_radius": tip_radius, "separate_contact_solid": False,
        "roof_clearance_mode": roof_clearance_mode, "terminal_cap": terminal_cap,
        "geometry_mode": geometry_mode, "branch_blend": branch_blend,
        "branch_sizing": sizing_report,
        "clearance_blend": clearance_blend, "rim_blend": rim_blend,
        "clearance_method": "compact terminal-neighborhood allowance and conservative smooth intersection" if geometry_mode == "smooth" else "per-primitive hard clearance intersection"}


def sample_solid(geometry, spacing, sample_clearance=True):
    extent = float(max(geometry.ra.max(), geometry.rb.max())) + 2 * spacing + geometry.blend_extent
    lower = np.minimum(geometry.a.min(axis=0), geometry.b.min(axis=0)) - extent
    upper = np.maximum(geometry.a.max(axis=0), geometry.b.max(axis=0)) + extent
    if hasattr(geometry, "roof_sample_bounds"):
        lower = np.minimum(lower, geometry.roof_sample_bounds[:, 0] - 2 * spacing)
        upper = np.maximum(upper, geometry.roof_sample_bounds[:, 1] + 2 * spacing)
    shape = np.ceil((upper - lower) / spacing).astype(int) + 1
    if np.prod(shape) > 3_000_000:
        raise ValueError("Tree geometry grid exceeds three million points.")
    raw = np.full(tuple(shape), 2 * spacing, dtype=float)
    for a, b, ra, rb in zip(geometry.a, geometry.b, geometry.ra, geometry.rb):
        border = max(ra, rb) + 2 * spacing + geometry.blend_extent
        lo = np.maximum(np.floor((np.minimum(a, b) - border - lower) / spacing).astype(int), 0)
        hi = np.minimum(np.ceil((np.maximum(a, b) + border - lower) / spacing).astype(int) + 1, shape)
        indices = np.indices(tuple(hi - lo)).transpose(1, 2, 3, 0) + lo
        points = lower + spacing * indices
        delta = b - a
        t = np.clip(np.sum((points - a) * delta, axis=-1) / max(float(delta @ delta), 1e-16), 0, 1)
        values = np.linalg.norm(points - a - t[..., None] * delta, axis=-1) - (ra + t * (rb - ra))
        slices = tuple(slice(x, y) for x, y in zip(lo, hi))
        raw[slices] = np.minimum(raw[slices], values)
    grid = pv.ImageData(dimensions=tuple(shape), spacing=(spacing,) * 3, origin=lower)
    near = raw.ravel(order="F") <= spacing * 1.8 + geometry.blend_extent
    if hasattr(geometry, "roof_sample_bounds"):
        bounds = geometry.roof_sample_bounds + np.array([-2 * spacing, 2 * spacing])
        near |= np.all((grid.points >= bounds[:, 0]) & (grid.points <= bounds[:, 1]), axis=1)
    if not sample_clearance:
        # Independent fallback layer seeds need only conservative sweep
        # bounds. Exact clearance is evaluated during adaptive trimming.
        grid['support_solid'] = raw.ravel(order='F')
        return grid
    field = np.full(grid.n_points, spacing * 2, dtype=float)
    points = np.asarray(grid.points)[near]
    # Batching limits temporary arrays and emits meaningful progress.
    near_ids = np.flatnonzero(near)
    for start in range(0, len(points), 20_000):
        ids = near_ids[start:start + 20_000]
        field[ids] = geometry.evaluate(points[start:start + 20_000])
    field[~np.isfinite(field)] = spacing * 2
    grid["support_solid"] = field
    return grid


def support_layer_source(geometry,field_evaluator,slice_evaluator,field_mode,transfer_domain,support_field_mode,layer_seed_spacing):
    """Build the manufacturing seed source once, also usable for online planning."""
    if field_mode in ('checkpoint','tetrahedral') and support_field_mode=='relaxed':
        from .layer_extraction import fem_layer_source
        return fem_layer_source(transfer_domain),'transferred_FEM_mesh'
    from .nonplanar import field_values
    grid=sample_solid(geometry,layer_seed_spacing,sample_clearance=False)
    grid['near_tree']=(grid['support_solid']<=layer_seed_spacing*1.8).astype(np.uint8)
    if field_mode in ('checkpoint','tetrahedral'):
        from scipy.ndimage import binary_dilation
        halo=binary_dilation(np.asarray(grid['near_tree']).reshape(grid.dimensions,order='F'),structure=np.ones((3,3,3))).ravel(order='F')
        values=np.zeros(grid.n_points);values[halo]=slice_evaluator.evaluate_scalars(grid.points[halo]);grid['scalar']=values
    else:grid['scalar']=field_values(field_evaluator,grid.points)[0] if field_mode!='analytic_y' else grid.points[:,1]
    return grid.threshold(.5,scalars='near_tree',all_scalars=False),'independent_layer_seed_grid'


def boundary_roots(lo, hi, lower_values, upper_values, geometry, iterations=14, method="bisect"):
    """Verified inside roots; adaptive secants stop per edge, with bisection backup."""
    lo, hi = lo.copy(), hi.copy()
    if method == "bisect":
        for _ in range(iterations):
            middle = (lo + hi) / 2
            keep = geometry.evaluate(middle) <= 0
            lo[keep], hi[~keep] = middle[keep], middle[~keep]
        return lo
    if method != "adaptive":
        raise ValueError("Unknown boundary root method.")
    lower_values, upper_values = lower_values.copy(), upper_values.copy()
    tolerance = np.linalg.norm(hi - lo, axis=1) / (2 ** iterations)
    # Secants exploit approximately linear distances near a boundary. Keep
    # only unresolved edges active, and finish with the original bracket
    # tolerance if a kink or noisy distance prevents field convergence.
    for step in range(iterations + 10):
        active = (np.linalg.norm(hi - lo, axis=1) > tolerance)
        # VTK's sampled distances are float32; 1e-7 mm avoids chasing its
        # quantization noise while remaining below the 1e-6 check threshold.
        active &= np.abs(lower_values) > 1e-7
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        fraction = np.full(len(ids), 0.5)
        if step < 8 and step % 4 != 3:
            left, right = lower_values[ids], upper_values[ids]
            finite = np.isfinite(left) & np.isfinite(right) & (right > left)
            fraction[finite] = np.clip(-left[finite] / (right[finite] - left[finite]), 0.001, 0.999)
        middle = lo[ids] + fraction[:, None] * (hi[ids] - lo[ids])
        values = geometry.evaluate(middle)
        keep = values <= 0
        lo[ids[keep]], lower_values[ids[keep]] = middle[keep], values[keep]
        hi[ids[~keep]], upper_values[ids[~keep]] = middle[~keep], values[~keep]
    return lo


def clip_layer(surface, geometry, iterations=14, root_method="bisect"):
    surface = surface.triangulate().clean()
    if not surface.n_cells:
        return pv.PolyData()
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.regular_faces)
    values = geometry.evaluate(points)
    inside = values <= 0
    face_edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1)
    crossing = inside[face_edges[..., 0]] != inside[face_edges[..., 1]]
    edges, inverse = unique_rows_inverse(np.sort(face_edges[crossing], axis=1))
    roots = np.empty((len(edges), 3))
    if len(edges):
        first_inside = inside[edges[:, 0]]
        lo = points[np.where(first_inside, edges[:, 0], edges[:, 1])].copy()
        hi = points[np.where(first_inside, edges[:, 1], edges[:, 0])].copy()
        lower_values = values[np.where(first_inside, edges[:, 0], edges[:, 1])]
        upper_values = values[np.where(first_inside, edges[:, 1], edges[:, 0])]
        roots[:] = boundary_roots(lo, hi, lower_values, upper_values, geometry, iterations, root_method)
    edge_ids = np.full(crossing.shape, -1, dtype=int)
    edge_ids[crossing] = len(points) + inverse
    vertices = np.column_stack((faces, edge_ids))
    cases = inside[faces] @ np.array([1, 2, 4])
    # Indices 0..2 are source vertices; 3..5 are successive crossing edges.
    templates = {1: [[0, 3, 5]], 2: [[3, 1, 4]], 4: [[4, 2, 5]],
                 3: [[0, 1, 4], [0, 4, 5]], 5: [[0, 3, 4], [0, 4, 2]],
                 6: [[3, 1, 2], [3, 2, 5]], 7: [[0, 1, 2]]}
    output = np.full((len(faces), 2, 3), -1, dtype=int)
    for case, template in templates.items():
        chosen = cases == case
        output[chosen, :len(template)] = vertices[chosen][:, np.asarray(template)]
    output = output.reshape(-1, 3)
    output = output[output[:, 0] >= 0]
    if not len(output):
        return pv.PolyData()
    mesh = pv.PolyData(np.vstack((points, roots)), np.column_stack((np.full(len(output), 3), output)).ravel())
    if _BATCH_LAYER_ID in surface.point_data:
        labels=np.asarray(surface[_BATCH_LAYER_ID])
        mesh[_BATCH_LAYER_ID]=np.concatenate((labels,labels[edges[:,0]]))
    return subset_triangles(mesh, np.ones(mesh.n_cells, dtype=bool))


def subset_triangles(surface, mask):
    # Rebuild selected faces directly. VTK extract_surface can attach stale
    # original-cell arrays after selection and turn degenerate faces into lines.
    faces = np.asarray(surface.regular_faces)[mask]
    if not len(faces):
        return pv.PolyData()
    coordinates = np.asarray(surface.points)[faces]
    area = np.linalg.norm(np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]), axis=1)
    faces = faces[area > 1e-12]
    if not len(faces):
        return pv.PolyData()
    result=pv.PolyData(np.asarray(surface.points), np.column_stack((np.full(len(faces), 3), faces)).ravel())
    if _BATCH_LAYER_ID in surface.point_data:
        result[_BATCH_LAYER_ID]=np.asarray(surface[_BATCH_LAYER_ID])
    return result.clean(polys_to_lines=False, lines_to_points=False)


def sampled_face_violations(surface, geometry):
    if not surface.n_cells:
        return np.zeros(0, dtype=bool)
    triangles = np.asarray(surface.points)[surface.regular_faces]
    probes = np.concatenate((triangles.mean(axis=1)[:, None, :], (triangles + np.roll(triangles, -1, axis=1)) / 2), axis=1)
    return np.any(geometry.evaluate(probes.reshape(-1, 3)).reshape(-1, 4) > 1e-6, axis=1)


def subdivide_preserving_coordinates(surface):
    # VTK subdivision may use float32 vtkPoints even for double input.
    # Interpolated double point attributes preserve the actual coordinates.
    surface = surface.copy()
    surface["__world_coordinates"] = np.asarray(surface.points, dtype=float).copy()
    result = surface.subdivide(1)
    result.points = np.asarray(result["__world_coordinates"], dtype=float).copy()
    del result.point_data["__world_coordinates"]
    return result


def subdivide_selected_faces(surface, selected):
    """Split marked face edges on both sides, including neighbouring faces.

    Refining an extracted patch and merging it back leaves hanging vertices:
    clean() merges coincident points but cannot split the neighbouring edge.
    Red/green triangle templates keep the original surface conforming while
    concentrating refinement around nonlinear clipping boundaries.
    """
    faces = np.asarray(surface.regular_faces)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(faces),):
        raise ValueError('Face refinement mask must match triangle count.')
    if not selected.any():
        return surface.copy()
    face_edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1)
    edges, inverse = unique_rows_inverse(np.sort(face_edges.reshape(-1, 2), axis=1))
    inverse = inverse.reshape(-1, 3)
    split = np.zeros(len(edges), dtype=bool)
    split[inverse[selected].ravel()] = True
    marked = edges[split]
    points = np.asarray(surface.points, dtype=float)
    mid_ids = np.full(len(edges), -1, dtype=int)
    mid_ids[split] = len(points) + np.arange(len(marked))
    vertices = np.column_stack((faces, mid_ids[inverse]))
    cases = split[inverse] @ np.array([1, 2, 4])
    templates = {
        0: [[0, 1, 2]],
        1: [[0, 3, 2], [3, 1, 2]],
        2: [[0, 1, 4], [0, 4, 2]],
        4: [[0, 1, 5], [5, 1, 2]],
        3: [[3, 1, 4], [0, 3, 2], [3, 4, 2]],
        5: [[0, 3, 5], [3, 1, 2], [3, 2, 5]],
        6: [[2, 5, 4], [0, 1, 5], [1, 4, 5]],
        7: [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]],
    }
    output = []
    for case, template in templates.items():
        chosen = vertices[cases == case]
        if len(chosen):
            output.append(chosen[:, np.asarray(template)].reshape(-1, 3))
    cells = np.concatenate(output)
    result = pv.PolyData(np.vstack((points, points[marked].mean(axis=1))),
                         np.column_stack((np.full(len(cells), 3), cells)).ravel())
    if _BATCH_LAYER_ID in surface.point_data:
        labels = np.asarray(surface[_BATCH_LAYER_ID])
        result[_BATCH_LAYER_ID] = np.concatenate((labels, labels[marked[:, 0]]))
    return result


def refine_clipped_layer(result, geometry, root_method, *, initial_bad=None):
    """Refine nonlinear excursions without tearing shared triangle edges."""
    # Refine locally before conservatively discarding any unresolved faces;
    # the old three-round patch split could leave visible missing strips.
    for _ in range(6):
        bad = sampled_face_violations(result, geometry) if initial_bad is None else initial_bad
        initial_bad = None
        if not np.any(bad):
            return result, 0
        result = clip_layer(subdivide_selected_faces(result, bad), geometry, root_method=root_method)
        if not result.n_cells:
            return pv.PolyData(), 0
    bad = sampled_face_violations(result, geometry)
    removed = int(np.count_nonzero(bad))
    if removed:
        result = subset_triangles(result, ~bad)
    return result, removed


def refined_clip_layer(surface, geometry, root_method="bisect", presubdivide=True):
    if not surface.n_cells:
        return pv.PolyData(), 0
    candidate = subdivide_preserving_coordinates(surface.triangulate()) if presubdivide else surface.triangulate()
    result = clip_layer(candidate, geometry, root_method=root_method)
    return refine_clipped_layer(result, geometry, root_method)


def scalar_clip_layer(surface, geometry):
    """VTK interpolation of one support-field evaluation at surface vertices."""
    if not surface.n_cells:
        return pv.PolyData()
    surface = surface.triangulate().copy()
    values = geometry.evaluate(surface.points)
    finite = np.isfinite(values)
    values[~finite] = max(float(np.max(np.abs(values[finite]), initial=0)), 1.0)
    surface["support_field"] = values
    surface.set_active_scalars("support_field")
    clip = vtkClipPolyData()
    clip.SetInputData(surface)
    clip.SetValue(0)
    clip.SetInsideOut(True)
    clip.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
    clip.Update()
    clipped = pv.wrap(clip.GetOutput()).triangulate()
    clipped.verts = np.empty(0, dtype=int)
    clipped.lines = np.empty(0, dtype=int)
    if not clipped.n_cells:
        return pv.PolyData()
    return subset_triangles(clipped, np.ones(clipped.n_cells, dtype=bool))


def scalar_face_violations(surface, geometry):
    if not surface.n_cells:
        return np.zeros(0, dtype=bool)
    outside = geometry.evaluate(surface.points) > 1e-6
    return np.any(outside[surface.regular_faces], axis=1) | sampled_face_violations(surface, geometry)


def surface_scalar_trim(surface, geometry, root_method="adaptive", presubdivide=True):
    """Cached vertex scalars, verified crossing roots, and checked local refinement."""
    if not surface.n_cells:
        return pv.PolyData(), 0
    candidate = subdivide_preserving_coordinates(surface.triangulate()) if presubdivide else surface.triangulate()
    # Linear interpolation can put an entire strip of boundary vertices
    # outside the true field, sending otherwise valid triangles into costly
    # subdivision. Resolve crossing edges once before checking interiors.
    result = clip_layer(candidate, geometry, root_method="adaptive") if root_method == "adaptive" else scalar_clip_layer(candidate, geometry)
    # ``clip_layer`` retains only source vertices with a non-positive field
    # value and crossing roots returned from the verified inside bracket.  A
    # second vertex evaluation here is therefore redundant; retain the
    # centroid and edge-midpoint checks, which are what detect nonlinear
    # field excursions inside a clipped face.
    bad = sampled_face_violations(result, geometry)
    if not np.any(bad):
        return result, 0
    # The first refinement check uses this exact unchanged mesh and field.
    # Reuse its mask; subsequent refinement rounds still perform every check.
    return refine_clipped_layer(result, geometry, root_method, initial_bad=bad)


class CachedLayerField:
    """Reuse exact scalar samples at shared surface vertices and face probes."""
    def __init__(self, geometry):
        self.geometry = geometry
        self.cache = {}
        self.requested = self.evaluated = 0
        # Exact accelerators may provide a compiled cache. Ordinary geometry
        # keeps the portable Python implementation and the same query policy.
        factory = getattr(geometry, 'sample_cache_factory', None)
        self._sample_cache = factory(geometry) if factory is not None else None

    def __getattr__(self, name):
        return getattr(self.geometry, name)

    def evaluate(self, points):
        if self._sample_cache is not None:
            values = self._sample_cache.evaluate(points)
            self.requested = self._sample_cache.requested
            self.evaluated = self._sample_cache.evaluated
            return values
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.requested += len(points)
        if not len(points):
            return np.empty(0)
        unique, inverse = unique_rows_inverse(points)
        keys = unique.view(np.dtype((np.void,24))).ravel().tolist()
        values = [self.cache.get(key) for key in keys]
        missing = np.array([i for i, value in enumerate(values) if value is None], dtype=int)
        if len(missing):
            computed = self.geometry.evaluate(unique[missing])
            self.evaluated += len(missing)
            for i, value in zip(missing, computed):
                values[i] = self.cache[keys[i]] = float(value)
        return np.asarray(values)[inverse]


def extract_one_layer(geometry, active, index, value, directory, spacing, roof_layer_spacing, layer_trim, layer_compression="zlib", platform_settings=None,layer_role_indices=None,spatial_platform=False,role_regions=None,planar_base_settings=None,adaptive_layer_tolerance=None,_prepared=None,_return_surface=False):
    """Independent task; optional surface handoff is for a local consumer only.

    Pooled callers keep the default metadata-only result to avoid mesh IPC.
    """
    started = time.perf_counter()
    if _prepared is not None:
        surface,local,removed,candidate_seconds,shared_trim_seconds=_prepared
    elif hasattr(geometry, "roof"):
        from .roof import direct_layer, geometry_for_layer
        local = geometry_for_layer(geometry, value)
        candidate = direct_layer(local, value, spacing, roof_layer_spacing)
    else:
        local = geometry
        candidate = active.contour([value], scalars="scalar")
    if _prepared is None and layer_trim in ("checked", "surface_scalar", "surface_scalar_bisect"):
        local = CachedLayerField(local)
    adaptive_report = None
    if _prepared is None and adaptive_layer_tolerance is not None and not hasattr(geometry, 'roof'):
        from .layer_extraction import adaptive_candidate
        candidate, adaptive_report = adaptive_candidate(candidate, local, adaptive_layer_tolerance)
    if _prepared is None:
        candidate_seconds = time.perf_counter() - started
    trim_started = time.perf_counter()
    if _prepared is not None:
        pass
    elif layer_trim == "checked":
        surface, removed = refined_clip_layer(candidate, local,
            root_method='adaptive' if adaptive_report is not None else 'bisect', presubdivide=adaptive_report is None)
    else:
        method = "bisect" if layer_trim == "surface_scalar_bisect" else "adaptive"
        surface, removed = surface_scalar_trim(candidate, local, root_method=method, presubdivide=adaptive_report is None)
    flange_report = None
    previous_requested = previous_evaluated = 0
    if platform_settings is not None and index == platform_settings["first_layer"]:
        if isinstance(local, CachedLayerField):
            previous_requested, previous_evaluated = local.requested, local.evaluated
        from .bambu_sizing import expand_platform_layer
        surface, local, flange_report, flange_removed = expand_platform_layer(surface, geometry, value,
            platform_settings["roots"], platform_settings["speed"])
        removed += flange_removed
    trim_seconds = (time.perf_counter() - trim_started if _prepared is None else shared_trim_seconds)
    if (planar_base_settings is not None and surface.n_points and
            np.min(np.asarray(surface.points)@geometry.direction)<=planar_base_settings['plate']+planar_base_settings['height']+1e-8):
        from .planar_bases import remove_from_field_layer
        base_cut_started=time.perf_counter()
        surface=remove_from_field_layer(surface,planar_base_settings)
        # The footprint/height cut creates interpolated vertices on existing
        # faces. A face spanning a nonconvex sweep boundary need not keep those
        # new points inside the continuous solid. Recheck the cut before role
        # classification and export, retaining the existing geometry tolerance.
        surface,base_removed=surface_scalar_trim(surface,local,root_method='adaptive',presubdivide=False)
        removed+=base_removed
        trim_seconds+=time.perf_counter()-base_cut_started
    validation_started = time.perf_counter()
    vertex_bad = centroid_bad = 0
    if surface.n_points:
        values = local.evaluate(surface.points)
        vertex_bad = int(np.count_nonzero(values > 1e-6))
        surface["support_field"] = values
        # Use the same arithmetic as interior probes so exact-cache entries
        # are reused instead of rounding-different VTK cell-center queries.
        centers = np.asarray(surface.points)[surface.regular_faces].mean(axis=1)
        centroid_bad = int(np.count_nonzero(local.evaluate(centers) > 1e-6))
    validation_seconds = time.perf_counter() - validation_started
    path = Path(directory) / f"{index}.vtp"
    from .layer_roles import roles_for,mark_surface
    roles=roles_for(int(index),layer_role_indices or {})
    if spatial_platform and surface.n_points and np.min(np.asarray(surface.points)@geometry.direction)<=geometry.plate+1e-6:
        roles=['platform',*roles]
    mark_surface(surface,dict(layer=int(index),scalar=float(value),roles=roles,
        is_platform_layer='platform' in roles,is_contact_layer='contact' in roles))
    model_base=int(index) in (layer_role_indices or {}).get('model_base_layer_indices',[])
    if role_regions is not None:
        from .spatial_roles import classify_patches
        flags=classify_patches(surface,int(index),role_regions)
        combined=int(np.bitwise_or.reduce(flags,initial=0))
        roles=[role for role,bit in (('platform',1),('contact',2)) if combined&bit]
        model_base=bool(combined&4)
        mark_surface(surface,dict(layer=int(index),scalar=float(value),roles=roles,
            is_platform_layer='platform' in roles,is_contact_layer='contact' in roles))
    surface.field_data['is_model_base_layer']=np.array([model_base],dtype=np.uint8)
    serialization_started = time.perf_counter()
    surface.save(path, compression=None if layer_compression == "none" else layer_compression)
    serialization_seconds = time.perf_counter() - serialization_started
    result = {"manifest": {"layer": int(index), "scalar":float(value),"roles":roles,
             "is_platform_layer":'platform' in roles,"is_contact_layer":'contact' in roles,
             "is_model_base_layer":model_base,
             "file": str(path.resolve()), "points": surface.n_points,
             "cells": surface.n_cells, "area": float(surface.area),
             "y_span": float(np.ptp(surface.points[:, 1])) if surface.n_points else 0},
            "timing": {"layer": int(index), "candidate_seconds": candidate_seconds,
             "trim_seconds": trim_seconds, "validation_seconds": validation_seconds,
             "serialization_seconds": serialization_seconds, "adaptive_extraction":adaptive_report,
             "total_seconds": time.perf_counter() - started+(candidate_seconds+trim_seconds if _prepared is not None else 0)},
            "removed": removed, "vertex_bad": vertex_bad, "centroid_bad": centroid_bad, "platform_flange": flange_report,
            "requested": previous_requested + (local.requested if isinstance(local, CachedLayerField) else 0),
            "evaluated": previous_evaluated + (local.evaluated if isinstance(local, CachedLayerField) else 0)}
    if _return_surface:
        result['surface']=surface
    return result


def extract_layer_batch(geometry,active,tasks,**settings):
    """Trim disconnected layer components together, retaining their identities."""
    if (len(tasks)<2 or hasattr(geometry,'roof') or settings.get('platform_settings') is not None
            or geometry.geometry_mode!='hard' or not settings['layer_trim'].startswith('surface_scalar')):
        return [extract_one_layer(geometry,active,index,value,**settings) for index,value in tasks]
    if len(set(value for _,value in tasks))!=len(tasks):
        raise ValueError('Batched layers must have distinct scalar values.')
    points=[];faces=[];labels=[];candidate_times=[];candidate_reports=[];offset=0
    local=CachedLayerField(geometry)
    for number,(index,value) in enumerate(tasks):
        started=time.perf_counter();candidate=active.contour([value],scalars='scalar').triangulate()
        adaptive_report=None
        if settings.get('adaptive_layer_tolerance') is not None:
            from .layer_extraction import adaptive_candidate
            candidate,adaptive_report=adaptive_candidate(candidate,local,settings['adaptive_layer_tolerance'])
        candidate_reports.append(adaptive_report)
        candidate_times.append(time.perf_counter()-started)
        if candidate.n_cells:
            points.append(np.asarray(candidate.points,dtype=float))
            faces.append(candidate.regular_faces+offset)
            labels.append(np.full(candidate.n_points,number,dtype=float))
            offset+=candidate.n_points
    started=time.perf_counter();removed=0
    surface=pv.PolyData()
    if points:
        triangles=np.vstack(faces)
        candidate=pv.PolyData(np.vstack(points),np.column_stack((np.full(len(triangles),3),triangles)).ravel())
        candidate[_BATCH_LAYER_ID]=np.concatenate(labels)
        method='bisect' if settings['layer_trim']=='surface_scalar_bisect' else 'adaptive'
        surface,removed=surface_scalar_trim(candidate,local,root_method=method,
            presubdivide=settings.get('adaptive_layer_tolerance') is None)
    trim_seconds=time.perf_counter()-started
    if surface.n_cells:
        ids=np.asarray(surface[_BATCH_LAYER_ID])[surface.regular_faces]
        if not np.all(np.abs(ids-ids[:,0,None])<1e-8) or not np.all(np.abs(ids-np.rint(ids))<1e-8):
            raise RuntimeError('Layer identities mixed during batch refinement.')
        face_ids=np.rint(ids[:,0]).astype(int)
    results=[]
    for number,(index,value) in enumerate(tasks):
        part=subset_triangles(surface,face_ids==number) if surface.n_cells else pv.PolyData()
        if _BATCH_LAYER_ID in part.point_data:del part.point_data[_BATCH_LAYER_ID]
        result=extract_one_layer(geometry,active,index,value,**settings,
            _prepared=(part,local,removed if number==0 else 0,candidate_times[number],trim_seconds/len(tasks)))
        result['timing'].update(batch_size=len(tasks),batch_trim_wall_seconds=trim_seconds)
        result['timing']['adaptive_extraction']=candidate_reports[number]
        results.append(result)
    # Counters and rejected-face totals belong to the shared batch, not each
    # output. Attribute counters to its first record so aggregate totals stay exact.
    for number,result in enumerate(results):
        result['requested']=local.requested if number==0 else 0
        result['evaluated']=local.evaluated if number==0 else 0
    return results


def initialize_layer_worker(payload, settings, active, threads):
    global _layer_worker_context
    configure_geometry_threads(threads)
    geometry = TaperedTree(**payload)
    roof_settings = settings["roof_settings"]
    settings = {key: value for key, value in settings.items() if key != "roof_settings"}
    if roof_settings is not None:
        from .roof import attach_roof
        geometry, _ = attach_roof(geometry, **roof_settings)
    _layer_worker_context = (geometry, active, settings)


def run_layer_worker(task):
    geometry, active, settings = _layer_worker_context
    index, value = task
    return extract_one_layer(geometry, active, index, value, **settings)


def run_layer_batch_worker(tasks):
    geometry,active,settings=_layer_worker_context
    return extract_layer_batch(geometry,active,tasks,**settings)


def export_geometry(source, mesh, schedule, indices, output, spacing=1, side_gap=0.4, contact_gap=0.2, tip_radius=0.6, roof_clearance_mode="plateau", terminal_cap="flat_disc",
                    geometry_mode="hard", branch_blend=0.3, clearance_blend=0.1, rim_blend=0.1,
                    roof_mode="none", roof_terminal=3939, roof_radius=3, roof_thickness=0.6, roof_rounding=0.1, roof_layer_spacing=0.1,
                    layer_trim="checked", geometry_threads=8, layer_workers=0, layer_compression="zlib",
                    branch_sizing="preserved", branch_diameter=5., branch_diameter_angle=5., platform_flange="none", support_speed=80., minimum_support_height=None, build_plate_projection=None,support_ceiling="none",tip_alignment="legacy",tip_length=3.,contact_layers_below=3,field_evaluator=None,field_mode='analytic_y',transfer_domain=None,layer_batch_size=1,model_distance_backend='exact',geometry_query_batch_size=2000,trim_pair_pruning=False,model_tree_mode='struts',skeleton_source=None,allow_stalled_skeleton=False,support_field_mode='native_blend',model_root_foot_radius=0.,model_root_foot_alignment=.95,model_root_foot_shape='conformal',platform_base_radius_factor=0.,platform_base_height=.6,build_direction=(0.,1.,0.),vtk_threads_per_worker=None,maximum_trunk_diameter=None,skip_unrooted_contacts=False,planar_platform_bases=False,platform_base_layer_height=.2,layer_seed_spacing=1.,adaptive_layer_tolerance=.05,maximum_platform_trunk_diameter=None,include_support_gradients=True,support_layer_spacing=.2,maximum_support_layer_height=.5,support_gradient_probe_spacing=2.,slicing_mode='sampled',model_layer_spacing=.2,_planning_model_volume=None):
    output = Path(output)
    started = time.perf_counter()
    if maximum_platform_trunk_diameter is not None:
        if not np.isfinite(maximum_platform_trunk_diameter) or maximum_platform_trunk_diameter<branch_diameter:
            raise ValueError('Platform trunk diameter cap must be finite and at least the minimum branch diameter.')
        if skeleton_source is None:
            raise ValueError('A platform trunk diameter cap requires an imported skeleton.')
    if not np.isfinite([layer_seed_spacing,adaptive_layer_tolerance]).all() or min(layer_seed_spacing,adaptive_layer_tolerance)<=0:
        raise ValueError('Layer seed spacing and adaptive tolerance must be finite and positive.')
    nonplanar=field_mode!='analytic_y'
    if nonplanar and (field_mode not in ('analytic_curved','checkpoint','tetrahedral') or field_evaluator is None):
        raise ValueError('Nonplanar support requires a callable analytic, checkpoint, or tetrahedral field.')
    if field_mode in ('checkpoint','tetrahedral') and transfer_domain is None:
        raise ValueError('Checkpoint/tetrahedral support requires a reconstructed transfer domain.')
    if nonplanar and (roof_mode!='none' or platform_flange!='none'):
        raise ValueError('Nonplanar experiment requires roof_mode=none and platform_flange=none; planar roof/flange construction is not compatible.')
    if skeleton_source is None:
        geometry, report = load_preserved_tree(source, mesh, side_gap, contact_gap, tip_radius, roof_clearance_mode, terminal_cap,
                                           geometry_mode, branch_blend, clearance_blend, rim_blend,
                                           branch_sizing, branch_diameter, branch_diameter_angle,minimum_support_height,build_plate_projection,support_ceiling,tip_alignment,tip_length,field_evaluator)
    if model_tree_mode not in ('struts','merged'):
        raise ValueError('Unknown model tree mode.')
    if skeleton_source is not None:
        if (side_gap!=contact_gap or terminal_cap!='rounded' or roof_mode!='none' or platform_flange!='none'
                or model_tree_mode!='struts'):
            raise ValueError('Imported skeleton currently requires uniform gaps, rounded ends, no roof/flange and no secondary merging.')
        from .skeleton_geometry import load_skeleton,sweep_rows
        maximum_body_radius = max(tip_radius, branch_diameter/2) if maximum_trunk_diameter is None else maximum_trunk_diameter/2
        if not np.isfinite(maximum_body_radius) or maximum_body_radius < branch_diameter/2:
            raise ValueError('Maximum trunk diameter must be finite and at least the minimum branch diameter.')
        skeleton,skeleton_report=load_skeleton(skeleton_source,allow_stalled=allow_stalled_skeleton,skip_unrooted=skip_unrooted_contacts)
        if abs(skeleton_report['clearance_mm']-contact_gap)>1e-8:
            raise ValueError('Skeleton and geometry clearance settings differ.')
        direction = np.asarray(build_direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError('Build direction must be finite and nonzero.')
        direction /= np.linalg.norm(direction)
        plate = float(np.min(mesh.points @ direction)) if build_plate_projection is None else float(build_plate_projection)
        rows,normals,roots,sizing = sweep_rows(skeleton,skeleton_report['seed_count'],
            branch_diameter/2,tip_radius,maximum_body_radius,direction,
            allow_stalled=allow_stalled_skeleton,maximum_contact_radius=max(model_root_foot_radius,tip_radius,maximum_body_radius),
            plate=plate,platform_base_radius_factor=platform_base_radius_factor,platform_base_height=platform_base_height,branch_sizing=branch_sizing,branch_diameter_angle=branch_diameter_angle,planar_platform_bases=planar_platform_bases,maximum_platform_radius=None if maximum_platform_trunk_diameter is None else maximum_platform_trunk_diameter/2)
        if support_ceiling not in ('none','contact_max'):
            raise ValueError('Unknown support ceiling mode.')
        maximum_projection = float(np.max(skeleton.points[:skeleton_report['seed_count']] @ direction))-contact_gap if support_ceiling=='contact_max' else None
        geometry = TaperedTree(rows,mesh,direction,plate,side_gap,contact_gap,
            roof_clearance_mode,normals,terminal_cap,geometry_mode,branch_blend,clearance_blend,rim_blend,
            maximum_projection=maximum_projection)
        report = dict(source=str(Path(skeleton_source).resolve()), side_clearance=side_gap,contact_gap=contact_gap,
            terminal_cap=terminal_cap,branch_sizing={'mode':branch_sizing,'branch_diameter_mm':branch_diameter},
            build_plate_projection_mm=plate,minimum_support_height_mm=minimum_support_height or 0.)
        if model_root_foot_radius>0:
            if field_evaluator is None:raise ValueError('Model-root foot sizing requires a callable model field.')
            from .root_foot import widen_roots
            oriented=mesh.compute_normals(cell_normals=True,point_normals=False,consistent_normals=True,auto_orient_normals=True)
            def query(points):
                ids,closest=oriented.find_closest_cell(points,return_closest_point=True)
                return closest,np.asarray(ids,dtype=int)
            skeleton,foot_report=widen_roots(skeleton,skeleton_report['seed_count'],query,oriented.cell_normals,
                field_evaluator,geometry.distance,maximum_radius=model_root_foot_radius,
                minimum_alignment=model_root_foot_alignment,gap=contact_gap,base_radius=branch_diameter/2,
                body_maximum=maximum_body_radius,direction=geometry.direction,footprint_mode=model_root_foot_shape)
            report['root_foot_sizing']=foot_report
        # Preserve actual plate-root labels from the skeleton and add optional
        # plate-clipped vertical feet before the nonplanar scalar is sliced.
        rows,normals,roots,sizing=sweep_rows(skeleton,skeleton_report['seed_count'],
            branch_diameter/2,tip_radius,maximum_body_radius,geometry.direction,
            allow_stalled=allow_stalled_skeleton,maximum_contact_radius=max(model_root_foot_radius,tip_radius,maximum_body_radius),
            plate=geometry.plate,platform_base_radius_factor=platform_base_radius_factor,platform_base_height=platform_base_height,branch_sizing=branch_sizing,branch_diameter_angle=branch_diameter_angle,planar_platform_bases=planar_platform_bases,maximum_platform_radius=None if maximum_platform_trunk_diameter is None else maximum_platform_trunk_diameter/2)
        old=geometry
        geometry=TaperedTree(rows,mesh,old.direction,old.plate,side_gap,contact_gap,
            roof_clearance_mode,normals,terminal_cap,geometry_mode,branch_blend,clearance_blend,rim_blend,
            distance_source=old,maximum_projection=old.maximum_projection,side_mesh=old.side_mesh)
        geometry.platform_roots=[];geometry.source_terminal_map={};geometry.model_contact_roots=roots
        report.update(skeleton_source=str(Path(skeleton_source).resolve()),imported_skeleton=skeleton_report,
            sweep_sizing=sizing,segment_count=len(rows),new_tree_planning=False,
            model_contact_planning='Saved progressive model-local skeleton; tapered sweeps')
        if skeleton_report.get('skipped_contacts'):
            report['skipped_contacts']=skeleton_report['skipped_contacts']
            report['skipped_contact_count']=len(skeleton_report['skipped_contacts'])
            (output/'skipped_contacts.json').write_text(json.dumps(skeleton_report['skipped_contacts'],indent=2))
    elif model_tree_mode=='merged':
        if field_evaluator is None or roof_mode!='none' or platform_flange!='none':
            raise ValueError('Merged model trees require a callable field and no planar roof/flange.')
        from .merged_model_tree import plan
        rows,normals,roots,planning=plan(geometry,field_evaluator,approach=tip_length)
        old=geometry
        geometry=TaperedTree(rows,mesh,old.direction,old.plate,side_gap,contact_gap,
            roof_clearance_mode,normals,terminal_cap,geometry_mode,branch_blend,clearance_blend,rim_blend,
            distance_source=old,maximum_projection=old.maximum_projection,side_mesh=old.side_mesh)
        geometry.platform_roots=[]
        geometry.model_contact_roots=roots
        geometry.source_terminal_map={}
        report.update(model_tree_plan=planning,segment_count=len(rows),new_tree_planning=True,
                      model_contact_planning='Experimental merged model-rooted forest')
        (output/'model_tree_plan.json').write_text(json.dumps(planning,indent=2))
    from .merged_model_tree import centerlines
    centerlines(geometry.segments).save(output/'support_centerlines.vtp')
    minimum_support_height=report["minimum_support_height_mm"]
    geometry.distance_backend=model_distance_backend
    geometry.query_batch_size=geometry_query_batch_size
    slice_evaluator=field_evaluator
    if field_mode in ('checkpoint','tetrahedral'):
        from .transferred_field import TransferredSupportField
        slice_evaluator=TransferredSupportField(transfer_domain,field_evaluator,geometry,mode=support_field_mode)
    if slicing_mode not in ('online','sampled'):raise ValueError('Unknown slicing mode.')
    active=None;source_name=None
    if include_support_gradients and slicing_mode=='online':
        from .online_slicing import plan_from_current_layers,remap_online_selection
        native_model=field_evaluator
        if native_model is None:
            from SupportGenerationVoxel.model_field import AnalyticYField
            native_model=AnalyticYField()
        planning_model=_planning_model_volume if _planning_model_volume is not None else getattr(native_model,'volume',None)
        if planning_model is None:
            if field_mode!='analytic_y':raise ValueError('Online slicing requires the model field volume; use sampled mode when no model volume is available.')
            planning_model=mesh
        planning_model=planning_model.copy(deep=False)
        model_values=(native_model.evaluate_scalars(planning_model.points) if hasattr(native_model,'evaluate_scalars') else native_model.evaluate_scalars_and_gradients(planning_model.points)[0])
        planning_model.point_data['__online_model_scalar']=model_values
        active,source_name=support_layer_source(geometry,field_evaluator,slice_evaluator,field_mode,transfer_domain,support_field_mode,layer_seed_spacing)
        old_schedule=np.asarray(schedule,dtype=float).copy()
        projections=geometry.a@geometry.direction
        base_points=geometry.a[projections<=float(projections.min())+.6]
        old_pruning=geometry.prune_pairs
        if geometry.geometry_mode=='hard':geometry.prune_pairs=bool(trim_pair_pruning)
        try:
            schedule,online_report=plan_from_current_layers(planning_model,active,native_model,slice_evaluator,geometry,
                old_schedule[0],old_schedule[-1],model_scalar='__online_model_scalar',base_points=base_points,
                model_spacing=model_layer_spacing,support_spacing=support_layer_spacing,maximum_support_spacing=maximum_support_layer_height)
        finally:geometry.prune_pairs=old_pruning
        indices=remap_online_selection(old_schedule,schedule,indices)
        online_report['selected_indices']=indices.tolist();report['online_slicing']=online_report
        report['support_slicing']=dict(enabled=True,mode='online',shared_schedule=True,levels=len(schedule))
        np.savetxt(output/'main_scalar_values.txt',schedule,fmt='%.17g')
        print(f"[online slicing] {len(schedule)} shared model/support levels; {online_report['seconds']:.3f}s planning",flush=True)
    elif include_support_gradients:
        from .support_schedule import sample_support_gradients,refine_support_schedule,select_support_indices
        model_schedule=np.asarray(schedule,dtype=float).copy()
        support_values,support_gradients,probe_report=sample_support_gradients(geometry,slice_evaluator,
            spacing=support_gradient_probe_spacing,analytic_y=not nonplanar)
        schedule,slicing_report=refine_support_schedule(model_schedule,support_values,support_gradients,
            target_spacing=support_layer_spacing,maximum_spacing=maximum_support_layer_height)
        indices=select_support_indices(model_schedule,schedule,indices)
        slicing_report['probe_sampling']=probe_report
        slicing_report['selected_support_indices']=indices.tolist()
        report['support_slicing']=slicing_report
        print(f"[support schedule] model levels preserved={len(model_schedule)}; support levels={len(schedule)}; "
              f"support gradient probes={len(support_values)}",flush=True)
    else:
        report['support_slicing']=dict(enabled=False,reason='Explicit legacy model-only schedule selection.')
    np.savetxt(output/'support_scalar_values.txt',schedule,fmt='%.17g')
    from .layer_roles import role_indices
    contact_scalars=geometry.b[geometry.terminal,1]
    if nonplanar:
        from .nonplanar import field_values
        contact_scalars=(slice_evaluator.evaluate_scalars(geometry.b[geometry.terminal]) if hasattr(slice_evaluator,'evaluate_scalars')
            else field_values(slice_evaluator,geometry.b[geometry.terminal])[0])
    layer_role_indices=role_indices(schedule,contact_scalars,platform_layer=-1 if nonplanar else 1,contact_layers_below=contact_layers_below)
    from .spatial_roles import region_specs
    role_regions=region_specs(schedule,indices,geometry,slice_evaluator if nonplanar else None,
                              contact_layers_below=contact_layers_below,margin=layer_seed_spacing)
    report.update(layer_role_indices)
    if np.any(geometry.model_bridge) or hasattr(geometry,'model_contact_roots'):
        base_points=getattr(geometry,'model_contact_roots',geometry.a[geometry.model_bridge])
        base_scalars=(slice_evaluator.evaluate_scalars(base_points) if nonplanar else base_points[:,1]) if len(base_points) else np.empty(0)
        base_indices=np.searchsorted(schedule,base_scalars,side='left')
        base_indices=sorted(set(int(i) for i in base_indices if 0<=i<len(schedule)))
        layer_role_indices['model_base_layer_indices']=base_indices
        layer_role_indices['contact_layer_indices']=sorted(set(layer_role_indices['contact_layer_indices'])|set(base_indices))
        report.update(layer_role_indices)
    report.update(field_mode=field_mode,support_scalar_source='shared global analytic callable' if nonplanar else 'analytic_y',
        tip_gradient_source=field_mode,
        clearance_mode='uniform' if side_gap==contact_gap else 'separate_contact_side',
        side_clearance_frame='model signed distance' if side_gap==contact_gap else 'physical build-height planes')
    if platform_flange not in ("none", "bambu_auto"):
        raise ValueError("Unknown platform flange mode.")
    if platform_flange != "none" and branch_sizing != "bambu":
        raise ValueError("Bambu auto flange currently requires Bambu route sizing.")
    if platform_flange != "none" and (not np.allclose(geometry.direction, [0.,1.,0.]) or not np.isfinite(support_speed) or support_speed < 0):
        raise ValueError("Platform flange requires analytic-Y build direction and finite nonnegative support speed.")
    report["parallelism"] = configure_geometry_threads(geometry_threads)
    physical_bases=None
    if planar_platform_bases:
        from .planar_bases import settings_for,export_bases
        physical_bases=settings_for(geometry)
        if physical_bases is not None:
            base_records=export_bases(geometry,physical_bases,platform_base_layer_height,schedule,output)
            report['planar_platform_bases']=dict(layer_count=len(base_records),layer_height_mm=platform_base_layer_height,
                base_height_mm=physical_bases['height'],scalar_independent=True)
            role_regions=[region for region in role_regions if region['flag']!=1]
    loading_seconds = time.perf_counter() - started
    started = time.perf_counter()
    if (roof_mode=="offset_patch" and minimum_support_height>0
            and roof_terminal in geometry.source_terminal_range and roof_terminal not in geometry.source_terminal_map):
        report["roof"]={"mode":"none","reason":"Configured roof contact excluded by minimum support height."}
        roof_mode="none"
    if roof_mode == "offset_patch":
        roof_terminal=geometry.source_terminal_map.get(roof_terminal,roof_terminal)
        from .roof import attach_roof
        geometry, roof_report = attach_roof(geometry, roof_terminal, roof_radius, roof_thickness, roof_rounding,minimum_support_height)
        geometry.roof.patch.save(output / "roof_patch.vtp")
        report["roof"] = {**roof_report, "mode": roof_mode, "thickness": roof_thickness,
                          "rounding": roof_rounding, "layer_spacing": roof_layer_spacing,
                          "layer_seeding": "direct analytic-y plane; independent of solid preview"}
    elif roof_mode != "none":
        raise ValueError("Unknown roof mode.")
    if layer_trim not in ("checked", "surface_scalar", "surface_scalar_bisect"):
        raise ValueError("Unknown layer trim mode.")
    preview_started = time.perf_counter()
    grid = sample_solid(geometry, spacing)
    grid.contour([0], scalars="support_solid").save(output / "support_solid.vtp")
    preview_seconds = time.perf_counter() - preview_started
    del grid
    layer_source_started=time.perf_counter()
    if active is None:
        active,source_name=support_layer_source(geometry,field_evaluator,slice_evaluator,field_mode,transfer_domain,support_field_mode,layer_seed_spacing)
    active.points = np.asarray(active.points, dtype=float)
    if field_mode in ('checkpoint','tetrahedral'):
        active.save(output/'support_scalar_grid.vtu')
        report['support_scalar_source']=slice_evaluator.report['source']
        coverage_points=np.unique(np.vstack((geometry.a,geometry.b,(geometry.a+geometry.b)*.5)),axis=0)
        solid_values=slice_evaluator.evaluate_scalars(coverage_points)
        if len(solid_values):
            tolerance = 1e-9 * max(1., float(np.max(np.abs(schedule))))
            below = int(np.count_nonzero(solid_values < schedule[0] - tolerance))
            above = int(np.count_nonzero(solid_values > schedule[-1] + tolerance))
            report['support_schedule_coverage'] = dict(
                solid_sample_count=len(solid_values),sample_source='sweep_endpoints_and_midpoints',below_schedule=below, above_schedule=above,
                support_scalar_range=[float(solid_values.min()), float(solid_values.max())],
                schedule_scalar_range=[float(schedule[0]), float(schedule[-1])])
            if below or above:
                raise ValueError(
                    f'Shared layer schedule omits {below} lower and {above} upper support samples. '
                    'Extend input.layer_values over the support scalar range, or leave it empty to derive a shared model/support schedule.')
    report.update(preview_grid_spacing=spacing,layer_seed_spacing=layer_seed_spacing,
        layer_extraction_source=source_name,adaptive_layer_tolerance_mm=adaptive_layer_tolerance,
        layer_source_seconds=time.perf_counter()-layer_source_started,
        layer_source_points=active.n_points,layer_source_cells=active.n_cells)
    # Broad positive primitive values are still needed when choosing active
    # grid cells above. Trimming itself only needs exact inside/outside tests.
    if geometry.geometry_mode=='hard':geometry.prune_pairs=bool(trim_pair_pruning)
    directory = output / "tree_layers"
    directory.mkdir()
    manifest = []
    vertex_violations = centroid_violations = 0
    removed_crossing_triangles = 0
    layer_started = time.perf_counter()
    clipping_seconds = 0
    field_requested = field_evaluated = 0
    layer_timings = []
    if layer_workers < 0:
        raise ValueError("Layer worker count must be nonnegative (zero selects automatic).")
    if layer_batch_size<1:
        raise ValueError('Layer batch size must be positive.')
    requested_batch_size=layer_batch_size
    if roof_mode!='none' or platform_flange!='none' or geometry_mode!='hard' or not layer_trim.startswith('surface_scalar'):
        layer_batch_size=1
    tasks = [(int(index), float(schedule[index])) for index in indices]
    batches=[tasks[start:start+layer_batch_size] for start in range(0,len(tasks),layer_batch_size)]
    workers = min(len(indices), layer_workers or (4 if len(indices) >= 32 else 1))
    workers = max(1, min(workers, multiprocessing.cpu_count()))
    workers=min(workers,max(1,len(batches)))
    if vtk_threads_per_worker is not None and (isinstance(vtk_threads_per_worker,bool) or not isinstance(vtk_threads_per_worker,int) or vtk_threads_per_worker < 1):
        raise ValueError('VTK threads per worker must be a positive integer.')
    requested_worker_threads = vtk_threads_per_worker if vtk_threads_per_worker is not None else max(1, geometry_threads // workers)
    worker_threads = effective_geometry_threads(requested_worker_threads)
    settings = {"directory": str(directory), "spacing": layer_seed_spacing,"role_regions":role_regions,"planar_base_settings":physical_bases,
                "adaptive_layer_tolerance":adaptive_layer_tolerance,
                "layer_role_indices":layer_role_indices,
                "spatial_platform":nonplanar,
                "roof_layer_spacing": roof_layer_spacing, "layer_trim": layer_trim,
                "layer_compression": layer_compression,
                "platform_settings": {"first_layer": 1, "roots": geometry.platform_roots, "speed": support_speed} if platform_flange == "bambu_auto" else None}
    executor = None
    if workers > 1:
        tree = geometry.tree if roof_mode == "offset_patch" else geometry
        payload = {"segments": tree.segments, "mesh": tree.mesh, "direction": tree.direction,
                   "plate": tree.plate, "side_gap": tree.side_gap, "contact_gap": tree.contact_gap,
                   "roof_clearance_mode": tree.roof_clearance_mode, "terminal_normals": tree.tip_normals,
                   "terminal_cap": tree.terminal_cap, "geometry_mode": tree.geometry_mode,
                   "branch_blend": tree.branch_blend, "clearance_blend": tree.clearance_blend,
                   "rim_blend": tree.rim_blend,"maximum_projection":tree.maximum_projection,"side_mesh":tree.side_mesh,
                   "distance_backend":tree.distance_backend,"query_batch_size":tree.query_batch_size,"prune_pairs":tree.prune_pairs}
        worker_settings = {**settings, "roof_settings": {"terminal": roof_terminal,
            "radius": roof_radius, "thickness": roof_thickness, "rounding": roof_rounding,
            "minimum_support_height":minimum_support_height} if roof_mode == "offset_patch" else None}
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_layer_worker,
            initargs=(payload, worker_settings, None if roof_mode == "offset_patch" else active, worker_threads))
        results = chain.from_iterable(executor.map(run_layer_batch_worker,batches,chunksize=1))
    else:
        if vtk_threads_per_worker is not None:
            configure_geometry_threads(worker_threads)
        results=chain.from_iterable(extract_layer_batch(geometry,active,batch,**settings) for batch in batches)
    print(f"Layer extraction: {workers} process(es), {worker_threads if workers > 1 or vtk_threads_per_worker is not None else effective_geometry_threads(geometry_threads)} VTK thread(s) per process, batch={layer_batch_size}", flush=True)
    try:
        for number, result in enumerate(results, 1):
            item, timing = result["manifest"], result["timing"]
            if nonplanar:
                from .nonplanar import slice_error
                item.update(slice_error(pv.read(item['file']).triangulate(),slice_evaluator,item['scalar']))
            manifest.append(item)
            layer_timings.append(timing)
            clipping_seconds += timing["trim_seconds"]
            removed_crossing_triangles += result["removed"]
            vertex_violations += result["vertex_bad"]
            centroid_violations += result["centroid_bad"]
            field_requested += result["requested"]
            field_evaluated += result["evaluated"]
            if result["platform_flange"] is not None:
                report["platform_flange"] = {"mode": platform_flange, "layer": item["layer"],
                    "support_speed_mm_s": support_speed, **result["platform_flange"]}
            print(f"Tree layers {number}/{len(indices)} (index {item['layer']}): {item['cells']} triangles; "
                  f"trim={timing['trim_seconds']:.3f}s, total={timing['total_seconds']:.3f}s", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    report.update({"geometry_spacing": spacing, "vertex_constraint_violations": vertex_violations,
        "support_role_classification":"connected_patch_endpoint_proximity_v1",
        "support_role_regions":role_regions,
        "layer_trim": layer_trim, "solid_preview_seconds": preview_seconds,
        "layer_compression": layer_compression,
        "boundary_root_method": "adaptive" if layer_trim == "surface_scalar" or (layer_trim=='checked' and roof_mode=='none') else "bisect",
        "layer_processing_seconds": time.perf_counter() - layer_started,
        "layer_clipping_seconds": clipping_seconds,
        "layer_workers": workers, "layer_batch_size":layer_batch_size,"requested_layer_batch_size":requested_batch_size,
        "model_distance_backend":model_distance_backend,"geometry_query_batch_size":geometry_query_batch_size,"trim_pair_pruning":trim_pair_pruning,
        "vtk_threads_per_worker": worker_threads if workers > 1 or vtk_threads_per_worker is not None else effective_geometry_threads(geometry_threads),
        "requested_vtk_threads_per_worker": requested_worker_threads,
        "layer_timing_scope": "processing is parent wall time including pool startup/shutdown; clipping and per-layer timings are summed worker elapsed times, which overlap in parallel runs",
        "layer_timings": layer_timings,
        "field_requested_points": field_requested,
        "field_evaluated_points": field_evaluated,
        "source_loading_seconds": loading_seconds,
        "geometry_and_layer_export_seconds": time.perf_counter() - started,
        "removed_crossing_triangles": removed_crossing_triangles,
        "centroid_constraint_violations": centroid_violations, "layer_count": len(manifest),
        "maximum_layer_y_span": max((x["y_span"] for x in manifest), default=0),
        "limitations": ["Preserved skeleton; no new Bambu planner.", "No mandatory contact-layer footprint or roof validation.", "Finite triangle sampling does not certify complete faces.", "Finite candidate resolution can miss small terminal components."]})
    report['platform_layer_indices']=[r['layer'] for r in manifest if r['is_platform_layer']]
    report['contact_layer_indices']=[r['layer'] for r in manifest if r['is_contact_layer']]
    report['model_base_layer_indices']=[r['layer'] for r in manifest if r['is_model_base_layer']]
    if nonplanar:
        report.update(platform_layer_indices=[r['layer'] for r in manifest if r['is_platform_layer']],
            maximum_vertex_scalar_error=max((r['vertex_scalar_error_max'] for r in manifest),default=0.),
            maximum_interior_scalar_error=max((r['interior_scalar_error_max'] for r in manifest),default=0.))
        report['limitations']+=[('Shared analytic field experiment.' if field_mode=='analytic_curved' else 'Neural near-model band joined to reconstructed exterior field; spacing/coverage not certified.'),
            'Nonlinear isosurfaces are piecewise-linear approximations; scalar residuals are reported.',
            ('Saved contact set and trunk routing were not recomputed for the curved field.' if field_mode=='analytic_curved' else 'Neural-model contact regions and scalar-ordered model-rooted supports.'),
            'Physical side gaps use build-height planes, not curved-layer tangent planes.']
        if field_mode in ('checkpoint','tetrahedral'):
            report['transferred_support_field']=slice_evaluator.report
    (output / "tree_layer_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "tree_geometry_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    import argparse
    import time
    from .output import prepare_output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--spacing", type=float, default=1)
    parser.add_argument('--field-mode',choices=('analytic_y','analytic_curved'),default='analytic_y')
    parser.add_argument('--curvature',type=float,default=.002)
    parser.add_argument("--side-clearance", type=float, default=0.4)
    parser.add_argument("--contact-gap", type=float, default=0.2)
    parser.add_argument("--tip-radius", type=float, default=0.6)
    parser.add_argument("--roof-clearance-mode", choices=("plateau", "legacy_blend"), default="plateau")
    parser.add_argument("--terminal-cap", choices=("flat_disc", "rounded"), default="flat_disc")
    parser.add_argument("--support-ceiling",choices=("none","contact_max"),default="none")
    parser.add_argument("--tip-alignment",choices=("legacy","gradient"),default="legacy")
    parser.add_argument("--tip-length",type=float,default=3.)
    parser.add_argument("--contact-layers-below",type=int,default=3)
    parser.add_argument("--geometry-mode", choices=("hard", "smooth"), default="hard")
    parser.add_argument("--branch-blend", type=float, default=0.3)
    parser.add_argument("--clearance-blend", type=float, default=0.1)
    parser.add_argument("--rim-blend", type=float, default=0.1)
    parser.add_argument("--roof-mode", choices=("none", "offset_patch"), default="none")
    parser.add_argument("--roof-terminal", type=int, default=3939)
    parser.add_argument("--roof-radius", type=float, default=3)
    parser.add_argument("--roof-thickness", type=float, default=0.6)
    parser.add_argument("--roof-rounding", type=float, default=0.1)
    parser.add_argument("--roof-layer-spacing", type=float, default=0.1)
    parser.add_argument("--layer-trim", choices=("checked", "surface_scalar", "surface_scalar_bisect"), default="checked")
    parser.add_argument('--layer-seed-spacing',type=float,default=1.)
    parser.add_argument('--adaptive-layer-tolerance',type=float,default=.05)
    parser.add_argument("--geometry-threads", type=int, default=8)
    parser.add_argument("--layer-compression", choices=("zlib", "lz4", "none"), default="zlib")
    parser.add_argument("--layer-workers", type=int, default=0, help="Layer processes; 0 selects four for >=32 layers, otherwise one.")
    parser.add_argument('--layer-batch-size',type=int,default=1)
    parser.add_argument("--branch-sizing", choices=("preserved", "bambu"), default="preserved")
    parser.add_argument("--branch-diameter", type=float, default=5.)
    parser.add_argument("--branch-diameter-angle", type=float, default=5.)
    parser.add_argument("--platform-flange", choices=("none", "bambu_auto"), default="none")
    parser.add_argument("--support-speed", type=float, default=80.)
    parser.add_argument("--minimum-layer", type=int, default=80)
    parser.add_argument("--minimum-support-height",type=float)
    parser.add_argument("--maximum-layer", type=int, default=90)
    args = parser.parse_args()
    if not np.isfinite(args.tip_length) or args.tip_length<=0:
        parser.error('Tip length must be finite and positive.')
    if args.minimum_support_height is not None and (not np.isfinite(args.minimum_support_height) or args.minimum_support_height<0):
        parser.error("Minimum support height must be finite and nonnegative.")
    if args.geometry_threads < 1:
        parser.error("Geometry thread count must be positive.")
    if args.layer_workers < 0:
        parser.error("Layer worker count must be nonnegative.")
    if args.spacing <= 0 or args.tip_radius <= 0 or not 0 <= args.contact_gap <= args.side_clearance:
        parser.error("Require positive spacing/radius and 0 <= contact gap <= side clearance.")
    if args.output.resolve() == args.source.resolve() or args.output.resolve() in args.source.resolve().parents:
        parser.error("Output must be separate from preserved source.")
    output = prepare_output(args.output, clean=args.clean_output)
    started = time.perf_counter()
    cfg = json.loads((args.source / "resolved_config.json").read_text())
    if cfg["input"]["field_mode"] != "analytic_y":
        parser.error("Standalone geometry experiment requires the analytic-y preserved tree.")
    mesh = pv.read(cfg["input"]["mesh"]).triangulate().clean().compute_normals(consistent_normals=True, auto_orient_normals=True)
    schedule = np.loadtxt(args.source / "main_scalar_values.txt")
    field_evaluator=None
    if args.field_mode=='analytic_curved':
        from .generate import AnalyticCurvedField
        field_evaluator=AnalyticCurvedField(args.curvature)
    indices = np.arange(max(0, args.minimum_layer), min(len(schedule) - 1, args.maximum_layer) + 1)
    report = export_geometry(args.source, mesh, schedule, indices, output, args.spacing, args.side_clearance, args.contact_gap, args.tip_radius, args.roof_clearance_mode, args.terminal_cap,
                             args.geometry_mode, args.branch_blend, args.clearance_blend, args.rim_blend,
                             args.roof_mode, args.roof_terminal, args.roof_radius, args.roof_thickness, args.roof_rounding, args.roof_layer_spacing,
                             args.layer_trim, args.geometry_threads, args.layer_workers, args.layer_compression,
                             args.branch_sizing, args.branch_diameter, args.branch_diameter_angle, args.platform_flange, args.support_speed,
                             args.minimum_support_height,None,args.support_ceiling,args.tip_alignment,args.tip_length,args.contact_layers_below,field_evaluator,args.field_mode,layer_batch_size=args.layer_batch_size,
                             layer_seed_spacing=args.layer_seed_spacing,adaptive_layer_tolerance=args.adaptive_layer_tolerance)
    elapsed = time.perf_counter() - started
    np.savetxt(output/'main_scalar_values.txt',schedule)
    (output / "summary.json").write_text(json.dumps({"mesh": cfg["input"]["mesh"], "field_mode":args.field_mode,"envelope": {"build_direction": cfg["domain"]["build_direction"]}, "tree_geometry": report, "total_seconds": elapsed}, indent=2))
    print(f"Geometry module: {elapsed:.2f}s", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

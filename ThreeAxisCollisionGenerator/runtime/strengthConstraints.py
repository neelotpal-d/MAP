"""Principal-direction alignment for the fast fixed-tool three-axis trainer."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from stressSampling import filtered_farthest_indices, FullPassBatches


class StrengthConstraints:
    def __init__(self, args, output):
        import igl
        vertices, faces = igl.read_triangle_mesh(str(args.mesh_path))
        self.center = (vertices.min(0)+vertices.max(0))/2
        self.scale = float((vertices.max(0)-vertices.min(0)).max()/2)
        rows = np.loadtxt(args.strength_file, ndmin=2)
        source_row_ids = np.arange(len(rows))
        columns = args.strength_point_columns+args.strength_direction_columns
        if not args.strength_all_signs:
            columns += [args.strength_signed_column]
        if min(columns) < 0 or max(columns) >= rows.shape[1]:
            raise ValueError('Strength column mapping exceeds the input file; tensile-only mode needs a signed principal-stress column')
        points, directions = rows[:, args.strength_point_columns], rows[:, args.strength_direction_columns]
        lengths = np.linalg.norm(directions, axis=1)
        good = np.isfinite(points).all(1) & np.isfinite(directions).all(1) & (lengths > 1e-10)
        signed = np.ones(len(rows)) if args.strength_all_signs else rows[:, args.strength_signed_column]
        good &= np.isfinite(signed)
        points, directions, signed = points[good], directions[good]/lengths[good, None], signed[good]
        source_row_ids = source_row_ids[good]
        inside = igl.signed_distance(points, vertices, faces)[0] <= 1e-4
        points, directions, signed = points[inside], directions[inside], signed[inside]
        source_row_ids = source_row_ids[inside]
        if len(points) < 5:
            raise ValueError('Fewer than five valid strength samples inside the mesh; check coordinates, units and column mapping')
        weights = np.ones(len(points)) if args.strength_all_signs else (signed > args.strength_min_tensile_stress).astype(float)
        if not weights.any():
            raise ValueError('No active tensile samples; check sign convention or explicitly select all-signs mode')
        order = np.random.default_rng(args.strength_seed).permutation(len(points))
        nval = min(2048, max(1, len(points)//5))
        val, train = order[:nval], order[nval:]
        if not weights[train].any():
            # Do not turn a very sparse tensile dataset into a zero-loss run.
            raise ValueError('No tensile samples remain in training after holdout; supply more tensile samples')
        self.tree = cKDTree(points[train])
        self.projectors = directions[train, :, None]*directions[train, None, :]*weights[train, None, None]
        self.sampling = getattr(args, 'strength_sampling', 'interior')
        self.requested_samples = getattr(args, 'strength_samples', 0)
        self.requested_batch = getattr(args, 'strength_batch_size', 0)
        self.seed = args.strength_seed
        self.fps_filter_order = getattr(args, 'strength_fps_filter_order', 'after')
        self.source_ids = source_row_ids[train]
        self.source_points = points[train]
        self.source_directions = directions[train]
        self.source_active = weights[train] > 0
        self.validation_points = points[val]
        self.validation_directions = directions[val]
        self.validation_weights = weights[val]
        self.k = min(args.strength_neighbors, len(train))
        self.distance = args.strength_max_distance_mm
        self.weight = args.strength_weight
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.metadata = dict(source=str(Path(args.strength_file).resolve()),
                             sha256=hashlib.sha256(Path(args.strength_file).read_bytes()).hexdigest(),
                             valid_inside_samples=len(points), training_sources=len(train), heldout_sources=len(val),
                             active_tensile_sources=int(weights.sum()), tensile_only=not args.strength_all_signs,
                             signed_column=None if args.strength_all_signs else args.strength_signed_column,
                             point_columns=args.strength_point_columns, direction_columns=args.strength_direction_columns,
                             min_tensile_stress=args.strength_min_tensile_stress, weight=self.weight,
                             neighbors=self.k, max_distance_mm=self.distance,
                             sampling=self.sampling,
                             interpolation='inverse-distance weighted direction projectors' if self.sampling == 'interior' else 'none; original source direction projectors',
                             objective='principal direction tangent to layer; not a separate in-layer toolpath field',
                             resampling=('fixed FPS pool; independent shuffled full pass each epoch; partial batches retained'
                                         if self.sampling == 'fps' else
                                         'query current shuffled/resampled interior training batch; direction projector interpolation'))
        (self.output/'strength_config.json').write_text(json.dumps(self.metadata, indent=2))

    def prepare_fps(self, interior_count, interior_batch_size, device):
        if self.sampling != 'fps':
            return
        count = self.requested_samples or interior_count
        pool_size = int(self.source_active.sum()) if self.fps_filter_order == 'before' else len(self.source_points)
        print(f'Selecting FPS stress sources: {min(count, pool_size)}/{pool_size}; tensile filter {self.fps_filter_order} FPS', flush=True)
        ids, before_filter = filtered_farthest_indices(self.source_points, self.source_active,
                                                       count, self.seed+502, self.fps_filter_order)
        points, directions = self.source_points[ids], self.source_directions[ids]
        self.fps_points = torch.tensor((points-self.center)/self.scale, dtype=torch.float32, device=device)
        self.fps_projectors = torch.tensor(directions[:, :, None]*directions[:, None, :], dtype=torch.float32, device=device)
        self.stream = FullPassBatches(len(ids), self.requested_batch or interior_batch_size, self.seed+503)
        self.metadata.update(selected_fps_samples=len(ids), requested_fps_samples=count,
                             fps_filter_order=self.fps_filter_order,
                             fps_source_pool_samples=pool_size,
                             fps_selected_before_filter=len(before_filter),
                             fps_excluded_after_filter=len(before_filter)-len(ids),
                             fps_batch_size=self.stream.batch_size,
                             selection=f'heldout split first; tensile filtering {self.fps_filter_order} FPS; original source rows; no replenishment',
                             resampling='fixed FPS pool; independent shuffled full pass each epoch; partial batches retained',
                             interpolation='none; original source direction projectors')
        np.savez_compressed(self.output/'strength_fps_sources.npz', source_ids=self.source_ids[ids],
                            source_ids_before_filter=self.source_ids[before_filter],
                            points_mm=points, directions=directions)
        print(f'FPS retained {len(ids)} tensile sources from {len(before_filter)} selected; '
              f'discarded {len(before_filter)-len(ids)} without replacement.', flush=True)
        (self.output/'strength_config.json').write_text(json.dumps(self.metadata, indent=2))

    def next_fps_batch(self):
        ids = self.stream.draw()
        self.batch_ids = torch.as_tensor(ids, device=self.fps_points.device)
        return self.fps_points[self.batch_ids]

    def begin_epoch(self, epoch):
        self.epoch, self.sum_loss, self.batches, self.accepted, self.rejected, self.active = epoch+1, None, 0, 0, 0, 0
        if self.sampling == 'fps':
            self.stream.begin_epoch(epoch)

    def _loss(self, gradients, matrices):
        normals = torch.nn.functional.normalize(gradients, dim=1)
        loss = torch.einsum('bi,bij,bj->b', normals, matrices, normals).mean()
        # Count points so the final partial batch is represented correctly in logs.
        value = loss.detach()*len(gradients)
        self.sum_loss = value if self.sum_loss is None else self.sum_loss+value
        self.batches += len(gradients)
        return self.weight*loss

    def __call__(self, model_points, gradients):
        if self.sampling == 'fps':
            self.accepted += len(gradients)
            self.active += len(gradients)
            return self._loss(gradients, self.fps_projectors[self.batch_ids])
        world = model_points.detach().cpu().numpy()*self.scale+self.center
        distances, ids = self.tree.query(world, k=self.k)
        distances, ids = distances.reshape(len(world), self.k), ids.reshape(len(world), self.k)
        weights = 1/np.maximum(distances, 1e-8)**2
        weights /= weights.sum(1, keepdims=True)
        matrices = (self.projectors[ids]*weights[:, :, None, None]).sum(1)
        valid = np.ones(len(world), dtype=bool) if self.distance is None else distances[:, 0] <= self.distance
        self.accepted += int(valid.sum()); self.rejected += int((~valid).sum())
        # Rejected queries are zero-weighted, not extrapolated. Keep a fixed
        # batch denominator so sparse support does not amplify the objective.
        matrices[~valid] = 0
        self.active += int((np.trace(matrices, axis1=1, axis2=2) > 1e-8).sum())
        matrices = torch.tensor(matrices, dtype=gradients.dtype, device=gradients.device)
        return self._loss(gradients, matrices)

    def end_epoch(self, scalar_net):
        if not self.accepted:
            raise ValueError('No strength queries within interpolation support this epoch; increase max distance or check source coverage')
        record = dict(epoch=self.epoch, weighted_strength_loss=self.weight*float(self.sum_loss/max(1, self.batches)),
                      accepted_queries=self.accepted, rejected_queries=self.rejected, active_queries=self.active)
        if self.sampling == 'fps':
            covered = int(self.stream.seen.sum())
            if covered != len(self.fps_points):
                raise RuntimeError('Incomplete FPS stress coverage in this epoch')
            record.update(fps_unique_sources_seen=covered, fps_sources=len(self.fps_points))
        device = next(scalar_net.parameters()).device
        with torch.enable_grad():
            x = torch.tensor((self.validation_points-self.center)/self.scale, dtype=torch.float32, device=device, requires_grad=True)
            values = scalar_net(x, compute_grads=False, compute_hess=False)['scalars']
            gradients = torch.autograd.grad(values.sum(), x, create_graph=False)[0]
        normals = torch.nn.functional.normalize(gradients.detach(), dim=1).cpu().numpy()
        errors = (normals*self.validation_directions).sum(1)**2
        record['heldout_weighted_mean_squared_dot'] = float((self.validation_weights*errors).mean())
        record['heldout_active_mean_squared_dot'] = float(errors[self.validation_weights > 0].mean()) if (self.validation_weights > 0).any() else None
        self.rows.append(record)
        with (self.output/'strength_metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        print('strength '+json.dumps(record), flush=True)

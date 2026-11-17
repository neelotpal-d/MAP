"""Sign-invariant principal-direction constraints and spatial resampling."""
import numpy as np
import torch
from scipy.spatial import cKDTree


class StressRegion:
    """Classify world-space queries by signed stress on their containing cell."""
    def __init__(self, volume, signed_field, region, threshold=0., top_percent=100.):
        import pyvista as pv
        if region not in ('all', 'tensile', 'compressive'):
            raise ValueError('Stress region must be all, tensile or compressive')
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('Stress threshold must be finite and nonnegative')
        if not np.isfinite(top_percent) or not 0 < top_percent <= 100:
            raise ValueError('Top stress percentage must be greater than zero and at most 100')
        self.mesh = pv.read(volume)
        if signed_field not in self.mesh.cell_data:
            raise ValueError(f'Signed cell stress field not found: {signed_field}')
        self.values = np.asarray(self.mesh.cell_data[signed_field], dtype=float)
        if self.values.shape != (self.mesh.n_cells,):
            raise ValueError('Signed stress must be a scalar cell field')
        if not self.mesh.n_cells:
            raise ValueError('Stress selection requires a nonempty volume mesh')
        self.region, self.threshold = region, threshold
        eligible = np.isfinite(self.values)
        if region == 'tensile':
            eligible &= self.values > threshold
        elif region == 'compressive':
            eligible &= self.values < -threshold
        # Region=all retains its existing behavior: the absolute threshold is
        # used only for tensile/compressive regions. Rank by magnitude, not sign.
        self.cell_mask = eligible.copy()
        magnitudes = np.abs(self.values[eligible])
        cutoff = None
        if top_percent < 100:
            if not len(magnitudes):
                raise ValueError('No finite FEA cells satisfy the selected stress region and threshold')
            count = max(1, int(np.ceil(len(magnitudes)*top_percent/100.)))
            cutoff = float(np.partition(magnitudes, len(magnitudes)-count)[len(magnitudes)-count])
            self.cell_mask &= np.abs(self.values) >= cutoff
        self.metadata = dict(region=region, threshold=float(threshold), top_percent=float(top_percent),
                             eligible_cells=int(eligible.sum()), selected_cells=int(self.cell_mask.sum()),
                             magnitude_cutoff=cutoff,
                             ranking='absolute signed stress on all eligible FEA cells, equal cell count',
                             ties='include all cells at the cutoff; selected fraction may exceed the requested percentage')

    def mask(self, points):
        if not len(points):
            return np.empty(0, dtype=bool)
        cells = np.asarray(self.mesh.find_containing_cell(np.asarray(points))).reshape(-1)
        valid = cells >= 0
        return valid & self.cell_mask[np.maximum(cells, 0)]


def farthest_indices(points, count, seed):
    """Deterministic spatial subset of original sources; never interpolate FPS targets."""
    count = min(count, len(points))
    chosen = np.empty(count, dtype=np.int64)
    distances = np.full(len(points), np.inf)
    index = int(np.random.default_rng(seed).integers(len(points)))
    for i in range(count):
        chosen[i] = index
        distances = np.minimum(distances, np.sum((points-points[index])**2, axis=1))
        distances[chosen[:i+1]] = -1
        index = int(np.argmax(distances))
    return chosen


def read_directions(path, point_columns=(0, 1, 2), direction_columns=(3, 4, 5)):
    rows = np.loadtxt(path, ndmin=2)
    columns = list(point_columns)+list(direction_columns)
    if len(point_columns) != 3 or len(direction_columns) != 3 or min(columns) < 0 or max(columns) >= rows.shape[1]:
        raise ValueError('Stress point/direction column indices must select three existing columns each')
    points, directions = rows[:, point_columns], rows[:, direction_columns]
    norms = np.linalg.norm(directions, axis=1)
    valid = np.isfinite(points).all(1) & np.isfinite(directions).all(1) & (norms > 1e-10)
    if valid.sum() < 5:
        raise ValueError('At least five finite, nonzero stress-direction samples are required')
    return points[valid], directions[valid]/norms[valid, None], int((~valid).sum())


class DirectionInterpolator:
    def __init__(self, points, directions, neighbors=8, max_distance_mm=None):
        self.tree = cKDTree(points)
        self.projectors = directions[:, :, None]*directions[:, None, :]
        self.neighbors = min(neighbors, len(points))
        self.max_distance_mm = max_distance_mm

    def query(self, points):
        distances, ids = self.tree.query(points, k=self.neighbors)
        distances = distances.reshape(len(points), self.neighbors)
        ids = ids.reshape(len(points), self.neighbors)
        weights = 1/np.maximum(distances, 1e-8)**2
        weights /= weights.sum(1, keepdims=True)
        matrices = (weights[:, :, None, None]*self.projectors[ids]).sum(1)
        valid = np.ones(len(points), dtype=bool) if self.max_distance_mm is None else distances[:, 0] <= self.max_distance_mm
        return matrices, valid, distances[:, 0]


class StressConstraints:
    def __init__(self, args, center, scale, device, vertices, faces):
        top_percent = float(getattr(args, 'stress_top_percent', 100.))
        if not np.isfinite(top_percent) or not 0 < top_percent <= 100:
            raise ValueError('Top stress percentage must be greater than zero and at most 100')
        import igl
        points, directions, removed = read_directions(args.stress_file, args.stress_point_columns, args.stress_direction_columns)
        if args.stress_coordinate_space == 'model':
            points = points*scale+center
        inside = igl.signed_distance(points, vertices, faces)[0] <= 1e-4
        if inside.sum() < 5:
            raise ValueError('Stress coordinates do not overlap the part; verify units, transforms and column mapping')
        points, directions = points[inside], directions[inside]
        self.region = None
        if getattr(args, 'stress_region', 'all') != 'all' or top_percent < 100:
            self.region = StressRegion(args.volume_mesh, args.stress_signed_field, args.stress_region,
                                       args.stress_threshold, top_percent)
        self.sampling = getattr(args, 'stress_sampling', 'resampled')
        self.seed = args.seed
        order = np.random.default_rng(args.seed+501).permutation(len(points))
        count = min(4096, max(1, len(points)//5))
        heldout, train = order[:count], order[count:]
        self.points, self.directions = points[train], directions[train]
        self.fps_points, self.fps_directions = self.points.copy(), self.directions.copy()
        if self.region is not None:
            keep = self.region.mask(self.points)
            self.points, self.directions = self.points[keep], self.directions[keep]
            heldout = heldout[self.region.mask(points[heldout])]
            if not len(self.points) or not len(heldout):
                raise ValueError('Selected stress region/percentage needs both training and held-out stress samples')
        self.interpolator = DirectionInterpolator(self.points, self.directions, args.stress_neighbors, args.stress_max_distance_mm)
        self.center, self.scale, self.device = center, scale, device
        self.count = args.stress_samples
        self.validation_points = torch.tensor((points[heldout]-center)/scale, dtype=torch.float32, device=device)
        self.validation_directions = torch.tensor(directions[heldout], dtype=torch.float32, device=device)
        self.metadata = dict(source_samples=len(points), training_source_samples=len(self.points), validation_source_samples=len(heldout),
                             invalid_rows_removed=removed, outside_rows_removed=int((~inside).sum()),
                             resampled_per_epoch=self.count, interpolation='inverse-distance weighted direction projectors',
                             neighbors=args.stress_neighbors, max_distance_mm=args.stress_max_distance_mm,
                             sampling=self.sampling, region=getattr(args, 'stress_region', 'all'),
                             top_percent=float(top_percent), selection=self.region.metadata if self.region else None,
                             objective='mean(n.T @ projector @ n); principal direction tangent to layer, not an in-layer toolpath direction')

    def resample(self, domain, generator):
        if self.sampling == 'fps':
            if hasattr(self, 'points_epoch'):
                return len(self.points_epoch)
            ids = farthest_indices(self.fps_points, self.count, self.seed+502)
            points, directions = self.fps_points[ids], self.fps_directions[ids]
            valid = self.region.mask(points) if self.region is not None else np.ones(len(points), dtype=bool)
            if not valid.any():
                raise ValueError('FPS selected no samples in the requested stress region/percentage; increase the percentage or reduce the interior grid spacing')
            points, directions = points[valid], directions[valid]
            self.points_epoch = torch.tensor((points-self.center)/self.scale, dtype=torch.float32, device=self.device)
            self.matrices_epoch = torch.tensor(directions[:, :, None]*directions[:, None, :], dtype=torch.float32, device=self.device)
            self.rejected = int((~valid).sum())
            self.max_query_distance_mm = 0.
            self.metadata.update(requested_fps_samples=self.count, selected_fps_samples=len(points),
                                 resampled_per_epoch=0, interpolation='none; original source direction projectors')
            return len(points)
        # Only the first half of domain.sample(2N) is uniform volume sampling.
        points = domain.sample(2*self.count, generator)[:self.count]
        world = points.cpu().numpy()*self.scale+self.center
        matrices, valid, distances = self.interpolator.query(world)
        if self.region is not None:
            valid &= self.region.mask(world)
        if not valid.any():
            raise ValueError('No resampled points within stress interpolation distance; increase the limit or check stress coverage')
        self.points_epoch = points[torch.tensor(valid, device=self.device)]
        self.matrices_epoch = torch.tensor(matrices[valid], dtype=torch.float32, device=self.device)
        self.rejected = int((~valid).sum())
        self.max_query_distance_mm = float(distances[valid].max())
        return len(self.points_epoch)

    def loss(self, gradients, indices):
        n = torch.nn.functional.normalize(gradients, dim=1)
        return torch.einsum('bi,bij,bj->b', n, self.matrices_epoch[indices], n).mean()

    def evaluate(self, field):
        from fields import first_order_output
        scores = []
        for points, directions in zip(self.validation_points.split(4096), self.validation_directions.split(4096)):
            n = torch.nn.functional.normalize(first_order_output(field, points)['grads'], dim=1)
            scores.append((n*directions).sum(1).square())
        return float(torch.cat(scores).mean())

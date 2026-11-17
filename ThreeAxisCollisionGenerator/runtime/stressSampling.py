"""Independent full-pass stress batches with original, sign-invariant targets."""
import math
import numpy as np


def _farthest_indices_full(points, count, seed):
    """Reference path for small pools, other dtypes and optional-SciPy fallback."""
    count = min(int(count), len(points))
    if count < 1:
        raise ValueError('FPS needs at least one point')
    selected = np.empty(count, dtype=np.int64)
    nearest = np.full(len(points), np.inf)
    index = int(np.random.default_rng(seed).integers(len(points)))
    for i in range(count):
        selected[i] = index
        delta = points-points[index]
        np.minimum(nearest, np.einsum('ij,ij->i', delta, delta), out=nearest)
        nearest[selected[:i+1]] = -1
        index = int(np.argmax(nearest))
    return selected


def farthest_indices(points, count, seed):
    """Exact original-row FPS, pruning distance updates with a spatial tree.

    The chosen row has the largest current nearest distance R. Every other
    nearest distance is <=R, so a point farther than R from the chosen row
    cannot improve. The tree only provides a conservative candidate set;
    squared distances, first-index argmax ties and the seed remain unchanged.
    """
    count=min(int(count),len(points))
    if count<1:raise ValueError('FPS needs at least one point')
    if (not isinstance(points,np.ndarray) or points.ndim!=2 or points.shape[1]!=3
            or points.dtype!=np.dtype('float64') or len(points)<512 or count<64
            or not np.isfinite(points).all() or np.max(abs(points))>1e100):
        return _farthest_indices_full(points,count,seed)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return _farthest_indices_full(points,count,seed)
    tree=cKDTree(points)
    selected=np.empty(count,dtype=np.int64)
    nearest=np.full(len(points),np.inf)
    index=int(np.random.default_rng(seed).integers(len(points)))
    coordinate_scale=max(1.,float(np.max(abs(points))))
    for i in range(count):
        selected[i]=index
        bound=nearest[index]
        if np.isinf(bound):
            delta=points-points[index]
            np.minimum(nearest,np.einsum('ij,ij->i',delta,delta),out=nearest)
        elif bound>0:
            radius=np.sqrt(bound)
            # Enlarge only the search, including a coordinate-scale roundoff
            # allowance. Extra candidates use the exact reference arithmetic.
            radius=np.nextafter(radius+32*np.finfo(float).eps*(radius+coordinate_scale),np.inf)
            ids=np.asarray(tree.query_ball_point(points[index],radius),dtype=np.int64)
            delta=points[ids]-points[index]
            nearest[ids]=np.minimum(nearest[ids],np.einsum('ij,ij->i',delta,delta))
        # Already-selected entries stay -1 under nonnegative squared distances.
        # A zero bound needs no update, including fully duplicated coordinates.
        nearest[index]=-1
        index=int(np.argmax(nearest))
    return selected


def filtered_farthest_indices(points, active, count, seed, filter_order='after'):
    """Return retained and pre-filter source IDs without replacing rejected FPS points."""
    active = np.asarray(active, dtype=bool)
    if active.shape != (len(points),):
        raise ValueError('FPS activity mask must match the source rows')
    if filter_order not in {'before', 'after'}:
        raise ValueError('FPS filter order must be before or after')
    candidates = np.flatnonzero(active) if filter_order == 'before' else np.arange(len(points))
    selected = candidates[farthest_indices(points[candidates], count, seed)]
    retained = selected[active[selected]]
    if not len(retained):
        raise ValueError('FPS retained no tensile sources; increase the FPS count or inspect stress signs')
    return retained, selected


class PartitionedEpochBatches:
    """Partition every source exactly once across a fixed number of optimizer steps."""
    def __init__(self, count, seed):
        if count < 1:
            raise ValueError('Coverage stream needs at least one source')
        self.count, self.seed = count, seed

    def begin_epoch(self, epoch, steps):
        if steps < 1:
            raise ValueError('Coverage stream needs a positive step count')
        order = np.random.default_rng(self.seed+epoch).permutation(self.count)
        self.batches = np.array_split(order, steps)
        self.position = 0
        self.seen = np.zeros(self.count, dtype=bool)

    def draw(self):
        if self.position >= len(self.batches):
            raise RuntimeError('Coverage stream exhausted before the epoch finished')
        ids = self.batches[self.position]
        self.position += 1
        self.seen[ids] = True
        return ids


class FullPassBatches:
    """Shuffle once per epoch, keep the partial batch, cycle shorter streams."""
    def __init__(self, count, batch_size, seed):
        if count < 1 or batch_size < 1:
            raise ValueError('Full-pass batching needs positive sizes')
        self.count, self.batch_size, self.seed = count, batch_size, seed

    def __len__(self):
        return math.ceil(self.count/self.batch_size)

    def begin_epoch(self, epoch):
        order = np.random.default_rng(self.seed+epoch).permutation(self.count)
        self.batches = [order[i:i+self.batch_size] for i in range(0, self.count, self.batch_size)]
        self.position = 0
        self.seen = np.zeros(self.count, bool)

    def draw(self):
        ids = self.batches[self.position % len(self.batches)]
        self.position += 1
        self.seen[ids] = True
        return ids

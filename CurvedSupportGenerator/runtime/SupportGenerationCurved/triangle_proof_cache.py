"""Bounded exact memoization for one immutable adaptive-refinement call."""
import numpy as np


class TriangleProofCache:
    """Keep exact ordered float64 triangle keys; never share across geometry.

    A fresh instance is required for every adaptive_candidate invocation. The
    evaluator and its geometry/clearance/inset stay fixed for that invocation.
    Reaching the cap stops insertion; it never changes proof decisions. The
    estimate includes conservative Python key/dict overhead, not process RSS.
    """
    def __init__(self, evaluate, maximum_entries=131072):
        if isinstance(maximum_entries, bool) or not isinstance(maximum_entries, int) or maximum_entries < 0:
            raise ValueError('Triangle proof cache limit must be a nonnegative integer.')
        self.evaluate, self.maximum_entries = evaluate, maximum_entries
        self.values = {}
        self.queries = self.hits = self.evaluated = 0

    def __call__(self, triangles):
        triangles = np.ascontiguousarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        self.queries += len(triangles)
        if not self.maximum_entries or not np.isfinite(triangles).all():
            self.evaluated += len(triangles)
            return self.evaluate(triangles)
        keys = triangles.reshape(-1, 9).view('V72').ravel().tolist()
        values = np.fromiter((self.values.get(key, -1) for key in keys), dtype=np.int8, count=len(keys))
        missing = np.flatnonzero(values < 0)
        self.hits += len(keys) - len(missing)
        if len(missing):
            # Preserve the input order for uncached triangles.
            results = np.asarray(self.evaluate(triangles[missing]), dtype=bool)
            values[missing] = results
            self.evaluated += len(missing)
            for index, value in zip(missing, results):
                if len(self.values) >= self.maximum_entries:
                    break
                self.values[keys[int(index)]] = bool(value)
        return values.astype(bool)

    def report(self):
        return dict(queries=self.queries, hits=self.hits, evaluated=self.evaluated,
                    entries=len(self.values), maximum_entries=self.maximum_entries,
                    estimated_retained_bytes=192 * len(self.values))

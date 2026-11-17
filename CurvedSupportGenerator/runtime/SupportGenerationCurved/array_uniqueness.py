"""Numeric row sorting without NumPy's expensive structured-record ordering."""
import numpy as np

def unique_rows_inverse(values):
    """Match ``np.unique(values, axis=0, return_inverse=True)`` for finite rows."""
    values=np.asarray(values)
    if values.ndim!=2:raise ValueError('Expected a two-dimensional array.')
    if not len(values):return values.copy(),np.empty(0,dtype=np.intp)
    if not np.isfinite(values).all():return np.unique(values,axis=0,return_inverse=True)
    order=np.lexsort(tuple(values[:,column] for column in reversed(range(values.shape[1]))))
    ordered=values[order]
    first=np.r_[True,np.any(ordered[1:]!=ordered[:-1],axis=1)]
    inverse=np.empty(len(values),dtype=np.intp);inverse[order]=np.cumsum(first)-1
    return np.ascontiguousarray(ordered[first]),inverse

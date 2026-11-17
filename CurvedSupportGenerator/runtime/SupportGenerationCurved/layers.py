"""Layer export shared by the pipeline and isolated module benchmarks."""
from pathlib import Path
import time
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersCore import vtkContour3DLinearGrid
from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm


def extend_schedule(schedule, scalar_range, maximum_count=10000):
    """Cover an exterior field while preserving every original model level."""
    schedule = np.asarray(schedule, dtype=float).reshape(-1)
    bounds = np.asarray(scalar_range, dtype=float).reshape(-1)
    if (len(schedule) < 2 or not np.isfinite(schedule).all() or np.any(np.diff(schedule) <= 0)
            or bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
        raise ValueError("Schedule extension requires increasing finite levels and a finite scalar range.")
    # Continue the existing edge spacing so extending support coverage does not
    # change model contours or insert new levels between existing model slices.
    before = max(0, int(np.ceil((schedule[0] - bounds[0]) / (schedule[1] - schedule[0]))))
    after = max(0, int(np.ceil((bounds[1] - schedule[-1]) / (schedule[-1] - schedule[-2]))))
    if len(schedule) + before + after > maximum_count:
        raise ValueError("Extended model/support schedule exceeds maximum layer count.")
    prefix = schedule[0] - np.arange(before, 0, -1) * (schedule[1] - schedule[0])
    suffix = schedule[-1] + np.arange(1, after + 1) * (schedule[-1] - schedule[-2])
    return np.concatenate((prefix, schedule, suffix)), dict(
        original_model_levels=len(schedule), prepended_levels=before, appended_levels=after,
        original_model_index_offset=before, coverage_scalar_range=bounds.tolist(),
        method="Continue model schedule edge spacing over transferred support field range")


def export_layers(domain, schedule, indices, output, analytic_y=False, contour_backend="auto", layer_compression="zlib"):
    # VTK contour interpolation follows input point precision. Float32 can
    # introduce several microns of span in an exactly affine planar field.
    # Use float64 geometry for all fields; do not project contours onto y.
    if domain.points.dtype != np.float64:
        domain.points = np.asarray(domain.points, dtype=np.float64)
    if contour_backend not in ("auto", "legacy", "linear_grid"):
        raise ValueError("Unknown envelope contour backend.")
    eligible = isinstance(domain, pv.UnstructuredGrid) and domain.n_cells > 0 and np.all(domain.celltypes == 10)
    if contour_backend == "linear_grid" and not eligible:
        raise ValueError("The linear-grid envelope path requires tetrahedral cells.")
    fast = None
    if eligible and contour_backend != "legacy":
        fast = vtkContour3DLinearGrid()
        fast.SetInputData(domain)
        fast.SetInputArrayToProcess(0, 0, 0, 0, "scalar")
        fast.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)
        fast.SetMergePoints(True)
        fast.SetInterpolateAttributes(True)
        fast.SetComputeScalars(True)
        fast.SetComputeNormals(False)
        fast.SetUseScalarTree(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    started = time.perf_counter()
    for number, index in enumerate(indices, start=1):
        if fast is None:
            layer = domain.contour([float(schedule[index])], scalars="scalar").triangulate().clean()
        else:
            fast.SetValue(0, float(schedule[index]))
            fast.Update()
            # The specialized filter already generates triangles. Cleaning
            # still merges vertices reached along different tetrahedral edges.
            layer = pv.wrap(fast.GetOutput()).clean()
        path = output / f"{index}.vtp"
        layer.save(path, compression=None if layer_compression == "none" else layer_compression)
        yspan = float(np.ptp(layer.points[:, 1])) if layer.n_points else 0.0
        if analytic_y and yspan > 1e-6:
            raise RuntimeError(f"Analytic-y layer {index} has Y span {yspan:g}.")
        manifest.append({"layer": int(index), "scalar": float(schedule[index]), "points": layer.n_points,
                         "cells": layer.n_cells, "area": float(layer.area), "y_span": yspan, "file": str(path.resolve())})
        if number == 1 or number % 50 == 0 or number == len(indices):
            print(f"Layers: {number}/{len(indices)} ({time.perf_counter() - started:.2f}s)", flush=True)
    return manifest

"""
Vectorized hybrid contour extraction wrapper.

This keeps `test_hybrid_contours.py` intact and patches only the hot-path
helpers used by its `generate_hybrid_toolpath()` implementation:

- vectorized triangle range pruning
- one-time triangle-to-level binning
- fused clipping path for vectorized extracted segments
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import test_hybrid_contours as base


_EDGE_START = np.asarray([0, 1, 2], dtype=np.int64)
_EDGE_END = np.asarray([1, 2, 0], dtype=np.int64)

_ORIGINAL_BUILD_SIGNED_LEVELS = base.build_signed_levels
_ORIGINAL_BUILD_UNSIGNED_LEVELS = base.build_unsigned_levels
_ORIGINAL_BUILD_PARALLEL_TRIANGLE_CACHE = base.build_parallel_triangle_cache
_ORIGINAL_CLIP_PARALLEL_SEGMENTS = base.clip_parallel_segments

_LAST_PARALLEL_LEVELS: np.ndarray | None = None
_CACHE_LEVEL_BINS: dict[int, "_LevelBinnedCache"] = {}


@dataclass(slots=True)
class _LevelBinnedCache:
    levels: np.ndarray
    level_face_indices: tuple[np.ndarray, ...]


@dataclass(slots=True)
class _VectorizedSegmentBundle:
    start_xyz: np.ndarray
    end_xyz: np.ndarray
    start_boundary: np.ndarray
    end_boundary: np.ndarray


def _record_signed_levels(*args, **kwargs) -> np.ndarray:
    global _LAST_PARALLEL_LEVELS
    levels = np.asarray(_ORIGINAL_BUILD_SIGNED_LEVELS(*args, **kwargs), dtype=np.float64)
    _LAST_PARALLEL_LEVELS = levels
    return levels


def _record_unsigned_levels(*args, **kwargs) -> np.ndarray:
    global _LAST_PARALLEL_LEVELS
    levels = np.asarray(_ORIGINAL_BUILD_UNSIGNED_LEVELS(*args, **kwargs), dtype=np.float64)
    _LAST_PARALLEL_LEVELS = levels
    return levels


def _build_binned_level_cache(
    cache: base.ParallelTriangleCache,
    levels: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> _LevelBinnedCache | None:
    level_values = np.asarray(levels, dtype=np.float64)
    if level_values.ndim != 1 or level_values.size == 0:
        return None

    tri_min = np.asarray(cache.tri_parallel_min, dtype=np.float64)
    tri_max = np.asarray(cache.tri_parallel_max, dtype=np.float64)

    start_idx = np.searchsorted(level_values, tri_min - tolerance, side="left")
    end_idx = np.searchsorted(level_values, tri_max + tolerance, side="right")

    bins: list[list[int]] = [[] for _ in range(level_values.size)]
    valid_triangles = np.flatnonzero(end_idx > start_idx)
    for tri_idx in valid_triangles.tolist():
        for level_idx in range(int(start_idx[tri_idx]), int(end_idx[tri_idx])):
            bins[level_idx].append(int(tri_idx))

    return _LevelBinnedCache(
        levels=level_values.copy(),
        level_face_indices=tuple(np.asarray(items, dtype=np.int64) for items in bins),
    )


def _build_parallel_triangle_cache_with_bins(
    V,
    F,
    parallel_field,
    boundary_field,
):
    cache = _ORIGINAL_BUILD_PARALLEL_TRIANGLE_CACHE(V, F, parallel_field, boundary_field)
    levels = _LAST_PARALLEL_LEVELS
    if levels is not None and np.asarray(levels).size > 0:
        binned = _build_binned_level_cache(cache, np.asarray(levels, dtype=np.float64))
        if binned is not None:
            _CACHE_LEVEL_BINS[id(cache)] = binned
    return cache


def _face_indices_for_level(
    cache: base.ParallelTriangleCache,
    level: float,
    *,
    tolerance: float,
) -> np.ndarray:
    binned = _CACHE_LEVEL_BINS.get(id(cache))
    if binned is not None and binned.levels.size > 0:
        idx = int(np.searchsorted(binned.levels, level, side="left"))
        if idx < binned.levels.size and abs(float(binned.levels[idx]) - float(level)) <= max(tolerance, 1e-12):
            return binned.level_face_indices[idx]
        if idx > 0 and abs(float(binned.levels[idx - 1]) - float(level)) <= max(tolerance, 1e-12):
            return binned.level_face_indices[idx - 1]

    candidate_mask = (
        (cache.tri_parallel_min <= level + tolerance)
        & (cache.tri_parallel_max >= level - tolerance)
    )
    return np.flatnonzero(candidate_mask)


def extract_parallel_segments_with_boundary_values_vectorized(
    cache: base.ParallelTriangleCache,
    level: float,
    *,
    tolerance: float,
) -> _VectorizedSegmentBundle:
    candidate_face_indices = _face_indices_for_level(cache, level, tolerance=tolerance)
    if candidate_face_indices.size == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return _VectorizedSegmentBundle(empty, empty, empty1, empty1)

    tri_points = np.asarray(cache.tri_points[candidate_face_indices], dtype=np.float64)
    tri_parallel = np.asarray(cache.tri_parallel[candidate_face_indices], dtype=np.float64)
    tri_boundary = np.asarray(cache.tri_boundary[candidate_face_indices], dtype=np.float64)

    s0 = tri_parallel[:, _EDGE_START]
    s1 = tri_parallel[:, _EDGE_END]
    b0 = tri_boundary[:, _EDGE_START]
    b1 = tri_boundary[:, _EDGE_END]
    p0 = tri_points[:, _EDGE_START, :]
    p1 = tri_points[:, _EDGE_END, :]

    exact0 = np.abs(s0 - level) <= tolerance
    exact1 = np.abs(s1 - level) <= tolerance
    both_exact = exact0 & exact1
    crossing_mask = ((s0 - level) * (s1 - level) <= 0.0) & (~both_exact)
    if not np.any(crossing_mask):
        empty = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return _VectorizedSegmentBundle(empty, empty, empty1, empty1)

    denom = s1 - s0
    t = np.zeros_like(s0, dtype=np.float64)

    exact0_only = exact0 & (~exact1)
    exact1_only = exact1 & (~exact0)
    general_mask = crossing_mask & (~exact0_only) & (~exact1_only)

    t[exact0_only] = 0.0
    t[exact1_only] = 1.0
    safe_general = general_mask & (np.abs(denom) > tolerance)
    t[safe_general] = (level - s0[safe_general]) / denom[safe_general]

    valid_mask = crossing_mask.copy()
    valid_mask[general_mask & (~safe_general)] = False
    valid_mask &= (t >= -tolerance) & (t <= 1.0 + tolerance)
    if not np.any(valid_mask):
        empty = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return _VectorizedSegmentBundle(empty, empty, empty1, empty1)

    t = np.clip(t, 0.0, 1.0)
    one_minus_t = 1.0 - t
    xyz = one_minus_t[..., None] * p0 + t[..., None] * p1
    boundary_values = one_minus_t * b0 + t * b1

    start_xyz_list: list[np.ndarray] = []
    end_xyz_list: list[np.ndarray] = []
    start_boundary_list: list[float] = []
    end_boundary_list: list[float] = []

    for local_face_index in range(candidate_face_indices.size):
        edge_indices = np.flatnonzero(valid_mask[local_face_index])
        if edge_indices.size == 0:
            continue

        samples: list[tuple[np.ndarray, float]] = []
        for edge_idx in edge_indices.tolist():
            sample_xyz = np.asarray(xyz[local_face_index, edge_idx], dtype=np.float64)
            sample_boundary = float(boundary_values[local_face_index, edge_idx])
            if not any(np.linalg.norm(sample_xyz - existing_xyz) <= tolerance for existing_xyz, _ in samples):
                samples.append((sample_xyz, sample_boundary))

        if len(samples) == 2:
            start_xyz_list.append(samples[0][0])
            end_xyz_list.append(samples[1][0])
            start_boundary_list.append(samples[0][1])
            end_boundary_list.append(samples[1][1])
            continue
        if len(samples) <= 2:
            continue

        max_pair: tuple[int, int] | None = None
        max_distance = -1.0
        for idx, (first_xyz, _) in enumerate(samples):
            for second_idx in range(idx + 1, len(samples)):
                second_xyz = samples[second_idx][0]
                distance = float(np.linalg.norm(first_xyz - second_xyz))
                if distance > max_distance:
                    max_distance = distance
                    max_pair = (idx, second_idx)
        if max_pair is not None and max_distance > tolerance:
            first_idx, second_idx = max_pair
            start_xyz_list.append(samples[first_idx][0])
            end_xyz_list.append(samples[second_idx][0])
            start_boundary_list.append(samples[first_idx][1])
            end_boundary_list.append(samples[second_idx][1])

    if not start_xyz_list:
        empty = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return _VectorizedSegmentBundle(empty, empty, empty1, empty1)

    return _VectorizedSegmentBundle(
        start_xyz=np.asarray(start_xyz_list, dtype=np.float64),
        end_xyz=np.asarray(end_xyz_list, dtype=np.float64),
        start_boundary=np.asarray(start_boundary_list, dtype=np.float64),
        end_boundary=np.asarray(end_boundary_list, dtype=np.float64),
    )


def clip_parallel_segments_vectorized(
    segments,
    boundary_cutoff: float,
    *,
    tolerance: float,
):
    if not isinstance(segments, _VectorizedSegmentBundle):
        return _ORIGINAL_CLIP_PARALLEL_SEGMENTS(
            segments,
            boundary_cutoff,
            tolerance=tolerance,
        )
    if segments.start_xyz.shape[0] == 0:
        return []

    start_inside = segments.start_boundary > boundary_cutoff + tolerance
    end_inside = segments.end_boundary > boundary_cutoff + tolerance

    kept: list[tuple[np.ndarray, np.ndarray]] = []

    both_inside = start_inside & end_inside
    if np.any(both_inside):
        for start_xyz, end_xyz in zip(segments.start_xyz[both_inside], segments.end_xyz[both_inside]):
            kept.append((np.asarray(start_xyz, dtype=np.float64), np.asarray(end_xyz, dtype=np.float64)))

    crossing_mask = start_inside ^ end_inside
    if np.any(crossing_mask):
        cross_start_xyz = segments.start_xyz[crossing_mask]
        cross_end_xyz = segments.end_xyz[crossing_mask]
        cross_start_boundary = segments.start_boundary[crossing_mask]
        cross_end_boundary = segments.end_boundary[crossing_mask]
        cross_start_inside = start_inside[crossing_mask]

        denom = cross_end_boundary - cross_start_boundary
        valid = np.abs(denom) > tolerance
        if np.any(valid):
            cross_start_xyz = cross_start_xyz[valid]
            cross_end_xyz = cross_end_xyz[valid]
            cross_start_boundary = cross_start_boundary[valid]
            cross_end_boundary = cross_end_boundary[valid]
            cross_start_inside = cross_start_inside[valid]

            t = (boundary_cutoff - cross_start_boundary) / (cross_end_boundary - cross_start_boundary)
            t = np.clip(t, 0.0, 1.0)
            crossing_xyz = (1.0 - t)[:, None] * cross_start_xyz + t[:, None] * cross_end_xyz

            kept_start = np.where(cross_start_inside[:, None], cross_start_xyz, crossing_xyz)
            kept_end = np.where(cross_start_inside[:, None], crossing_xyz, cross_end_xyz)
            for start_xyz, end_xyz in zip(kept_start, kept_end):
                kept.append((np.asarray(start_xyz, dtype=np.float64), np.asarray(end_xyz, dtype=np.float64)))

    return kept


def apply_vectorized_extractor_patch() -> None:
    base.build_signed_levels = _record_signed_levels
    base.build_unsigned_levels = _record_unsigned_levels
    base.build_parallel_triangle_cache = _build_parallel_triangle_cache_with_bins
    base.extract_parallel_segments_with_boundary_values = extract_parallel_segments_with_boundary_values_vectorized
    base.clip_parallel_segments = clip_parallel_segments_vectorized


def main() -> None:
    apply_vectorized_extractor_patch()
    base.main()


if __name__ == "__main__":
    main()

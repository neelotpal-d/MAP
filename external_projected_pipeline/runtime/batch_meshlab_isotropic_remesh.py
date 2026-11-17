"""
Batch-remesh OBJ meshes in a folder using PyMeshLab's
"Isotropic Explicit Remeshing" filter.

This version avoids `meshlabserver` and calls the MeshLab filter directly
through `pymeshlab`, which is often simpler on Windows.
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor
import shutil
import time
from pathlib import Path

import numpy as np

from batch_expand_boundary_support_meshes import save_obj_mesh
from mesh_distance_workflow import build_vertex_adjacency
from mesh_distance_workflow import find_boundary_vertices

try:
    import pymeshlab
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "PyMeshLab is required for this utility. Install it in the active environment "
        "or run the script from an environment where `import pymeshlab` works."
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch remesh OBJ files with PyMeshLab isotropic explicit remeshing.",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing OBJ meshes to remesh.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where remeshed OBJ files will be written.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended to each output stem before .obj. Default keeps the original stem.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of isotropic remeshing iterations.",
    )
    parser.add_argument(
        "--targetlen-pct",
        type=float,
        default=0.5,
        help="Target edge length percentage for PyMeshLab PercentageValue.",
    )
    parser.add_argument(
        "--targetlen-abs",
        type=float,
        default=None,
        help="Optional absolute target edge length in mesh units. If set, this overrides --targetlen-pct.",
    )
    parser.add_argument(
        "--featuredeg",
        type=float,
        default=30.0,
        help="Crease angle used to preserve features.",
    )
    parser.add_argument(
        "--check-surfdist",
        action="store_true",
        help="Enable surface-distance checking during remeshing.",
    )
    parser.add_argument(
        "--max-surfdist-pct",
        type=float,
        default=1.4142,
        help="Maximum surface distance percentage for PyMeshLab PercentageValue.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive isotropic remeshing.",
    )
    parser.add_argument(
        "--selectedonly",
        action="store_true",
        help="Run the filter only on selected faces/vertices.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Disable the refine/split step.",
    )
    parser.add_argument(
        "--no-collapse",
        action="store_true",
        help="Disable the collapse step.",
    )
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Disable the edge-swap step.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable the smoothing step.",
    )
    parser.add_argument(
        "--no-reproject",
        action="store_true",
        help="Disable the reproject step.",
    )
    parser.add_argument(
        "--save-vertex-normal",
        action="store_true",
        help="Preserve vertex normals in the saved OBJ. Default is false.",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of worker processes used to remesh separate OBJ files in parallel. Default keeps the original serial behavior.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in the output directory before writing.",
    )
    parser.add_argument(
        "--post-smooth-iterations",
        type=int,
        default=0,
        help="Optional number of boundary-pinned Laplacian smoothing iterations applied after remeshing. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--post-smooth-alpha",
        type=float,
        default=0.2,
        help="Per-iteration smoothing strength in [0, 1] for the optional post-remesh boundary-pinned smoothing pass.",
    )
    parser.add_argument(
        "--report-timing",
        action="store_true",
        help="Print load, preparation, remesh, and save times for each layer.",
    )
    return parser


def clear_directory(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _remesh_job(
    *,
    obj_path: Path,
    output_path: Path,
    iterations: int,
    adaptive: bool,
    selectedonly: bool,
    targetlen_pct: float,
    targetlen_abs: float | None,
    featuredeg: float,
    checksurfdist: bool,
    max_surfdist_pct: float,
    splitflag: bool,
    collapseflag: bool,
    swapflag: bool,
    smoothflag: bool,
    reprojectflag: bool,
    save_vertex_normal: bool,
    post_smooth_iterations: int,
    post_smooth_alpha: float,
    report_timing: bool = False,
) -> tuple[Path, int, int, int, int]:
    v0, f0, v1, f1 = remesh_file(
        obj_path,
        output_path,
        iterations=iterations,
        adaptive=adaptive,
        selectedonly=selectedonly,
        targetlen_pct=targetlen_pct,
        targetlen_abs=targetlen_abs,
        featuredeg=featuredeg,
        checksurfdist=checksurfdist,
        max_surfdist_pct=max_surfdist_pct,
        splitflag=splitflag,
        collapseflag=collapseflag,
        swapflag=swapflag,
        smoothflag=smoothflag,
        reprojectflag=reprojectflag,
        save_vertex_normal=save_vertex_normal,
        post_smooth_iterations=post_smooth_iterations,
        post_smooth_alpha=post_smooth_alpha,
        report_timing=report_timing,
    )
    return obj_path, v0, f0, v1, f1


def iter_obj_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".obj")


def smooth_mesh_with_fixed_boundary(
    V: np.ndarray,
    F: np.ndarray,
    *,
    iterations: int,
    alpha: float,
) -> np.ndarray:
    if iterations <= 0:
        return np.asarray(V, dtype=np.float64).copy()
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("post_smooth_alpha must be in [0, 1].")

    vertices = np.asarray(V, dtype=np.float64).copy()
    faces = np.asarray(F, dtype=np.int64)
    adjacency = build_vertex_adjacency(faces)
    boundary_vertices = set(find_boundary_vertices(faces).tolist())
    interior_vertices = [
        vertex_idx
        for vertex_idx in range(vertices.shape[0])
        if vertex_idx not in boundary_vertices and adjacency.get(vertex_idx)
    ]
    if not interior_vertices:
        return vertices

    for _ in range(iterations):
        updated = vertices.copy()
        for vertex_idx in interior_vertices:
            neighbors = np.asarray(sorted(adjacency[vertex_idx]), dtype=np.int64)
            neighbor_mean = np.mean(vertices[neighbors], axis=0)
            updated[vertex_idx] = (1.0 - alpha) * vertices[vertex_idx] + alpha * neighbor_mean
        vertices = updated
    return vertices


def remesh_file(
    obj_path: Path,
    output_path: Path,
    *,
    iterations: int,
    adaptive: bool,
    selectedonly: bool,
    targetlen_pct: float,
    targetlen_abs: float | None,
    featuredeg: float,
    checksurfdist: bool,
    max_surfdist_pct: float,
    splitflag: bool,
    collapseflag: bool,
    swapflag: bool,
    smoothflag: bool,
    reprojectflag: bool,
    save_vertex_normal: bool,
    post_smooth_iterations: int = 0,
    post_smooth_alpha: float = 0.2,
    report_timing: bool = False,
) -> tuple[int, int, int, int]:
    start = time.perf_counter()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(obj_path))
    load_elapsed = time.perf_counter() - start

    prepare_start = time.perf_counter()
    try:
        ms.meshing_poly_to_tri()
    except Exception:
        pass
    prepare_elapsed = time.perf_counter() - prepare_start

    mesh_before = ms.current_mesh()
    v0 = mesh_before.vertex_number()
    f0 = mesh_before.face_number()

    # PyMeshLab expects a wrapped parameter object here:
    # PercentageValue for relative mode, PureValue for absolute mesh-unit mode.
    targetlen_value = (
        pymeshlab.PureValue(float(targetlen_abs))
        if targetlen_abs is not None
        else pymeshlab.PercentageValue(targetlen_pct)
    )

    remesh_start = time.perf_counter()
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        adaptive=adaptive,
        selectedonly=selectedonly,
        targetlen=targetlen_value,
        featuredeg=featuredeg,
        checksurfdist=checksurfdist,
        maxsurfdist=pymeshlab.PercentageValue(max_surfdist_pct),
        splitflag=splitflag,
        collapseflag=collapseflag,
        swapflag=swapflag,
        smoothflag=smoothflag,
        reprojectflag=reprojectflag,
    )
    remesh_elapsed = time.perf_counter() - remesh_start

    mesh_after = ms.current_mesh()
    v1 = mesh_after.vertex_number()
    f1 = mesh_after.face_number()

    save_start = time.perf_counter()
    if post_smooth_iterations > 0:
        vertices = np.asarray(mesh_after.vertex_matrix(), dtype=np.float64)
        faces = np.asarray(mesh_after.face_matrix(), dtype=np.int64)
        smoothed_vertices = smooth_mesh_with_fixed_boundary(
            vertices,
            faces,
            iterations=post_smooth_iterations,
            alpha=post_smooth_alpha,
        )
        save_obj_mesh(output_path, smoothed_vertices, faces)
    else:
        ms.save_current_mesh(str(output_path), save_vertex_normal=save_vertex_normal)
    save_elapsed = time.perf_counter() - save_start
    if report_timing:
        total_elapsed = time.perf_counter() - start
        print(
            f"[remesh timing] {obj_path.name}: load={load_elapsed:.2f}s "
            f"prepare={prepare_elapsed:.2f}s native_remesh={remesh_elapsed:.2f}s "
            f"save={save_elapsed:.2f}s total={total_elapsed:.2f}s",
            flush=True,
        )
    return v0, f0, v1, f1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    obj_files = iter_obj_files(args.input_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {args.input_dir}")
    if args.file_workers <= 0:
        raise ValueError("file_workers must be positive.")
    if args.post_smooth_iterations < 0:
        raise ValueError("post_smooth_iterations must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(args.output_dir)
        print(f"Cleared output directory: {args.output_dir}")

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"OBJ count: {len(obj_files)}")
    print("Backend: pymeshlab")
    print(f"File workers: {args.file_workers}")
    if args.post_smooth_iterations > 0:
        print(
            f"Post-remesh boundary-pinned smoothing: iterations={args.post_smooth_iterations}, "
            f"alpha={args.post_smooth_alpha}",
        )

    if args.file_workers == 1:
        for obj_path in obj_files:
            output_path = args.output_dir / f"{obj_path.stem}{args.suffix}.obj"
            v0, f0, v1, f1 = remesh_file(
                obj_path,
                output_path,
                iterations=args.iterations,
                adaptive=args.adaptive,
                selectedonly=args.selectedonly,
                targetlen_pct=args.targetlen_pct,
                targetlen_abs=args.targetlen_abs,
                featuredeg=args.featuredeg,
                checksurfdist=args.check_surfdist,
                max_surfdist_pct=args.max_surfdist_pct,
                splitflag=not args.no_split,
                collapseflag=not args.no_collapse,
                swapflag=not args.no_swap,
                smoothflag=not args.no_smooth,
                reprojectflag=not args.no_reproject,
                save_vertex_normal=args.save_vertex_normal,
                post_smooth_iterations=args.post_smooth_iterations,
                post_smooth_alpha=args.post_smooth_alpha,
                report_timing=bool(args.report_timing),
            )
            print(f"{obj_path.name}: V {v0} -> {v1}, F {f0} -> {f1}")
    else:
        with ProcessPoolExecutor(max_workers=args.file_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    _remesh_job,
                    obj_path=obj_path,
                    output_path=args.output_dir / f"{obj_path.stem}{args.suffix}.obj",
                    iterations=args.iterations,
                    adaptive=args.adaptive,
                    selectedonly=args.selectedonly,
                    targetlen_pct=args.targetlen_pct,
                    targetlen_abs=args.targetlen_abs,
                    featuredeg=args.featuredeg,
                    checksurfdist=args.check_surfdist,
                    max_surfdist_pct=args.max_surfdist_pct,
                    splitflag=not args.no_split,
                    collapseflag=not args.no_collapse,
                    swapflag=not args.no_swap,
                    smoothflag=not args.no_smooth,
                    reprojectflag=not args.no_reproject,
                    save_vertex_normal=args.save_vertex_normal,
                    post_smooth_iterations=args.post_smooth_iterations,
                    post_smooth_alpha=args.post_smooth_alpha,
                    report_timing=bool(args.report_timing),
                )
                for obj_path in obj_files
            ]
            results = [future.result() for future in futures]
        results.sort(key=lambda item: item[0].name)
        for obj_path, v0, f0, v1, f1 in results:
            print(f"{obj_path.name}: V {v0} -> {v1}, F {f0} -> {f1}")


if __name__ == "__main__":
    main()

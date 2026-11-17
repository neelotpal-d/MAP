"""
Batch-compute first-pass 5-axis support depths for ordered path files.

Inputs:
- layer meshes named like <main>_<sub>.obj
- matching path files named like <main>_<sub>.txt

Outputs:
- augmented path files named like <main>_<sub>_5axis.txt
  with rows: x y z nx ny nz height
"""

from __future__ import annotations

from multiprocessing import get_context

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import shutil
from pathlib import Path

import numpy as np

from extrusion_height import LayerKey
from extrusion_height import compute_point_heights_5axis
from extrusion_height import load_path_rows
from extrusion_height import load_reference_mesh_5axis
from extrusion_height import save_path_with_heights
from batch_compute_heights import copy_width_profile_for_layer
from batch_compute_heights import infer_model_lift_from_metadata


LAYER_PATTERN = re.compile(r"^(?P<main>\d+)_(?P<sub>\d+)\.(?P<ext>obj|txt)$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute first-pass 5-axis support depths for layer path files.",
    )
    parser.add_argument(
        "mesh_dir",
        type=Path,
        help="Folder containing layer meshes named like <main>_<sub>.obj.",
    )
    parser.add_argument(
        "path_dir",
        type=Path,
        help="Folder containing path files named like <main>_<sub>.txt.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where <main>_<sub>_5axis.txt files will be written.",
    )
    parser.add_argument(
        "--previous-main-window",
        type=int,
        default=4,
        help="How many previous main-index layer groups to consider as support candidates.",
    )
    parser.add_argument(
        "--layer-workers",
        type=int,
        default=1,
        help="Number of worker processes used to compute separate layers in parallel. Default keeps the original serial behavior.",
    )
    parser.add_argument(
        "--major-alignment-threshold",
        type=float,
        default=0.5,
        help="Alignment threshold below which a point is counted as a major deviation.",
    )
    parser.add_argument(
        "--report-major-deviations",
        action="store_true",
        help="Print per-layer major deviation counts and alignment summary statistics.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing files in the output directory before writing.",
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


def parse_layer_name(path: Path) -> LayerKey | None:
    match = LAYER_PATTERN.match(path.name)
    if match is None:
        return None
    return LayerKey(main=int(match.group("main")), sub=int(match.group("sub")))


def collect_layer_files(folder: Path, expected_ext: str) -> dict[LayerKey, Path]:
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")
    files: dict[LayerKey, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = LAYER_PATTERN.match(path.name)
        if match is None or match.group("ext").lower() != expected_ext:
            continue
        key = parse_layer_name(path)
        if key is not None:
            files[key] = path
    return files


def previous_main_indices(keys: list[LayerKey], current_main: int, window: int) -> list[int]:
    unique_mains = sorted({key.main for key in keys if key.main < current_main})
    if window <= 0:
        return []
    return unique_mains[-window:]


def load_existing_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def find_layer_metadata_entry(
    batch_metadata: dict[str, object],
    key: LayerKey,
) -> dict[str, object] | None:
    layers = batch_metadata.get("layers")
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if int(layer.get("layer_main", -1)) != key.main or int(layer.get("layer_sub", -1)) != key.sub:
            continue
        return layer
    return None


def copy_width_profile_for_layer(
    *,
    source_batch_metadata: dict[str, object],
    key: LayerKey,
    source_dir: Path,
    output_dir: Path,
) -> str | None:
    layer_meta = find_layer_metadata_entry(source_batch_metadata, key)
    if layer_meta is None:
        return None
    width_profile_name = layer_meta.get("width_profile_file")
    if not isinstance(width_profile_name, str) or not width_profile_name:
        return None
    source_path = source_dir / width_profile_name
    if not source_path.exists():
        return None
    output_path = output_dir / source_path.name
    if source_path.resolve() != output_path.resolve():
        shutil.copyfile(source_path, output_path)
    return output_path.name


def write_height_metadata(
    *,
    mesh_dir: Path,
    path_dir: Path,
    output_dir: Path,
    previous_main_window: int,
    layer_workers: int,
    major_alignment_threshold: float,
    report_major_deviations: bool,
    layer_outputs: list[dict[str, object]],
) -> dict[str, object]:
    width = None
    for layer in layer_outputs:
        source_metadata = layer.get("source_metadata")
        if isinstance(source_metadata, dict):
            width_value = source_metadata.get("width", source_metadata.get("contour_spacing"))
            if width_value is not None:
                width = width_value
                break
    return {
        "batch_kind": "5axis_path_heights",
        "input_mesh_dir": str(mesh_dir),
        "input_path_dir": str(path_dir),
        "output_dir": str(output_dir),
        "output_format": "x y z nx ny nz height",
        "width_profile_mode": "per_point_sidecar",
        "height_mode": "5axis",
        "height_definition": "euclidean_distance_to_nearest_surface_point",
        "normal_alignment_check": {
            "enabled": True,
            "major_alignment_threshold": major_alignment_threshold,
            "report_major_deviations": report_major_deviations,
        },
        "width": width,
        "previous_main_window": previous_main_window,
        "layer_workers": layer_workers,
        "layers": layer_outputs,
    }


def _compute_layer_heights_5axis_job(
    *,
    key: LayerKey,
    path_path: Path,
    output_dir: Path,
    reference_items: list[tuple[LayerKey, Path]],
    major_alignment_threshold: float,
    model_lift: float = 0.0,
    model_lift_axis: str = "y",
) -> dict[str, object]:
    reference_meshes = [
        load_reference_mesh_5axis(
            reference_path,
            reference_key,
            coordinate_lift=model_lift,
            coordinate_lift_axis=model_lift_axis,
        )
        for reference_key, reference_path in reference_items
    ]
    path_rows = load_path_rows(path_path)
    heights, diagnostics = compute_point_heights_5axis(
        path_rows[:, 0:3],
        path_rows[:, 3:6],
        reference_meshes,
        major_alignment_threshold=major_alignment_threshold,
        return_diagnostics=True,
    )

    output_path = output_dir / f"{key.stem}_5axis.txt"
    save_path_with_heights(output_path, path_rows, heights)
    valid_alignment = diagnostics.alignment[~np.isnan(diagnostics.alignment)]
    return {
        "layer_key": key,
        "output_path": output_path,
        "reference_layers": [reference_key.stem for reference_key, _ in reference_items],
        "max_height": float(np.nanmax(heights)) if heights.size else 0.0,
        "major_deviation_count": diagnostics.major_deviation_count,
        "alignment_min": float(np.min(valid_alignment)) if valid_alignment.size else None,
        "alignment_max": float(np.max(valid_alignment)) if valid_alignment.size else None,
        "alignment_mean": float(np.mean(valid_alignment)) if valid_alignment.size else None,
    }


def main() -> None:
    args = build_parser().parse_args()

    mesh_files = collect_layer_files(args.mesh_dir, "obj")
    path_files = collect_layer_files(args.path_dir, "txt")
    shared_keys = sorted(set(mesh_files) & set(path_files), key=lambda key: (key.main, key.sub))
    if not shared_keys:
        raise FileNotFoundError("No matching <main>_<sub> mesh/path pairs were found.")
    if args.layer_workers <= 0:
        raise ValueError("layer_workers must be positive.")

    source_batch_metadata = load_existing_metadata(args.path_dir / "batch_paths.meta.json")
    model_lift, model_lift_axis = infer_model_lift_from_metadata(source_batch_metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_output:
        clear_directory(args.output_dir)
        print(f"Cleared output directory: {args.output_dir}")

    all_mesh_keys = sorted(mesh_files.keys(), key=lambda key: (key.main, key.sub))

    print(f"Mesh folder: {args.mesh_dir}")
    print(f"Path folder: {args.path_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Matched layers: {len(shared_keys)}")
    print(f"Previous main window: {args.previous_main_window}")
    print(f"Layer workers: {args.layer_workers}")
    print(f"Model lift: {model_lift} along {model_lift_axis}")

    layer_jobs = []
    for key in shared_keys:
        candidate_mains = previous_main_indices(all_mesh_keys, key.main, args.previous_main_window)
        reference_keys = [mesh_key for mesh_key in all_mesh_keys if mesh_key.main in candidate_mains]
        layer_jobs.append(
            {
                "key": key,
                "path_path": path_files[key],
                "reference_items": [(reference_key, mesh_files[reference_key]) for reference_key in reference_keys],
            }
        )

    if args.layer_workers == 1:
        results = [
            _compute_layer_heights_5axis_job(
                key=job["key"],
                path_path=job["path_path"],
                output_dir=args.output_dir,
                reference_items=job["reference_items"],
                major_alignment_threshold=args.major_alignment_threshold,
                model_lift=model_lift,
                model_lift_axis=model_lift_axis,
            )
            for job in layer_jobs
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.layer_workers, mp_context=get_context('spawn')) as executor:
            futures = [
                executor.submit(
                    _compute_layer_heights_5axis_job,
                    key=job["key"],
                    path_path=job["path_path"],
                    output_dir=args.output_dir,
                    reference_items=job["reference_items"],
                    major_alignment_threshold=args.major_alignment_threshold,
                    model_lift=model_lift,
                    model_lift_axis=model_lift_axis,
                )
                for job in layer_jobs
            ]
            results = [future.result() for future in futures]

    results.sort(key=lambda item: (item["layer_key"].main, item["layer_key"].sub))

    layer_outputs: list[dict[str, object]] = []
    for result in results:
        key = result["layer_key"]
        output_path = result["output_path"]
        width_profile_file = copy_width_profile_for_layer(
            source_batch_metadata=source_batch_metadata,
            key=key,
            source_dir=args.path_dir,
            output_dir=args.output_dir,
        )
        layer_record = {
            "layer_main": key.main,
            "layer_sub": key.sub,
            "source_mesh": mesh_files[key].name,
            "source_path_file": path_files[key].name,
            "output_path_file": output_path.name,
            "source_metadata_file": "batch_paths.meta.json",
            "source_metadata": source_batch_metadata,
            "width_profile_file": width_profile_file,
            "reference_layers": result["reference_layers"],
            "major_deviation_count": result["major_deviation_count"],
            "alignment_min": result["alignment_min"],
            "alignment_max": result["alignment_max"],
            "alignment_mean": result["alignment_mean"],
        }
        layer_outputs.append(layer_record)
        print(
            f"Saved: {output_path.name} "
            f"(references={len(result['reference_layers'])}, max_height={result['max_height']:.6g})"
        )
        if args.report_major_deviations:
            print(
                f"  major_deviations={result['major_deviation_count']} "
                f"alignment_min={result['alignment_min']} "
                f"alignment_mean={result['alignment_mean']} "
                f"alignment_max={result['alignment_max']}"
            )

    batch_metadata = write_height_metadata(
        mesh_dir=args.mesh_dir,
        path_dir=args.path_dir,
        output_dir=args.output_dir,
        previous_main_window=args.previous_main_window,
        layer_workers=args.layer_workers,
        major_alignment_threshold=args.major_alignment_threshold,
        report_major_deviations=args.report_major_deviations,
        layer_outputs=layer_outputs,
    )
    batch_metadata_path = args.output_dir / "batch_5axis.meta.json"
    batch_metadata_path.write_text(json.dumps(batch_metadata, indent=2), encoding="utf-8")
    print(f"Saved batch metadata: {batch_metadata_path.name}")


if __name__ == "__main__":
    main()

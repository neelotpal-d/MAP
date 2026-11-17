"""Experimental XZ-projected hybrid pipeline.

This is intentionally a separate entry point. It reuses every stage of the
established surface pipeline, but replaces only hybrid path generation with
the direct 2D projected planner.
"""

from __future__ import annotations

from pathlib import Path

print("[startup] Loading projected-hybrid pipeline modules...", flush=True)
import run_surface_pipeline as _surface_pipeline

print("[startup] Loading projected 2D planner...", flush=True)
from batch_generate_projected_hybrid_paths import process_projected_hybrid_layer


def _reuse_existing_stage(
    stage_name: str,
    original_stage: object,
) -> object:
    def run(*, input_dir: Path, output_dir: Path, clean_output: bool, **kwargs: object) -> Path:
        stage_cfg = next((value for value in kwargs.values() if isinstance(value, dict)), {})
        reuse_raw = stage_cfg.get("reuse_existing_output") if isinstance(stage_cfg, dict) else None
        reuse_enabled = bool(stage_cfg.get("reuse_existing", False)) if isinstance(stage_cfg, dict) else False
        if not reuse_enabled:
            return original_stage(input_dir=input_dir, output_dir=output_dir, clean_output=clean_output, **kwargs)  # type: ignore[operator]
        if reuse_raw is None:
            raise ValueError(f"{stage_name}.reuse_existing is true but reuse_existing_output is not configured.")
        reuse_dir = Path(str(reuse_raw)).resolve()
        if reuse_dir.is_dir() and any(reuse_dir.glob("*.obj")):
            print(f"{stage_name} reused; starting from existing output: {reuse_dir}", flush=True)
            return reuse_dir
        print(
            f"Saved {stage_name.lower()} output unavailable at {reuse_dir}; generating it instead.",
            flush=True,
        )
        return original_stage(input_dir=input_dir, output_dir=output_dir, clean_output=clean_output, **kwargs)  # type: ignore[operator]

    return run


def _reuse_existing_hybrid_paths(original_stage: object) -> object:
    def run(*, output_dir: Path, hybrid_cfg: dict[str, object], **kwargs: object) -> Path:
        if not bool(hybrid_cfg.get("reuse_existing_paths", False)):
            return original_stage(output_dir=output_dir, hybrid_cfg=hybrid_cfg, **kwargs)  # type: ignore[operator]
        metadata_path = output_dir / "batch_paths.meta.json"
        has_paths = output_dir.is_dir() and any(output_dir.glob("*_*.txt"))
        if has_paths and metadata_path.is_file():
            print(f"Hybrid path stage reused; starting from saved paths: {output_dir}", flush=True)
            return output_dir
        print(
            f"Saved projected paths unavailable at {output_dir}; generating paths instead.",
            flush=True,
        )
        return original_stage(output_dir=output_dir, hybrid_cfg=hybrid_cfg, **kwargs)  # type: ignore[operator]

    return run


def main() -> None:
    print("[startup] Projected-hybrid pipeline started.", flush=True)
    _surface_pipeline.process_hybrid_layer = process_projected_hybrid_layer
    _surface_pipeline.run_remesh_stage = _reuse_existing_stage(
        "Remesh stage",
        _surface_pipeline.run_remesh_stage,
    )
    _surface_pipeline.run_support_expansion_stage = _reuse_existing_stage(
        "Support-expansion stage",
        _surface_pipeline.run_support_expansion_stage,
    )
    _surface_pipeline.run_hybrid_path_stage = _reuse_existing_hybrid_paths(
        _surface_pipeline.run_hybrid_path_stage,
    )
    _surface_pipeline.main()


if __name__ == "__main__":
    main()

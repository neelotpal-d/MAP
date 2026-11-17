"""Run the curved support generator from one portable job JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


PATH_INPUTS = {
    "mesh", "checkpoint", "normalization_mesh", "layer_values", "samples",
    "tree_source", "model_volume", "model_layers_source", "skeleton_source", "tetrahedral_volume", "support_field_source", "shared_layer_values", "tool",
}


def _absolute(value, base):
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def load_job(path: Path, output_override: Path | None = None, clean_override: bool | None = None):
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"output", "clean_output", "pipeline"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown job keys: {sorted(unknown)}")
    if not isinstance(raw.get("pipeline"), dict):
        raise ValueError("Job requires a 'pipeline' object with input/network/settings sections.")
    pipeline = raw["pipeline"]
    if set(pipeline) - {"input", "network", "settings"}:
        raise ValueError("pipeline only accepts input, network, and settings.")
    inputs = dict(pipeline.get("input", {}))
    for key in PATH_INPUTS & set(inputs):
        if inputs[key] is not None:
            inputs[key] = _absolute(inputs[key], path.parent)
    # The core's preserved-tree exporter is gated by tree_source even when an
    # imported skeleton is the actual geometry source.  Hide that historical
    # switch from public jobs: one skeleton path is sufficient.
    if inputs.get("skeleton_source") and not inputs.get("tree_source"):
        inputs["tree_source"] = inputs["skeleton_source"]
    resolved = {
        "input": inputs,
        "network": dict(pipeline.get("network", {})),
        "settings": dict(pipeline.get("settings", {})),
    }
    output = output_override or raw.get("output")
    if output is None:
        raise ValueError("Job requires an output path, or pass --output.")
    output = Path(output)
    output = (output if output.is_absolute() else path.parent / output).resolve()
    clean = bool(raw.get("clean_output", True)) if clean_override is None else clean_override
    return output, clean, resolved


def main(argv=None):
    # This is a compute subprocess: native failures must reach the parent as a
    # nonzero exit, rather than waiting indefinitely behind a Windows dialog.
    # Do not change the error mode merely when this module is imported by a GUI.
    import faulthandler
    faulthandler.enable()
    if sys.platform == 'win32':
        import ctypes
        import os
        # Calling an environment's python.exe directly need not activate Conda.
        # Resolve its own native DLLs before NumPy/VTK load delayed libraries.
        # Keep handles alive for this complete job; children inherit PATH.
        prefix=Path(sys.executable).resolve().parent
        dll_handles=[]
        if (prefix/'conda-meta').is_dir():
            directories=[prefix,prefix/'Library/bin',prefix/'Library/usr/bin',prefix/'Library/mingw-w64/bin',prefix/'Scripts']
            directories=[path for path in directories if path.is_dir()]
            os.environ['PATH']=os.pathsep.join(map(str,directories))+os.pathsep+os.environ.get('PATH','')
            os.environ['CONDA_PREFIX']=str(prefix)
            if hasattr(os,'add_dll_directory'):
                dll_handles=[os.add_dll_directory(str(path)) for path in directories]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetErrorMode.restype = ctypes.c_uint
        kernel.SetErrorMode.argtypes = [ctypes.c_uint]
        kernel.SetErrorMode(kernel.GetErrorMode() | 0x0001 | 0x0002)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, help="Job JSON; relative input/output paths are relative to this file.")
    parser.add_argument("--output", type=Path, help="Override job output path.")
    parser.add_argument("--no-clean", action="store_true", help="Refuse a non-empty output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved job without running.")
    parser.add_argument("--core-root", type=Path, help="Override the bundled runtime directory containing SupportGenerationCurved and SupportGenerationVoxel.")
    args = parser.parse_args(argv)

    output, clean, pipeline = load_job(args.job, args.output, False if args.no_clean else None)
    if args.dry_run:
        print(json.dumps({"output": str(output), "clean_output": clean, "pipeline": pipeline}, indent=2))
        return 0

    package_root = Path(__file__).resolve().parent
    core_root = (args.core_root or package_root / "runtime").resolve()
    if not (core_root / "SupportGenerationCurved" / "generate.py").is_file():
        raise FileNotFoundError("--core-root must contain SupportGenerationCurved/generate.py.")
    if not (core_root / "SupportGenerationVoxel").is_dir():
        raise FileNotFoundError("--core-root must also contain SupportGenerationVoxel (model-field runtime).")
    sys.path.insert(0, str(core_root))
    from SupportGenerationCurved.generate import parse_args, run

    # The core parser validates every supported setting and protects --clean-output
    # from cleaning an input.  The temporary config remains outside output.
    with tempfile.TemporaryDirectory(prefix="curved_support_job_") as temporary:
        config_path = Path(temporary) / "resolved_config.json"
        config_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
        forwarded = ["--config", str(config_path), "--output", str(output)]
        if clean:
            forwarded.append("--clean-output")
        run(parse_args(forwarded))
    (output / "resolved_config.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    print(f"Curved support job complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

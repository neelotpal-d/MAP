"""Create a progressive support skeleton from a completed envelope-transfer run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

if __package__:
    from .cli import load_job
else:
    from cli import load_job


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Any unrecognised options are forwarded to the planner; for example --routing-mode near-model --step 0.6.",
    )
    parser.add_argument("job", type=Path, help="Public generator job JSON. It supplies mesh, field, and settings.")
    parser.add_argument("--run", type=Path, help="Completed transfer run for the legacy contact-source route.")
    parser.add_argument("--from-model", action="store_true", help="Grow directly from a model and field without an existing tree.")
    parser.add_argument("--output", required=True, type=Path, help="New directory for support_centerlines.vtp and skeleton reports.")
    parser.add_argument("--clean-output", action="store_true")
    args, forwarded = parser.parse_known_args(argv)

    _, _, pipeline = load_job(args.job)
    package_root = Path(__file__).resolve().parent
    runtime = package_root / "runtime"
    sys.path.insert(0, str(runtime))
    if args.from_model:
        from SupportGenerationCurved.model_skeleton import main as model_main
        return model_main(pipeline, args.output.resolve(), forwarded, clean=args.clean_output)
    if args.run is None:
        parser.error("Supply --from-model or a completed --run directory.")
    from SupportGenerationCurved.skeleton_planner import main as planner_main

    with tempfile.TemporaryDirectory(prefix="curved_support_skeleton_") as temporary:
        config = Path(temporary) / "planner_config.json"
        config.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
        command = [
            "skeleton_planner.py", "--run", str(args.run.resolve()), "--config", str(config),
            "--output", str(args.output.resolve()),
        ]
        if args.clean_output:
            command.append("--clean-output")
        old_argv = sys.argv
        try:
            sys.argv = command + forwarded
            planner_main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()

"""Run the banded pipeline with local mesh-edge scalar consistency by default."""

from __future__ import annotations

import sys

import runBandedSurfaceQualityPipeline


def main() -> None:
    if not any(
        argument == "--scalar-consistency-mode"
        or argument.startswith("--scalar-consistency-mode=")
        for argument in sys.argv
    ):
        sys.argv.extend(("--scalar-consistency-mode", "local-edge"))
    if not any(argument == "--checkpoint-tag" or argument.startswith("--checkpoint-tag=") for argument in sys.argv):
        sys.argv.extend(("--checkpoint-tag", "local_edge"))
    runBandedSurfaceQualityPipeline.main()


if __name__ == "__main__":
    main()

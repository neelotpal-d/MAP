"""Prepare, inspect, and train a model-specific banded surface-quality slicer run.

The script is intentionally an orchestrator: SDF fitting, scalar initialization,
layer inspection, and interactive surface selection keep using the existing
project implementations. It stores every generated artifact under one model
folder so changing models or SIREN frequencies does not overwrite prior work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

import sdfField
import visualizeHybridSurfaceSampling as surface_sampling
import visualizeHeightClippedSurfaceSampling as height_clipping
import visualizeBoundaryVertexNormalSampling as boundary_smoothing


REPOSITORY_ROOT = Path(__file__).resolve().parent
TEMPLATE_CONFIG_PATH = REPOSITORY_ROOT / "experiments" / "convex_hard_collision" / "convex_surface_quality.py"
TRAINING_MODULE_CONFIGURATOR = None
SURFACE_FACE_SAMPLER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Name used for this model's artifact folder.")
    parser.add_argument("--mesh-path", type=Path, required=True, help="Closed surface OBJ/mesh to slice.")
    parser.add_argument("--node-path", type=Path, default=None, help="Optional TetGen .node file.")
    parser.add_argument("--ele-path", type=Path, default=None, help="Optional TetGen .ele file.")
    parser.add_argument("--run-root", type=Path, default=REPOSITORY_ROOT / "model_runs")
    parser.add_argument("--w0", type=float, required=True, help="SIREN hidden-layer frequency.")
    parser.add_argument("--w0-initial", type=float, default=None, help="SIREN first-layer frequency; defaults to --w0.")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of SIREN hidden layers.")
    parser.add_argument("--up-direction", nargs=3, type=float, default=(0.0, 1.0, 0.0), metavar=("UX", "UY", "UZ"))
    parser.add_argument(
        "--affine-y-residual",
        action="store_true",
        help="Train phi = y_normalized + a gated learned residual, initially using the exact +y gradient.",
    )
    parser.add_argument(
        "--affine-residual-initial-scale", type=float, default=0.0,
        help="Initial trainable multiplier on the learned residual when --affine-y-residual is enabled; 0 starts exactly as y.",
    )
    parser.add_argument("--in-plane-u", nargs=3, type=float, default=(1.0, 0.0, 0.0), metavar=("UX", "UY", "UZ"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--render-proxy-target-faces",
        type=int,
        default=0,
        help=(
            "If positive, PyVista preview windows render a decimated temporary mesh with at most this "
            "many faces. It never changes SDF, constraints, training, or slicing geometry."
        ),
    )
    parser.add_argument(
        "--initializer-preview-clip-proxy-target-faces",
        type=int,
        default=0,
        help=(
            "If positive, use a saved decimated mesh only to bound and clip the initializer-layer preview. "
            "This makes that preview approximate; training and final slicing still use the full mesh."
        ),
    )
    parser.add_argument(
        "--initializer-lightweight-preview",
        action="store_true",
        help="Use decimated render-only geometry for the initializer-layer viewer to avoid memory exhaustion on large parts.",
    )
    parser.add_argument("--initializer-preview-mesh-target-faces", type=int, default=50_000)
    parser.add_argument("--initializer-preview-layer-target-cells", type=int, default=12_000)

    parser.add_argument("--sdf-epochs", type=int, default=150)
    parser.add_argument("--sdf-learning-rate", type=float, default=1e-5)
    parser.add_argument("--sdf-grid-spacing", type=float, default=2.0)
    parser.add_argument(
        "--sdf-grid-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional per-axis SDF training-grid spacings; overrides --sdf-grid-spacing.",
    )
    parser.add_argument("--sdf-min-grid-samples-per-axis", type=int, default=8)
    parser.add_argument("--sdf-padding", type=float, default=2.0)
    parser.add_argument("--sdf-point-limit", type=int, default=150_000)
    parser.add_argument(
        "--sdf-surface-point-limit",
        type=int,
        default=0,
        help="Maximum surface point/normal SDF constraints; 0 keeps every proxy-mesh vertex.",
    )
    parser.add_argument(
        "--sdf-surface-sample-seed",
        type=int,
        default=1234,
        help="Seed used when subsampling SDF surface point/normal constraints.",
    )
    parser.add_argument(
        "--sdf-proxy-target-faces",
        type=int,
        default=0,
        help=(
            "If positive, decimate a temporary mesh to at most this many faces for SDF "
            "supervision only.  The original mesh remains authoritative for constraints and slicing."
        ),
    )
    parser.add_argument("--sdf-batch-size", type=int, default=2000, help="Per-set SDF batch size; fused mode evaluates roughly twice this many points per forward.")
    parser.add_argument(
        "--sdf-save-every",
        type=int,
        default=0,
        help="Save an SDF epoch snapshot every N epochs in sdf/checkpoints; 0 saves only the final SDF.",
    )
    parser.add_argument("--sdf-query-batch-size", type=int, default=5000, help="SDF inference batch size used by the verification display.")
    parser.add_argument("--no-sdf-fuse-batches", action="store_true", help="Keep separate domain and surface SDF forwards for comparison/debugging.")
    parser.add_argument("--sdf-visualization-samples", type=int, default=40)
    parser.add_argument("--sdf-visualization-noise", type=float, default=0.3)

    parser.add_argument("--initializer-epochs", type=int, default=150)
    parser.add_argument("--initializer-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--initializer-lr-warmup-epochs", type=int, default=0,
        help="Linearly increase the initializer learning rate to --initializer-learning-rate over these first epochs; 0 disables it.",
    )
    parser.add_argument(
        "--initializer-lr-warmup-factor", type=float, default=1e-3,
        help="Fraction of --initializer-learning-rate used at the first initializer warm-up epoch.",
    )
    parser.add_argument("--initializer-batch-size", type=int, default=4000)
    parser.add_argument(
        "--initializer-max-input-points",
        type=int,
        default=100_000,
        help="Maximum retained initializer points after the interior/near-surface sampling grid is evaluated.",
    )
    parser.add_argument("--initializer-curvature-weight", type=float, default=0.0)
    parser.add_argument(
        "--initializer-near-surface-band",
        type=float,
        default=1.0,
        help="Exterior SDF shell retained with all interior points during initializer warm start; 0 keeps only interior points.",
    )
    parser.add_argument("--initializer-grid-spacing", type=float, default=2.0)
    parser.add_argument(
        "--initializer-grid-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional per-axis initializer sampling spacings; overrides --initializer-grid-spacing.",
    )
    parser.add_argument("--input-min-grid-samples-per-axis", type=int, default=8)
    parser.add_argument("--layer-distance", type=float, default=0.2)
    parser.add_argument("--voxel-spacing", type=float, default=1.0)
    parser.add_argument("--voxel-padding", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", "--lr", type=float, default=1e-5, help="Scalar-field optimizer learning rate.")
    parser.add_argument('--learning-rate-decay-every-epochs', type=int, default=250,
                        help='Main field StepLR interval in epochs; 0 disables decay. Separate from SDF training.')
    parser.add_argument('--learning-rate-decay-factor', type=float, default=.1,
                        help='Main field StepLR multiplier in (0, 1].')
    parser.add_argument("--start-checkpoint", default=None, help="Checkpoint path, or 'previous' to resume the newest earlier checkpoint in this model folder.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--input-grid-spacing",
        type=float,
        default=2.0,
        help="Spacing of the interior/domain sampling grid in world units.",
    )
    parser.add_argument(
        "--input-grid-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional per-axis interior/domain grid spacings; overrides --input-grid-spacing.",
    )
    parser.add_argument(
        "--max-input-points",
        type=int,
        default=150_000,
        help=(
            "Maximum distinct interior-domain points retained for scalar and norm control. "
            "Raise this when refining --input-grid-spacing so the finer grid is not randomly capped."
        ),
    )
    parser.add_argument("--input-points-per-epoch", type=int, default=30_000, help="Fresh without-replacement interior points used by each training epoch.")
    parser.add_argument("--collision-roots-per-step", type=int, default=128)
    parser.add_argument("--no-fuse-regular-losses", action="store_true", help="Use separate regular-loss forwards instead of the default fused forward.")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Initial low-learning-rate epochs before restoring --learning-rate.")
    parser.add_argument("--warmup-factor", type=float, default=1e-5, help="Warm-up learning rate as a fraction of --learning-rate.")
    parser.add_argument(
        "--learning-rate-ramp-epochs", type=int, default=0,
        help="Linearly increase from the warm-up learning rate to --learning-rate after warm-up; 0 restores it immediately.",
    )
    parser.add_argument("--surface-group-scalar-delay-epochs", type=int, default=20, help="Delay before enabling the surface-group scalar-consistency loss.")
    parser.add_argument(
        "--surface-scalar-ramp-epochs",
        type=int,
        default=0,
        help="Linearly ramp group/edge scalar loss from zero after the delay; zero means immediate full weight.",
    )
    parser.add_argument(
        "--scalar-consistency-mode",
        choices=("group", "local-edge"),
        default="group",
        help="Use whole-group scalar consistency or local mesh-edge consistency for preserved surfaces.",
    )
    parser.add_argument("--local-edge-scalar-weight", type=float, default=50.0, help="Weight for local mesh-edge scalar consistency.")
    parser.add_argument("--local-edge-alignment-degrees", type=float, default=6.0, help="Strongly weight an edge only when both endpoint gradients are this close to their target normals.")
    parser.add_argument(
        "--edge-alignment-pair-weight",
        choices=("product", "min"),
        default="product",
        help="Combine endpoint alignment confidences by product (existing behavior) or the weakest endpoint.",
    )
    parser.add_argument("--local-edge-neighbor-radius-factor", type=float, default=2.5)
    parser.add_argument("--local-edge-neighbor-normal-degrees", type=float, default=20.0)
    parser.add_argument(
        "--long-range-pairs-per-group",
        type=int,
        default=0,
        help="Optional random distant scalar-consistency pairs per preserved group; 0 disables them.",
    )
    parser.add_argument(
        "--long-range-min-distance-factor",
        type=float,
        default=10.0,
        help="Minimum long-pair separation as this factor times the surface-sample spacing.",
    )
    parser.add_argument("--long-range-pair-seed", type=int, default=1234)
    parser.add_argument("--save-every", type=int, default=10, help="Scalar-field checkpoint interval in epochs; use 0 to save only the final checkpoint.")
    parser.add_argument(
        "--checkpoint-tag",
        default="banded",
        help="Suffix for the output checkpoint within this model run, without the .pt extension.",
    )
    parser.add_argument("--debug-loss-breakdown", action="store_true", help="Print synchronized per-loss diagnostics every epoch; disabled by default for throughput.")
    parser.add_argument(
        "--interactive-loss-controls",
        action="store_true",
        help="Open a checkbox table to enable or disable individual losses before training.",
    )
    parser.set_defaults(balanced_norm_batches=True)
    parser.add_argument("--balanced-norm-batches", dest="balanced_norm_batches", action="store_true", help="Use per-band queues covering every control point each epoch (default).")
    parser.add_argument("--no-balanced-norm-batches", dest="balanced_norm_batches", action="store_false", help="Use ordinary input batches against the full-set intervals, without full epoch training coverage.")
    parser.add_argument("--norm-pool-size", type=int, default=20000, help="Chunk size for the full control-set reference scan each epoch; does not cap its coverage.")
    parser.add_argument("--norm-points-per-band", type=int, default=1000, help="Minimum per-band batch budget; increased automatically when needed for full epoch coverage.")
    parser.add_argument("--max-layer-height-ratio", type=float, default=2.0, help="Maximum local perpendicular layer-spacing ratio within each sampled scalar band.")
    parser.add_argument("--gradient-band-count", type=int, default=8, help="Scalar slabs on an epoch-wide reference range; half-offset slabs also constrain band boundaries.")
    parser.add_argument('--gradient-band-ratio-weight', type=float, help='Override the within-band gradient-norm ratio penalty weight')
    parser.add_argument('--gradient-band-floor-weight', type=float, help='Override the gradient-norm floor penalty weight')
    parser.add_argument(
        "--disable-norm-banded-control",
        action="store_true",
        help="Disable both the within-band gradient-ratio and gradient-floor penalties for comparison runs.",
    )
    parser.add_argument(
        "--show-norm-control-batches",
        action="store_true",
        help="Show the exact balanced norm-control points for every batch after each epoch.",
    )
    parser.add_argument("--show-local-edge-pairs", action="store_true", help="Display sampled neighbour pairs before training.")
    parser.add_argument("--initial-gradient-diagnostics", action="store_true", help="Run expensive parameter-gradient diagnostics during early warm-up epochs.")
    parser.set_defaults(enable_input_curvature=False, enable_tangent_twist=False)
    parser.add_argument("--enable-input-curvature", dest="enable_input_curvature", action="store_true", help="Enable sampled curvature regularization with the selected mode; disabled by default for throughput.")
    parser.add_argument("--disable-input-curvature", dest="enable_input_curvature", action="store_false", help="Disable sampled curvature regularization.")
    parser.add_argument('--input-curvature-mode', choices=['raw-hessian', 'geometric'], default='raw-hessian')
    parser.add_argument('--input-curvature-weight', type=float, default=1e-3)
    parser.add_argument('--input-curvature-sample-fraction', type=float, default=.2)
    parser.add_argument('--input-curvature-radius-mm', type=float, default=35.)
    parser.add_argument('--input-curvature-boundary-coverage', action='store_true',
                        help='Cover every boundary vertex each epoch; requires curvature fraction 1 and fused training')
    parser.add_argument('--regularization-profile', choices=['banded', 'integrated-five-axis'], default='banded')
    parser.add_argument('--gradient-variation-weight', type=float, default=.02)
    parser.add_argument('--small-gradient-weight', type=float, default=1.)
    parser.add_argument("--enable-tangent-twist", dest="enable_tangent_twist", action="store_true", help="Enable periodic tangent-twist gradient-cosine probes; disabled by default for throughput.")
    parser.add_argument("--disable-tangent-twist", dest="enable_tangent_twist", action="store_false", help="Explicitly keep tangent-twist probes disabled.")
    parser.add_argument("--surface-normal-max-angle", type=float, default=40.0)
    parser.add_argument(
        "--boundary-vertex-normal-smoothing",
        action="store_true",
        help="Smooth automatic angle-selection boundaries with one-ring, crease-aware vertex-normal clipping before uniform sampling.",
    )
    parser.add_argument(
        "--boundary-face-normal-similarity-degrees",
        type=float,
        default=15.0,
        help="Exclude incident faces beyond this normal angle when computing a transition face's local vertex normals.",
    )
    parser.add_argument(
        "--exclude-surface-transition-samples",
        action="store_true",
        help=(
            "Keep the angle-transition collar in the selected patch and cap geometry, but do not place "
            "surface-preservation samples on its clipped triangles."
        ),
    )
    parser.add_argument(
        "--surface-guide-normal-mode",
        choices=("parent-face", "selected-patch-smoothed"),
        default="parent-face",
        help=(
            "Guide normal assigned to surface-preservation samples. selected-patch-smoothed uses only "
            "same-patch selected faces and barycentric interpolation; parent-face preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--surface-guide-normal-similarity-degrees",
        type=float,
        default=20.0,
        help="Crease threshold for --surface-guide-normal-mode selected-patch-smoothed.",
    )
    parser.add_argument("--surface-height-min", type=float, default=None)
    parser.add_argument("--surface-height-max", type=float, default=None)
    parser.add_argument("--height-selection-mode", choices=("face-center", "clipped"), default="face-center")
    parser.add_argument(
        "--surface-region-area-ratio-min",
        type=float,
        default=0.0,
        help="Discard automatic connected regions smaller than this fraction of the largest; zero preserves every selected region.",
    )
    parser.add_argument("--skip-automatic-selection-window", action="store_true")
    parser.add_argument(
        "--fast-selection-preview",
        action="store_true",
        help="Use a whole-face, no-clipping preview while moving automatic-selection sliders; final constraint clipping remains exact.",
    )
    parser.add_argument("--base-height", type=float, default=1.0)
    parser.add_argument("--base-normal-max-angle", type=float, default=40.0, help="Keep base constraints only on faces this close to -up.")
    parser.add_argument("--base-direction-weight", type=float, default=100.0, help="Weight of the base gradient-alignment loss.")
    parser.add_argument(
        "--base-direction-parallel-only",
        action="store_true",
        help="Penalize only the base-gradient component perpendicular to up; do not force magnitude one.",
    )
    parser.add_argument(
        "--disable-scalar-gradient-division",
        action="store_true",
        help="Do not divide group/edge scalar consistency by gradient magnitude.",
    )
    parser.add_argument(
        "--detach-scalar-gradient-denominator",
        action="store_true",
        help="Keep gradient-normalized group/edge scalar loss, but stop its denominator from updating gradient magnitude.",
    )
    parser.add_argument(
        "--surface-sample-density",
        type=float,
        default=1.25,
        help="Uniform surface-preservation samples per world-area unit; set zero to retain vertex samples.",
    )
    parser.add_argument(
        "--base-sample-density",
        type=float,
        default=None,
        help="Uniform base samples per world-area unit; defaults to --surface-sample-density.",
    )
    parser.add_argument("--constraint-sample-candidate-oversampling", type=int, default=12)
    parser.add_argument("--min-constraint-samples-per-region", type=int, default=0)
    parser.add_argument("--max-constraint-samples-per-region", type=int, default=50000)
    parser.add_argument("--constraint-sample-seed", type=int, default=1234)
    parser.add_argument("--constraint-normal-arrow-scale", type=float, default=None)
    parser.add_argument("--skip-constraint-sample-preview", action="store_true")
    parser.add_argument("--disable-collision", action="store_true")
    parser.add_argument("--skip-surface-selection", action="store_true")
    parser.add_argument(
        "--select-surface-interactively",
        action="store_true",
        help="Open the preserved-region picker without asking first; accepting it replaces this run's saved regions.",
    )
    parser.add_argument(
        "--reuse-saved-preserved-regions",
        action="store_true",
        help=(
            "Load this model run's saved constraints/manual_surface_regions.json and bypass the "
            "fresh-region picker. Fails clearly if that file contains no accepted regions."
        ),
    )
    parser.add_argument(
        "--reuse-generated-constraint-samples",
        action="store_true",
        help=(
            "Reuse a matching cached post-selection surface-constraint dataset, including its guide normals "
            "and cap patch. The cache is created automatically after a fresh generation."
        ),
    )
    parser.add_argument("--no-train", action="store_true", help="Prepare and inspect artifacts without training.")
    parser.add_argument("--skip-sdf-verification", action="store_true")
    parser.add_argument(
        "--skip-sdf",
        action="store_true",
        help="Skip SDF fitting and verification; allowed only together with --disable-collision.",
    )
    parser.add_argument("--skip-initializer-verification", action="store_true")
    parser.add_argument("--create-missing", action="store_true", help="Create absent SDF/initializer artifacts without prompting.")
    parser.add_argument("--retrain-existing", action="store_true", help="Archive and rebuild existing SDF and initializer artifacts without prompting.")
    parser.add_argument(
        "--retrain-initializer",
        action="store_true",
        help="Archive and rebuild only the scalar initializer; combine with --skip-initializer-verification for a direct warm start.",
    )
    parser.add_argument("--yes", action="store_true", help="Automatically continue after visual checks as well as creation prompts.")
    parser.add_argument('--stress-orientation-record', type=Path)
    parser.add_argument('--disable-orientation', action='store_true', default=True)
    parser.add_argument('--optimize-yaw', action='store_true', help='Optimize one global rotation about the fixed tool axis; keep the platform normal fixed.')
    parser.add_argument('--yaw-update-epochs',type=int,nargs='+',help='Optional completed field epochs followed by a bounded yaw-only burst; keeps all field epochs.')
    parser.add_argument('--yaw-update-steps',type=int,default=10)
    parser.add_argument('--orientation-learning-rate', type=float, default=1e-3)
    parser.add_argument('--orientation-max-rotation', type=float, default=60.)
    parser.add_argument('--orientation-initial-quaternion', type=float, nargs=4)
    parser.add_argument('--orientation-schedule', nargs='+', default=['1:joint'])
    parser.add_argument('--orientation-refresh-steps', type=int, default=10)
    parser.add_argument('--collision-weight', type=float, default=1e6)
    parser.add_argument('--collision-weight-schedule', nargs='+')
    parser.add_argument('--collision-topk-fraction', type=float, default=.3)
    parser.add_argument('--collision-tool-weights', type=float, nargs='+', default=[1., 1.])
    parser.add_argument('--tool-plane-config', type=Path)
    parser.add_argument('--tool-box-config', type=Path)
    parser.add_argument('--field-query-batch-size', type=int, default=16384)
    parser.add_argument('--mining-candidate-count', type=int, default=50000)
    parser.add_argument('--mining-refresh-epochs', type=int, default=5)
    parser.add_argument('--hard-pool-size', type=int, default=8192)
    parser.add_argument('--scoring-root-batch-size', type=int, default=128)
    parser.add_argument('--sdf-gate-batch-size', type=int, default=50000)
    parser.add_argument('--collision-validation-count', type=int, default=4096)
    parser.add_argument('--normalize-collision-by-gradient', action='store_true')
    parser.add_argument('--collision-gradient-epsilon', type=float, default=1e-6)
    parser.add_argument('--disable-surface-quality-loss', action='store_true')
    parser.add_argument('--enable-support-free-loss', action='store_true')
    parser.add_argument('--disable-surface-scalar-loss', action='store_true')
    parser.add_argument('--surface-direction-weight', type=float, default=100.)
    parser.add_argument('--objective', choices=['surface-quality', 'support-free', 'strength', 'combined'], default='surface-quality')
    parser.add_argument('--support-free-weight', type=float, default=200.)
    parser.add_argument('--support-free-angle', type=float, default=45., help='Allowed overhang angle beyond 90 degrees')
    parser.add_argument('--support-free-height-min', type=float)
    parser.add_argument('--support-free-height-max', type=float)
    parser.add_argument('--show-final-preview', action='store_true')
    parser.add_argument('--strength-file', type=Path)
    parser.add_argument('--strength-weight', type=float, default=1.)
    parser.add_argument('--strength-sampling', choices=['interior', 'fps'], default='interior')
    parser.add_argument('--strength-fps-filter-order', choices=['before', 'after'], default='after',
                        help='FPS: apply tensile filtering before selection or discard compression points after FPS without replacement')
    parser.add_argument('--strength-samples', type=int, default=0, help='FPS source count; 0 matches the interior pool, capped by active training sources')
    parser.add_argument('--strength-batch-size', type=int, default=0, help='FPS batch size; 0 uses the interior batch size')
    parser.add_argument('--strength-neighbors', type=int, default=8)
    parser.add_argument('--strength-max-distance-mm', type=float)
    parser.add_argument('--strength-point-columns', type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument('--strength-direction-columns', type=int, nargs=3, default=[3, 4, 5])
    parser.add_argument('--strength-signed-column', type=int, default=6)
    parser.add_argument('--strength-all-signs', action='store_true')
    parser.add_argument('--strength-min-tensile-stress', type=float, default=0.)
    parser.add_argument('--strength-seed', type=int, default=42)
    parser.add_argument('--initializer-seed', type=int, default=None)
    args = parser.parse_args()
    if args.learning_rate_decay_every_epochs < 0:
        parser.error('--learning-rate-decay-every-epochs must be nonnegative.')
    if not 0 < args.learning_rate_decay_factor <= 1:
        parser.error('--learning-rate-decay-factor must be in (0, 1].')
    if args.optimize_yaw:
        args.disable_orientation = False
        if args.orientation_initial_quaternion is not None:
            parser.error('Fixed-axis yaw uses the initial in-plane U frame; do not combine it with a free-rotation quaternion.')
    if args.yaw_update_epochs is not None:
        if (not args.optimize_yaw or not args.yaw_update_epochs or args.yaw_update_steps<1 or
            len(set(args.yaw_update_epochs))!=len(args.yaw_update_epochs) or
            any(epoch<1 or epoch>=args.epochs for epoch in args.yaw_update_epochs)):
            parser.error('Yaw milestones need optimize-yaw, positive step count and distinct epochs before the final field epoch.')
    if args.show_final_preview and not args.disable_orientation:
        parser.error('Use --view-orientation after training to display the learned pose')
    if args.tool_plane_config is not None and args.tool_box_config is not None:
        parser.error('Choose either tool-plane-config or tool-box-config')
    if args.no_fuse_regular_losses and args.tool_box_config is not None:
        parser.error('Full box sampling requires the fused loop')
    if args.no_fuse_regular_losses and args.strength_sampling == 'fps':
        parser.error('Independent FPS stress batches require the fused loop')
    if args.input_curvature_boundary_coverage and (
        args.no_fuse_regular_losses or not args.enable_input_curvature or args.input_curvature_sample_fraction != 1.
    ):
        parser.error('Boundary curvature coverage requires enabled curvature, fraction 1, and fused training')
    if args.regularization_profile == 'integrated-five-axis' and (
        args.no_fuse_regular_losses or not args.enable_input_curvature
        or args.input_curvature_sample_fraction != 1. or args.input_curvature_mode != 'geometric'
        or args.objective != 'strength' or not args.disable_orientation
    ):
        parser.error('Integrated five-axis losses require fixed-frame strength workflow, fused training and full geometric curvature sampling')
    if any(not np.isfinite(w) or w < 0 for w in [args.gradient_variation_weight, args.small_gradient_weight]):
        parser.error('Gradient regularization weights must be finite and nonnegative')
    if any(w is not None and (not np.isfinite(w) or w < 0)
           for w in (args.gradient_band_ratio_weight, args.gradient_band_floor_weight)):
        parser.error('Gradient band weights must be finite and nonnegative')
    if args.strength_samples < 0 or args.strength_batch_size < 0:
        parser.error('Strength sample count and batch size must be nonnegative')
    if not np.isfinite(args.input_curvature_weight) or args.input_curvature_weight < 0:
        parser.error('Curvature weight must be finite and nonnegative')
    if not np.isfinite(args.input_curvature_radius_mm) or args.input_curvature_radius_mm <= 0:
        parser.error('Curvature radius must be finite and positive')
    if not 0 < args.input_curvature_sample_fraction <= 1:
        parser.error('Curvature sample fraction must be in (0,1]')
    if args.no_fuse_regular_losses and args.collision_weight_schedule:
        parser.error('Collision weight schedules require the fused loop in this prototype')
    if not args.disable_orientation:
        if args.disable_collision or args.no_fuse_regular_losses:
            parser.error('Orientation search requires collision and the fused training loop; use --disable-orientation for a baseline')
        if not np.isfinite(args.orientation_learning_rate) or args.orientation_learning_rate <= 0:
            parser.error('orientation-learning-rate must be finite and positive')
        if not 0 < args.orientation_max_rotation <= 180 or args.orientation_refresh_steps < 1:
            parser.error('Invalid orientation rotation bound or refresh interval')
    if not np.isfinite(args.collision_weight) or args.collision_weight < 0 or not 0 < args.collision_topk_fraction <= 1:
        parser.error('Invalid collision weight or top-k fraction')
    if not args.collision_tool_weights or any(not np.isfinite(x) or x < 0 for x in args.collision_tool_weights) or sum(args.collision_tool_weights) <= 0:
        parser.error('Collision tool weights must be finite/nonnegative with positive total')
    from globalOrientation import parse_schedule, scheduled_value
    args.orientation_schedule_parsed = parse_schedule(args.orientation_schedule)
    args.collision_weight_schedule_parsed = parse_schedule(args.collision_weight_schedule or [f'1:{args.collision_weight}'], kind='weight')
    if args.yaw_update_epochs and any(scheduled_value(args.collision_weight_schedule_parsed,epoch)<=0 for epoch in args.yaw_update_epochs):
        parser.error('Every intermediate yaw milestone requires positive collision weight.')
    if not args.disable_orientation and any(phase in ['frame','joint'] and not any(weight > 0 for _, weight in args.collision_weight_schedule_parsed) for _, phase in args.orientation_schedule_parsed):
        parser.error('Orientation optimization needs positive collision weight')
    if not np.isfinite(args.collision_gradient_epsilon) or args.collision_gradient_epsilon <= 0:
        parser.error('collision-gradient-epsilon must be finite and positive')
    if args.objective in ['strength', 'combined'] and args.strength_file is None:
        parser.error('Strength/combined objectives require --strength-file')
    if args.strength_file is not None:
        args.strength_file = args.strength_file.resolve()
        if not args.strength_file.is_file():
            parser.error('Strength data file does not exist')
    if args.strength_neighbors < 1 or args.strength_weight <= 0 or args.support_free_weight < 0 or not 0 <= args.support_free_angle < 90:
        parser.error('Invalid strength/support objective parameters')
    if args.strength_max_distance_mm is not None and args.strength_max_distance_mm <= 0:
        parser.error('Strength interpolation distance must be positive')
    if args.disable_collision:
        args.skip_sdf = True

    args.mesh_path = args.mesh_path.resolve()
    args.run_root = args.run_root.resolve()
    args.w0_initial = args.w0 if args.w0_initial is None else args.w0_initial
    if not args.mesh_path.is_file():
        parser.error(f"--mesh-path does not exist: {args.mesh_path}")
    if (args.node_path is None) != (args.ele_path is None):
        parser.error("Pass both --node-path and --ele-path, or neither.")
    if args.node_path is not None:
        args.node_path = args.node_path.resolve()
        args.ele_path = args.ele_path.resolve()
        if not args.node_path.is_file() or not args.ele_path.is_file():
            parser.error("The supplied TetGen --node-path/--ele-path files must exist.")
    if args.start_checkpoint is not None:
        if args.start_checkpoint != "previous":
            args.start_checkpoint = Path(args.start_checkpoint).resolve()
            if not args.start_checkpoint.is_file():
                parser.error(f"--start-checkpoint does not exist: {args.start_checkpoint}")
    if args.w0 <= 0.0 or args.w0_initial <= 0.0:
        parser.error("--w0 and --w0-initial must be positive.")
    if np.linalg.norm(args.up_direction) <= 1e-12:
        parser.error("--up-direction must be non-zero.")
    if np.linalg.norm(args.in_plane_u) <= 1e-12:
        parser.error("--in-plane-u must be non-zero.")
    normalized_up = np.asarray(args.up_direction, dtype=float) / np.linalg.norm(args.up_direction)
    normalized_u = np.asarray(args.in_plane_u, dtype=float) / np.linalg.norm(args.in_plane_u)
    if abs(float(np.dot(normalized_up, normalized_u))) > 1e-5:
        parser.error("--in-plane-u must be perpendicular to --up-direction.")
    if args.surface_sample_density < 0.0 or (args.base_sample_density is not None and args.base_sample_density < 0.0):
        parser.error("Constraint sample densities must be non-negative.")
    if args.surface_region_area_ratio_min < 0.0:
        parser.error("--surface-region-area-ratio-min must be non-negative.")
    if args.initializer_lr_warmup_epochs < 0:
        parser.error("--initializer-lr-warmup-epochs cannot be negative.")
    if not 0.0 < args.initializer_lr_warmup_factor <= 1.0:
        parser.error("--initializer-lr-warmup-factor must be in (0, 1].")
    if not np.isfinite(args.base_direction_weight) or args.base_direction_weight < 0.0:
        parser.error("--base-direction-weight must be finite and non-negative.")
    if args.constraint_sample_candidate_oversampling < 1:
        parser.error("--constraint-sample-candidate-oversampling must be at least one.")
    if args.sdf_min_grid_samples_per_axis < 1 or args.input_min_grid_samples_per_axis < 1:
        parser.error("Minimum grid samples per axis must be at least one.")
    if args.max_input_points < 1:
        parser.error("--max-input-points must be positive.")
    if args.initializer_near_surface_band < 0.0:
        parser.error("--initializer-near-surface-band cannot be negative.")
    if (args.sdf_point_limit < 1 or args.sdf_surface_point_limit < 0
            or args.sdf_proxy_target_faces < 0 or args.render_proxy_target_faces < 0
            or args.initializer_preview_clip_proxy_target_faces < 0
            or args.sdf_save_every < 0):
        parser.error("SDF point limit must be positive; surface/proxy limits and save interval cannot be negative.")
    if args.initializer_preview_mesh_target_faces < 1 or args.initializer_preview_layer_target_cells < 1:
        parser.error("Initializer lightweight-preview face/cell budgets must be positive.")
    if not np.isfinite(args.max_layer_height_ratio) or args.max_layer_height_ratio < 1.0:
        parser.error("--max-layer-height-ratio must be finite and at least one.")
    if args.gradient_band_count < 1 or args.norm_pool_size < 1 or args.norm_points_per_band < 1:
        parser.error("Gradient band count, reference scan chunk size, and points per band must be positive.")
    return args


def safe_name(value: str) -> str:
    """Keep model folder names portable while preserving readable identifiers."""

    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def frequency_name(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def ask_to_continue(question: str, args: argparse.Namespace, creation: bool = False) -> bool:
    if args.yes or (creation and args.create_missing):
        return True
    return input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}


def ask_to_retrain(artifact_name: str, args: argparse.Namespace) -> bool:
    """Do not let --yes overwrite reusable artifacts; retraining stays explicit."""

    if args.retrain_existing:
        return True
    if args.yes:
        return False
    return input(f"Retrain {artifact_name}? The current file will be archived first [y/N]: ").strip().lower() in {"y", "yes"}


def initializer_compatibility_issue(path: Path, args: argparse.Namespace) -> str | None:
    """Return a human-readable incompatibility reason for an initializer checkpoint."""

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return f"cannot be read ({exc})"
    if checkpoint.get("field_backend", "siren") != "siren":
        return "it was created for a different field backend"
    checkpoint_layers = int(checkpoint.get("num_layers", 5))
    if checkpoint_layers != args.num_layers:
        return f"it has num_layers={checkpoint_layers}, but this run requires num_layers={args.num_layers}"
    checkpoint_hidden = int(checkpoint.get("dim_hidden", 128))
    if checkpoint_hidden != 128:
        return f"it has dim_hidden={checkpoint_hidden}, but this run requires dim_hidden=128"
    return None


def has_nonempty_manual_regions(path: Path) -> bool:
    """Return whether a saved picker session contains an accepted region."""

    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("regions", []))


def load_saved_automatic_selection(path: Path, mesh_path: Path, args: argparse.Namespace) -> bool:
    """Restore the saved automatic selector settings for a matching mesh.

    Manual groups live in ``manual_surface_regions.json``.  The all-groups
    metadata additionally records the automatic selection window, so replaying
    both files reproduces the complete preserved set without reopening sliders.
    """

    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metadata_type") != "surface_constraint_regions" or not payload.get("regions"):
        return False
    current_point_count = surface_sampling.pv.read(mesh_path).triangulate().n_points
    if int(payload.get("n_points", -1)) != current_point_count:
        raise ValueError(
            f"Saved constraint metadata {path} has {payload.get('n_points')} mesh vertices, "
            f"but the current mesh has {current_point_count}."
        )
    saved = payload.get("automatic_selection") or {}
    if saved.get("max_angle_degrees") is not None:
        args.surface_normal_max_angle = float(saved["max_angle_degrees"])
    if saved.get("min_height_world") is not None:
        args.surface_height_min = float(saved["min_height_world"])
    if saved.get("max_height_world") is not None:
        args.surface_height_max = float(saved["max_height_world"])
    if saved.get("min_region_area_ratio") is not None:
        args.surface_region_area_ratio_min = float(saved["min_region_area_ratio"])
    print(
        f"Reusing saved automatic preserved regions from {path}: "
        f"angle={args.surface_normal_max_angle:g} deg, "
        f"height=[{args.surface_height_min}, {args.surface_height_max}], "
        f"regions={len(payload['regions'])}."
    )
    return True


def copy_input_artifact(source: Path, destination_dir: Path) -> Path:
    """Stage immutable source geometry into the model folder without silent replacement."""

    destination = destination_dir / source.name
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(
                f"{destination} already exists but differs from {source}. "
                "Use a new --model-name or remove the stale model folder deliberately."
            )
        return destination
    shutil.copy2(source, destination)
    return destination


def archive_artifact(path: Path, archive_dir: Path) -> Path:
    """Move an artifact aside, replacing the prior single archived version."""

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / path.name
    if archive_path.exists():
        archive_path.unlink()
    shutil.move(str(path), str(archive_path))
    print(f"Archived {path.name} to {archive_path}")
    return archive_path


def release_transient_cuda_memory() -> None:
    """Release completed SDF/visualization allocations before scalar training."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, check=True)


def train_sdf(mesh_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    """Fit the existing project SDF model with model-specific paths."""

    mesh, vertices, faces = sdfField.load_mesh_data(str(mesh_path))
    if args.sdf_proxy_target_faces and mesh.n_cells > args.sdf_proxy_target_faces:
        original_face_count = mesh.n_cells
        reduction = 1.0 - args.sdf_proxy_target_faces / original_face_count
        mesh = mesh.triangulate().decimate(reduction)
        vertices = np.asarray(mesh.points)
        faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]
        print(
            f"SDF proxy mesh: {mesh_path.name}: original faces={original_face_count}, "
            f"proxy faces={mesh.n_cells}. The original mesh is still used for scalar constraints and slicing."
        )
    sdf_spacing = tuple(args.sdf_grid_spacing_xyz) if args.sdf_grid_spacing_xyz is not None else args.sdf_grid_spacing
    samples = sdfField.prepare_training_samples(
        mesh, vertices, faces, device=args.device, padding=args.sdf_padding, spacing=sdf_spacing,
        min_samples_per_axis=args.sdf_min_grid_samples_per_axis,
    )
    model = sdfField.sdfModel(device=args.device)
    point_limit = min(args.sdf_point_limit, samples["points_dist"].shape[0])
    surface_limit = args.sdf_surface_point_limit
    if surface_limit and samples["points_surf"].shape[0] > surface_limit:
        # The SDF currently supervises mesh vertices.  Randomized selection is
        # preferable to taking an input-order prefix, which can spatially bias
        # common OBJ/PLY exports.  A seed makes comparisons reproducible.
        selected = np.random.default_rng(args.sdf_surface_sample_seed).choice(
            samples["points_surf"].shape[0], surface_limit, replace=False
        )
        selected = torch.as_tensor(selected, dtype=torch.long, device=samples["points_surf"].device)
        samples["points_surf"] = samples["points_surf"][selected].detach().requires_grad_(True)
        samples["normals_surf"] = samples["normals_surf"][selected]
    print(
        f"SDF training constraints: domain={point_limit}/{samples['points_dist'].shape[0]}, "
        f"surface={samples['points_surf'].shape[0]}."
    )
    model.train(
        samples["points_dist"][:point_limit],
        samples["scalars_dist"][:point_limit],
        samples["points_surf"],
        samples["normals_surf"],
        lr=args.sdf_learning_rate,
        epochNum=args.sdf_epochs,
        savePath=str(output_path),
        batch_size=args.sdf_batch_size,
        fuse_point_batches=not args.no_sdf_fuse_batches,
        save_every=args.sdf_save_every,
        checkpoint_dir=output_path.parent / "checkpoints",
    )


def visualize_sdf(mesh_path: Path, checkpoint_path: Path, args: argparse.Namespace) -> None:
    mesh, vertices, _ = sdfField.load_mesh_data(str(mesh_path))
    model = sdfField.sdfModel(device=args.device, model_load_path=str(checkpoint_path))
    sdfField.run_visualization(
        visualization_mode=1,
        model=model,
        mesh=mesh,
        mesh_vertices=vertices,
        padding=args.sdf_padding,
        samples_per_axis=args.sdf_visualization_samples,
        noise_scale=args.sdf_visualization_noise,
        query_batch_size=args.sdf_query_batch_size,
    )


def build_initializer(initializer_path: Path, artifact_dir: Path, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "generateScalarInitializer.py"),
        "--model-name", safe_name(args.model_name),
        "--mesh-path", str(args.mesh_path),
        "--w0", str(args.w0),
        "--w0-initial", str(args.w0_initial),
        "--num-layers", str(args.num_layers),
        "--device", args.device,
        "--epochs", str(args.initializer_epochs),
        "--lr", str(args.initializer_learning_rate),
        "--lr-warmup-epochs", str(args.initializer_lr_warmup_epochs),
        "--lr-warmup-factor", str(args.initializer_lr_warmup_factor),
        "--batch-size", str(args.initializer_batch_size),
        "--max-input-points", str(args.initializer_max_input_points),
        "--grid-spacing", str(args.initializer_grid_spacing),
        "--min-grid-samples-per-axis", str(args.input_min_grid_samples_per_axis),
        "--curvature-weight", str(args.initializer_curvature_weight),
        "--near-surface-band", str(args.initializer_near_surface_band),
        "--target-direction", *(str(value) for value in args.up_direction),
        "--output-dir", str(artifact_dir),
    ]
    if args.initializer_grid_spacing_xyz is not None:
        command.extend(("--grid-spacing-xyz", *(str(value) for value in args.initializer_grid_spacing_xyz)))
    if args.initializer_seed is not None:
        command.extend(['--seed', str(args.initializer_seed)])
    run_command(command)
    if not initializer_path.is_file():
        raise FileNotFoundError(f"Initializer command completed but did not create {initializer_path}")


def initializer_preview_mesh_path(args: argparse.Namespace) -> Path:
    """Build/reuse a display-only clip mesh for the initializer layer viewer."""

    target = args.initializer_preview_clip_proxy_target_faces
    if target <= 0:
        return args.mesh_path
    proxy_path = args.mesh_path.with_name(f"{args.mesh_path.stem}_initializer_preview_{target}faces.vtp")
    if proxy_path.is_file():
        proxy = surface_sampling.pv.read(proxy_path)
        if proxy.n_cells <= target:
            return proxy_path
    mesh = surface_sampling.pv.read(args.mesh_path).triangulate()
    if mesh.n_cells <= target:
        return args.mesh_path
    proxy = mesh.decimate(1.0 - target / mesh.n_cells)
    proxy.save(proxy_path)
    print(
        f"Initializer preview clip proxy: {mesh.n_cells} -> {proxy.n_cells} faces: {proxy_path}. "
        "It is not used by optimization or final slicing."
    )
    return proxy_path


def visualize_initializer(initializer_path: Path, args: argparse.Namespace) -> None:
    preview_mesh_path = initializer_preview_mesh_path(args)
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "visualizeGradientHeightLayers.py"),
        "--field-type", "neural",
        "--checkpoint-path", str(initializer_path),
        "--distance", str(args.layer_distance),
        "--w0", str(args.w0),
        "--w0-initial", str(args.w0_initial),
        "--num-layers", str(args.num_layers),
        "--device", args.device,
        "--color-seed", "42",
    ]
    if args.node_path is not None:
        command.extend(("--contour-source", "tet", "--node-path", str(args.node_path), "--ele-path", str(args.ele_path)))
    else:
        command.extend(
            (
                "--contour-source", "voxel", "--mesh-path", str(preview_mesh_path),
                # The network was trained in the full mesh's normalization;
                # keep that coordinate frame when using a coarse preview clip.
                "--normalization-model-path", str(args.mesh_path),
                "--voxel-spacing", str(args.voxel_spacing), "--voxel-padding", str(args.voxel_padding),
                "--voxel-preclip-volume",
            )
        )
    if args.initializer_lightweight_preview:
        command.extend(
            (
                "--lightweight-preview",
                "--preview-mesh-target-faces", str(args.initializer_preview_mesh_target_faces),
                "--preview-layer-target-cells", str(args.initializer_preview_layer_target_cells),
            )
        )
    run_command(command)


def load_banded_template():
    spec = importlib.util.spec_from_file_location("pipeline_banded_surface_quality", TEMPLATE_CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load banded template from {TEMPLATE_CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_grouped_surface_faces(
    mesh,
    selected_faces: np.ndarray,
    face_group_ids: np.ndarray,
    density: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create density-controlled face samples with a parent-face normal per point."""

    if density <= 0.0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    sampler_args = SimpleNamespace(
        samples_per_square_unit=density,
        interior_sample_area=1.0 / density,
        min_samples_per_region=args.min_constraint_samples_per_region,
        max_samples_per_region=args.max_constraint_samples_per_region,
        sample_count=None,
    )
    sampled_points, sampled_normals, sampled_group_ids = [], [], []
    regions = surface_sampling.allocate_region_sample_counts(
        mesh, selected_faces, face_group_ids, sampler_args
    )
    # Area-uniform candidates are strictly interior barycentric samples, so
    # their normal is exactly the parent-face normal.  Calling the generic
    # boundary-normal routine for every clipped component rebuilds adjacency
    # maps repeatedly; Swan can have hundreds of components after smoothing.
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    faces = surface_sampling.faces_of(mesh)
    face_vertices = np.asarray(mesh.points, dtype=np.float64)[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]), axis=1
    )
    smoothed_corner_normals = None
    if args.surface_guide_normal_mode == "selected-patch-smoothed":
        incident = boundary_smoothing.incident_faces_by_vertex(faces, mesh.n_points)
        selected_mask = np.asarray(selected_faces, dtype=bool)
        threshold = float(np.cos(np.deg2rad(args.surface_guide_normal_similarity_degrees)))
        selected_face_indices = np.flatnonzero(selected_mask)
        print(
            "Building selected-patch-smoothed surface guides: "
            f"{len(selected_face_indices)} face(s), crease threshold "
            f"{args.surface_guide_normal_similarity_degrees:.3f} deg."
        )
        # Only the sampled patch is populated. float32 halves the temporary
        # footprint without affecting the float64 training tensors downstream.
        smoothed_corner_normals = np.zeros((mesh.n_cells, 3, 3), dtype=np.float32)
        progress_step = max(25_000, len(selected_face_indices) // 20)
        for processed, face_index in enumerate(selected_face_indices, start=1):
            group_id = int(face_group_ids[face_index])
            reference = face_normals[face_index]
            for corner, vertex in enumerate(faces[face_index]):
                candidates = incident[int(vertex)]
                keep = (
                    selected_mask[candidates]
                    & (face_group_ids[candidates] == group_id)
                    & ((face_normals[candidates] @ reference) >= threshold)
                )
                compatible = candidates[keep]
                if compatible.size == 0:
                    compatible = np.asarray([face_index], dtype=np.int64)
                normal = np.sum(face_normals[compatible] * face_areas[compatible, None], axis=0)
                smoothed_corner_normals[face_index, corner] = normal / (np.linalg.norm(normal) + 1e-12)
            if processed % progress_step == 0 or processed == len(selected_face_indices):
                print(f"  Smoothed guide corners: {processed}/{len(selected_face_indices)} face(s).")
    for region_index, (group_id, group_mask, group_count) in enumerate(regions, start=1):
        result = surface_sampling.poisson_surface_samples(
            mesh,
            group_mask,
            sample_area=1.0 / density,
            sample_count=group_count,
            oversampling=args.constraint_sample_candidate_oversampling,
            radius=None,
            seed=args.constraint_sample_seed + group_id,
        )
        candidates, candidate_faces, candidate_weights, kept_indices, _, requested_count = result
        if len(kept_indices) < requested_count:
            print(f"Surface group {group_id}: retained {len(kept_indices)}/{requested_count} Poisson samples.")
        points = candidates[kept_indices]
        if smoothed_corner_normals is None:
            normals = face_normals[candidate_faces[kept_indices]]
        else:
            corners = smoothed_corner_normals[candidate_faces[kept_indices]]
            normals = np.einsum("ni,nij->nj", candidate_weights[kept_indices], corners)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        sampled_points.append(points)
        sampled_normals.append(normals)
        sampled_group_ids.append(np.full(len(points), group_id, dtype=np.int64))
        if len(regions) >= 20 and (region_index % 25 == 0 or region_index == len(regions)):
            print(f"Surface Poisson sampling: {region_index}/{len(regions)} clipped groups.")
    if not sampled_points:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.concatenate(sampled_points).astype(np.float32),
        np.concatenate(sampled_normals).astype(np.float32),
        np.concatenate(sampled_group_ids),
    )


def mesh_coordinate_key(point: np.ndarray) -> tuple[float, float, float]:
    """Match float32 slicer constraints back to their float64 PyVista vertices."""

    return tuple(np.round(np.asarray(point, dtype=np.float64), decimals=5))


def automatic_angle_transition_vertices(
    faces: np.ndarray, automatic_faces: np.ndarray, angle_selected: np.ndarray, point_count: int
) -> np.ndarray:
    """Find vertices on automatic faces where the hard angle test changes."""

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        if not automatic_faces[face_index]:
            continue
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    transition_vertices = np.zeros(point_count, dtype=bool)
    for edge, incident in edge_faces.items():
        if len(incident) != 2:
            continue
        if bool(angle_selected[incident[0]]) != bool(angle_selected[incident[1]]):
            transition_vertices[list(edge)] = True
    return transition_vertices


def split_clipped_groups_by_edge_connectivity(
    triangles: list[np.ndarray],
    group_ids: list[int],
    reserved_group_ids: np.ndarray,
) -> np.ndarray:
    """Split a source group when clipping leaves edge-disconnected patches.

    Selection starts from mesh vertices, whereas height and angle clipping can
    remove the bridge that originally connected those vertices.  Surface
    scalar constraints must use the final clipped topology instead.
    """

    if not triangles:
        return np.empty((0,), dtype=np.int64)
    group_ids_array = np.asarray(group_ids, dtype=np.int64)
    remapped = group_ids_array.copy()
    next_group_id = int(np.max(reserved_group_ids)) + 1 if reserved_group_ids.size else 0
    for group_id in np.unique(group_ids_array):
        triangle_indices = np.flatnonzero(group_ids_array == group_id)
        if len(triangle_indices) < 2:
            continue
        edge_owners: dict[tuple[tuple[float, float, float], tuple[float, float, float]], list[int]] = {}
        for local_index, triangle_index in enumerate(triangle_indices):
            triangle = triangles[int(triangle_index)]
            keys = [mesh_coordinate_key(point) for point in triangle]
            for first, second in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
                edge_owners.setdefault(tuple(sorted((first, second))), []).append(local_index)
        neighbours = [set() for _ in triangle_indices]
        for owners in edge_owners.values():
            for owner in owners:
                neighbours[owner].update(other for other in owners if other != owner)
        components: list[list[int]] = []
        remaining = set(range(len(triangle_indices)))
        while remaining:
            component = [remaining.pop()]
            stack = component.copy()
            while stack:
                current = stack.pop()
                adjacent = neighbours[current] & remaining
                remaining.difference_update(adjacent)
                stack.extend(adjacent)
                component.extend(adjacent)
            components.append(sorted(component))
        if len(components) < 2:
            continue
        # Retain the original ID for the earliest component, then allocate
        # fresh IDs without colliding with manual or other automatic groups.
        components.sort(key=lambda component: int(triangle_indices[component[0]]))
        for component in components[1:]:
            for local_index in component:
                remapped[triangle_indices[local_index]] = next_group_id
            next_group_id += 1
        print(f"Split clipped surface group {int(group_id)} into {len(components)} edge-connected patch(es).")
    return remapped


def resample_surface_constraints(
    mesh,
    points,
    normals,
    group_ids,
    automatic_group_ids,
    automatic_angle: float,
    automatic_min_height: float | None,
    automatic_max_height: float | None,
    args: argparse.Namespace,
):
    """Replace vertex-density-biased constraints while preserving auto face tests."""

    if args.surface_sample_density <= 0.0:
        return points, normals, group_ids
    surface_face_sampler = SURFACE_FACE_SAMPLER or sample_grouped_surface_faces
    # The slicer already supplies a triangulated mesh. Avoid re-triangulating:
    # PyVista may reorder duplicated vertices, invalidating constraint indices.
    sampling_mesh = mesh.copy(deep=True)
    sampling_mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    mesh_points = np.asarray(sampling_mesh.points, dtype=np.float64)
    point_to_index = {mesh_coordinate_key(point): index for index, point in enumerate(mesh_points)}
    vertex_groups = np.full(sampling_mesh.n_points, -1, dtype=np.int64)
    for point, group_id in zip(points, group_ids):
        point_index = point_to_index.get(mesh_coordinate_key(point))
        if point_index is not None:
            vertex_groups[point_index] = int(group_id)
    faces = surface_sampling.faces_of(sampling_mesh)
    groups_per_face = vertex_groups[faces]
    selected_faces = np.all(groups_per_face == groups_per_face[:, :1], axis=1) & (groups_per_face[:, 0] >= 0)
    face_group_ids = np.where(selected_faces, groups_per_face[:, 0], -1)

    # The automatic selector works on faces, but its output is represented as
    # vertex groups before this hook.  Without restoring this test, an
    # unqualified triangle can be sampled if each of its vertices happens to
    # be shared with separately qualified neighbouring faces.
    automatic_group_ids = np.asarray(sorted(automatic_group_ids), dtype=np.int64)
    automatic_faces = np.zeros(sampling_mesh.n_cells, dtype=bool)
    automatic_height_eligible = np.ones(sampling_mesh.n_cells, dtype=bool)
    automatic_angle_selected = np.zeros(sampling_mesh.n_cells, dtype=bool)
    if automatic_group_ids.size:
        up = np.asarray(args.up_direction, dtype=np.float64)
        up /= np.linalg.norm(up) + 1e-12
        face_normals = np.asarray(sampling_mesh.cell_data["Normals"], dtype=np.float64)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
        # Reproduce the interactive selector's face tests directly. Rebuilding
        # automatic faces from vertex group IDs drops partial boundary faces,
        # making cap geometry visibly holey even though the selector preview is
        # continuous.
        automatic_faces = np.ones(sampling_mesh.n_cells, dtype=bool)
        automatic_angle_selected = face_normals @ up > np.cos(np.deg2rad(automatic_angle))
        automatic_eligible = automatic_angle_selected.copy()
        vertex_heights = mesh_points[faces] @ up - float(np.min(mesh_points @ up))
        face_heights = np.mean(vertex_heights, axis=1)
        if automatic_min_height is not None:
            automatic_height_eligible &= (
                np.max(vertex_heights, axis=1) >= float(automatic_min_height)
                if args.height_selection_mode == "clipped" else face_heights >= float(automatic_min_height)
            )
        if automatic_max_height is not None:
            automatic_height_eligible &= (
                np.min(vertex_heights, axis=1) <= float(automatic_max_height)
                if args.height_selection_mode == "clipped" else face_heights <= float(automatic_max_height)
            )
        automatic_eligible &= automatic_height_eligible
        automatic_eligible = filter_small_selected_face_regions(
            sampling_mesh, automatic_eligible, args.surface_region_area_ratio_min
        )
        manual_faces = selected_faces & ~np.isin(face_group_ids, automatic_group_ids)
        selected_faces = manual_faces | automatic_eligible
        for face_index in np.flatnonzero(automatic_eligible):
            candidate_groups = vertex_groups[faces[face_index]]
            candidate_groups = candidate_groups[np.isin(candidate_groups, automatic_group_ids)]
            if candidate_groups.size:
                values, counts = np.unique(candidate_groups, return_counts=True)
                face_group_ids[face_index] = int(values[np.argmax(counts)])
            else:
                # This is only possible at a newly admitted partial boundary.
                # It will be skipped unless a neighbouring selected vertex can
                # supply a valid automatic group in the transition pass below.
                face_group_ids[face_index] = -1

    transition_faces = np.zeros(sampling_mesh.n_cells, dtype=bool)
    if args.boundary_vertex_normal_smoothing and automatic_group_ids.size:
        transition_vertices = automatic_angle_transition_vertices(
            faces, automatic_faces, automatic_angle_selected, sampling_mesh.n_points
        )
        transition_faces = automatic_faces & np.any(transition_vertices[faces], axis=1) & automatic_height_eligible
        if np.any(transition_faces):
            print(f"Boundary vertex-normal smoothing: {int(transition_faces.sum())} automatic one-ring face(s).")

    generated_sampling_faces = selected_faces | transition_faces
    use_generated_mesh = args.height_selection_mode == "clipped" or np.any(transition_faces)
    cap_triangles: list[np.ndarray] = []
    cap_group_ids: list[int] = []
    cap_parent_face_ids: list[int] = []
    cap_transition_flags: list[bool] = []
    if use_generated_mesh:
        height_clipping_active = args.height_selection_mode == "clipped"
        minimum_projection = None
        lower_clip_world = None
        upper_clip_world = None
        if height_clipping_active:
            minimum_projection = float(np.min(mesh_points @ up))
            relative_vertex_heights = mesh_points @ up - minimum_projection
            lower_height = 0.0 if automatic_min_height is None else float(automatic_min_height)
            upper_height = float(np.max(relative_vertex_heights)) if automatic_max_height is None else float(automatic_max_height)
            lower_clip_world = minimum_projection + lower_height
            upper_clip_world = minimum_projection + upper_height
        else:
            relative_vertex_heights = None
        clipped_triangles, clipped_groups = [], []
        # A face can be copied whenever it is wholly inside the requested
        # height interval and away from the angle-transition ring.  This
        # vectorized test avoids polygon clipping for the overwhelming
        # majority of faces, including when clipped height selection is on.
        direct_face_mask = selected_faces & ~transition_faces
        if height_clipping_active:
            vertex_relative_heights = relative_vertex_heights[faces]
            fully_inside_height = (
                np.min(vertex_relative_heights, axis=1) >= lower_height - 1e-10
            ) & (
                np.max(vertex_relative_heights, axis=1) <= upper_height + 1e-10
            )
            direct_face_mask &= fully_inside_height
        direct_face_indices = np.flatnonzero(direct_face_mask)
        if direct_face_indices.size:
            direct_triangles = mesh_points[faces[direct_face_indices]]
            direct_groups = face_group_ids[direct_face_indices]
            valid_direct = direct_groups >= 0
            direct_triangles = direct_triangles[valid_direct]
            direct_groups = direct_groups[valid_direct]
            direct_parent_ids = direct_face_indices[valid_direct]
            clipped_triangles.extend(direct_triangles)
            clipped_groups.extend(direct_groups.tolist())
            cap_triangles.extend(direct_triangles)
            cap_group_ids.extend(direct_groups.tolist())
            cap_parent_face_ids.extend(direct_parent_ids.tolist())
            cap_transition_flags.extend([False] * len(direct_triangles))
        faces_to_clip = generated_sampling_faces & ~direct_face_mask
        generated_face_indices = np.flatnonzero(faces_to_clip)
        print(
            "Building clipped surface constraint patch: "
            f"{len(generated_face_indices)} face(s) requiring polygon clipping; "
            f"bulk-copied {len(clipped_triangles)} unchanged selected triangle(s) "
            f"({int(selected_faces.sum())} selected, {int(transition_faces.sum())} transition)."
        )
        if np.any(transition_faces):
            print("Building source-mesh vertex adjacency for the transition ring.")
            face_vertices = mesh_points[faces]
            face_areas = 0.5 * np.linalg.norm(
                np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]), axis=1
            )
            incident_faces = boundary_smoothing.incident_faces_by_vertex(faces, sampling_mesh.n_points)
        else:
            face_areas = None
            incident_faces = None
        progress_step = max(25_000, len(generated_face_indices) // 20)
        for processed, face_index in enumerate(generated_face_indices, start=1):
            face = faces[face_index]
            group_id = face_group_ids[face_index]
            # A transition-only face was excluded by the hard face-normal test,
            # so restore its automatic group before partial clipping.
            if group_id < 0:
                candidate_groups = vertex_groups[face]
                candidate_groups = candidate_groups[np.isin(candidate_groups, automatic_group_ids)]
                if not candidate_groups.size:
                    continue
                values, counts = np.unique(candidate_groups, return_counts=True)
                group_id = int(values[np.argmax(counts)])
            polygon = mesh_points[face]
            if transition_faces[face_index]:
                local_normals = boundary_smoothing.face_local_vertex_normals(
                    face_index, faces, face_normals, face_areas, incident_faces,
                    args.boundary_face_normal_similarity_degrees,
                )
                scores = local_normals @ up - np.cos(np.deg2rad(automatic_angle))
                polygon, _ = boundary_smoothing.clip_polygon_by_values(polygon, scores)
                if len(polygon) < 3:
                    continue
            if args.height_selection_mode == "clipped":
                polygon = height_clipping.clip_polygon_to_height(
                    polygon, up, lower_clip_world, keep_above=True
                )
                polygon = height_clipping.clip_polygon_to_height(
                    polygon, up, upper_clip_world, keep_above=False
                )
            if len(polygon) >= 3:
                pieces = height_clipping.triangulate_polygon(polygon)
                clipped_triangles.extend(pieces)
                clipped_groups.extend([group_id] * len(pieces))
                cap_triangles.extend(pieces)
                cap_group_ids.extend([int(group_id)] * len(pieces))
                cap_parent_face_ids.extend([int(face_index)] * len(pieces))
                cap_transition_flags.extend([bool(transition_faces[face_index])] * len(pieces))
            if processed % progress_step == 0 or processed == len(generated_face_indices):
                print(
                    f"  Clipped constraint patch: {processed}/{len(generated_face_indices)} source face(s), "
                    f"{len(clipped_triangles)} triangle(s)."
                )
        print(f"Splitting {len(clipped_triangles)} clipped triangle(s) into edge-connected patch(es).")
        source_clipped_groups = np.asarray(clipped_groups, dtype=np.int64)
        remapped_groups = split_clipped_groups_by_edge_connectivity(
            clipped_triangles,
            clipped_groups,
            np.asarray(group_ids, dtype=np.int64),
        )
        clipped_groups = remapped_groups.tolist()
        cap_group_ids = remapped_groups.tolist()
        remapped_automatic_groups = set(
            remapped_groups[np.isin(source_clipped_groups, automatic_group_ids)].tolist()
        )
        clipped_mesh = height_clipping.polydata_from_triangles(clipped_triangles)
        print(f"Sampling {clipped_mesh.n_cells} clipped surface triangle(s) for surface constraints.")
        clipped_mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
        clipped_mesh.cell_data["AngleTransition"] = np.asarray(cap_transition_flags, dtype=np.uint8)
        sampleable_faces = np.ones(clipped_mesh.n_cells, dtype=bool)
        if args.exclude_surface_transition_samples:
            sampleable_faces &= ~np.asarray(clipped_mesh.cell_data["AngleTransition"], dtype=bool)
            print(
                "Excluding "
                f"{int((~sampleable_faces).sum())} angle-transition triangle(s) from surface-preservation samples."
            )
        sampled = surface_face_sampler(
            clipped_mesh,
            sampleable_faces,
            np.asarray(clipped_groups, dtype=np.int64),
            args.surface_sample_density,
            args,
        )
    else:
        sampled = surface_face_sampler(sampling_mesh, selected_faces, face_group_ids, args.surface_sample_density, args)
        for face_index in np.flatnonzero(selected_faces):
            cap_triangles.append(mesh_points[faces[face_index]])
            cap_group_ids.append(int(face_group_ids[face_index]))
            cap_parent_face_ids.append(int(face_index))
            cap_transition_flags.append(False)
        remapped_automatic_groups = set(automatic_group_ids.tolist())
    # Keep the precise selected patch for post-processing caps. The trainer
    # persists this after resampling, before clipping information is lost.
    resample_surface_constraints.cap_geometry = {
        "triangles": np.asarray(cap_triangles, dtype=np.float32),
        "group_ids": np.asarray(cap_group_ids, dtype=np.int64),
        "parent_face_ids": np.asarray(cap_parent_face_ids, dtype=np.int64),
        "angle_transition": np.asarray(cap_transition_flags, dtype=bool),
    }
    resample_surface_constraints.automatic_group_ids = remapped_automatic_groups
    if sampled[0].size == 0:
        print("Uniform surface sampler found no fully constrained faces; retaining the original vertex constraints.")
        return points, normals, group_ids
    sampler_kind = "Adaptive" if SURFACE_FACE_SAMPLER is not None else "Uniform"
    print(f"{sampler_kind} surface constraints: {len(points)} vertices -> {len(sampled[0])} samples.")
    return sampled


def surface_constraint_cache_fingerprint(
    mesh,
    group_ids: np.ndarray,
    automatic_group_ids,
    automatic_angle: float,
    automatic_min_height: float | None,
    automatic_max_height: float | None,
    args: argparse.Namespace,
) -> str:
    """Fingerprint every input that changes the generated surface samples."""

    digest = hashlib.sha256()
    faces = np.ascontiguousarray(surface_sampling.faces_of(mesh), dtype=np.int64)
    vertices = np.ascontiguousarray(np.asarray(mesh.points), dtype=np.float32)
    input_groups = np.ascontiguousarray(group_ids, dtype=np.int64)
    for values in (vertices, faces, input_groups):
        digest.update(memoryview(values).cast("B"))
    settings = {
        "automatic_group_ids": sorted(int(value) for value in automatic_group_ids),
        "automatic_angle": float(automatic_angle),
        "automatic_min_height": automatic_min_height,
        "automatic_max_height": automatic_max_height,
        "height_selection_mode": args.height_selection_mode,
        "boundary_vertex_normal_smoothing": bool(args.boundary_vertex_normal_smoothing),
        "boundary_face_normal_similarity_degrees": float(args.boundary_face_normal_similarity_degrees),
        "exclude_surface_transition_samples": bool(args.exclude_surface_transition_samples),
        "surface_guide_normal_mode": args.surface_guide_normal_mode,
        "surface_guide_normal_similarity_degrees": float(args.surface_guide_normal_similarity_degrees),
        "surface_sample_density": float(args.surface_sample_density),
        "candidate_oversampling": int(args.constraint_sample_candidate_oversampling),
        "minimum_samples_per_region": int(args.min_constraint_samples_per_region),
        "maximum_samples_per_region": int(args.max_constraint_samples_per_region),
        "sample_seed": int(args.constraint_sample_seed),
    }
    digest.update(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def resample_base_constraints(mesh, points, target_scalars, up_direction, base_height, base_normal_angle, scalar_scale, args: argparse.Namespace):
    """Use area-density samples on downward faces wholly inside the base height band."""
    if base_height <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 1), dtype=np.float32)

    density = args.surface_sample_density if args.base_sample_density is None else args.base_sample_density
    if density <= 0.0:
        return points, target_scalars
    sampling_mesh = mesh.copy(deep=True)
    sampling_mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    mesh_points = np.asarray(sampling_mesh.points, dtype=np.float32)
    up = np.asarray(up_direction, dtype=np.float32).reshape(3)
    up /= np.linalg.norm(up) + 1e-8
    relative_heights = mesh_points @ up - float(np.min(mesh_points @ up))
    faces = surface_sampling.faces_of(sampling_mesh)
    face_normals = np.asarray(sampling_mesh.cell_data["Normals"], dtype=np.float32)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    # Keep every generated sample inside the original base band. Triangles cut
    # by the height plane are intentionally omitted rather than extrapolated.
    selected_faces = np.all(relative_heights[faces] <= float(base_height), axis=1)
    selected_faces &= face_normals @ (-up) > np.cos(np.deg2rad(base_normal_angle))
    face_group_ids = np.where(selected_faces, 0, -1)
    sampled_points, _, _ = sample_grouped_surface_faces(sampling_mesh, selected_faces, face_group_ids, density, args)
    if sampled_points.size == 0:
        print("Uniform base sampler found no faces fully inside the base band; retaining original base vertices.")
        return points, target_scalars
    sampled_heights = sampled_points @ up - float(np.min(mesh_points @ up))
    sampled_scalars = (sampled_heights / float(scalar_scale)).reshape(-1, 1).astype(np.float32)
    print(f"Uniform base constraints: {len(points)} vertices -> {len(sampled_points)} area-density samples.")
    return sampled_points, sampled_scalars


def mesh_for_rendering(mesh, args: argparse.Namespace):
    """Return a visual-only decimated mesh, never an optimization input."""

    target = args.render_proxy_target_faces
    if target <= 0 or mesh.n_cells <= target:
        return mesh
    reduction = 1.0 - target / mesh.n_cells
    proxy = mesh.triangulate().decimate(reduction)
    print(f"Render proxy: {mesh.n_cells} -> {proxy.n_cells} faces (calculation mesh is unchanged).")
    return proxy


def preview_constraint_samples(mesh, surface_points, surface_normals, _group_ids, base_points, up_direction, args: argparse.Namespace) -> None:
    """Show exactly the point sets that will be supplied to the surface/base losses."""

    if args.skip_constraint_sample_preview:
        return
    plotter = surface_sampling.pv.Plotter(title="Pipeline constraint samples")
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(mesh_for_rendering(mesh, args), color="lightgray", opacity=0.16, show_edges=True, edge_color="gray")
    if len(surface_points):
        plotter.add_mesh(
            surface_sampling.pv.PolyData(surface_points), color="gold", point_size=6.0,
            render_points_as_spheres=True, label="Surface preservation (automatic + manual)"
        )
    if len(base_points):
        plotter.add_mesh(
            surface_sampling.pv.PolyData(base_points), color="deepskyblue", point_size=6.0,
            render_points_as_spheres=True, label="Base constraints"
        )
    arrow_length = args.constraint_normal_arrow_scale
    if arrow_length is None and args.surface_sample_density > 0.0:
        arrow_length = 0.35 / np.sqrt(args.surface_sample_density)
    if arrow_length is not None and arrow_length > 0.0:
        if len(surface_points):
            plotter.add_arrows(surface_points, surface_normals, mag=arrow_length, color="midnightblue")
        if len(base_points):
            base_normals = np.repeat(np.asarray(up_direction, dtype=np.float32).reshape(1, 3), len(base_points), axis=0)
            plotter.add_arrows(base_points, base_normals, mag=arrow_length, color="royalblue")
    plotter.add_legend(bcolor="white")
    plotter.add_text(
        f"surface preservation={len(surface_points)}\nbase={len(base_points)}\n"
        f"Gold arrows: {args.surface_guide_normal_mode} surface guides. Blue arrows: target up direction.",
        position="upper_left", font_size=10,
    )
    plotter.show()
    if not ask_to_continue("Constraint-sample preview closed. Start training?", args):
        raise SystemExit("Pipeline stopped after constraint-sample preview.")


def filter_small_selected_face_regions(mesh, selected_faces: np.ndarray, area_ratio: float) -> np.ndarray:
    """Match the trainer's edge-connected automatic-region area filtering."""

    if area_ratio <= 0.0 or not np.any(selected_faces):
        return selected_faces
    faces = surface_sampling.faces_of(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = points[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_to_faces.setdefault(tuple(sorted((int(first), int(second)))), []).append(face_index)
    neighbors: list[set[int]] = [set() for _ in range(len(faces))]
    for incident_faces in edge_to_faces.values():
        for face_index in incident_faces:
            neighbors[face_index].update(other for other in incident_faces if other != face_index)

    kept = np.zeros_like(selected_faces)
    visited = np.zeros_like(selected_faces)
    components: list[list[int]] = []
    for start in np.flatnonzero(selected_faces):
        if visited[start]:
            continue
        visited[start] = True
        component, stack = [], [int(start)]
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for neighbor in neighbors[face_index]:
                if selected_faces[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    largest_area = max(float(np.sum(areas[component])) for component in components)
    for component in components:
        if float(np.sum(areas[component])) >= area_ratio * largest_area:
            kept[np.asarray(component, dtype=np.int64)] = True
    return kept


def configure_automatic_surface_selection(
    mesh,
    up_direction,
    initial_angle: float,
    initial_min_height: float | None,
    initial_max_height: float | None,
    args: argparse.Namespace,
) -> tuple[float, float | None, float | None]:
    """Interactively tune automatic face selection before its constraints are built."""

    if args.skip_automatic_selection_window:
        return initial_angle, initial_min_height, initial_max_height
    selection_mesh = mesh.copy(deep=True).triangulate()
    selection_mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    faces = surface_sampling.faces_of(selection_mesh)
    points = np.asarray(selection_mesh.points, dtype=np.float64)
    normals = np.asarray(selection_mesh.cell_data["Normals"], dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    up = np.asarray(up_direction, dtype=np.float64).reshape(3)
    up /= np.linalg.norm(up) + 1e-12
    minimum_projection = float(np.min(points @ up))
    vertex_heights = points @ up - minimum_projection
    heights = np.mean(vertex_heights[faces], axis=1)
    maximum_height = max(float(vertex_heights.max()), 1e-6)
    face_vertices = points[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]), axis=1
    )
    incident_faces = boundary_smoothing.incident_faces_by_vertex(faces, selection_mesh.n_points)
    state = {
        "angle": float(initial_angle),
        "min_height": 0.0 if initial_min_height is None else float(initial_min_height),
        "max_height": maximum_height if initial_max_height is None else float(initial_max_height),
        "area_ratio": float(args.surface_region_area_ratio_min),
        "boundary_smoothing": bool(args.boundary_vertex_normal_smoothing),
        "height_smoothing": args.height_selection_mode == "clipped",
        "fast_preview": bool(args.fast_selection_preview),
    }
    plotter = surface_sampling.pv.Plotter(title="Automatic surface-quality selection")
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(
        mesh_for_rendering(selection_mesh, args),
        color="lightgray",
        opacity=0.14,
        show_edges=not state["fast_preview"],
        edge_color="gray",
    )

    def smoothed_selection_patch(
        angle_selected: np.ndarray, selected: np.ndarray, height_eligible: np.ndarray,
        use_angle_smoothing: bool, use_height_smoothing: bool,
    ) -> tuple[pv.PolyData, int]:
        transition_vertices = automatic_angle_transition_vertices(
            faces, np.ones(selection_mesh.n_cells, dtype=bool), angle_selected, selection_mesh.n_points
        )
        transition_faces = (
            np.any(transition_vertices[faces], axis=1) & height_eligible
            if use_angle_smoothing else np.zeros(selection_mesh.n_cells, dtype=bool)
        )
        candidates = selected | transition_faces
        threshold = float(np.cos(np.deg2rad(state["angle"])))
        polygons: list[np.ndarray] = []
        for face_index in np.flatnonzero(candidates):
            polygon = points[faces[face_index]]
            if transition_faces[face_index]:
                local_normals = boundary_smoothing.face_local_vertex_normals(
                    face_index, faces, normals, face_areas, incident_faces,
                    args.boundary_face_normal_similarity_degrees,
                )
                polygon, _ = boundary_smoothing.clip_polygon_by_values(polygon, local_normals @ up - threshold)
                if len(polygon) < 3:
                    continue
            if use_height_smoothing:
                polygon = height_clipping.clip_polygon_to_height(
                    polygon, up, minimum_projection + state["min_height"], keep_above=True
                )
                polygon = height_clipping.clip_polygon_to_height(
                    polygon, up, minimum_projection + state["max_height"], keep_above=False
                )
            if len(polygon) >= 3:
                polygons.append(polygon)
        return height_clipping.polydata_from_triangles(
            [triangle for polygon in polygons for triangle in height_clipping.triangulate_polygon(polygon)]
        ), int(np.count_nonzero(transition_faces))

    def refresh(_value=None) -> None:
        angle_selected = normals @ up > np.cos(np.deg2rad(state["angle"]))
        if state["height_smoothing"]:
            height_eligible = np.max(vertex_heights[faces], axis=1) >= state["min_height"]
            height_eligible &= np.min(vertex_heights[faces], axis=1) <= state["max_height"]
        else:
            height_eligible = (heights >= state["min_height"]) & (heights <= state["max_height"])
        raw_selected = angle_selected & height_eligible
        selected = filter_small_selected_face_regions(selection_mesh, raw_selected, state["area_ratio"])
        if state["fast_preview"]:
            # Fast interaction path: avoid expensive polygon clipping and VTK
            # mesh rebuilding for every slider callback. Exact clipping occurs
            # later when constraints are generated after acceptance.
            selected_mesh = selection_mesh.extract_cells(np.flatnonzero(selected))
            transition_count = 0
        elif state["boundary_smoothing"] or state["height_smoothing"]:
            selected_mesh, transition_count = smoothed_selection_patch(
                angle_selected, selected, height_eligible,
                state["boundary_smoothing"], state["height_smoothing"],
            )
        else:
            selected_mesh = selection_mesh.extract_cells(np.flatnonzero(selected))
            transition_count = 0
        if selected_mesh.n_cells:
            plotter.add_mesh(
                mesh_for_rendering(selected_mesh, args), color="tomato", opacity=0.75, show_edges=not state["fast_preview"],
                edge_color="darkred", name="automatic_selection", reset_camera=False,
            )
        else:
            plotter.remove_actor("automatic_selection", render=False)
        plotter.add_text(
            f"angle <= {state['angle']:.1f} deg\nheight = [{state['min_height']:.3f}, {state['max_height']:.3f}]\n"
            f"selected faces = {int(selected.sum())}/{int(raw_selected.sum())}\n"
            f"minimum region area ratio = {state['area_ratio']:.3f}\n"
            f"angle smoothing = {state['boundary_smoothing']}; height smoothing = {state['height_smoothing']}\n"
            f"fast whole-face preview = {state['fast_preview']}\n"
            f"transition faces = {transition_count}\nClose window to accept.",
            position="upper_left", font_size=10, name="automatic_selection_text",
        )
        plotter.render()

    def set_angle(value: float) -> None:
        state["angle"] = float(value)
        refresh()

    def set_min_height(value: float) -> None:
        state["min_height"] = float(value)
        refresh()

    def set_max_height(value: float) -> None:
        state["max_height"] = float(value)
        refresh()

    def set_area_ratio(value: float) -> None:
        state["area_ratio"] = float(value)
        refresh()

    def set_boundary_smoothing(enabled: bool) -> None:
        state["boundary_smoothing"] = bool(enabled)
        refresh()

    def set_height_smoothing(enabled: bool) -> None:
        state["height_smoothing"] = bool(enabled)
        refresh()

    def set_fast_preview(enabled: bool) -> None:
        state["fast_preview"] = bool(enabled)
        refresh()

    plotter.add_slider_widget(set_angle, rng=(0.0, 90.0), value=state["angle"], title="Maximum normal angle (deg)", pointa=(0.02, 0.10), pointb=(0.32, 0.10))
    plotter.add_slider_widget(set_min_height, rng=(0.0, maximum_height), value=state["min_height"], title="Minimum height", pointa=(0.36, 0.10), pointb=(0.64, 0.10))
    plotter.add_slider_widget(set_max_height, rng=(0.0, maximum_height), value=state["max_height"], title="Maximum height", pointa=(0.68, 0.10), pointb=(0.98, 0.10))
    plotter.add_slider_widget(set_area_ratio, rng=(0.0, 0.50), value=state["area_ratio"], title="Minimum connected region area ratio", pointa=(0.02, 0.02), pointb=(0.32, 0.02))
    plotter.add_checkbox_button_widget(set_boundary_smoothing, value=state["boundary_smoothing"], position=(560, 22), size=24)
    plotter.add_text("Angle boundary smoothing", position=(595, 24), font_size=10, name="boundary_smoothing_label")
    plotter.add_checkbox_button_widget(set_height_smoothing, value=state["height_smoothing"], position=(860, 22), size=24)
    plotter.add_text("Height partial coverage", position=(895, 24), font_size=10, name="height_smoothing_label")
    plotter.add_checkbox_button_widget(set_fast_preview, value=state["fast_preview"], position=(20, 55), size=24)
    plotter.add_text("Fast whole-face preview", position=(55, 57), font_size=10, name="fast_preview_label")
    refresh()
    plotter.show()
    if state["min_height"] > state["max_height"]:
        raise ValueError("Automatic selection minimum height exceeds maximum height.")
    args.surface_normal_max_angle = state["angle"]
    args.surface_height_min = state["min_height"]
    args.surface_height_max = state["max_height"]
    args.surface_region_area_ratio_min = state["area_ratio"]
    args.boundary_vertex_normal_smoothing = state["boundary_smoothing"]
    args.height_selection_mode = "clipped" if state["height_smoothing"] else "face-center"
    print(
        "Automatic selection accepted: "
        f"angle <= {state['angle']:.3f} deg, "
        f"height = [{state['min_height']:.6f}, {state['max_height']:.6f}]."
        f" minimum region area ratio = {state['area_ratio']:.6f}; "
        f"height selection mode = {'clipped' if state['height_smoothing'] else 'face-center'}."
    )
    return state["angle"], state["min_height"], state["max_height"]


def configure_training_module(module, paths: dict[str, Path], args: argparse.Namespace) -> None:
    """Point the existing banded slicer at this pipeline's isolated artifacts."""

    up_direction = np.asarray(args.up_direction, dtype=np.float32)
    up_direction /= np.linalg.norm(up_direction)
    module.SURFACE_MESH_PATH = str(args.mesh_path)
    module.VOLUME_NODE_PATH = None if args.node_path is None else str(args.node_path)
    module.VOLUME_ELE_PATH = None if args.ele_path is None else str(args.ele_path)
    module.CHECKPOINT_PATH = str(args.start_checkpoint or paths["initializer"])
    module.STRESS_ORIENTATION_INITIALIZATION = json.loads(args.stress_orientation_record.read_text()) if args.stress_orientation_record else None
    module.DEVICE = args.device
    module.OUTPUT_CHECKPOINT_PATH = str(paths["output_checkpoint"])
    module.SDF_GATE_CHECKPOINT_PATH = None if args.skip_sdf else str(paths["sdf"])
    module.MANUAL_SURFACE_REGIONS_PATH = str(paths["manual_regions"])
    module.SURFACE_CONSTRAINT_METADATA_PATH = str(paths["constraint_metadata"])
    module.SLICER_VIS_CONFIG_PATH = str(paths["visualization_config"])
    has_saved_manual_regions = has_nonempty_manual_regions(paths["manual_regions"])
    module.SAVE_MANUAL_SURFACE_REGIONS = True
    # A fresh picker run intentionally replaces the saved manual selection on
    # acceptance. Declining the prompt below loads the saved regions instead.
    module.LOAD_MANUAL_SURFACE_REGIONS = has_saved_manual_regions and not args.select_surface_interactively
    module.SAVE_SURFACE_CONSTRAINT_METADATA = True
    module.ENABLE_INTERACTIVE_MANUAL_SURFACE_REGIONS = (
        args.select_surface_interactively or has_saved_manual_regions
    )
    # The pipeline already provides explicit SDF/initializer verification and
    # interactive patch selection. Leave the template's batch-debug windows off.
    module.SHOW_SURFACE_QUALITY_POINTS_BEFORE_TRAINING = False
    module.SHOW_SURFACE_QUALITY_NORMALS_BEFORE_TRAINING = False
    module.SHOW_SURFACE_CONSTRAINT_BATCHES_BEFORE_TRAINING = False
    module.SHOW_ALL_LOSS_EVALUATION_POINTS_BEFORE_TRAINING = False
    module.SHOW_SURFACE_EDGE_PAIRS_BEFORE_TRAINING = args.show_local_edge_pairs
    module.SHOW_INPUT_REGULARIZATION_BATCHES_BEFORE_TRAINING = False

    module.SCALAR_W0 = args.w0
    module.SCALAR_W0_INITIAL = args.w0_initial
    if args.num_layers < 1:
        raise ValueError("--num-layers must be positive.")
    module.SCALAR_NUM_LAYERS = args.num_layers
    module.UP_DIRECTION = up_direction.reshape(1, 3)
    module.BUILD_DIRECTION_WORLD = tuple(float(value) for value in up_direction)
    module.IN_PLANE_U_WORLD = tuple(float(value) for value in args.in_plane_u)
    module.BATCH_SIZE = args.batch_size
    if args.input_grid_spacing <= 0.0:
        raise ValueError("--input-grid-spacing must be positive.")
    if args.input_grid_spacing_xyz is not None and any(value <= 0.0 for value in args.input_grid_spacing_xyz):
        raise ValueError("--input-grid-spacing-xyz values must all be positive.")
    module.GRID_SPACING = (
        tuple(args.input_grid_spacing_xyz)
        if args.input_grid_spacing_xyz is not None
        else args.input_grid_spacing
    )
    module.MAX_INPUT_POINTS = args.max_input_points
    module.EPOCHS = args.epochs
    module.LEARNING_RATE = args.learning_rate
    module.STEP_SIZE = args.learning_rate_decay_every_epochs
    module.STEP_GAMMA = args.learning_rate_decay_factor
    module.AFFINE_RESIDUAL_DIRECTION = [0.0, 1.0, 0.0] if args.affine_y_residual else None
    module.AFFINE_RESIDUAL_INITIAL_SCALE = args.affine_residual_initial_scale
    if args.affine_y_residual:
        print(
            "Affine y residual enabled: phi = y_normalized + residual_scale * residual_net; "
            f"initial residual scale={args.affine_residual_initial_scale:.3e}."
        )
    module.LEARNING_RATE_WARMUP_EPOCHS = args.warmup_epochs
    if args.surface_scalar_ramp_epochs < 0:
        raise ValueError("--surface-scalar-ramp-epochs cannot be negative.")
    if not 0.0 < args.warmup_factor <= 1.0:
        raise ValueError("--warmup-factor must be in (0, 1].")
    if args.learning_rate_ramp_epochs < 0:
        raise ValueError("--learning-rate-ramp-epochs cannot be negative.")
    module.LEARNING_RATE_WARMUP_FACTOR = args.warmup_factor
    module.LEARNING_RATE_RAMP_EPOCHS = args.learning_rate_ramp_epochs
    module.SURFACE_GROUP_SCALAR_START_EPOCH = args.surface_group_scalar_delay_epochs
    module.SURFACE_SCALAR_RAMP_EPOCHS = args.surface_scalar_ramp_epochs
    module.ENABLE_SURFACE_GROUP_SCALAR_LOSS = args.scalar_consistency_mode == "group"
    module.ENABLE_SURFACE_EDGE_SCALAR_LOSS = args.scalar_consistency_mode == "local-edge"
    module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS = args.scalar_consistency_mode == "local-edge"
    module.SURFACE_EDGE_SCALAR_WEIGHT = args.local_edge_scalar_weight
    module.SURFACE_EDGE_SCALAR_START_EPOCH = args.surface_group_scalar_delay_epochs
    module.SURFACE_EDGE_ALIGNMENT_MAX_ANGLE_DEGREES = args.local_edge_alignment_degrees
    module.SURFACE_EDGE_ALIGNMENT_PAIR_WEIGHT = args.edge_alignment_pair_weight
    module.SURFACE_EDGE_USE_SAMPLED_NEIGHBORS = args.scalar_consistency_mode == "local-edge"
    module.SURFACE_EDGE_NEIGHBOR_RADIUS_WORLD = args.local_edge_neighbor_radius_factor / np.sqrt(max(args.surface_sample_density, 1e-8))
    module.SURFACE_EDGE_NEIGHBOR_NORMAL_ANGLE_DEGREES = args.local_edge_neighbor_normal_degrees
    if args.long_range_pairs_per_group < 0 or args.long_range_min_distance_factor < 0.0:
        raise ValueError("Long-range pair count and minimum-distance factor must be non-negative.")
    module.SURFACE_EDGE_LONG_RANGE_PAIRS_PER_GROUP = args.long_range_pairs_per_group
    module.SURFACE_EDGE_LONG_RANGE_MIN_DISTANCE_WORLD = (
        args.long_range_min_distance_factor / np.sqrt(max(args.surface_sample_density, 1e-8))
    )
    module.SURFACE_EDGE_LONG_RANGE_SEED = args.long_range_pair_seed
    module.SAVE_EVERY = args.save_every
    module.DEBUG_PRINT_LOSS_BREAKDOWN = args.debug_loss_breakdown
    module.SHOW_NORM_CONTROL_BATCHES_EACH_EPOCH = args.show_norm_control_batches
    module.ENABLE_BALANCED_NORM_BATCHES = args.balanced_norm_batches
    module.NORM_POOL_SIZE = args.norm_pool_size
    module.NORM_POINTS_PER_BAND = args.norm_points_per_band
    module.GRADIENT_BAND_MAX_SQUARED_RATIO = args.max_layer_height_ratio ** 2
    module.GRADIENT_BAND_COUNT = args.gradient_band_count
    if args.gradient_band_ratio_weight is not None:
        module.GRADIENT_BAND_RATIO_WEIGHT = args.gradient_band_ratio_weight
    if args.gradient_band_floor_weight is not None:
        module.GRADIENT_BAND_FLOOR_WEIGHT = args.gradient_band_floor_weight
    if args.disable_norm_banded_control:
        module.GRADIENT_BAND_RATIO_WEIGHT = 0.0
        module.GRADIENT_BAND_FLOOR_WEIGHT = 0.0
    module.PRINT_INITIAL_LOSS_GRADIENT_CONTRIBUTIONS = args.initial_gradient_diagnostics
    module.ENABLE_INPUT_CURVATURE_LOSS = args.enable_input_curvature
    module.INPUT_CURVATURE_MODE = args.input_curvature_mode
    module.INPUT_CURVATURE_WEIGHT = args.input_curvature_weight
    module.INPUT_CURVATURE_SAMPLE_FRACTION = args.input_curvature_sample_fraction
    module.INPUT_CURVATURE_RADIUS_MM = args.input_curvature_radius_mm
    module.INPUT_CURVATURE_BOUNDARY_COVERAGE = args.input_curvature_boundary_coverage
    module.ENABLE_TANGENT_TWIST_LOSS = args.enable_tangent_twist
    module.BASE_DIRECTION_PARALLEL_ONLY = args.base_direction_parallel_only
    module.SURFACE_SCALAR_DIVIDE_BY_GRADIENT = not args.disable_scalar_gradient_division
    module.SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH = args.detach_scalar_gradient_denominator
    print(
        "Scalar consistency: "
        f"mode={args.scalar_consistency_mode} "
        f"delay_epochs={args.surface_group_scalar_delay_epochs} "
        f"ramp_epochs={args.surface_scalar_ramp_epochs} "
        f"first_active_printed_epoch={args.surface_group_scalar_delay_epochs + 1}"
    )
    print(
        "Learning-rate warm-up: "
        f"epochs={args.warmup_epochs} factor={args.warmup_factor:.3e} "
        f"initial_lr={args.learning_rate * args.warmup_factor:.3e} "
        f"target_lr={args.learning_rate:.3e}"
    )
    if args.learning_rate_ramp_epochs:
        print(f"Learning-rate ramp: {args.learning_rate_ramp_epochs} epoch(s) after warm-up.")
    print(
        "Norm-banded control: "
        f"enabled={not args.disable_norm_banded_control} "
        f"ratio_weight={module.GRADIENT_BAND_RATIO_WEIGHT:.3e} "
        f"floor_weight={module.GRADIENT_BAND_FLOOR_WEIGHT:.3e}"
    )
    print(
        "Scalar/base comparison: "
        f"gradient_division={module.SURFACE_SCALAR_DIVIDE_BY_GRADIENT} "
        f"gradient_denominator_detached={module.SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH} "
        f"base_parallel_only={module.BASE_DIRECTION_PARALLEL_ONLY}"
    )
    module.SURFACE_NORMAL_MAX_ANGLE_DEGREES = args.surface_normal_max_angle
    module.SURFACE_HEIGHT_MIN_WORLD = args.surface_height_min
    module.SURFACE_HEIGHT_MAX_WORLD = args.surface_height_max
    module.HEIGHT_SELECTION_MODE = args.height_selection_mode
    module.SURFACE_REGION_AREA_RATIO_MIN = args.surface_region_area_ratio_min
    module.BASE_HEIGHT_WORLD = args.base_height
    module.BASE_NORMAL_MAX_ANGLE_DEGREES = args.base_normal_max_angle
    module.BASE_DIRECTION_WEIGHT = args.base_direction_weight
    module.ENABLE_BASE_DIRECTION_LOSS = module.ENABLE_BASE_DIRECTION_LOSS and args.base_direction_weight > 0. and args.base_height > 0.
    if args.base_height <= 0:
        module.ENABLE_BASE_SCALAR_LOSS = False
        module.BASE_DIRECTION_WEIGHT = module.BASE_SCALAR_WEIGHT = 0.
    module.MIN_INPUT_GRID_SAMPLES_PER_AXIS = args.input_min_grid_samples_per_axis
    module.ENABLE_COLLISION_LOSS = not args.disable_collision
    def surface_constraint_resampler(mesh, points, normals, group_ids, automatic_group_ids, automatic_angle, automatic_min_height, automatic_max_height):
        fingerprint = surface_constraint_cache_fingerprint(
            mesh, group_ids, automatic_group_ids, automatic_angle,
            automatic_min_height, automatic_max_height, args,
        )
        cache_path = paths["surface_constraint_cache"]
        cache_metadata_path = paths["surface_constraint_cache_metadata"]
        if args.reuse_generated_constraint_samples and cache_path.is_file() and cache_metadata_path.is_file():
            with cache_metadata_path.open("r", encoding="utf-8") as handle:
                cache_metadata = json.load(handle)
            if cache_metadata.get("fingerprint") == fingerprint:
                with np.load(cache_path, allow_pickle=False) as cache:
                    result = (
                        np.asarray(cache["surface_points"], dtype=np.float32),
                        np.asarray(cache["surface_normals"], dtype=np.float32),
                        np.asarray(cache["surface_group_ids"], dtype=np.int64),
                    )
                    surface_constraint_resampler.cap_geometry = {
                        "triangles": np.asarray(cache["cap_triangles"], dtype=np.float32),
                        "group_ids": np.asarray(cache["cap_group_ids"], dtype=np.int64),
                        "parent_face_ids": np.asarray(cache["cap_parent_face_ids"], dtype=np.int64),
                        "angle_transition": np.asarray(cache["cap_angle_transition"], dtype=bool),
                    }
                    surface_constraint_resampler.automatic_group_ids = set(
                        np.asarray(cache["automatic_group_ids"], dtype=np.int64).tolist()
                    )
                print(
                    "Reusing generated surface constraints: "
                    f"{len(result[0])} samples from {cache_path}."
                )
                return result
            print("Generated surface-constraint cache does not match current selection/settings; regenerating it.")
        elif args.reuse_generated_constraint_samples:
            print("No generated surface-constraint cache found; generating it now.")
        result = resample_surface_constraints(
            mesh, points, normals, group_ids, automatic_group_ids, automatic_angle,
            automatic_min_height, automatic_max_height, args,
        )
        surface_constraint_resampler.cap_geometry = getattr(resample_surface_constraints, "cap_geometry", None)
        surface_constraint_resampler.automatic_group_ids = getattr(
            resample_surface_constraints, "automatic_group_ids", set(automatic_group_ids)
        )
        cap_geometry = surface_constraint_resampler.cap_geometry
        if cap_geometry is not None:
            np.savez(
                cache_path,
                surface_points=np.asarray(result[0], dtype=np.float32),
                surface_normals=np.asarray(result[1], dtype=np.float32),
                surface_group_ids=np.asarray(result[2], dtype=np.int64),
                cap_triangles=np.asarray(cap_geometry["triangles"], dtype=np.float32),
                cap_group_ids=np.asarray(cap_geometry["group_ids"], dtype=np.int64),
                cap_parent_face_ids=np.asarray(cap_geometry["parent_face_ids"], dtype=np.int64),
                cap_angle_transition=np.asarray(cap_geometry["angle_transition"], dtype=bool),
                automatic_group_ids=np.asarray(sorted(surface_constraint_resampler.automatic_group_ids), dtype=np.int64),
            )
            with cache_metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"fingerprint": fingerprint, "sample_count": int(len(result[0]))},
                    handle, indent=2,
                )
            print(f"Saved generated surface constraints to {cache_path}.")
        return result

    module.SURFACE_CONSTRAINT_RESAMPLER = surface_constraint_resampler
    module.BASE_CONSTRAINT_RESAMPLER = lambda mesh, points, targets, up, height, angle, scale: resample_base_constraints(
        mesh, points, targets, up, height, angle, scale, args
    )
    module.PRE_TRAIN_CONSTRAINT_PREVIEW_CALLBACK = lambda mesh, surface_points, surface_normals, group_ids, base_points, up: preview_constraint_samples(
        mesh, surface_points, surface_normals, group_ids, base_points, up, args
    )
    def configure_automatic_selection_and_filter(mesh, up, angle, minimum, maximum):
        selection = configure_automatic_surface_selection(mesh, up, angle, minimum, maximum, args)
        # The selection window can change this slider after the module's
        # initial configuration has been installed.
        module.SURFACE_REGION_AREA_RATIO_MIN = args.surface_region_area_ratio_min
        return selection

    module.AUTOMATIC_SELECTION_CONFIGURATOR = configure_automatic_selection_and_filter
    module.NORMALIZE_COLLISION_BY_GRADIENT = args.normalize_collision_by_gradient
    module.COLLISION_GRADIENT_EPSILON = args.collision_gradient_epsilon
    module.COLLISION_WEIGHT = args.collision_weight
    module.COLLISION_WEIGHT_SCHEDULE = args.collision_weight_schedule_parsed
    module.COLLISION_TOPK_FRACTION = args.collision_topk_fraction
    module.COLLISION_TOOL_WEIGHTS = tuple(args.collision_tool_weights)
    module.TOOL_AXISYMMETRIC_SPEC = None
    if args.tool_plane_config is not None:
        planes = json.loads(args.tool_plane_config.read_text())
        if isinstance(planes, dict) and planes.get('kind') == 'axisymmetric':
            from axisymmetricTool import sample_tool
            module.TOOL_AXISYMMETRIC_SPEC = planes['training']
            sample_tool(planes['training'], 'cpu', seed=0)  # validate before training
            module.COLLISION_TOOL_WEIGHTS = (1., 1., 1.)
            if args.collision_tool_weights != [1., 1.]:
                if len(args.collision_tool_weights) != 3 or sum(args.collision_tool_weights) <= 0:
                    raise ValueError('Axisymmetric tools require three component weights')
                module.COLLISION_TOOL_WEIGHTS = tuple(args.collision_tool_weights)
            planes = []
        if module.TOOL_AXISYMMETRIC_SPEC is None and (not isinstance(planes, list) or not 1 <= len(planes) <= 2):
            raise ValueError('tool-plane-config must contain one or two rectangular sample-plane objects')
        expected = {'x_min','x_max','z_min','z_max','y_value','num_x','num_z'}
        for index, plane in enumerate(planes):
            if set(plane) != expected or any(not np.isfinite(v) for v in plane.values()):
                raise ValueError('Invalid tool sample-plane keys or values')
            if plane['x_min'] >= plane['x_max'] or plane['z_min'] >= plane['z_max'] or plane['y_value'] <= 0:
                raise ValueError('Invalid tool plane bounds or nozzle clearance')
            if any(plane[name] < 2 or int(plane[name]) != plane[name] for name in ['num_x','num_z']):
                raise ValueError('Tool plane grid counts must be integers >=2')
            prefix = '' if index == 0 else 'SECOND_'
            for name, value in plane.items():
                setattr(module, prefix+'TOOL_'+name.upper(), int(value) if name.startswith('num_') else value)
        module.SECOND_TOOL_SURFACE_ENABLED = len(planes) == 2
    module.TOOL_BOX_SPEC = None
    if args.tool_box_config is not None:
        from boxToolSampling import validate_box
        spec = json.loads(args.tool_box_config.read_text())
        validate_box(spec)
        module.TOOL_BOX_SPEC = spec
        # One top-k component per face plus one interior component.
        module.COLLISION_TOOL_WEIGHTS = (1.,)*7
        if args.collision_tool_weights != [1.,1.]:
            if len(args.collision_tool_weights) != 7 or sum(args.collision_tool_weights) <= 0:
                raise ValueError('Custom box weights require seven face/interior weights')
            module.COLLISION_TOOL_WEIGHTS = tuple(args.collision_tool_weights)
    count = 2 if module.SECOND_TOOL_SURFACE_ENABLED else 1
    if len(args.collision_tool_weights) < count or sum(args.collision_tool_weights[:count]) <= 0:
        raise ValueError('Supply a positive total weight for active tool sample planes')
    module.GLOBAL_ORIENTATION = None
    if not args.disable_orientation:
        if args.optimize_yaw:
            from axialOrientation import AxialOrientation
            module.GLOBAL_ORIENTATION = AxialOrientation(args, paths['output_checkpoint'].parent.parent/'orientation')
        else:
            from globalOrientation import GlobalOrientation
            module.GLOBAL_ORIENTATION = GlobalOrientation(args, paths['output_checkpoint'].parent.parent/'orientation')
    quality = args.objective in ['surface-quality', 'combined'] and not args.disable_surface_quality_loss
    module.SURFACE_DIRECTION_WEIGHT = args.surface_direction_weight if quality else 0.
    module.ENABLE_SURFACE_DIRECTION_LOSS = quality
    module.ENABLE_SURFACE_GROUP_SCALAR_LOSS &= quality
    module.ENABLE_SURFACE_EDGE_SCALAR_LOSS &= quality
    module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS &= quality
    module.ENABLE_TANGENT_TWIST_LOSS &= quality
    if args.disable_surface_scalar_loss:
        module.ENABLE_SURFACE_GROUP_SCALAR_LOSS = module.ENABLE_SURFACE_EDGE_SCALAR_LOSS = False
        module.SURFACE_GROUP_SCALAR_WEIGHT = module.SURFACE_EDGE_SCALAR_WEIGHT = 0.
    module.ENABLE_SUPPORT_FREE_SURFACE_LOSS = (args.objective in ['support-free', 'combined'] or args.enable_support_free_loss) and args.support_free_weight > 0
    module.SUPPORT_FREE_SURFACE_WEIGHT = args.support_free_weight
    module.SUPPORT_FREE_SURFACE_EXTRA_ANGLE_DEGREES = args.support_free_angle
    module.SUPPORT_FREE_SURFACE_HEIGHT_MIN_WORLD = args.support_free_height_min
    module.SUPPORT_FREE_SURFACE_HEIGHT_MAX_WORLD = args.support_free_height_max
    module.SHOW_FINAL_CONTOUR_VIS = args.show_final_preview
    module.REGULARIZATION_PROFILE = args.regularization_profile
    module.GRADIENT_VARIATION_WEIGHT = args.gradient_variation_weight if args.regularization_profile == 'integrated-five-axis' else 0.
    module.SMALL_GRADIENT_WEIGHT = args.small_gradient_weight if args.regularization_profile == 'integrated-five-axis' else 0.
    if args.regularization_profile == 'integrated-five-axis':
        module.GRADIENT_BAND_RATIO_WEIGHT = module.GRADIENT_BAND_FLOOR_WEIGHT = 0.
        module.ENABLE_BASE_DIRECTION_LOSS = module.ENABLE_BASE_SCALAR_LOSS = False
        module.BASE_DIRECTION_WEIGHT = module.BASE_SCALAR_WEIGHT = 0.
    if args.objective in ['strength', 'combined']:
        from strengthConstraints import StrengthConstraints
        module.STRENGTH_LOSS_PROVIDER = StrengthConstraints(args, paths['output_checkpoint'].parent)


def interactive_loss_controls(module, args: argparse.Namespace) -> None:
    """Let a user toggle loss components before the training loop starts."""
    import tkinter as tk
    from tkinter import messagebox

    entries = [
        ("surface_direction", "ENABLE_SURFACE_DIRECTION_LOSS", "SURFACE_DIRECTION_WEIGHT", True),
        ("support_free", "ENABLE_SUPPORT_FREE_SURFACE_LOSS", "SUPPORT_FREE_SURFACE_WEIGHT", True),
        ("group_scalar", "ENABLE_SURFACE_GROUP_SCALAR_LOSS", "SURFACE_GROUP_SCALAR_WEIGHT", True),
        ("edge_scalar", "ENABLE_SURFACE_EDGE_SCALAR_LOSS", "SURFACE_EDGE_SCALAR_WEIGHT", True),
        ("base_direction", "ENABLE_BASE_DIRECTION_LOSS", "BASE_DIRECTION_WEIGHT", True),
        ("base_scalar", "ENABLE_BASE_SCALAR_LOSS", "BASE_SCALAR_WEIGHT", True),
        ("gradient_ratio", None, "GRADIENT_BAND_RATIO_WEIGHT", True),
        ("gradient_floor", None, "GRADIENT_BAND_FLOOR_WEIGHT", True),
        ("input_curvature", "ENABLE_INPUT_CURVATURE_LOSS", "INPUT_CURVATURE_WEIGHT", True),
        ("build_slope", "ENABLE_BUILD_DIRECTION_SLOPE_LOSS", "BUILD_DIRECTION_SLOPE_WEIGHT", True),
        ("collision", "ENABLE_COLLISION_LOSS", "COLLISION_WEIGHT", True),
        ("tangent_twist", "ENABLE_TANGENT_TWIST_LOSS", "TANGENT_TWIST_WEIGHT", True),
    ]
    original_weights = {name: float(getattr(module, weight_attr)) for name, _, weight_attr, _ in entries}
    root = tk.Tk()
    root.title("Loss controls")
    root.resizable(False, False)
    tk.Label(root, text="Enable losses for this run", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=3, padx=12, pady=(10, 4)
    )
    tk.Label(root, text="Loss", anchor="w", width=20).grid(row=1, column=0, padx=8, sticky="w")
    tk.Label(root, text="Enabled", width=10).grid(row=1, column=1)
    tk.Label(root, text="Current weight", width=16).grid(row=1, column=2)
    variables = {}
    weight_variables = {}
    for row, (name, flag_attr, weight_attr, _) in enumerate(entries, start=2):
        enabled = bool(getattr(module, flag_attr)) if flag_attr else original_weights[name] > 0.0
        if name == "group_scalar" and args.scalar_consistency_mode == "local-edge":
            enabled = False
        variable = tk.BooleanVar(value=enabled)
        variables[name] = variable
        tk.Label(root, text=name, anchor="w", width=20).grid(row=row, column=0, padx=8, sticky="w")
        tk.Checkbutton(root, variable=variable).grid(row=row, column=1)
        weight_variable = tk.StringVar(value=f"{original_weights[name]:g}")
        weight_variables[name] = weight_variable
        tk.Entry(root, textvariable=weight_variable, width=16).grid(row=row, column=2, padx=8)

    def apply_and_close() -> None:
        for name, flag_attr, weight_attr, _ in entries:
            enabled = bool(variables[name].get())
            try:
                weight = float(weight_variables[name].get())
            except ValueError as exc:
                raise ValueError(f"Invalid weight for {name}: {weight_variables[name].get()!r}") from exc
            if weight < 0.0:
                raise ValueError(f"Weight for {name} cannot be negative.")
            if flag_attr:
                setattr(module, flag_attr, enabled)
            if not enabled:
                setattr(module, weight_attr, 0.0)
            else:
                setattr(module, weight_attr, weight)
            if name == "edge_scalar":
                module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS = (
                    enabled and args.scalar_consistency_mode == "local-edge"
                )
        root.destroy()

    def cancel() -> None:
        if messagebox.askyesno("Cancel", "Stop before training?"):
            root.destroy()
            raise SystemExit("Pipeline stopped from loss controls.")

    tk.Button(root, text="Apply and continue", command=apply_and_close).grid(
        row=len(entries) + 2, column=0, columnspan=3, pady=(8, 10)
    )
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()


def json_value(value):
    """Convert argparse values into JSON-safe data without losing paths."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def write_training_run_record(record_path: Path, args: argparse.Namespace, module, paths: dict[str, Path]) -> None:
    """Persist the resolved interactive loss state beside its output checkpoint."""
    loss_controls = (
        ("surface_direction", "ENABLE_SURFACE_DIRECTION_LOSS", "SURFACE_DIRECTION_WEIGHT"),
        ("support_free", "ENABLE_SUPPORT_FREE_SURFACE_LOSS", "SUPPORT_FREE_SURFACE_WEIGHT"),
        ("group_scalar", "ENABLE_SURFACE_GROUP_SCALAR_LOSS", "SURFACE_GROUP_SCALAR_WEIGHT"),
        ("edge_scalar", "ENABLE_SURFACE_EDGE_SCALAR_LOSS", "SURFACE_EDGE_SCALAR_WEIGHT"),
        ("base_direction", "ENABLE_BASE_DIRECTION_LOSS", "BASE_DIRECTION_WEIGHT"),
        ("base_scalar", "ENABLE_BASE_SCALAR_LOSS", "BASE_SCALAR_WEIGHT"),
        ("gradient_ratio", None, "GRADIENT_BAND_RATIO_WEIGHT"),
        ("gradient_floor", None, "GRADIENT_BAND_FLOOR_WEIGHT"),
        ("input_curvature", "ENABLE_INPUT_CURVATURE_LOSS", "INPUT_CURVATURE_WEIGHT"),
        ("build_slope", "ENABLE_BUILD_DIRECTION_SLOPE_LOSS", "BUILD_DIRECTION_SLOPE_WEIGHT"),
        ("collision", "ENABLE_COLLISION_LOSS", "COLLISION_WEIGHT"),
        ("tangent_twist", "ENABLE_TANGENT_TWIST_LOSS", "TANGENT_TWIST_WEIGHT"),
    )
    losses = {}
    for name, enabled_attr, weight_attr in loss_controls:
        weight = float(getattr(module, weight_attr))
        losses[name] = {
            "enabled": bool(getattr(module, enabled_attr)) if enabled_attr else weight > 0.0,
            "weight": weight,
        }
    provider = getattr(module, 'STRENGTH_LOSS_PROVIDER', None)
    losses['input_curvature'].update(mode=args.input_curvature_mode,
        radius_mm=args.input_curvature_radius_mm, sample_fraction=args.input_curvature_sample_fraction,
        boundary_coverage=args.input_curvature_boundary_coverage)
    losses['strength'] = dict(enabled=provider is not None, weight=args.strength_weight,
                              metadata=provider.metadata if provider is not None else None)
    for name, weight in [('gradient_variation', module.GRADIENT_VARIATION_WEIGHT),
                         ('small_gradient', module.SMALL_GRADIENT_WEIGHT)]:
        losses[name] = dict(enabled=weight > 0., weight=weight)
    record = {
        'regularization_profile': args.regularization_profile,
        "arguments": {key: json_value(value) for key, value in vars(args).items()},
        "checkpoint_input": str(module.CHECKPOINT_PATH),
        "checkpoint_output": str(module.OUTPUT_CHECKPOINT_PATH),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest"},
        "losses": losses,
        "orientation_search": getattr(module, 'GLOBAL_ORIENTATION', None).metadata if getattr(module, 'GLOBAL_ORIENTATION', None) is not None else {"enabled": False},
        "collision_normalization": {
            "enabled": args.normalize_collision_by_gradient,
            "gradient_epsilon": args.collision_gradient_epsilon,
            "denominator_detached": False,
            "coordinates": "normalized model coordinates",
        },
        "scalar_consistency": {
            "mode": args.scalar_consistency_mode,
            "edge_replaces_group": bool(module.SURFACE_EDGE_REPLACES_GROUP_SCALAR_LOSS),
            "start_epoch_zero_based": int(module.SURFACE_EDGE_SCALAR_START_EPOCH),
            "ramp_epochs": int(module.SURFACE_SCALAR_RAMP_EPOCHS),
            "divide_by_gradient": bool(module.SURFACE_SCALAR_DIVIDE_BY_GRADIENT),
            "gradient_denominator_detached": bool(module.SURFACE_SCALAR_GRADIENT_DENOMINATOR_DETACH),
        },
        "base_direction_parallel_only": bool(module.BASE_DIRECTION_PARALLEL_ONLY),
        "norm_control": {
            "balanced_batches": bool(module.ENABLE_BALANCED_NORM_BATCHES),
            "band_count": int(module.GRADIENT_BAND_COUNT),
            "max_squared_ratio": float(module.GRADIENT_BAND_MAX_SQUARED_RATIO),
            "squared_floor": float(module.GRADIENT_BAND_SQUARED_FLOOR),
        },
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="ascii")
    print(f"Saved resolved training configuration to {record_path}")


def install_fast_training_sampler(module, args: argparse.Namespace) -> None:
    """Use the existing Convex hard sampler for fast, representative epochs."""

    from experiments.convex_hard_collision.hard_collision_sampling import HardCollisionConfig, install

    install(
        module,
        config=HardCollisionConfig(
            input_points_per_epoch=args.input_points_per_epoch,
            collision_roots_per_step=args.collision_roots_per_step,
            mining_candidate_count=args.mining_candidate_count,
            refresh_every_epochs=args.mining_refresh_epochs,
            hard_pool_size=args.hard_pool_size,
            scoring_root_batch_size=args.scoring_root_batch_size,
            sdf_gate_batch_size=args.sdf_gate_batch_size,
            field_query_batch_size=args.field_query_batch_size,
            validation_count=args.collision_validation_count,
            fuse_regular_losses=not args.no_fuse_regular_losses,
        ),
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.boundary_face_normal_similarity_degrees <= 180.0:
        raise ValueError("--boundary-face-normal-similarity-degrees must lie in [0, 180].")
    if args.skip_sdf and not args.disable_collision:
        raise ValueError("--skip-sdf requires --disable-collision because collision checking needs the SDF gate.")
    original_mesh_path = args.mesh_path
    original_node_path = args.node_path
    original_ele_path = args.ele_path
    model_dir = args.run_root / safe_name(args.model_name)
    layer_suffix = "" if args.num_layers == 5 else f"_l{args.num_layers}"
    initializer_curvature_suffix = (
        f"_curv{frequency_name(args.initializer_curvature_weight)}"
        if args.initializer_curvature_weight > 0.0 else ""
    )
    paths = {
        "model_dir": model_dir,
        "source_dir": model_dir / "source",
        "archive_dir": model_dir / "archive",
        "sdf": model_dir / "sdf" / f"{safe_name(args.model_name)}_sdf.pt",
        "initializer": model_dir / "initializers" / (
            f"scalar_init_{safe_name(args.model_name)}_w{frequency_name(args.w0_initial)}_w{frequency_name(args.w0)}"
            f"{layer_suffix}{initializer_curvature_suffix}.pt"
        ),
        "manual_regions": model_dir / "constraints" / "manual_surface_regions.json",
        "constraint_metadata": model_dir / "constraints" / "surface_constraint_regions.json",
        "surface_constraint_cache": model_dir / "constraints" / "generated_surface_constraints.npz",
        "surface_constraint_cache_metadata": model_dir / "constraints" / "generated_surface_constraints.json",
        "output_checkpoint": model_dir / "checkpoints" / (
            f"parameters_{safe_name(args.model_name)}_{safe_name(args.checkpoint_tag)}.pt"
        ),
        "visualization_config": model_dir / "slicer_visualization_config.json",
        "manifest": model_dir / "pipeline_manifest.json",
    }
    paths["training_record"] = paths["output_checkpoint"].with_name(
        f"{paths['output_checkpoint'].stem}_run_config.json"
    )
    for path in paths.values():
        if isinstance(path, Path) and path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths["source_dir"].mkdir(parents=True, exist_ok=True)
    if args.start_checkpoint == "previous":
        candidates = [path for path in paths["output_checkpoint"].parent.glob("*.pt") if path != paths["output_checkpoint"]]
        # A default-tag run may have only its own prior output. In that case it
        # is still the intended checkpoint to resume from.
        if not candidates and paths["output_checkpoint"].is_file():
            candidates = [paths["output_checkpoint"]]
        if not candidates:
            raise FileNotFoundError(f"No previous checkpoint exists in {paths['output_checkpoint'].parent}.")
        args.start_checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)
        print(f"Resuming from previous checkpoint: {args.start_checkpoint}")
    # From this point onward every stage uses the copied files, not a mutable
    # external input path.
    args.mesh_path = copy_input_artifact(original_mesh_path, paths["source_dir"])
    if args.stress_orientation_record is not None:
        args.stress_orientation_record = copy_input_artifact(args.stress_orientation_record, paths['source_dir'])
    if args.tool_plane_config is not None:
        args.tool_plane_config = copy_input_artifact(args.tool_plane_config, paths['source_dir'])
    if args.tool_box_config is not None:
        args.tool_box_config = copy_input_artifact(args.tool_box_config, paths['source_dir'])
    if args.strength_file is not None:
        args.strength_file = copy_input_artifact(args.strength_file, paths['source_dir'])
    if original_node_path is not None:
        args.node_path = copy_input_artifact(original_node_path, paths["source_dir"])
        args.ele_path = copy_input_artifact(original_ele_path, paths["source_dir"])
    paths["manifest"].write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "mesh_path": str(args.mesh_path),
                "original_mesh_path": str(original_mesh_path),
                "node_path": None if args.node_path is None else str(args.node_path),
                "ele_path": None if args.ele_path is None else str(args.ele_path),
                "w0": args.w0,
                "w0_initial": args.w0_initial,
                "start_checkpoint": None if args.start_checkpoint is None else str(args.start_checkpoint),
                "up_direction": list(args.up_direction),
                "layer_thickness_control": {
                    "method": "shared_band_pointwise_interval",
                    "max_height_ratio": args.max_layer_height_ratio,
                    "band_count": args.gradient_band_count,
                    "balanced_sampling": args.balanced_norm_batches,
                    "reference_scan_batch_size": args.norm_pool_size,
                    "points_per_band_minimum": args.norm_points_per_band,
                },
                "paths": {name: str(path) for name, path in paths.items() if name != "manifest"},
            },
            indent=2,
        ),
        encoding="ascii",
    )

    if args.skip_sdf:
        print("Skipping SDF fitting and verification because collision loss is disabled.")
    else:
        sdf_exists = paths["sdf"].is_file()
        if not sdf_exists:
            if not ask_to_continue(f"SDF checkpoint is missing. Create {paths['sdf']}?", args, creation=True):
                raise SystemExit("Pipeline stopped before SDF training.")
            train_sdf(args.mesh_path, paths["sdf"], args)
            release_transient_cuda_memory()
        else:
            print(f"Reusing SDF checkpoint: {paths['sdf']}")
        if not args.skip_sdf_verification:
            visualize_sdf(args.mesh_path, paths["sdf"], args)
            release_transient_cuda_memory()
            if sdf_exists and ask_to_retrain("the displayed SDF", args):
                archive_artifact(paths["sdf"], paths["archive_dir"] / "sdf")
                train_sdf(args.mesh_path, paths["sdf"], args)
                release_transient_cuda_memory()
                visualize_sdf(args.mesh_path, paths["sdf"], args)
                release_transient_cuda_memory()
            if not ask_to_continue("SDF verification closed. Continue?", args):
                raise SystemExit("Pipeline stopped after SDF verification.")

    initializer_exists = paths["initializer"].is_file()
    initializer_rebuilt = False
    if initializer_exists and args.retrain_initializer:
        print(f"Rebuilding scalar initializer: {paths['initializer']}")
        archive_artifact(paths["initializer"], paths["archive_dir"] / "initializers")
        build_initializer(paths["initializer"], paths["initializer"].parent, args)
        initializer_rebuilt = True
    elif initializer_exists:
        incompatibility = initializer_compatibility_issue(paths["initializer"], args)
        if incompatibility is not None:
            print(f"Initializer is incompatible: {incompatibility}.")
            if not ask_to_retrain("the incompatible scalar initializer", args):
                raise SystemExit("Pipeline stopped because the initializer is incompatible.")
            archive_artifact(paths["initializer"], paths["archive_dir"] / "initializers")
            build_initializer(paths["initializer"], paths["initializer"].parent, args)
            initializer_exists = True
            initializer_rebuilt = True
    if not initializer_exists:
        if not ask_to_continue(f"Initializer is missing. Create {paths['initializer']}?", args, creation=True):
            raise SystemExit("Pipeline stopped before scalar initialization.")
        build_initializer(paths["initializer"], paths["initializer"].parent, args)
    elif not initializer_rebuilt:
        print(f"Reusing scalar initializer: {paths['initializer']}")
    if not args.skip_initializer_verification:
        visualize_initializer(paths["initializer"], args)
        if initializer_exists and not initializer_rebuilt and ask_to_retrain("the displayed scalar initializer", args):
            archive_artifact(paths["initializer"], paths["archive_dir"] / "initializers")
            build_initializer(paths["initializer"], paths["initializer"].parent, args)
            visualize_initializer(paths["initializer"], args)
        if not ask_to_continue("Initializer-layer visualization closed. Continue to surface selection/training?", args):
            raise SystemExit("Pipeline stopped after initializer verification.")

    if args.no_train:
        print(f"Preparation complete. Artifacts are in {model_dir}")
        return

    if args.reuse_saved_preserved_regions:
        manual_regions_exist = has_nonempty_manual_regions(paths["manual_regions"])
        metadata_exists = load_saved_automatic_selection(
            paths["constraint_metadata"], args.mesh_path, args
        )
        if not manual_regions_exist and not metadata_exists:
            raise FileNotFoundError(
                "--reuse-saved-preserved-regions requested, but no accepted manual or automatic "
                f"regions were found in {paths['manual_regions']} or {paths['constraint_metadata']}"
            )
        args.select_surface_interactively = False
        # Do not reopen the automatic slider window after its saved settings
        # have been restored above.
        args.skip_automatic_selection_window = True
        if manual_regions_exist:
            print(f"Reusing saved manual preserved regions: {paths['manual_regions']}")
    elif args.select_surface_interactively:
        print("Opening a fresh preserved-region picker; its accepted regions will replace the saved file.")
    elif args.skip_surface_selection:
        args.select_surface_interactively = False
    elif has_nonempty_manual_regions(paths["manual_regions"]):
        args.select_surface_interactively = ask_to_continue(
            "Open a fresh surface-selection picker? Saving it will replace the existing manual regions.", args
        )
    else:
        args.select_surface_interactively = ask_to_continue(
            "Open the interactive surface-selection picker before training?", args
        )

    module = load_banded_template()
    configure_training_module(module, paths, args)
    if TRAINING_MODULE_CONFIGURATOR is not None:
        TRAINING_MODULE_CONFIGURATOR(module, paths, args)
    if args.interactive_loss_controls:
        interactive_loss_controls(module, args)
        print(
            "Loss controls applied: "
            f"gradient_ratio_weight={module.GRADIENT_BAND_RATIO_WEIGHT:.3e} "
            f"gradient_floor_weight={module.GRADIENT_BAND_FLOOR_WEIGHT:.3e}"
        )
    write_training_run_record(paths["training_record"], args, module, paths)
    install_fast_training_sampler(module, args)
    print(f"Starting banded surface-quality training. Artifacts: {model_dir}")
    module.main()


if __name__ == "__main__":
    main()

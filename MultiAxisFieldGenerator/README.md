# Multi-axis field generator

Portable configurable snapshot of the five-axis cone/cylinder collision
workflow, with support-free, stress-tangent and combined objectives. The
original Fertility experiment is unchanged. Python and dependencies are
required; this is a source package, not a Windows executable.

## Start a model job

Copy `example_job.json`, edit paths and tool dimensions, and use the same
Python/CUDA environment as the experiment. Commands below run from the parent
directory containing `MultiAxisFieldGenerator/`:

```powershell
python -m MultiAxisFieldGenerator MultiAxisFieldGenerator/example_job.json --dry-run
python -m MultiAxisFieldGenerator MultiAxisFieldGenerator/example_job.json
```

All input paths and `output` resolve relative to the job JSON. `--output`
overrides the run directory relative to the current working directory.
Existing runs are protected. A dry run checks paths and supported option
names and prints the resolved command; checkpoint compatibility and numerical
settings are checked by the runner. Extra runner flags may be appended.

Model inputs:

- Closed triangular surface mesh, in millimetres.
- Compatible trained SIREN SDF checkpoint (negative inside), **only when
  collision training is enabled**.
- Compatible initial layer-field checkpoint, supplied or built below.
- Explicit enlarged training-tool and nominal checking-tool presets for
  collision-enabled jobs and analytical checking.

The package includes SDF and initializer builders; either build those inputs
or reuse compatible checkpoints.
Self-describing checkpoints include `model_state_dict`, `field_spec` with
`width`, `layers`, `frequency`, and preferably normalization metadata. For
legacy checkpoints without `field_spec`, specify the field/SDF width, depth
and frequency in `model`; defaults match the Fertility example. Only the
bundled SIREN state-dict layout is supported. Arbitrary network architectures
are not automatically converted.

Normalization is the surface bounding-box center and half its maximum extent.
Input checkpoint normalization, when present, must match it. Legacy files
without metadata must have been trained in these coordinates; their matching
coordinates cannot be inferred from weights alone. `up_axis` can be x, y or z
(positive direction); orient the model before training if another build
direction is needed. `base_band_mm` selects the low band on that axis.

By default, this base band is excluded from the support-angle objective.
For an experiment with no prescribed platform base, set `training.base_weight`
and `training.anchor_weight` to zero and `training.support_on_base` to true
(`--support-on-base` on the runner CLI). This includes the complete boundary
in both support-angle sampling and the inward-surface orientation reference.
Setting the two weights to zero alone retains the default base-band exemption.

## Build the prerequisites

With a mesh path and desired checkpoint paths in your job, build each input:

```powershell
python -m MultiAxisFieldGenerator my_job.json --action build-initializer
python -m MultiAxisFieldGenerator my_job.json --action build-sdf
python -m MultiAxisFieldGenerator my_job.json
```

Builder actions permit missing target checkpoints and do not require the other
checkpoint to exist. If paths are omitted, outputs default to
`output/inputs/initializer.pt` and (when needed) `output/inputs/sdf.pt`.
Already-existing checkpoint targets are protected. These are portable builder
implementations using the same bundled SIREN layout, not copies of the old
scripts with machine-specific paths.

Optional top-level builder settings:

```json
"builders": {
  "initializer": {"steps": 2000, "samples": 32768, "batch": 2048},
  "sdf": {"steps": 5000, "samples": 65536, "batch": 2048, "gradient_weight": 0.1}
}
```

Width, depth and frequency default to the corresponding model field/SDF
settings. The initializer fits build height above the mesh minimum (divided
by normalization scale), with input gradients encouraged toward the chosen
up axis. The SDF fits exact mesh signed-distance labels, normalized by scale,
plus optional Eikonal regularization. Its training pool mixes bounding-box,
surface and near-surface samples. A closed outward-oriented manifold mesh is
required. Builder checkpoints embed architecture, normalization and source
hashes; a sibling `.build.json` records sampled errors and fit history.
Inspect SDF sign accuracy and error metrics before collision training; step
counts are settings, not accuracy guarantees.

Direct script equivalents, without a job file:

```powershell
python MultiAxisFieldGenerator/build_initializer.py --mesh part.obj --output initializer.pt --up-axis y
python MultiAxisFieldGenerator/build_sdf.py --mesh part.obj --output sdf.pt
```

## Training without collision

Start with `example_no_collision_job.json`: it needs only the mesh path,
then run `--action build-initializer` followed by normal training.

Set `training.collision_weight` to 0 and omit the collision schedule (or make
every active stage zero). Then **no SDF is required or loaded**, even if the
job retains an unused missing SDF path. Collision benchmarks and SDF checks
are skipped. Intermediate collision metrics are null with `gate: not_checked`,
not a false zero-collision result. Ordinary support/stress/base/regularization
training continues. Tool presets can be omitted for this training-only job.

A schedule that activates collision later still requires an SDF at startup;
a positive stage beyond the requested run length does not. Merely setting the
base weight to zero does not override a positive scheduled stage.

The optional final `exact_validation` uses the surface mesh directly and
does not require an SDF. The nominal analytical checker also never requires
one. Viewer `--gate auto` (default) selects mesh inclusion when no active SDF
is available; `--gate mesh` makes that choice explicit.

The repository-local `examples/fertility_job.json` references the existing
experiment assets, so it can be dry-run here without duplicating them:

```powershell
python -m MultiAxisFieldGenerator MultiAxisFieldGenerator/examples/fertility_job.json --dry-run
```

Those assets are not included in the portable ZIP. Update paths when moving
that example to another machine.

## Optional tetrahedra

Set `model.volume_mesh` to a matching `.ele`/`.node` pair or linear-tet `.vtu`.
If omitted, a sibling `.ele` is used when present. Tetrahedra support efficient
uniform volume samples and guaranteed-inside near-face samples.

Without tetrahedra, coverage mode uses verified surface insets and exact
mesh-membership rejection sampling of the bounding box. Closed manifold
triangles with consistent outward orientation are required. Thin models can
make rejection inefficient; the sampler stops with a clear error rather than
looping indefinitely. Tetrahedra are preferable for such models. Analytical
validation also has a surface-only sampling fallback; its deterministic
seeds are surface vertices/centroids rather than tetrahedral seeds. Therefore
reports from different backends do not have identical coverage.

## Tool dimensions

`tools/abb_training.json` retains the enlarged training enclosure. Its editable
fields specify cone angle/distances, inner/outer radii, axial sample levels,
cap sampling and outer-height jitter. Fresh interior points are added to the
cone and both cylinders during training.

`tools/abb_nominal.json` defines finite analytical cone/cylinder inclusion for
checking: half-angle, cone radial cap, radii and axial intervals, all in mm.
The included preset retains the experiment's assumed axial bounds. It is
**not measured CAD and not identical to the legacy unbounded checker**.
Fit both presets to the actual tool. They are explicit separate inputs;
clearances are not automatically subtracted or inferred.

The supported family is an axisymmetric cone tip and two coaxial cylinders.
This is not arbitrary tool/robot CAD, box tooling, robot-link checking or a
kinematic planner. Axial symmetry makes roll irrelevant.

## Training configuration

`training` maps to runner options with underscores in JSON. It exposes batch
sizes, collision roots, mining intervals, query chunks, tool-volume samples,
learning rate, loss weights, curvature radius, base band, device, schedules
and epoch mode. Defaults retain the coverage sampler and 256 roots/step.
The finite coverage stream, fresh part/tool samples and automatically retained
failing roots discover collisions without an external analytical report.

The example uses collision weights 10k, 20k, 30k and 50k beginning at epochs
1, 26, 51 and 76. `validate_every: 100` performs initial/final monitoring for
100 epochs. Validation does not update parameters. `exact_validation` adds
the mesh-gated dense sampled-tool check at the end. The separate analytical
checker below tests nominal-volume inclusion and is independent of training.

Regularization includes curvature, spatial variation of gradient norm,
small-gradient suppression, base alignment and scalar anchoring. Hessians
are restricted to the interior regularization batch. Support, base,
collision-root and stress first-derivative queries are grouped. Inspection
uses first derivatives without higher derivative graphs. Placed tool-point
queries preserve orientation derivatives during training.

Collision training screens all active tool samples without a derivative graph,
then builds tool-query graphs only where the squared ReLU penalty can be
nonzero, including a floating-point guard around its boundary. Sample counts,
component top-k reductions, and orientation derivatives of penalized queries
are retained. `disable_collision_pruning: true` selects full backpropagation for
comparison. `field_pairs` counts unique queried pairs; `field_evaluations` also
counts scalar screening/re-evaluation; `gradient_pairs` counts queries with a
derivative graph. Mining, validation, and sampled SDF membership are unchanged.

Hessian components are evaluated together using batched gradient vector
products, including the higher derivatives needed by curvature training.
Set `separate_hessian_passes: true` to use the three-call reference.

Zero-weight support and base terms skip their network evaluations during
training. Surface queries return when the late orientation constraint needs
them, and random sampling draws are retained. Disabled training terms are
logged as zero; validation still reports surface support failures.

Training records include `mining_seconds`, `epoch_wall_seconds` (mining and
optimization, before periodic validation), and mean active/gradient collision
pair counts. The legacy `seconds` field measures the optimization loop.
Query chunk sizes control memory/throughput independently of collision pose
count. Reducing poses also reduces collision coverage per epoch.

## Reuse and inspect a fitted SDF

The application exposes Collision SDF source = Load saved and a saved .pt
checkpoint. The worker reuses an existing SDF after checking its kind, source
mesh fingerprint, coordinate normalization, and finite weights. SDF geometry
is independent of the layer-field initializer and its build-axis label.
Changed geometry or coordinates require a new SDF. Optional Save built SDF as
creates a protected reusable copy. Compatible existing job SDFs are also reused.

Action = Build / validate SDF prepares only the SDF and its validation view;
it skips layer-field initialization and optimization. The standalone tester
exposes the same action. Validation uses independent points (default seed
2026), compares against mesh signed distances, reports wrong inside/outside
classifications by distance to the surface, and exports a zero-surface overlay,
three error slices, wrong-sign points, a preview image, and a JSON report.
The preview grid only affects the displayed zero surface; metrics use direct
network evaluations at independent samples. A good overall sign score can
hide poor classification close to the surface; inspect the distance bins.

The portable CLI also supports --action validate-sdf. To reopen an existing
validation view without inference, use runtime/validate_sdf.py --view FOLDER
--interactive. This validates the SDF approximation, independently of the
nominal-tool collision checker.

## Stress mode and resampling

The interface and standalone tester expose **Align top stress (%)**. Set
Stress region to tensile and this percentage to 30 to align the highest
30% of tensile FEA cells after the absolute-threshold filter. The default
100 keeps the full selected region. In a portable job, use
`stress.top_percent: 30`; the runner flag is `--stress-top-percent 30`.
Ranking uses absolute signed cell stress and equal cell counts, rather than
cell volume or a fraction of the maximum stress value. All cutoff ties are
included. The resolved cutoff and eligible/selected cell counts are saved
in stress_metadata.selection in run/config.json.

The same region is used for training targets, held-out stress validation,
new stress-informed planar initialization, and exported alignment masks. FPS
still runs once on the original training pool before this region filter.
Other material retains the support, collision and regularization objectives.
Existing saved initializers retain their previous orientation.

Start with `example_stress_job.json`. Objectives:

- `support_free`: support, collision and geometric regularization.
- `stress`: stress, collision and geometric regularization; support weight zero.
- `combined`: stress plus support, collision and geometric regularization.

The stress file is whitespace-delimited numeric text. Point/direction columns
are explicit zero-based triples; defaults use columns 0-2 for world-mm
positions and 3-5 for one principal direction. Other columns are ignored:
they are **not assumed to be magnitudes or tensor components**. Invalid or
zero directions are filtered, and source points outside the model are removed
with counts reported. `coordinate_space: model` converts normalized positions
back to world mm; directions are normalized in either case. Arbitrary input
coordinate rotations must be applied consistently before import.

A deterministic 20% of source rows is held out. The application interface and
standalone tester default to `stress.sampling: fps`: farthest-point sampling
runs once at setup, targeting the interior training-point count, capped by the
available training sources, then applies the selected stress-region filter.
It retains original stress directions without interpolation. With
`training.epoch_mode: full-pass`, the fixed interior and stress pools are shuffled
and divided into batches each epoch, including final partial batches. The
longest stream sets epoch length; shorter streams cycle if batch counts differ.
No FPS selection is repeated during training.

The optional `stress.sampling: resampled` mode (the raw CLI default) draws fresh
uniform interior query points each epoch. A CPU KD-tree finds training-source neighbors. The
interpolator averages direction projectors `d*d.T` with inverse-distance
weights, avoiding cancellation between `d` and `-d`. The loss is
`mean(n.T @ interpolated_projector @ n)` where `n` is the layer normal.
Thus principal directions are encouraged to lie **tangent to layers**. It
does not optimize a second in-layer toolpath field, predict strength, or
interpolate a full stress tensor. Those are separate extensions.

In resampled mode, `stress.samples` sets the requested count per epoch, `batch` its
minibatch size, `neighbors` the interpolation neighborhood, and `weight` the
objective multiplier. `max_distance_mm` rejects unsupported queries; set it
for the source sampling density. Omitting it permits nearest-neighbor
extrapolation over the whole model. Partial accepted pools are fully visited;
all rejected queries cause a clear failure. Accepted/rejected counts and
maximum source distance are logged each epoch. Held-out source-direction
alignment is reported at validation; it is not a mechanical strength test.

## Check and visualize saved runs

```powershell
python -m MultiAxisFieldGenerator my_job.json --action check --seed 2026
python -m MultiAxisFieldGenerator my_job.json --action view-collisions
python -m MultiAxisFieldGenerator my_job.json --action view --compare --slicing-mode gradient --layer-height-mm 1
```

Use `--action check --max-roots 2000` for a quick subset. Checker reports are
protected; append `--output` directly to the runtime checker for a custom
report directory (the wrapper's `--output` always selects the run directory).
The viewer shows initial/final fields, physical gradient-based slice previews,
support diagnostics and sampled tool collisions. Surface-only previews use a
clipped grid (`--viz-grid-mm`, default 2 mm); tet previews use the supplied
volume mesh. Target layer spacing is a first-order approximation, not uniform
thickness or production toolpaths. `view-collisions` shows all failing roots
from the saved nominal report. Add `--off-screen` to either viewer for PNGs.

Runs save resolved input settings, hashes, normalization, network specs,
`metrics.jsonl`, `latest.pt`, held-out roots, and optional analytical reports
and viewer exports. Source models/checkpoints remain external. No analytical
witness replay is used unless explicitly requested through the optional
`replay_results` runner setting.

## Package and tests

```powershell
python -m unittest discover -s MultiAxisFieldGenerator/tests -v
python MultiAxisFieldGenerator/pack.py --output artifacts/MultiAxisFieldGenerator.zip
```

The ZIP contains source, configuration examples, presets, tests and a SHA-256
manifest. It excludes models, checkpoints, caches, run results and Python/CUDA.
Install dependencies from `requirements.txt` in a compatible environment.
Synthetic integration tests exercise tet support training, surface-only
combined stress training, saved-model checking, and sign-invariant stress
interpolation. They do not establish convergence on an engineering stress
dataset. No full manufacturing/stress optimization run was launched while
packaging this version.

Finite point checks are not continuous collision certificates. Confirm tool
dimensions, input coordinate conventions and resulting geometry for each job;
robot motion between poses and fixtures are outside this package's scope.

## Optional collision normalization

Set the following options in `training`:

```json
"normalize_collision_by_gradient": true,
"collision_gradient_epsilon": 1e-6
```

The default is `false` for compatibility. With this enabled, each positive
scalar-order violation is divided by its root gradient norm before squaring
and top-k reduction. The denominator stays differentiable and uses a positive
numerical floor. With zero scalar margin and norms above the floor, positive
rescaling of the scalar field leaves the loss unchanged. Keep the norm and
layer-height regularizers enabled; this is a local distance approximation.
The gradient uses normalized model coordinates, so the loss is not in mm².
Collision weights may need recalibration. Five-axis `margin` remains a scalar
margin, so a fixed nonzero margin breaks exact field-scale invariance.

Hard-root scoring and sampled diagnostics use the same setting. The optional
five-axis witness replay does too. Analytical point-in-tool collision tests
remain geometric and unchanged. No full real-part optimization was run for
this update.

Backend source: [MAP](https://github.com/neelotpal-d/MAP). Third-party notices remain applicable.

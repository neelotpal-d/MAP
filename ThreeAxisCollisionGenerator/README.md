# Three-Axis Collision Generator

Portable fixed-axis neural-field training for `strength`, `support-free`,
`surface-quality`, and `combined` objectives. The folder includes its runtime,
SDF/initializer builders, stress importer, analytical collision checker and
initial/final layer viewer. Run it from the directory containing the package.

## Updated strength configuration (24 October 2026)

Copy `example_strength_job.json` and edit its mesh, stress and output paths.
Keep the job beside the supplied tool JSONs, or update `tool_plane_config`
when moving it. It contains the latest tested batch-500 setup. `example_strength_batch256_job.json`
contains the corresponding batch-256 alternative. Input coordinates are mm.

| Setting | Strength preset |
|---|---|
| Network | 128 wide, 10 hidden layers |
| Frequency | `w0: "auto"`, maximum world-space mesh extent / 20; first layer follows |
| Stress loss | 1; original FPS source directions, sign invariant |
| Curvature | 0.1, geometric radius 35 mm |
| Curvature coverage | Every retained interior point and every boundary vertex per epoch |
| Banded gradient-norm ratio / floor | 10 / 1; spacing-ratio target 2 is soft |
| Base-direction / base-scalar alignment | Disabled in this strength preset |
| Collision | 1e6 from epoch 1; attached gradient normalization, epsilon 1e-6 |
| Collision sampling | 256 roots per step, top 30% tool-point violations |
| Optimization | 50 epochs, Adam 1e-5, no warm-up |

Numeric frequencies remain supported. `auto` is resolved by the public JSON
wrapper on the mesh actually supplied to training. In the separate stress-
orientation wrapper it is resolved again after the mesh has been rotated.
The preset uses the banded norm losses; the additional five-axis norm-variation
and small-gradient losses are disabled.

```powershell
python -m ThreeAxisCollisionGenerator my_strength_job.json --dry-run
python -m ThreeAxisCollisionGenerator my_strength_job.json
```

Paths in JSON resolve relative to that JSON. `output/model_name/` contains the
run. The examples have placeholder input paths: supply real mesh/stress files
before running even a dry-run. `--dry-run` prints the resolved runtime command.

The full-coverage examples set both `max_input_points` and
`input_points_per_epoch` to 150000. Every point retained after that input cap
is visited each epoch, with shuffled order and the final partial batch retained.
If raising the cap, raise the epoch budget too to preserve complete coverage.
Boundary curvature points are partitioned across the steps. Stress batches
shuffle independently and visit every retained FPS source; a shorter stress
stream cycles until the interior epoch finishes. Full coverage does not imply
continuous-volume coverage beyond the finite input pool.

Changing batch size changes updates per epoch. On Tshape, 256 gives 47 steps
and 500 gives 25; both cover the same point pools. Collision-root count is a
separate setting, so these runs draw 12032 versus 6400 roots per epoch.

## Tool configuration and collision checking

The tested strength preset uses one underside plane, fixed along machine +Y:

- X and Z each span -25 to +25 mm.
- Training plane Y is 7.6 mm: `tool_y8_xz25_training_margin04.json`.
- Nominal clearance is 8 mm, with a 0.4 mm training offset.
- Tool sampling uses 20 x 22 points plus the existing perturbation.

`tool_y8_xz25_check.json` describes the nominal rectangular body Y=8..100 mm
for the analytical point-in-tool postcheck. It has the same X/Z bounds and a
1 mm point grid. Change these files to match another tool. The trainer samples
the underside; the checker tests the nominal body independently of the SDF.
The supplied dimensions are an example for the tested tool, not a universal tool.

```powershell
python -m ThreeAxisCollisionGenerator --run-dir model_runs/part_strength --collision-check --tool-config ThreeAxisCollisionGenerator/tool_y8_xz25_check.json --device cuda
```

With `--run-dir`, tool paths resolve from the current directory. Job-based
checking also works; tool paths then resolve from the job's directory. A finite
point-grid check reports detected collisions; it does not certify the continuous
geometry. Configure the body height to cover the actual tool.

With `pipeline.collision.disable_collision: true`, the wrapper skips SDF fitting
and loading automatically. The scalar initializer still runs. Both builders
remain bundled for collision-enabled jobs. Gradient normalization is optional
for legacy jobs (runtime default false); the new strength presets enable it.
It divides positive scalar-order violations by the attached root gradient norm
before squaring and top-k reduction. Norms use normalized model coordinates,
so collision loss is not a squared world-mm distance.

## Stress input and FPS order

Use whitespace-delimited numeric data in the mesh's world coordinates:

```text
x y z principal_dir_x principal_dir_y principal_dir_z signed_selected_principal_stress
```

Column mappings are zero based and explicit in `strength`. The signed value
must correspond to the supplied principal direction. Legacy nine-column files
can mean different things; verify their schema.

The new preset uses `sampling: "fps"`, `samples: 0`, `fps_filter_order: "after"`:
finite/inside filtering and heldout split happen first, then FPS requests the
same number as the interior input pool, then the tensile sign filter removes
nonpositive selected sources. Removed sources are not replenished, so the final
stress pool can be smaller than the interior pool. Source directions and source
IDs are preserved. FPS mode performs no projection/interpolation onto interior
points. The loss is mean((normalized_gradient dot unit_stress_direction)^2),
which places the stress direction tangent to the layer and ignores vector sign.

For tensor-valued tetrahedral VTU input, the bundled importer selects the
largest-absolute principal eigenvalue, retains its sign, and exports the paired
direction. Positive values are tensile; an exact absolute tie prefers the larger
algebraic value. Geometry includes the complete tetrahedral solid.

```powershell
python -m ThreeAxisCollisionGenerator.import_fea part.vtu --output imported_part
```

Use `imported_part/mesh.obj` (and optional `.node`/`.ele`) plus
`imported_part/stress_all_cells.txt` for training. That full stress file lets the
trainer do FPS first and tensile filtering second. The importer's separate
`stress_tensile_fps.txt` is a smaller, tensile-only orientation aid; using it as
the training source would change the sampling protocol. `largest-algebraic`
selection remains an explicit legacy importer option.

Legacy `sampling: "interior"` remains available. It queries inverse-distance
averaged direction projectors using `neighbors` and `max_distance_mm`; compressed
sources retain zero projector weight. This is a separate interpolation mode,
not the default in the updated strength example. Directions alone require
`tensile_only: false`. Alignment is a layer objective, not an in-layer toolpath
field or mechanical stiffness calculation.

## View initial and final layers

```powershell
python -m ThreeAxisCollisionGenerator --run-dir model_runs/part_strength --view --device cuda --layer-height 2
python -m ThreeAxisCollisionGenerator --run-dir model_runs/part_strength --view --device cpu --off-screen --screenshot layers.png
```

The linked interactive views include a visible-layer slider. Display layers
advance using the field gradient norm. Their requested spacing is approximate;
this viewer does not generate validated printable toolpaths. The saved run
provides input copies and field metadata; original job files are unnecessary.
Saved internal paths are absolute, so relocation of existing runs needs rebasing.

## Other workflows and compatibility

| Workflow | Surface quality | Support-free | Stress alignment |
|---|---|---|---|
| `surface-quality` (legacy default) | yes | no | no |
| `support-free` | no | yes | no |
| `strength` | no | no | yes |
| `combined` | yes | yes | yes |

`example_job.json`, `example_support_free_job.json` and
`example_combined_job.json` retain their existing task-specific loss choices.
The updated strength preset explicitly disables base alignment. Other jobs can
set `surface_quality.base_direction_weight: 0` to disable its field evaluations,
or retain a positive weight when a base constraint is desired. Geometric
curvature, full boundary coverage and norm-weight overrides are available to all
fused workflows. FPS requires the fused trainer; legacy interior interpolation
also supports the separate loss loop.

The public package always uses a fixed tool axis. The optional stress-selected
planar pose remains in the separate `ThreeAxisStressOrientation` wrapper; its
strength examples now match these settings. Quaternion pose learning remains a
separate prototype. This folder executes without either sibling package.

## Outputs, checks and provenance

Each run stores copied inputs, SDF/initializer as applicable, the final field,
network and loss configuration, and a pipeline manifest. Strength mode records
`strength_config.json`, `strength_metrics.jsonl` and `strength_fps_sources.npz`.
Full geometric curvature records `curvature_coverage.jsonl`. Source hashes and
original FPS row IDs support later audits. Gradient/SIREN metadata accompany
saved fields, so their actual frequencies/depth are used when reloading.

The source Tshape trials used the stress-selected pose and 12007 interior,
8788 boundary and 10165 retained tensile FPS points. Both batch sizes detected
0 collisions on the 78821-point nominal 1 mm check. At batch256/500, heldout
stress MSE was 0.04608/0.04528 versus planar 0.05896; P95 curvature was
0.03687/0.03438 per mm. Maximum curvature was 0.04982/0.06478. These are specific
trial results, not performance or stiffness guarantees for another model.

```powershell
python -m unittest discover -s ThreeAxisCollisionGenerator/tests -v
```

Backend source: [MAP](https://github.com/neelotpal-d/MAP). Third-party notices remain applicable.

# Standalone Bambu Layer-Path Slicer

This bundle assembles already-generated layer-path G-code snippets into Bambu
`.gcode.3mf` jobs. It is not a mesh slicer: an external path generator must
produce the layer snippets first.

It contains two independent front ends:

- `h2d_right_nozzle/`: H2D, single material on the physical right nozzle.
- `h2d_dual_nozzle/`: H2D, two materials across the physical left and right nozzles.
- `a1_single_nozzle/`: A1, single material on `T0`.

Both front ends use `common/` for the assembly, motion-test, 3MF packaging,
and archive-packing code. The bundle uses only the Python standard library.

## Inputs

Place ordered snippets in the relevant `inputPaths/` folder.

```text
0_0.gcode
0_1.gcode
1_0.gcode
```

The first number is the main layer index; the second is the sub-path index.
H2D accepts optional `_0` material suffixes but this bundled right-nozzle
profile accepts only material `0`. A1 accepts either no material suffix or
`_0` only.

The source paths must already use the correct printer plate coordinates. The
H2D profile expects its path data near plate centre `X175, Y160`; A1 expects
coordinates on its `0..256` plate.

## H2D Right Nozzle

Put the matching OBJ at `h2d_right_nozzle/models/model.obj`, add snippets to
`h2d_right_nozzle/inputPaths/`, then run:

```powershell
cd h2d_right_nozzle
python run_pipeline_motion_test.py --config pipeline_config.json
```

The packaged result is `GeneratedH2DRightNozzleJob_motion_test.gcode.3mf`.
The profile maps material `0` to logical `T1`, which is the physical right
nozzle in this setup. It uses the included `right_white` Bambu template.

## A1 Single Nozzle

Put snippets in `a1_single_nozzle/inputPaths/`, then run:

```powershell
cd a1_single_nozzle
python run_pipeline_motion_test.py --config pipeline_config.json
```

The packaged result is `GeneratedA1Job.gcode.3mf`. The included A1 reference
archive supplies the original Bambu startup and shutdown G-code at generation
time.

## H2D Dual Nozzle

Place a matching OBJ at `h2d_dual_nozzle/models/model.obj`. Use explicit
material-suffixed snippets in `h2d_dual_nozzle/inputPaths/`:

```text
0_0_0.gcode   # material 0: logical T0, physical left nozzle
0_1_1.gcode   # material 1: logical T1, physical right nozzle
1_0_0.gcode
```

Run:

```powershell
cd h2d_dual_nozzle
python run_pipeline_motion_test.py --config pipeline_config.json
```

The bundled template/profile is `black_left_white_right`: material 0 is black
on the left nozzle and material 1 is white on the right nozzle. The result is
`GeneratedH2DDualNozzleJob_motion_test.gcode.3mf`.

## Normal Versus Motion-Test Runs

`run_pipeline.py` uses the `assemble` configuration as written.
`run_pipeline_motion_test.py` applies the `motion_test` values, including
role-specific feedrate and acceleration overrides. Use the motion-test entry
point for the profiles that have been exercised most recently.

Use `--assemble-only` on either front end when only the assembled G-code and
summary JSON are needed.

## Safety

These jobs contain custom machine motion and nonplanar paths. Validate a new
path set with a small, low-risk print first. Check the selected physical
nozzle, material, temperatures, machine calibration, plate coordinate system,
and first-layer clearance before sending a production job.

Backend source: [MAP](https://github.com/neelotpal-d/MAP). Third-party notices remain applicable.

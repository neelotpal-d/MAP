# External Projected Pipeline 2.0

Standalone snapshot of the current projected toolpath stack. Copy this folder
or extract `external_projected_pipeline.zip` beside your own project. The older
`external_surface_pipeline` bundle remains separate.

Two pipelines share the same CLI and Python API:

- `projected`: general model layer OBJ files; projected hybrid/sparse/raster paths
  with nearest-triangle heights.
- `support-model`: compatible upstream model/support exports, one material,
  support-before-model schedule, platform/contact roles and contact-gap correction.

## Use Without Installing

Run from the folder containing `external_projected_pipeline`:

```powershell
python -m external_projected_pipeline --help
python -m external_projected_pipeline doctor
```

Or call `python C:\path\external_projected_pipeline\pipeline_cli.py ...` from
any directory. For the integrated interface, use the combined environment described
in [Windows CUDA setup](../WINDOWS_CUDA_SETUP.md). Standalone use can select any
Python environment containing the required dependencies.
Nothing installs dependencies automatically. Required packages are listed in
`requirements.txt`; `tested_dependencies.json` records the build environment.
PyMeshLab is currently required by the eagerly imported stage dispatcher even
when remeshing is disabled. PyVista splits disconnected layer patches and supports preview; potpourri3d is for
direct intrinsic/heat helpers. A libigl build with batched `AABB` ray methods is
needed for contact model-normal heights.

## Generate and Run a General Config

```powershell
python -m external_projected_pipeline init-config --pipeline projected --config jobs\model.json --input-surfaces model_layers --output-root jobs\model_output --model-pattern sparse --density 0.25 --layer-workers 2 --clean-output-dirs
python -m external_projected_pipeline run jobs\model.json
```

`--model-pattern` accepts `hybrid`, `sparse`, `contours`, `raster`. All parameters
remain editable in JSON. Repeat `--set KEY=JSON` to set any nested option while
generating, for example `--set hybrid_contours.hatch_angle_degrees=45` with
`--axis-mode fixed_angle`. Relative CLI inputs resolve from the calling folder;
generated paths are absolute. Relative paths manually written in JSON resolve
from the config location. Keep outputs outside the package and input folders.

## Model and Same-Material Supports

```powershell
python -m external_projected_pipeline init-config --pipeline support-model --config jobs\support.json --export-root compatible_export --output-root jobs\support_output --model-pattern hybrid --layer-workers 2 --clean-output-dirs
python -m external_projected_pipeline run jobs\support.json
```

The export root contains `model_layers/`, `support_layers/`, their two manifests,
and `compatible_export.json`. Duplicate filenames are reassigned combined
sub-indices; main indices, scalar order and original identities are preserved.
Default support settings: two regular contours, solid platform hybrid, sparse
contact roofs, one fewer contour on filled contact patches and 0.05 mm wall
overlap. Small patches retain contours. Contact gap is the current user-selected
trial value of 0.2 mm along the model normal; edit it for your export.

## Export and Preview

```powershell
python -m external_projected_pipeline export-bare jobs\support_output\heights jobs\bare_layers --machine-name bambuH2D --center-model --bead-cross-section rounded_rectangle
python -m external_projected_pipeline preview jobs\support_output\combined_input_surfaces jobs\support_output\heights --randomize-consecutive-colors --mesh-opacity 0.15
```

The exporter infers the main-index range and automatically uses contact-role
flow handling for support-model batches. All standard bare-export flags pass
through, including explicit range, speeds, first-layer scaling and centering.
The viewer accepts `--layer <main>_<sub>`, `--show-points`, width highlighting
and nearest-raster diagnostics. When remeshing is enabled, preview against the
remeshed folder instead of the combined input folder.

## Python Integration

```python
from external_projected_pipeline import create_config, run_config, export_bare

config = create_config("jobs/model.json", pipeline="projected",
                       input_surfaces="model_layers", output_root="jobs/output",
                       model_pattern="hybrid")
result = run_config(config, on_output=lambda line: print(line, end=""))
export_bare(result.outputs["heights"], "jobs/bare",
            options=["--center-model", "--bead-cross-section", "rounded_rectangle"])
```

Call from a normal guarded Python entry point (`if __name__ == '__main__':`) in
Windows host applications. The API starts a fresh interpreter for each run;
it does not add runtime modules to the host's imports. It returns `RunResult`
with exit code, log, JSON report and output paths. Failures raise `PipelineError`
whose `.result` preserves diagnostics.

Every run writes a timestamped log, effective config snapshot and job report.
Processed outputs include batch metadata, role/width/travel sidecars and, for
support runs, the original-to-combined print schedule. Package installation is
optional via the included `pyproject.toml`; no installation is needed to copy
and call the bundle.

## Extrusion cross-section default

Machine and bare exports default to `rounded_rectangle`: bead area is `width * height - (1 - pi/4) * min(width, height)**2`, with the corner radius limited by the smaller measured dimension. The configured flow multiplier is applied to this area. Use `--bead-cross-section rectangular` explicitly for the legacy `width * height` model.

New A1 interface exports enable **Calculate model bead width from spacing** by default. The equivalent CLI option is `--model-width-from-spacing`, available for `export` and `export-bare` with rounded geometry. It interprets model width profiles as the assigned material-strip spacing and derives a rounded bead with area `spacing * measured_height`. For ordinary flat beads, `bead_width = spacing + (1-pi/4)*height`; tall beads use the inverse of the existing symmetric capsule model. The exporter uses the equivalent area directly, avoiding per-move width reconstruction. Every model wall and hatch uses this fast rule; there is no new wall classifier, mesh read, path movement or sidecar pass. Existing model/support labels are read once per layer. Support layers and tagged ironing retain explicit bead widths. Sparse-model spacing/density and all role/first-layer/global flow multipliers remain separate; the large sparse hatch pitch is not substituted for its recorded flow width.

Centrelines and existing boundary-inset choices stay fixed, so the wider inferred bead can extend farther outside the model and reduce nominal clearances. At 0.5 mm spacing and 0.2 mm physical height, the derived width is approximately 0.54292 mm before flow scaling. This does not establish collision freedom or physical bead shape. Disable the A1 checkbox, or omit the CLI option, for the previous explicit-width calculation. New export settings/reports and paired print schedules record the option. Existing paths, saved exports and print archives are not rewritten.

## Contact-interface heights

When `heights_3axis.contact_support_height.enabled` is true, air-gap subtraction
applies in both directions: model on contact support, and support marked
`is_model_base_layer` on model. It applies only where the selected underlying
surface is the opposite part. Ordinary support growth, platform bases and
same-part support retain their ordinary height rules. Same-part references win
equal-distance ties.

The corrected value is `max(0, measured interface distance - gap)`. The legacy
`direction_mode=model_normal` setting measures along the current path's downward
surface normal, including support-footing normals for the reverse interface.
Directional hits are refined in double precision. A missed native ray may use
a nearest triangle only after a directional ray/triangle containment test;
proximity alone does not establish support across a hole.
`direction_mode=nearest` retains its guard against modifying platform/fixed-axis
fallback heights. A configured `gap_metadata_field` is read from the support
side's upstream metadata in either direction. The configured gap is the fallback.
Coordinates, normals, path order and contour offsets are unchanged. Existing
saved height files must be regenerated to receive this correction.

Backend source: [MAP](https://github.com/neelotpal-d/MAP), developed by Neelotpal Dutta. Additional author details: see AUTHORS.md. Third-party notices remain applicable.

# Curved Support Generator

This is the small public wrapper around the curved support pipeline. It accepts one job JSON and writes all generated artifacts beneath the job's `output` directory.

Run from the repository root:

```powershell
python -m CurvedSupportGenerator .\CurvedSupportGenerator\example_job.json --dry-run
python -m CurvedSupportGenerator .\my_support_job.json
```

Relative paths in the job are resolved relative to the job JSON, not the shell's current directory. A completed run saves its fully resolved input configuration as `resolved_config.json` in the output directory.

The default `clean_output: true` clears only recognized generated artifacts in that output directory. Use `--no-clean` to fail instead when it already contains generated files.

Compatible model and support layer files use `<main>_<sub>.obj`, with one
connected triangle component per file. Main indices and scalar levels remain
fixed. Support components retain their platform/contact/model-base roles;
physical base components retain their original height sequence. Native model
VTPs, manifests, and viewer entries follow the same component split. Downstream
planning can order equal-scalar components by proximity while preserving
support-before-model order. Regenerate layers and downstream paths/heights to
use this convention for an older saved job.

## Skeleton planner

The sibling app's direct **Create Curved Support Tree** path remains available
through `skeleton.py --from-model`, without a saved transfer/tree source for
tetrahedral inputs. **Generate Curved Support Layers** preserves the 4-worker /
2-thread controls and separate minimum branch / maximum trunk diameters. Its
new platform base multiplier and height controls use the examples' 1.8 / 0.6 mm
defaults and act only at plate roots. See the workspace-root
`CURVED_SUPPORT_INTEGRATION.md` for the application workflow.

`tree_spacing` controls the support solid preview only. Relaxed support layers
use the reconstructed FEM mesh directly, with local conforming refinement and
checked model-clearance trimming. `adaptive_layer_tolerance` defaults to
0.05 mm. Other scalar modes use an independent `layer_seed_spacing` of 1 mm;
changing preview resolution does not change exported layers. Candidate
refinement has bounded passes and a triangle budget instead of a dense fine
volume grid, and reports include its per-layer counts.

Direct-model routing defaults to `--prefer-platform`: merges and steering avoid
modelward detours when a checked downward step offers more clearance, and
explicit downward retries precede field-direction fallback. Bounded recovery
tries a complete platform route before model attachments. The fixed contact
direction, transition, local angle and clearance constraints remain active.
Exterior FEM Taylor values are not used as scalar descent barriers in this
path, since they can jump between nearest tetrahedra; the planner checks build
height descent and local direction cones instead. `--no-prefer-platform`
disables the new route preference for comparisons.

After ordinary steering retries fail on a mature branch, routing checks a
bounded set of directions around the local cone at the existing four step
lengths. It ranks free corridors by clearance and downward progress, then
applies the normal edge checks. The protected contact-gradient and transition
sections are retained. Edges within half a degree of the cone limit also get
33-point midpoint refinement before acceptance. Reports count `cone_escape_attempts` and
`cone_escape_steps`.

Platform recovery can use the same azimuth coverage and rewind up to 3 mm of
its temporary trial path, with at most 24 rewinds and two detours per anchor.
Saved-tail rewind anchors share the platform portion of the existing global
recovery budget. It never crosses a saved junction or rewinds into a protected
contact section, and commits only a complete checked route to the platform.
Reports include per-anchor platform attempts, detours, rewinds and termination
reasons. No extra interface settings are required.

The package includes the progressive field-guided skeleton planner and its
contact-direction, landing-check, stalled-recovery, and gradual trunk-shift
modules. Create a skeleton with a completed transfer run, then use that new
skeleton directory as `input.skeleton_source` in the final generator job:

```powershell
python -m CurvedSupportGenerator.skeleton .\my_support_job.json `
  --run .\envelope_transfer_run `
  --output .\saved_skeleton `
  --clean-output `
  --routing-mode near-model --step 0.6 --angle 45 --merge-distance 6
```

The planner output contains `support_centerlines.vtp`, `summary.json`, and
`module_report.json`. It is the same experimental progressive planner used in
the development work. It still needs a compatible completed transfer run and a
preserved contact/tree source from that run; it is not yet a universal
mesh-only Bambu-style tree planner.

## Platform attachments

For imported progressive skeletons, set `platform_base_radius_factor` above
zero to add a short vertical primitive at each root that actually reaches the
build plate. The build-plate constraint clips its lower face into an exactly
horizontal disc before support layers are extracted. The disc radius is the
local trunk radius multiplied by this factor; `1.8` and a
`platform_base_height` of `0.6` mm are the packaged examples. Model-rooted
branches do not receive a platform disc. A factor of `0` disables it.

`runtime/` is self-contained: it bundles the execution modules for curved support generation, the progressive skeleton planner, the small neural-field runtime subset, and the SIREN implementation. It excludes benchmark scripts, previews, tests, source-model assets, generated results, and reference repositories. `--core-root` is only an override for a different compatible runtime snapshot.

## Model-tetrahedral scalar input

The interface enables **Accelerate support trimming** (`trim_pair_pruning`)
for new layer jobs. It uses conservative primitive bounds to skip irrelevant
branches during checked clipping. Checked clipping also caches repeated exact
point queries within each layer. Grid resolution, layer spacing and clearance
checks are unchanged. Existing JSON jobs retain their saved pruning setting.

When no layer schedule is supplied, tree generation extends the derived model
schedule over the transferred exterior field range. This includes support
below the model's minimum scalar after lifting or rotating it. Original model
scalar levels remain intact, but prepended levels shift their main indices;
`summary.json` records the offset in `schedule_extension`. A supplied schedule
is kept unchanged and must cover the support field too. Coverage checks reject
sampled support outside the schedule instead of silently omitting its layers.

For a scalar field available only at vertices of the model tetrahedral mesh,
use `field_mode: "tetrahedral"`. The VTU must contain only tetrahedra and one
finite point-data scalar per vertex. The model surface and VTU must describe
the same model boundary. The package linearly interpolates that scalar in each
tetrahedron and uses its constant per-tetrahedron gradient for the boundary
transfer; it does not require a neural field.

Start from [example_tetrahedral_job.json](example_tetrahedral_job.json):

```powershell
python -m CurvedSupportGenerator .\CurvedSupportGenerator\example_tetrahedral_job.json --dry-run
```

Set `input.tetrahedral_volume` to the VTU and `input.tetrahedral_scalar` to its
point-data array name. Set `input.model_volume` to the same VTU when exporting
model OBJ layers. Tetrahedral fields automatically use the reconstructed
support scalar (`support_field_mode: relaxed`), because the source field is
defined only inside the model.

## 28 October 2026 verification and Windows geometry policy

The interface defaults to four layer processes and one VTK/SDF thread per
process. Windows uses one effective VTK geometry thread per process even for
older multi-thread requests; reports retain requested/effective values.
Independent layer processes still run concurrently. Other platforms honor the
request. This mitigates measured contour/clean repeatability problems; it does
not claim every native/classic VTK thread pool is capped.

Backend source: [MAP](https://github.com/neelotpal-d/MAP). Third-party notices remain applicable.

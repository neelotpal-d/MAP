# Model Path Choices

The model settings below apply to both pipelines. In `support-model` they affect
model slices only; `support_paths` separately controls regular/platform/contact
support paths. Full editable examples are in `configs/`.

## Support/Model Sublayer Order

`compatible_inputs.sublayer_order="proximity"` visits equal-scalar components
within each main index and part by nearest surface-area centroid. It retains a
change only when the centroid route, including the incoming and outgoing jumps,
is shorter. Main indices, support-before-model order, distinct scalar levels,
and physical planar-base order remain fixed. Original source indices are kept
in the schedule; combined sub-indices are assigned after ordering, before path
and height generation. This does not change loop starts or patch order inside
a sublayer. Actual travel-time savings depend on the generated path endpoints
and clearance lifts.

New support/model configs use `proximity`. Use `"index"` to retain source index
order; saved configs without the setting also retain index order. The chosen
mode is recorded in `print_schedule.json` as `equal_scalar_sublayer_order`.

## Main Pattern

| Choice | Settings | Result |
| --- | --- | --- |
| Solid hybrid | `disable_parallel=false`, `planar_infill.enabled=false` | Contours plus dense greedy-chained raster |
| Sparse zig-zag | `planar_infill.enabled=true`, `density` in (0,1] | Contours plus clipped sparse zig-zag with boundary-following turns |
| Contours only | `disable_parallel=true` | Boundary offsets without interior raster |
| Raster only | `boundary_count=0`, `disable_parallel=false` | Dense raster or sparse zig-zag, according to infill settings |

These fields live under `hybrid_contours`. `spacing` controls contour/bead width;
`parallel_spacing` optionally sets the raster's nominal bead width. Sparse line
pitch is nominal width/density, and its larger separation does not inflate E.
`resample_spacing` and `parallel_resample_spacing` control waypoint intervals.

`boundary_start_offset_count` omits that many outer offset levels.
`first_layers_lowest_y_start_count` chooses a low-Y start for early model main
layers. Boundaries use the existing nearest-opening chaining and quarter-width
seam treatment. Raster fragments remain separate until greedy ordering.
`parallel_direction_mode` accepts `connect_nearest` and legacy `one_direction`.
Disconnected infill blocks carry explicit travel breaks; connectors inside a
valid block retain the existing projected boundary-following logic.

## Dense Base and Top Bands

```json
"planar_infill": {
  "enabled": true,
  "density": 0.25,
  "solid_base_main_layers": 3,
  "solid_top_main_layers": 3,
  "solid_top_target": "last_main_layers",
  "angle_degrees": 0.0,
  "alternate_angle_degrees": 90.0,
  "boundary_link_pitch_factor": 2.0
}
```

Base and top bands use the established dense hybrid raster, not sparse gaps.
`last_main_layers` works without a top classifier. `top_classified` instead
places bands ending at detected top main indices. Base/top counts use the
available distinct model mains. Main-index parity determines alternating hatch
directions; missing main indices are not compacted for this parity.

Sparse-infill knobs also include `pattern_origin_xz`, `overlap`,
`max_link_length`, `max_boundary_link_length` (or pitch factor plus extra),
and `min_segment_length`. These are the established projected clipper choices.

## Raster Directions

| `axis_mode` | Other fields | Behavior |
| --- | --- | --- |
| `auto_tangent_alternating` | none | Alternate X/Z rasters by original main-index parity |
| `fixed` | `axis="x"` or `"z"` | Fixed raster axis |
| `fixed_angle` | `hatch_angle_degrees=45` | Fixed hatch angle in XZ |
| `alternating_angle` | `hatch_angle_degrees=45` | Dense hatch alternates 45°/135° by original main-index parity |

Dense rasters use these fields. Sparse zig-zag uses
`planar_infill.angle_degrees` and `alternate_angle_degrees`; set the latter to
null for one fixed hatch angle. Ironing has its own direction controls.

## Progressive Changes

```json
"progress_overrides": [
  {"start_fraction": 0.0, "boundary_count": 3, "disable_parallel": false},
  {"start_fraction": 0.8, "boundary_count": 2, "disable_parallel": true}
]
```

Progress is `(main rank + 1) / number of distinct model mains`. Matching rules
apply in ascending start fraction. Rules support `boundary_count`,
`parallel_spacing`, `resample_spacing`, `parallel_resample_spacing`,
`boundary_start_offset_count`, `parallel_direction_mode`, `disable_parallel`.
Count/spacing/resampling rules also accept their `_factor` variants, computed
from the base count or base bead spacing. Explicit values take priority over a
factor in the same rule. Initial and later blocks can independently enable or
disable the interior. Model top overrides take priority over progressive rules.

## Top Classification and Widths

`top_layer_classifier.enabled` runs the existing coverage classifier. Options
include `lookahead_main_count`, `min_uncovered_fraction`, `sample_mode`
(`face_centroids`, or other modes supported by that classifier), worker counts,
and optional `global_boundary_mesh` distance precheck. Global model-base
normalization is independently optional. Use `top_layer_overrides` to load a
saved classification JSON or override selected top contour count, raster
spacing and resampling. Width checks and `disable_parallel` can be configured
for those top layers. Supply an actual global model mesh before enabling that
precheck; the templates contain no machine-specific input model path.

`projected_width_check` can target `last_main_layers` with `main_layer_count`,
or `top_classified`. Its `parallel_spacing` changes generation width/pitch on
the selected dense rasters (for example 0.42 mm), while its nearest-other-pass
measurement supplies local per-point width metadata. `max_factor` bounds the
adjustment. This dense-raster width measurement is not applied to sparse-infill
families. Export uses stored width sidecars; visualization can separately
recalculate diagnostic distances.

## Ironing and Postprocessing

`ironing.enabled` adds tagged ironing passes. Select `target=top_classified` or
`last_main_layers`, `main_layer_count`, direction `perpendicular`, `x`, `z` or
`angle`, `angle_degrees`, `spacing`, `resample_spacing`, `boundary_inset`,
`flow_width` and optional ironing `clearance_lift`. Ironing is disabled in the
templates. Its flow width must be configured for any print trial.

`short_fragment_filter` removes only short travel-separated raster runs before
heights; boundaries are protected. `heights_3axis.adaptive_point_decimation`
optionally reduces post-height waypoints while retaining tags, widths and
travel breaks. Boundary roles are protected. Nozzle-clearance lifting can be
run afterward with the optional `lift` command, including current-surface and
immediate-prior-surface checks; it preserves nominal extrusion segment lengths.

The projected planner ignores heat/Poisson tuning and the legacy contour
smoothing/sharp-angle fields it accepts for interface compatibility. It uses
polygon offsets and clipping. Do not interpret those legacy fields as active
controls for smoothing projected geometry.

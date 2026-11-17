# Grid support-field collision modes

In **Slicing → Field generation → Support → Optimize Conservative Support Field**,
use **Tool motion for collision**:

- `auto`: read the selected model's saved generator/support-field provenance.
- `fixed_axis`: preserve the existing fixed rectangular-head search for 3-axis.
- `field_gradient`: orient the nominal head along the root gradient for 5-axis.

Imported fields without provenance require an explicit selection in the GUI.
Old runtime jobs without a collision-mode setting remain fixed-axis; an explicitly
recorded `input.model_generator_axes: 5` selects gradient-following if the setting
is omitted. New GUI jobs save the resolved mode. Reloading a support field retains
its mode and nominal tool link. Project bundles relocate that link.

**From model** reads the saved nominal validation tool of a 5-axis job. It never
silently substitutes a Bambu rectangular preset for a 5-axis model with no tool.
Select a preset or custom nominal JSON explicitly when necessary. Supported
5-axis geometry is a rectangular box or the existing nominal cone/inner-cylinder/
outer-cylinder enclosure. The support tool editor exposes nominal geometry;
neural training sample controls do not apply to this optimizer.

Public field-job settings:

```json
{"collision_mode": "field_gradient", "rounds": 4, "iterations": 200}
```

The normal field workflow (`workflow.py JOB.json --stage field`) saves the mode,
tool source and normalized tool fingerprint in `support_field.json`. Generated
reports include all three support-related directions and, in 5-axis mode,
model-to-model collisions. Model-to-model violations are diagnostic: this solver
does not change the original model field.

At root p, n = grad(f(p))/norm(grad(f(p))). Model roots use original tetrahedral
gradients; support roots use current trilinear-grid gradients. Box U is the saved
reference U projected perpendicular to n, with saved V as the parallel fallback;
V = U cross n. Transform q-p into (U,n,V) coordinates for exact box membership.
Axisymmetric tools use axial distance and radius, so roll has no effect. The
world-aligned enclosing box is only a BVH pruning bound, never the final
membership test. Zero/nonfinite pose gradients raise an error rather than
silently removing roots from the check.

During LBFGS, retained-pair membership is recomputed using current support
gradients in each objective closure. Pairs outside the current tool contribute
zero collision loss. Fresh searches between replay rounds discover new pairs.
This is a hard-membership active-set scalar-order optimizer; it does not
differentiate tool geometry or optimize robot joint angles. Model values remain
fixed. The existing loss weights, scalar normalization and support shadow are
unchanged.

These checks cover finite potential-material samples. They do not certify
continuous collision freedom, travel sweeps, platform/head collisions, robot
kinematics or achievable roll for an asymmetric head. The existing downward
shadow is not a general omnidirectional support domain. Actual routed solids,
toolpaths and machine motion need separate verification. Old fixed-axis reports
are not upgraded to 5-axis reports by loading them; prepare a fresh job.

## Optional collision objective

**Optimize for collision** is enabled by default. Turn it off to solve only the
existing field regularization/interface objectives (plus any separately configured
route/contact constraints). Its effective collision weight becomes zero even if
the stored Advanced **Collision weight** is nonzero. The stored weight is preserved
for re-enabling the checkbox. Collision pair replay/mining rounds and warm-start
collision pairs are not used when it is off.

A final collision search still reports the result; **Validate independent sample
pool** controls the additional holdout search. Thus disabling the loss does not
mean the result was collision-free or that every collision query is skipped. Tool
motion and nominal-tool inputs remain required for those diagnostics. Saved jobs,
optimization history and completion reports record both the configured and
effective collision weights. JSON setting: `"collision_training": false`.

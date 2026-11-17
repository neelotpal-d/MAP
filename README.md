# MAP — multi-axis process-planning tools

Developed by **Neelotpal Dutta**. This repository contains the backend source
snapshot bundled with the MultiAxisPrinting combined interface dated
17 November 2026. The same backend files are included in
https://github.com/RyanTaoLiu/MultiAxisPrinting/tree/dev2 .

| Directory | Purpose |
|---|---|
| `MultiAxisFieldGenerator/` | Five-axis neural scalar-field optimization |
| `ThreeAxisCollisionGenerator/` | Fixed-axis field optimization and collision checks |
| `CurvedSupportGenerator/` | Support field optimization, routing, solids and layers |
| `external_projected_pipeline/` | Layer-to-path planning, heights and bare export |
| `standalone_bambu_layer_slicer/` | Bambu program assembly and 3MF packaging |

Keep these directory names and sibling layout. Read each component README for
its inputs and CLI. With the existing compatible numerical environment, from
this repository root:

```text
python -m MultiAxisFieldGenerator --help
python -m ThreeAxisCollisionGenerator --help
python -m CurvedSupportGenerator --help
python -m external_projected_pipeline --help
python -m external_projected_pipeline doctor
```

This is the algorithm source repository. For the complete GUI, FEA and the
locked cross-platform installation workflow, use MultiAxisPrinting/dev2.
It currently bundles a snapshot directly; no new package/dependency mechanism
or numerical workflow is introduced here. `MAP_BUNDLE.json` records the exact
backend hashes shared between the repositories. Algorithm/process source and
configuration files are unchanged from the supplied combined ZIP.

See AUTHORS.md for author details/placeholders. Original SIREN, Bambu-derived
and other third-party authorship/licences remain applicable; see
THIRD_PARTY_NOTICES.md and LICENSE_STATUS.md. Discussion/agent handovers,
generated experiments, checkpoints and environments are not included.

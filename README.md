# MAP — multi-axis process-planning tools

## Install and launch

### Recommended: use the interface installation packages

For the complete GUI, FEA and end-to-end workflows, download the current
[MultiAxisPrinting installer release](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-preview-2026-11-27-map):

| Platform | Package | Start |
|---|---|---|
| Windows x86-64 | [MultiAxisPrinting-Setup-Windows.exe](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-map/MultiAxisPrinting-Setup-Windows.exe) | Open the EXE, choose **Install / Check**, then **Launch interface**. |
| macOS 14+, Apple Silicon or Intel | [MultiAxisPrinting-Setup-macOS.zip](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-map/MultiAxisPrinting-Setup-macOS.zip) | Extract the whole ZIP, open **Install MultiAxisPrinting.app**, then launch **MultiAxisPrinting.app** from your user Applications folder. |
| Linux x86-64 | [application.zip](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-map/application.zip) | Extract into a fresh folder and follow **INSTALL.md** inside it. This is a Linux/source package. |

Windows/macOS setup creates its own private Miniforge/Python environment and
installs pinned dependencies; no separate Conda/Python installation or manual
environment activation is needed. Mac setup may request Apple Command Line Tools.
Windows/Linux support NVIDIA CUDA with CPU fallback; Mac profiles use CPU.

The interface repository is currently **private**: GitHub installer downloads
require repository access. Installers received directly work without a GitHub
login. The MAP backend release below is public and needs no account.

### How the installers obtain MAP

First-time setup downloads the exact [tested MAP backend release](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27)
from **this repository**, rather than fetching changing code from `main` or
`latest`. `MAP_BACKEND_LOCK.json` in the interface package pins the 3.3 MB ZIP's
SHA256, size and complete 382-member inventory. Setup verifies every file before
installing the five component folders; the 380 component code/reference files
match the preceding tested interface snapshot, with two release/notices files added.

Valid installed source and cached archives are reused. Normal launches and
check-only do not download code. Modified existing backend source is preserved
and blocks replacement. This makes backend installation a genuine MAP release
download; release-download statistics are separate from page views and clones.

For an offline backend, download the exact ZIP named by the interface's lock from
the [MAP release](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27). From the extracted **interface** folder, run:

```text
python map_backend.py --archive /path/to/MAP-backends-2026-11-27.zip
```

Then run the normal interface setup. `map_backend.py`, `setup.py`, GUI code and
dependency locks belong to the interface package, not to this backend-only ZIP.
See the [interface installation notes](https://github.com/RyanTaoLiu/MultiAxisPrinting/blob/dev2/INSTALLERS.md).

These are development prereleases. Actual backend download/offline reuse was
verified on Windows and Linux. Complete fresh-environment installation, native
Mac setup and physical printing are not claimed verified; read the matching
[release verification notes](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-map/VERIFICATION.json).

### Manual/source installation (fallback)

For manual GUI setup, extract **application.zip from the same interface release**,
use a native-architecture Miniforge/Conda terminal, and follow its **INSTALL.md**
and pinned dependency profiles. From the folder containing `setup.py`:

```text
python setup.py --device auto
python launch.py --installed-runtime
```

For backend-only development or component CLIs, keep the component folders as
siblings and use a compatible numerical environment. Read each component README
for its inputs, dependencies and CLI. The existing commands from this repository
root remain available below.

The uploaded MAP backend ZIP and its `MAP_BACKEND_RELEASE.json` identify the
exact tested source with maintenance patches. GitHub's automatic source archives
at the coordination tag may differ; use the named release artifact to reproduce
the installer backend. The repository's `MAP_BUNDLE.json` retains the original
17 November source-snapshot provenance and maintenance history.

MAP provides backend algorithms for neural-field slicing, support generation,
toolpaths, supporting heights and printer-program export. Current interface
installers obtain their exact tested backend from [MAP Releases](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27).
GUI and existing FEA functionality are provided by
[MultiAxisPrinting/dev2](https://github.com/RyanTaoLiu/MultiAxisPrinting/tree/dev2).

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

See AUTHORS.md for author details/placeholders. Original SIREN, Bambu-derived
and other third-party authorship/licences remain applicable; see
THIRD_PARTY_NOTICES.md and LICENSE_STATUS.md. Discussion/agent handovers,
generated experiments, checkpoints and environments are not included.

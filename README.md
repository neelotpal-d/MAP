# MAP — multi-axis process-planning tools

## Install and launch

### Recommended: use the interface installation packages

For the complete GUI, FEA and end-to-end workflows, choose the installer for
your operating system from the table below.

| Platform | Package | Start |
|---|---|---|
| Windows x86-64 | [MultiAxisPrinting-Setup-Windows.exe](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-r2/MultiAxisPrinting-Setup-Windows.exe) | Open the EXE, choose **Install / Check**, then **Launch interface**. |
| macOS 14+, Apple Silicon or Intel | [MultiAxisPrinting-Setup-macOS.zip](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-r2/MultiAxisPrinting-Setup-macOS.zip) | Extract the whole ZIP, open **Install MultiAxisPrinting.app**, then launch **MultiAxisPrinting.app** from your user Applications folder. |
| Linux x86-64 | [MultiAxisPrinting-Setup-Linux.sh](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-linux-preview-2026-11-27/MultiAxisPrinting-Setup-Linux.sh) | Run `bash MultiAxisPrinting-Setup-Linux.sh` in Terminal, then open **MultiAxisPrinting** from the application menu. |

**Windows, macOS and Linux installers:** setup creates a dedicated Miniforge/Python
environment and installs the pinned dependencies. You do not need to install
Conda/Python separately or activate an environment manually. Mac setup may request
Apple Command Line Tools. Windows/Linux use verified prebuilt native accelerators
and select usable NVIDIA CUDA with CPU fallback; Mac profiles use CPU.

Linux download: [installer and installation notes](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-linux-preview-2026-11-27).
Run `bash "$HOME/Downloads/MultiAxisPrinting-Setup-Linux.sh"` using the path to
your downloaded file. See [Linux setup details](INSTALLERS.md#linux-installer).

### Backend downloads

Setup obtains the fixed, hash-verified [MAP backend release](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27-r2)
automatically and reuses verified installed source on later launches. No manual
backend copying is needed for normal installation. For local ZIP/offline-source
options, use the [backend guide](https://github.com/RyanTaoLiu/MultiAxisPrinting/blob/dev2/MAP_BACKEND_DOWNLOAD.md).
See the matching [Windows/Mac verification record](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-preview-2026-11-27-r2/VERIFICATION.json)
or [Linux verification record](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-linux-preview-2026-11-27/VERIFICATION.json) for the downloaded version.

### Downloads, installed files and later launches

The setup packages include the interface, starter models and local documentation.
Setup obtains the locked MAP backends from MAP Releases and pinned dependencies
from conda-forge/PyPI and the official PyTorch wheel indexes. All three platform installers
also install Miniforge; the manual/source fallback uses an existing manager.

| Platform | Default installed application root | Launch again |
|---|---|---|
| Windows bootstrap | `%LOCALAPPDATA%\MultiAxisPrinting\versions\<release-id>\` | **MultiAxisPrinting** Desktop/Start-menu shortcut; its Properties show the exact source root. |
| macOS bootstrap | `~/.local/share/multiaxisprinting/versions/<release-id>/` | `~/Applications/MultiAxisPrinting.app`; `current-app.txt` in the installation base identifies the active source root. |
| Linux bootstrap | `~/.local/share/multiaxisprinting/versions/<release-id>/` (or the selected/XDG base) | **MultiAxisPrinting** application-menu entry; `current-app.txt` identifies the active root. |
| Manual/source ZIP | The folder where you extracted `application.zip` | From that folder, `python launch.py --installed-runtime`. |

Windows can use the destination selected in setup; Linux accepts `--destination`.
Mac/Linux setup can use a custom `MULTIAXIS_INSTALL_DIR`. Python lives in the application root's `.runtime/`;
`.runtime/selection.json` records the active interpreter. Starter models are in
`MultiAxisPrinting-dev/data/mesh/`. Use **File → Run reports…** for generated
results/logs; project saves and final exports use the destination you choose.

See [download sources, folder locations and launch instructions](INSTALLERS.md)
for each component, Finder/File Explorer navigation, caches, setup/launch logs
and manual/offline installation.

### Manual/source installation (fallback)

Use these steps only if you choose the source ZIP instead of an automatic installer. The shared source ZIP includes all four platform profiles.

**1. Install Miniforge once.** Download the installer for your OS/CPU from the
[official Miniforge page](https://github.com/conda-forge/miniforge#install).
On Windows, open **Miniforge Prompt** after installation. On Mac/Linux, complete
the shell-initialization prompt, then open a new Terminal. If Conda already works,
use that installation. [Miniforge installation steps for each OS](INSTALLERS.md#step-1-install-miniforge-once).

**2. Download and extract the application.** From the
[shared manual source download](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-linux-preview-2026-11-27/application.zip),
download the file named **application.zip**. Extract it to `%USERPROFILE%\MultiAxisPrinting`
on Windows, or `~/MultiAxisPrinting` on Mac/Linux. The extracted folder must
contain `setup.py` and `launch.py`. If you choose another folder, use its path
in the commands below.

**3. Create the application environment and launch.** Run the block for your OS:

Windows — in **Miniforge Prompt**:

```cmd
conda activate base
cd /d "%USERPROFILE%\MultiAxisPrinting"
python setup.py --device auto
python launch.py --installed-runtime
```

Linux or macOS — in **Terminal**:

```bash
conda activate base
cd "$HOME/MultiAxisPrinting"
python setup.py --device auto
python launch.py --installed-runtime
```

**`setup.py` creates the application environment for you.** It installs the
profile's Python 3.10, pinned dependencies and matching MAP backends, then checks
the installation. Application dependencies go into this folder's `.runtime/`,
separately from Conda's `base` environment. Windows/Linux select usable CUDA
or CPU fallback; Mac profiles use CPU. Continue to launch after setup reports
**Setup passed**. Platform prerequisites are listed in the package's `INSTALL.md`.

**4. Launch on later visits.** Open the same terminal, activate `base`, change
to the same application folder, and run only:

```text
python launch.py --installed-runtime
```

You do not need to repeat setup. The launcher uses the recorded application
environment. See the [manual setup guide](INSTALLERS.md#manual-installation-optional)
for explicit Conda activation, installation checks and common setup messages. Close the interface before running setup, checks or repair.


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
installers obtain their exact tested backend from [MAP Releases](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27-r2).
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
its inputs and CLI. After manual setup, activate the application environment
using the commands in [INSTALLERS.md](INSTALLERS.md#manual-installation-optional).
From the application folder containing these components, run:

```text
python -m MultiAxisFieldGenerator --help
python -m ThreeAxisCollisionGenerator --help
python -m CurvedSupportGenerator --help
python -m external_projected_pipeline --help
python -m external_projected_pipeline doctor
```

See AUTHORS.md for authorship details. Original SIREN, Bambu-derived
and other third-party authorship/licences remain applicable; see
THIRD_PARTY_NOTICES.md and LICENSE_STATUS.md. Generated results, checkpoints and runtime environments are stored separately from source.

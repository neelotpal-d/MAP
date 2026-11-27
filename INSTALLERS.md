# Install, launch and locate MultiAxisPrinting

Download the [Windows/macOS package](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-preview-2026-11-27-r2).
For Linux, download the [automatic installer](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-linux-preview-2026-11-27).
The packages install one interface with five backend component folders. The
backends are modules used by the interface, not five separate desktop applications.

## What setup downloads and where it comes from

| Component | Source | Installed location |
|---|---|---|
| Interface, starter models and offline wiki/course pages | The named MultiAxisPrinting release above: embedded in the Windows EXE, or in `payload/application.zip` inside the complete Mac setup ZIP. The Linux `.sh` installer embeds its matching `application.zip`; the separate ZIP is the manual fallback. | The application root, including `MultiAxisPrinting-dev/`, `docs/` and `course_reporting/`. |
| MAP algorithms for fields, supports, toolpaths, heights and printer assembly | The exact [MAP backend release](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27-r2) specified in `MAP_BACKEND_LOCK.json`. Setup checks the archive size, SHA256 and every member. | The five folders listed below, directly inside the application root. |
| Miniforge, used to create/manage the Python environment | Official [conda-forge/Miniforge releases](https://github.com/conda-forge/miniforge/releases/tag/26.9.4-2). The bootstrap selects the OS/CPU-specific installer and checks its pinned SHA256. | Windows/Mac/Linux bootstrap: `<installation-base>/managers/miniforge-<version>-<attempt>/`. Source setup uses your existing Conda/Mamba installation. |
| Python and native dependencies | Exact package URLs in `installation/locks/*.native.txt`, from [conda-forge](https://conda.anaconda.org/conda-forge/). | The selected application runtime under `<application-root>/.runtime/`. |
| Python packages, including Qt bindings, VTK/PyVista, NumPy/SciPy and geometry libraries | Hash-locked packages from [PyPI](https://pypi.org/simple/). Windows/Linux PyTorch profiles additionally use the official [CPU](https://download.pytorch.org/whl/cpu) or [CUDA 12.8](https://download.pytorch.org/whl/cu128) wheel index. | The selected runtime's Python package directories. They are not installed into system Python. |
| Apple Silicon docking binding | Pinned [Qt Advanced Docking System 4.3.1 source](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System/tree/4.3.1), downloaded from GitHub's `codeload.github.com` service and hash-checked before building. | Source staging: `<application-root>/.runtime/build/osx-arm64/`; the built binding is installed into the selected runtime. |
| Two support-query C++ accelerators | Matching verified assets included with the interface package when available; otherwise built locally from the included C++ source using the OS compiler. | `<application-root>/CurvedSupportGenerator/runtime/SupportGenerationCurved/native_build/`. |

The five MAP folders are:

| Folder at the application root | Purpose |
|---|---|
| `MultiAxisFieldGenerator/` | 5-axis model fields |
| `ThreeAxisCollisionGenerator/` | 3-axis model fields |
| `CurvedSupportGenerator/` | Support fields, trees and paired layers |
| `external_projected_pipeline/` | Toolpaths and supporting heights |
| `standalone_bambu_layer_slicer/` | Specialized printer-program/3MF assembly |

The frontend is extracted from the downloaded setup package; setup does not fetch
its GUI from a changing branch. MAP is fetched from its fixed release when
needed, then valid installed source or cached downloads are reused. Normal
launch does not download application code or install dependencies. Models and
course pages included in the package are local files.

Windows/Linux use NVIDIA CUDA when its runtime probe succeeds, with CPU fallback.
Mac profiles use CPU. Bootstrap setup does not install a GPU driver. Windows and
Linux packages include verified support-query binaries for their supported
profiles; the default route needs no C++ compiler there. Mac setup may request
Apple Command Line Tools for the accelerator/docking builds.

Exact versions and hashes are recorded in the extracted package's
`MAP_BACKEND_LOCK.json`, `installation/profiles.json`, `installation/locks/` and
`installation/bootstrap/downloads.json`. `SOURCE_ATTRIBUTION.txt`, `AUTHORS.md`
and the included licence/third-party notices describe ownership and attribution.

## Installation folders

**Installation base** means the parent directory used by a platform
bootstrap. **Application root** means the versioned directory containing
`launch.py`, `setup.py`, `MultiAxisPrinting-dev/` and the backend folders.
The downloaded installer and the launcher shortcut are separate from this root.

| Platform and installation route | Installation base | Application root |
|---|---|---|
| Windows bootstrap | `%LOCALAPPDATA%\MultiAxisPrinting` by default, or the writable destination selected in setup | `<installation-base>\versions\<release-id>\` |
| Mac bootstrap | `~/.local/share/multiaxisprinting` by default; `MULTIAXIS_INSTALL_DIR` can select a different base when running setup | `<installation-base>/versions/<release-id>/`; `current-app.txt` records the active root |
| Linux bootstrap | `${XDG_DATA_HOME:-~/.local/share}/multiaxisprinting` by default (normally `~/.local/share/multiaxisprinting`), or `--destination` / `MULTIAXIS_INSTALL_DIR` | `<installation-base>/versions/<release-id>/`; `current-app.txt` records the active root |
| Manually extracted source on any OS | The folder you choose when extracting the package | That extracted folder, containing `launch.py` |

Windows/Mac/Linux bootstrap runtimes normally use
`<application-root>/.runtime/<profile>-<attempt>/`. Manual `python setup.py`
normally uses `<application-root>/.runtime/<profile>/`, unless `--prefix` is
specified. Profiles are `win-64`, `linux-64`, `osx-arm64` or `osx-64`.
The authoritative interpreter, platform and numerical device are recorded in
`<application-root>/.runtime/selection.json`; do not guess the active runtime
from folder dates. The Miniforge manager and the application runtime are distinct.

Bootstrap download caches are under `<installation-base>/downloads/`. MAP's ZIP
cache is under `<application-root>/.runtime/map-backend/downloads/`, and Conda
package caching uses `<application-root>/.runtime/conda-cache/`. These caches
are separate from generated results and saved projects.

## First installation and subsequent launches

### Windows

1. Open `MultiAxisPrinting-Setup-Windows.exe`.
2. Select a writable installation base and choose **Install / Check**. Leave
   automatic device selection enabled, or select **CPU only**.
3. After setup passes, select **Launch interface**.
4. Later, use the **MultiAxisPrinting** Desktop or Start-menu shortcut. The
   **MultiAxisPrinting Instructions** shortcut opens the local course guide.

No separate Python/Conda installation or environment activation is needed for
this bootstrap route. Shortcuts target
`<installation-base>\MultiAxisPrinting.exe --launch "<application-root>"`.
Opening the installer without `--launch` opens setup.

To find the installed source, right-click the application shortcut and open
**Properties**: its **Target** contains the `--launch` application root and its
**Start in** field points to that root. **Open file location** on the launcher
EXE takes you to the installation base, not necessarily the versioned source.
For the default base, paste `%LOCALAPPDATA%\MultiAxisPrinting` into File
Explorer's address bar, then open the appropriate `versions` folder.

### macOS

1. Extract the entire `MultiAxisPrinting-Setup-macOS.zip` and keep its contents together.
2. Open **Install MultiAxisPrinting.app** and complete the Terminal prompts,
   including Apple's Command Line Tools prompts if requested.
3. After setup passes, open **MultiAxisPrinting.app** in your Home folder's
   **Applications** directory (`~/Applications`). This is different from the
   system-wide `/Applications` directory.
4. Use **MultiAxisPrinting Instructions.app** there for the local course guide.

To find the source/data in Finder, choose **Go → Go to Folder** (**⌘ Shift G**),
enter `~/.local/share/multiaxisprinting/`, and open `current-app.txt` to find the
active application root. For the default installation, Terminal can open it:

```bash
open "$(cat "$HOME/.local/share/multiaxisprinting/current-app.txt")"
```

For a custom base, use its `current-app.txt`. The launcher `.app` in
`~/Applications` is separate from the source/data folder.

### Linux installer

Download [MultiAxisPrinting-Setup-Linux.sh](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/download/interface-linux-preview-2026-11-27/MultiAxisPrinting-Setup-Linux.sh), open Terminal, and run:

```bash
bash "$HOME/Downloads/MultiAxisPrinting-Setup-Linux.sh"
```

Use the actual downloaded file path if different. Press Return to install.
No existing Python, Miniforge or Conda installation is needed. Setup checks its
embedded application, downloads and verifies Miniforge, then creates the pinned
application environment and checks CUDA/CPU and native accelerators.

After **Setup passed**, open **MultiAxisPrinting** from the application menu.
**MultiAxisPrinting Instructions** opens the local course guide. You can also
run `bash "<application-root>/Launch MultiAxisPrinting.sh"` from Terminal.
The installer prints the exact source, launch and log paths.

Default installation base: `${XDG_DATA_HOME:-~/.local/share}/multiaxisprinting`
(normally `~/.local/share/multiaxisprinting`). `current-app.txt` there records
the active version under `versions/`. Menu entries are in
`${XDG_DATA_HOME:-~/.local/share}/applications`; this does not create a Desktop icon.

Requires Linux x86-64, glibc 2.35+ (for example Ubuntu 22.04+), Bash, `flock`
(util-linux), standard core utilities, and `curl` or `wget`. The GUI needs a
working desktop/OpenGL driver and the system libraries listed in `INSTALL.md`.
The installer uses verified prebuilt support-query accelerators and does not
install system packages or GPU drivers.

<details>
<summary>Linux installer options and repair</summary>

```bash
bash MultiAxisPrinting-Setup-Linux.sh --device cpu
bash MultiAxisPrinting-Setup-Linux.sh --destination "$HOME/Applications/MultiAxisPrinting"
bash MultiAxisPrinting-Setup-Linux.sh --headless --no-shortcuts
bash MultiAxisPrinting-Setup-Linux.sh --fresh
bash MultiAxisPrinting-Setup-Linux.sh --plan
bash MultiAxisPrinting-Setup-Linux.sh --verify-payload
```

`--headless` performs installation/checks without prompts and uses offscreen Qt
checks. `--fresh` retains earlier runtime attempts and creates a new one. A repeated
normal setup checks/reuses the recorded runtime. `--plan` and `--verify-payload`
do not install anything. Installation logs are `setup-*.log` in the base folder;
launcher output is `<application-root>/.runtime/launch.log`.

</details>

### Linux and manual/source installation

Follow this procedure when choosing manual setup instead of a platform installer.
It creates the application's Python environment and installs its dependencies.

#### Step 1: install Miniforge once

If `conda --version` already works in your terminal, continue to Step 2.
Otherwise download the matching installer from the
[official Miniforge release](https://github.com/conda-forge/miniforge/releases/latest).

| Your computer | Miniforge file | What to do |
|---|---|---|
| Windows x86-64 | `Miniforge3-Windows-x86_64.exe` | Open the EXE, follow the prompts and keep Start-menu shortcuts enabled. After installation, open **Miniforge Prompt** from Start. |
| Linux x86-64 | `Miniforge3-Linux-x86_64.sh` | Run the Linux command below in Terminal, follow the prompts and accept Conda initialization. Close and reopen Terminal. |
| Mac with Apple Silicon | `Miniforge3-MacOSX-arm64.sh` | Run the Apple Silicon command below in Terminal, follow the prompts and accept Conda initialization. Close and reopen Terminal. |
| Mac with Intel CPU | `Miniforge3-MacOSX-x86_64.sh` | Run the Intel command below in Terminal, follow the prompts and accept Conda initialization. Close and reopen Terminal. |

For a file saved in your Downloads folder, run the one matching command:

Linux x86-64:

```bash
bash "$HOME/Downloads/Miniforge3-Linux-x86_64.sh"
```

Apple Silicon Mac:

```bash
bash "$HOME/Downloads/Miniforge3-MacOSX-arm64.sh"
```

Intel Mac:

```bash
bash "$HOME/Downloads/Miniforge3-MacOSX-x86_64.sh"
```

In the new terminal, run `conda --version`. A version number confirms that Conda
is available. These steps follow the
[official Miniforge instructions](https://github.com/conda-forge/miniforge#install).

#### Step 2: extract the application

Download **application.zip** from the
[MultiAxisPrinting release](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-preview-2026-11-27-r2)
and extract it into a new folder you will keep at the same location.
These examples use `C:\MultiAxisPrinting` on Windows and `~/MultiAxisPrinting`
on Mac/Linux. Open that folder and check that `setup.py`, `launch.py` and
`MultiAxisPrinting-dev/` are directly inside it. If extraction adds another
folder level, use that inner folder in the following commands.

For Linux, the supported profile is x86-64 with glibc 2.35+ (for example
Ubuntu 22.04+), plus a working desktop/OpenGL driver. The default prebuilt
accelerators need no C++ compiler. On Mac, use macOS 14+ and install Apple Command
Line Tools if requested for native builds. See `INSTALL.md` in the extracted
package for system-library requirements.

#### Step 3: create the environment and install dependencies

Windows — open **Miniforge Prompt**:

```cmd
conda activate base
cd /d "C:\MultiAxisPrinting"
python setup.py --device auto
```

Linux/macOS — open **Terminal**:

```bash
conda activate base
cd "$HOME/MultiAxisPrinting"
python setup.py --device auto
```

Replace the example application path if you extracted elsewhere. Leave the
terminal open until it reports **Setup passed**. Setup performs all of these:

1. Downloads and verifies the matching MAP backend source when needed.
2. Creates a separate application environment with the profile's Python 3.10.
3. Installs the pinned Conda/pip dependencies into that environment.
4. Installs or builds the matching native accelerators, checks dependencies,
   Qt and native operations, and probes CUDA/CPU as applicable.
5. Records the application interpreter in `.runtime/selection.json`.

`base` provides the Python/Conda used to run setup. The application's dependencies
are installed separately under the extracted application's `.runtime/` folder.
You do not need a separate `conda create` command or a manual `pip install` list.

| Platform | Default environment created by this procedure |
|---|---|
| Windows | `<application-folder>\.runtime\win-64\` |
| Linux | `<application-folder>/.runtime/linux-64/` |
| Apple Silicon Mac | `<application-folder>/.runtime/osx-arm64/` |
| Intel Mac | `<application-folder>/.runtime/osx-64/` |

Windows/Linux use usable NVIDIA CUDA with CPU fallback; Mac uses CPU. For a
deliberate CPU-only setup, use `python setup.py --device cpu` for the first setup.
This setup script uses the Miniforge/Conda installed in Step 1; it does not install
the environment manager itself.

#### Step 4: launch now and on later visits

After setup passes, run from the same application folder:

```text
python launch.py --installed-runtime
```

For later launches, open Miniforge Prompt/Terminal, activate `base`, change to the
application folder and run that launch command. Skip the setup command.
The launcher selects the recorded application environment automatically.
Linux/source setup does not create a desktop shortcut automatically.

<details>
<summary>Optional: activate the application environment explicitly</summary>

From the extracted application folder, use the command for your OS:

| Platform | Conda activation command |
|---|---|
| Windows, in Miniforge Prompt | `conda activate ".\.runtime\win-64"` |
| Linux | `conda activate "./.runtime/linux-64"` |
| Apple Silicon Mac | `conda activate "./.runtime/osx-arm64"` |
| Intel Mac | `conda activate "./.runtime/osx-64"` |

Then launch using `python launch.py`. This is an alternative to the automatic
`--installed-runtime` launch above, after the same setup has completed.
For a custom `--prefix`, use the environment path recorded in
`.runtime/selection.json` instead of the default table.

</details>

<details>
<summary>Common setup messages and installation checks</summary>

- **`conda: command not found`:** reopen the terminal after Miniforge installation.
  With the default Mac/Linux Miniforge location, running
  `source "$HOME/miniforge3/etc/profile.d/conda.sh"` enables Conda for that terminal.
  Use your chosen Miniforge path if different.
- **`setup.py` or `launch.py` not found:** change to the extracted folder containing
  those files, rather than the parent folder or the ZIP itself.
- **`Runtime destination exists`:** setup has already created that environment.
  To verify it, run `python setup.py --check-only`; to launch, use Step 4.
  If setup was interrupted and checking fails, retain its log and use a fresh
  application folder for another installation.
- **Missing Linux display/system libraries:** follow the Linux system-library
  section in the package's `INSTALL.md`; numerical CPU fallback still needs a
  working desktop for the GUI.

Run `python launch.py --installed-runtime --check` from the application folder
for an installed-runtime environment check. Keep the runtime at its original
path; choose a new setup folder rather than moving an existing Conda environment.

</details>


## Find models, generated results, saved projects and logs

| Item | Location or action |
|---|---|
| Starter meshes | `<application-root>/MultiAxisPrinting-dev/data/mesh/`; see `TEST_MODELS.md` in the package. |
| Wiki and course guide | `<application-root>/docs/index.html` and `<application-root>/course_reporting/web/instructions.html`, also accessible through the interface's Help menu or instructions shortcut. |
| Recorded operations | **File → Run reports…** (also available above the log) opens the run/report location. The log prints full job paths. |
| Default run files | `<application-root>/MultiAxisPrinting-dev/data/sessions/<session-id>/runs/<run-id>/`: `report.html`, `run.log`, `run.json` and the `work/` folder. Older versions may place job folders directly under `data/` or `data/sessions/<session-id>/`. |
| Fields, layers, paths and height reports | Inside the corresponding job's work/output folders. A custom **Output folder** takes priority; follow the actual full path printed in the interface log. |
| Saved projects and final exports | The destination selected in the relevant Save/Export dialog. A project save is separate from a job's automatic working outputs. |
| Windows/Mac/Linux bootstrap setup log | `<installation-base>/setup-*.log` |
| Windows/Mac/Linux shortcut launch log | `<application-root>/.runtime/launch.log` |
| Selected environment and bootstrap check result | `<application-root>/.runtime/selection.json` and `.runtime/bootstrap-result.json` (the latter is produced by bootstrap setup). |

Open the exact run report when locating an output or diagnosing a failure;
do not assume the most recently modified version folder is the active one.
Keep existing projects and referenced job files when changing installations.

## Offline MAP source and verification

Download the exact backend ZIP named by the extracted interface's
`MAP_BACKEND_LOCK.json`. From the interface application root, with Python available:

```text
python map_backend.py --archive /path/to/MAP-backends-2026-11-27-r2.zip
```

Then run setup normally. This supplies backend source offline; Python/dependency
installation can still need network access unless its packages are already
available locally. Full development checkouts can use their included backends.

Check the release's downloads against `SHA256SUMS.txt`, and read
`VERIFICATION.json` for tested platforms and installation limits. Use the named
`application.zip` and its locked MAP ZIP for the matching source; automatic
GitHub tag-source archives can contain a different snapshot.

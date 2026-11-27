# Install, launch and locate MultiAxisPrinting

Download the [Windows, macOS or Linux/source package](https://github.com/RyanTaoLiu/MultiAxisPrinting/releases/tag/interface-preview-2026-11-27-r2).
The package installs one interface with five backend component folders. The
backends are modules used by the interface, not five separate desktop applications.

## What setup downloads and where it comes from

| Component | Source | Installed location |
|---|---|---|
| Interface, starter models and offline wiki/course pages | The named MultiAxisPrinting release above: embedded in the Windows EXE, or in `payload/application.zip` inside the complete Mac setup ZIP. Linux uses the separate `application.zip`. | The application root, including `MultiAxisPrinting-dev/`, `docs/` and `course_reporting/`. |
| MAP algorithms for fields, supports, toolpaths, heights and printer assembly | The exact [MAP backend release](https://github.com/neelotpal-d/MAP/releases/tag/backends-2026-11-27-r2) specified in `MAP_BACKEND_LOCK.json`. Setup checks the archive size, SHA256 and every member. | The five folders listed below, directly inside the application root. |
| Miniforge, used to create/manage the Python environment | Official [conda-forge/Miniforge releases](https://github.com/conda-forge/miniforge/releases/tag/26.9.4-2). The bootstrap selects the OS/CPU-specific installer and checks its pinned SHA256. | Windows/Mac bootstrap: `<installation-base>/managers/miniforge-<version>-<attempt>/`. Source setup uses your existing Conda/Mamba installation. |
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

**Installation base** means the parent directory used by the Windows/Mac
bootstrap. **Application root** means the versioned directory containing
`launch.py`, `setup.py`, `MultiAxisPrinting-dev/` and the backend folders.
The downloaded installer and the launcher shortcut are separate from this root.

| Platform and installation route | Installation base | Application root |
|---|---|---|
| Windows bootstrap | `%LOCALAPPDATA%\MultiAxisPrinting` by default, or the writable destination selected in setup | `<installation-base>\versions\<release-id>\` |
| Mac bootstrap | `~/.local/share/multiaxisprinting` by default; `MULTIAXIS_INSTALL_DIR` can select a different base when running setup | `<installation-base>/versions/<release-id>/`; `current-app.txt` records the active root |
| Linux/source ZIP, or manually extracted source on any OS | The folder you choose when extracting the package | That extracted folder, containing `launch.py` |

Windows/Mac bootstrap runtimes normally use
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

### Linux and manual/source installation

Use a native-architecture Miniforge/Conda terminal in the extracted folder and
follow `INSTALL.md` from `application.zip`. With a working Conda/Mamba manager
and Python available, run:

```text
python setup.py --device auto
python launch.py --installed-runtime
```

For subsequent launches, return to the application root and run the second
command. `--installed-runtime` uses the recorded application interpreter. If
you deliberately use your own complete compatible environment instead, activate
it and run `python launch.py`. Linux/source setup does not create a desktop
shortcut automatically. A graphical Linux bootstrap installer is not supplied.

Keep an installed Conda runtime at its original path; do not move it between
application folders, computers or operating systems. For download/security
prompts and supported desktop requirements, see the package's
`DOWNLOAD_SECURITY.md` and `INSTALL.md`.

## Find models, generated results, saved projects and logs

| Item | Location or action |
|---|---|
| Starter meshes | `<application-root>/MultiAxisPrinting-dev/data/mesh/`; see `TEST_MODELS.md` in the package. |
| Wiki and course guide | `<application-root>/docs/index.html` and `<application-root>/course_reporting/web/instructions.html`, also accessible through the interface's Help menu or instructions shortcut. |
| Recorded operations | **File → Run reports…** (also available above the log) opens the run/report location. The log prints full job paths. |
| Default run files | `<application-root>/MultiAxisPrinting-dev/data/sessions/<session-id>/runs/<run-id>/`: `report.html`, `run.log`, `run.json` and the `work/` folder. Older versions may place job folders directly under `data/` or `data/sessions/<session-id>/`. |
| Fields, layers, paths and height reports | Inside the corresponding job's work/output folders. A custom **Output folder** takes priority; follow the actual full path printed in the interface log. |
| Saved projects and final exports | The destination selected in the relevant Save/Export dialog. A project save is separate from a job's automatic working outputs. |
| Windows/Mac bootstrap setup log | `<installation-base>/setup-*.log` |
| Windows/Mac shortcut launch log | `<application-root>/.runtime/launch.log` |
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

# Armor Penetration Lab

An experimental smoothed-particle hydrodynamics (SPH) code for simulating projectile impact on armor. The C++17/OpenMP solver builds in 2D or 3D from one source file. A Tkinter control GUI, a PySide6 particle viewer, and optional HDF5/XDMF, VTK-surface, and MP4 export tools are included.

> **Research software, not validated engineering software.** The constitutive models and material parameters are simplified and have not been independently validated or calibrated here. Do not use results as a basis for real-world armor design, safety decisions, or certification. See [Model scope and limitations](#model-scope-and-limitations).

## Features

- One C++17 solver source for 2D and 3D builds, with OpenMP parallelism.
- Hybrid Total-Lagrangian / Updated-Lagrangian SPH mode.
- Johnson–Cook plasticity and D1–D5 damage parameters, plus a Mie–Grüneisen equation of state.
- Particle output in `frames.bin` and `meta.txt`; optional legacy VTK output for ParaView.
- Tkinter GUI for setting up and launching runs, browsing frames, running velocity sweeps, and estimating V50 by bisection.
- PySide6 viewer with a rotatable 3D camera, slicing, scalar-field display, particle inspection, playback, and PNG export.
- Optional converters for HDF5/XDMF, smoothed VTK surfaces, and MP4 animations.

## Repository layout

| File | Purpose |
|---|---|
| `sph_core2.cpp` | Shared 2D/3D C++ solver source |
| `gui2.py` | Tkinter run-control GUI, 2D/3D frame reader and CPU renderer |
| `viewer_qt.py` | Interactive PySide6 particle viewer |
| `convert_hdf5.py` | Convert `frames.bin` runs to HDF5 and XDMF |
| `surface_export.py` | Export a smoothed isosurface from an HDF5 frame as `.vtp` |
| `render_video.py` | Render a run as an MP4 animation |
| `build_windows.bat` | Build the 2D and 3D solver executables on Windows |
| `run_gui.bat` | Launch the Tkinter GUI on Windows |
| `requirements.txt` | Core Python dependency (NumPy) |
| `requirements-viewer.txt` | Dependencies for all optional viewer/export tools |
| `requirements-test.txt` | Python test dependencies, including HDF5 conversion tests |
| `tests/` | Python unit tests for frame parsing, rendering, and HDF5 conversion |
| `.github/workflows/ci.yml` | Build both solver dimensions and run end-to-end smoke tests on GitHub |

Compiled solver binaries and generated simulation outputs are not intended for distribution. The solver's local `summary.csv` run log is ignored by Git. Build the binaries for your machine using the instructions below.

## Requirements

- C++17 compiler with OpenMP support (GCC is the primary tested toolchain).
- Python 3.10 or newer for the Python tools.
- NumPy for the Tkinter GUI and frame-rendering utilities.
- Tkinter for `gui2.py` (often installed separately on Linux, for example with `python3-tk`).
- PySide6 for `viewer_qt.py`.
- Optional export dependencies: `h5py`, `vtk`, `imageio`, and `imageio-ffmpeg`.

### Linux / WSL build

From the repository root:

```bash
# Build the 2D solver (default DIM=2).
g++ -O3 -fopenmp -std=c++17 sph_core2.cpp -o sph_core2

# Build the 3D solver.
g++ -O3 -fopenmp -std=c++17 -DDIM=3 sph_core2.cpp -o sph_core3

# Optional: install the base Python dependency and launch the GUI.
python3 -m pip install -r requirements.txt
python3 gui2.py
```

You may add `-march=native` for a local build. Such a binary may not run on older or different CPU architectures, so omit that flag when building portable artifacts.

### Windows build

Install Python and an MSYS2/MinGW-w64 GCC toolchain with OpenMP support, and make `g++` available in `PATH`. From the project directory, run:

```bat
build_windows.bat
run_gui.bat
```

The GUI selects the matching 2D or 3D executable. On Windows, if the Windows executable is absent, the GUI can also use a Linux solver through WSL when one is available.

### Python dependencies

Install the dependencies for every viewer/export utility with:

```bash
python -m pip install -r requirements-viewer.txt
```

For a smaller installation, start with `requirements.txt` and install only the optional packages you need:

```bash
python -m pip install -r requirements.txt
python -m pip install PySide6                 # Qt viewer
python -m pip install h5py                    # HDF5/XDMF conversion
python -m pip install vtk h5py                # VTK surface export
python -m pip install imageio imageio-ffmpeg  # MP4 export
```

## Run a simulation

The command-line solver accepts `key=value` arguments. Example 2D and 3D runs:

```bash
./sph_core2 v=800 angle=60 t_plate=15 proj_mat=steel arm_mat=rha \
    nose=flat dx=0.5 t_end=10 frames=20 out=run_2d

./sph_core3 v=800 angle=0 t_plate=6 h_plate=50 w_plate=30 \
    proj_l=18 proj_d=5 dx=0.8 t_end=10 frames=20 out=run_3d
```

The example dimensions and times are for a quick demonstration, not a recommended physical setup. Smaller `dx` and larger 3D geometries can increase run time and memory use substantially.

### Command-line parameters

| Parameter | Meaning | Units / accepted values |
|---|---|---|
| `v` | Impact speed | m/s |
| `angle` | Plate obliquity; `0` is normal incidence | degrees |
| `t_plate` | Plate thickness | mm |
| `h_plate` | Plate extent in the impact plane | mm |
| `w_plate` | Plate width along z (3D only) | mm |
| `proj_l`, `proj_d` | Projectile length and diameter | mm |
| `dx` | Initial particle spacing | mm |
| `t_end` | End time | μs |
| `frames` | Requested number of output frames | integer |
| `proj_mat`, `arm_mat` | Projectile and armor materials | `steel`, `tungsten`, `rha`, `hha`, `aluminum` |
| `nose` | Projectile nose shape | `flat`, `ogive`, `pointed` |
| `out` | Output directory | path |
| `tl` | Hybrid TL/UL switch | `1` on, `0` off |
| `cfl` | Courant factor | positive number |
| `xs` | XSPH smoothing coefficient | `0` disables it |
| `as` | Artificial-stress coefficient | `0` disables it |
| `sw` | TL-to-UL strain-rate threshold | 1/s |
| `vtk` | Write legacy VTK particle frames | `1` on, `0` off |
| `z` | Reorder particles by Morton/Z-order every N steps | `0` disables it |
| `prof` | Print solver-pass timing information | `1` on, `0` off |

The solver prints progress lines and a final record in the form:

```text
RESULT verdict=PENETRATION|RICOCHET|EMBEDDED vres=... angle_out=... intact=... tl=...
```

Each run directory contains `meta.txt` and `frames.bin`; `out/vtk/` is also created when `vtk=1`. A `summary.csv` row is appended in the solver's current working directory.

## View and export results

### Tkinter GUI

```bash
python gui2.py
```

Use the controls to configure a run, select 2D or 3D, launch the solver, and review the scalar-field animation. The 3D canvas supports rotation, panning, zooming, z-slicing, and preset views. The GUI also provides a velocity sweep, an approximate V50 bisection search, a residual-speed plot, VTK output, and a headless self-test:

```bash
python gui2.py --selftest run_2d
```

### Qt particle viewer

```bash
python viewer_qt.py run_3d
```

The Qt viewer reads `meta.txt` and `frames.bin` directly and loads one frame at a time. It supports playback, scalar fields, camera presets, mouse rotation/panning, wheel zoom, z-slicing, click-to-inspect particle values, and PNG export. The viewer is CPU-rendered and can be slow for large particle counts. It currently reads `frames.bin`; it does not open HDF5 files directly.

### HDF5 and XDMF

```bash
python convert_hdf5.py run_3d run_3d/results.h5
```

This creates `results.h5` and `results.h5.xdmf` for scientific visualization tools that support XDMF/HDF5.

### Smoothed VTK surface

```bash
python surface_export.py run_3d/results.h5 10 run_3d/surface_0010.vtp --field damage
```

The frame index is zero-based. The generated surface is a Gaussian-splatter isosurface intended for visualization; it is not an exact material boundary. Available scalar fields include `damage`, `speed`, `pressure`, `temperature`, `epsp`, and `svm`.

### MP4 animation

```bash
python render_video.py run_3d run_3d/animation.mp4 damage 25
```

The optional arguments are the field name and frames per second. The renderer supports `damage`, `speed`, `pressure`, `temperature`, `plastic`, and `mises` (or raw field keys such as `dmg`, `spd`, `P`, `T`, `epsp`, and `svm`).

## Output format and units

Each `frames.bin` record is little-endian `float32` data in field-major order:

```text
[time][X for N particles][Y for N particles][Z for N particles, 3D only]
      [speed][damage][pressure][temperature][plastic strain][von Mises stress]
```

`time` is stored in seconds. Coordinates are stored in meters; speed is in m/s; damage is dimensionless; pressure and von Mises stress are stored in Pa; temperature is in K; plastic strain is dimensionless. The GUIs display pressure and von Mises stress in GPa. The 2D format has 8 fields and the 3D format has 9 fields. `meta.txt` records the dimension, particle counts, verdict, and run parameters.

## Model scope and limitations

The solver implements an experimental hybrid SPH formulation with elastic stress evolution, a Johnson–Cook-style plasticity/damage model, a Mie–Grüneisen-style equation of state, artificial viscosity/stress, and optional XSPH smoothing. These implementation details should not be mistaken for demonstrated predictive accuracy.

- The 2D build represents plane strain; it is not equivalent to a 3D projectile and armor geometry.
- Material constants are illustrative defaults and require material-specific calibration.
- No independent validation, mesh/particle convergence study, or cross-code benchmark is supplied with this repository.
- Energy and momentum conservation have not been established as acceptance criteria for the current solver.
- Melting, phase change, and adiabatic shear-band physics are not modeled.
- The TL-to-UL transition is abrupt and may introduce local numerical artifacts.
- The V50 search reports the threshold implied by this solver's verdict rule; it is not a certified ballistic limit.
- Large 3D runs can require substantial memory and CPU time.

Use the software for exploration, teaching, and code development only until it has been independently verified for a specific application.

## Tests

Install the test dependencies and run the Python unit tests from the repository root:

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The GitHub Actions workflow also compiles both solver dimensions and runs small 2D/3D end-to-end smoke simulations. These checks verify buildability and file-format/rendering plumbing; they do **not** validate the physical model.

## License

No license has been selected or included yet. Until the maintainer adds a license, the source remains under its default copyright status; public availability on GitHub does not by itself grant permission to reuse or redistribute it. Add a license before inviting external reuse.

# Bowstring arm: notes for Claude Code

Simulation, control and a live web console for a three-segment tendon-driven elastica arm whose
tendons run as straight strings between disks (the "bowstring" arm). README.md is the dated
lab notebook: newest sections first, with every result and how it was produced. PLAN.md is the
original plan. PRODUCT.md describes the console.

## Setting up on a new machine

The project is a public GitHub repo (`grraxx/bowstring_arm`, since 2026-11-24), cloned on each
machine, to run the console and its Live tab (keyboard and gamepad driving). What's needed:

1. **Python 3.10+ with numpy and scipy, plus a C compiler.** Either:
   - `conda env create -f environment.yml && conda activate bowstring` (includes a C compiler), or
   - `pip install -r requirements.txt` with gcc or clang on PATH (macOS:
     `xcode-select --install`).

   The two hot loops (`bowstring/_rk4.c`, `bowstring/_chain.c`) are compiled automatically on
   first import into `bowstring/_build/`. The compiler search order is $CC, gcc, cc, clang. If
   none works there is a RuntimeWarning and a pure-Python fallback, which is far too slow for
   real time, so fix the compiler instead of ignoring it.
2. **Check it:** `python tests/test_realtime.py` (about 30 s). It confirms the C library matches
   the Python reference, the implicit plant matches RK4, and the closed loop runs faster than real
   time. On the original machine it ran at about 2.7× real time.
3. **Run the console:** `bash console/run.sh` (some unzip tools drop the executable bit), then
   open http://localhost:8765.
   - Interpreter: `BOWSTRING_PYTHON=/path/to/python` overrides the default choice.
   - **Live** tab → Start.
   - Keys: WASD or the arrows move, Q and E turn, Esc stops.
   - **Gamepad:** uses the browser's standard Gamepad API (Chrome or Edge), no drivers. Put the
     8BitDo in XInput (X) mode and press any button once with the page open. Either stick moves
     the payload, LT and RT turn it, Start toggles the session.

**Platform:**
- Linux and macOS work natively.
- On Windows, run the project in **WSL2**. `run.sh` is bash and the build makes `.so` files. Open
  the page in the Windows browser (WSL forwards localhost). The browser reads the gamepad, so no
  USB passthrough to WSL is needed.
- Native Windows would need MSVC or MinGW changes in `segment.build_shared` and a replacement for
  `run.sh`. Not done.

**Optional: SOFA plants** (independent test simulators; not needed for the model plant or the
Live tab, which refuses SOFA as slower than real time):
- A conda env `sofa-env` (Python 3.12, `sofa-python3` 25.12 from conda-forge).
- SoftRobots v25.12 and the Cosserat plugin v25.12 built from source into
  `~/opt/SoftRobots/install` and `~/opt/Cosserat/install` (cmake with the conda compilers).
- Other locations: set `SOFA_ROOT` and `SOFA_PLUGIN_PATH`, see `sofa/scene.py`.
- `run.sh` uses `~/miniforge3/envs/sofa-env` automatically if it exists.

**Not in git** (see `.gitignore`): `data/`, about 6 GB of scan outputs, regenerable with
`scripts/` and `fenics/` (FEniCSx, a separate environment); the README says how. Copy by hand
what a machine needs. The Till et al. PDF is also left out, since the repo is public.
- The console creates `data/console/runs/` itself.
- The clamped-elastica table `data/segment_table.npz` was deleted on 2026-11-24 as no longer used
  by the project. `bowstring/table.py`, `tests/test_chain.py` check 2 and
  `scripts/scan_failures.py` still read it and fail without it; `fenics/merge_table.py` rebuilds it.
- The plotting scripts need their scans.

## Project facts to keep in mind

- **Units are non-dimensional:** lengths in segment lengths L, bending stiffness in EI₀, forces in
  EI₀/L². Time stays in seconds, because the massless arm has no time scale of its own.
- **A tendon crossing the backbone is physical by construction.** The backbone is two stacked
  strips with a gap, and the tendons pass between them. Do not flag rod–tendon "contact" as
  invalid. The README section on contact limits is marked superseded. Coinciding holes, at
  segment bends near 173°, are still a model singularity.
- **Speed:** the model plant is stepped implicitly (BDF-α, Till et al. 2019, the PDF in the root),
  with the chain evaluation in C. See README "Real-time solver". "explicit RK4" is kept as the
  reference.
- **Working report:** a journal-style report on the arm lives in a Claude Doc (bowstring arm
  working report). Edit it rather than starting a new one.

## How the user likes to work

- **Keep changes lean.** An earlier full execution of PLAN.md was thrown away as overcomplicated.
  Ask for the minimal deliverable first, and don't fan out sweeps, extra solvers or designed
  pages unless asked.
- **When a result looks weak**, trace it to the binding constraint (a modelling choice, a limit, a
  geometry parameter) and report that with numbers before shrinking demos to fit it.
- **Report outcomes plainly** with the numbers. Write simple language and no fluff.

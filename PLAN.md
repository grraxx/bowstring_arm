# Plan: online tracking with a bowstring tendon-driven elastica arm

## Goal

An **online, open-loop controller** for the arm in `robot_math_28786.pdf` (ME246). The arm has n = 3
elastica segments (n = 4 is out of scope for now). Its tendons are straight strings between disks, and it carries a payload.

Each control step, the controller takes the reference pose of the payload, X*, with Ẋ* and Ẍ*, and
returns the tendon tensions T. It works from the model alone and has no measurement of the arm. These
arms are meant for tight spaces, where a sensor on the tip is unlikely to fit. Nothing is precomputed
for a particular trajectory: the controller must keep up with a reference it sees for the first time.

Demos:
1. **Trajectory:** a circle and a line with given timing.
2. **Path:** a geometric path whose speed the controller picks online, as fast as the tension limits allow.
3. **Live input:** the user drives the payload with a gamepad (8BitDo Ultimate 2C Wireless), and the arm
   follows. The right stick sets the direction and speed of motion in x, y. The triggers rotate the
   payload clockwise and counterclockwise.

## Decisions

- **Self-contained project** in `~/bowstring_arm/`; nothing reads `~/elastica` any more. `fenics/` holds
  copies of the elastica solver, and `data/production/fields.npz` holds the production fields.
- **Open loop.** The controller sees only the reference. The arm's own stiffness pulls the payload back
  towards the reference, which is why open loop can work at all. Closed loop is out of scope.
- **Fully online.** No offline schedules, so the AUTO-07p feedforward is dropped. The only offline work
  describes the arm, not any trajectory: the segment table and the workspace map.
- **No tension limits.** Minimal co-contraction: the slack side keeps a small pretension, and there
  is no upper limit. Feasibility is stability and reach only.
- **Segments by shooting** (changed 2026-11-22, see step 3). The FEniCSx table gives the energy
  accurately but not the axial load of a nearly straight segment, which is the arm's usual state.
  So the chain integrates each segment directly. The table stays as a reference. No ML surrogate.
- **No no-contact condition.** Tendons are straight chords whether or not they cross the rod.
- **SOFA is the test plant.** It is an independent simulation of the arm, so it shows whether the
  controller works on mechanics it was not built from.
- **Environment:**
  - The arm code needs only numpy and scipy, and runs in `elastica-env`.
  - SOFA gets its own environment. Installing it needs the user's approval.
  - **The gamepad is read on Windows, not in WSL.** WSL2 does not see USB devices, and its kernel may
    lack the joystick drivers.
    - A small Windows Python script (`scripts/gamepad_bridge.py`, pygame) reads the controller through
      its 2.4 GHz dongle or a USB-C cable.
    - It sends stick and button states about 100 times a second over a local TCP socket to the demo
      running in WSL.
    - Installing pygame on Windows needs the user's approval.
    - Fallback: attach the dongle to WSL with usbipd-win, if the WSL kernel exposes it as a joystick.

Units: segment length ℓ = 1 and bending stiffness B = 1. Forces are in B/ℓ², moments in B/ℓ.

## Layout
```
bowstring/   segment.py table.py tendons.py chain.py controller.py reference.py plant.py
             rod.py (existing: the whole arm as one discretised rod, used as a cross-check)
fenics/      FEniCSx solver copies, layers.py, merge_table.py (done)
sofa/        scene.py, bridge.py, run_sofa.py
scripts/     experiments.py, make_figures.py, live_demo.py, gamepad_bridge.py (runs on Windows)
data/        segment_table.npz, runs/, sofa/          (not in git)
figures/
tests/
```

## Steps

### 1. One segment by shooting: `bowstring/segment.py` — DONE (2026-11-22)
- Given the pose of a segment's end relative to its start, find the segment's shape and loads.
- Unknowns: the base load (nx, ny, M0).
- Integrate the elastica with RK4, about 200 steps. Newton matches the end pose, warm-started from the
  last solution so it stays on the same branch.
- Returns:
  - the energy V;
  - the end wrench g = ∂V/∂pose;
  - the stiffness K = ∂g/∂pose;
  - the branch (sign of p);
  - stability (`fenics/stability.py`).
- The chain uses only the forward integration (load → pose, wrench and their derivatives). Matching
  a given pose is badly conditioned when nearly straight, so it falls back to continuation.

### 2. Segment table: `fenics/` — DONE (2026-11-22)
- **What it is:** `data/segment_table.npz`, the up branch on 66 layers, d 0.70–0.9999.
  - Production supplies d 0.70–0.96.
  - New runs supply d 0.970–0.9999 (1000-element beam).
  - It is computed only up to U = 10; the arm reaches U ≤ 3.5.
- **How it was made and checked:** see the README.

### 3. Reading the table: `bowstring/table.py` — DONE (2026-11-22), not used by the chain
- **Interpolation:**
  - bicubic in (φ1, φ2) within a layer;
  - across layers, cubic Hermite in σ = √(1 − d) with the exact slopes the solver stores
    (∂U/∂d = −fx, and ∂M/∂d from symmetry).
- **Finding:** V is accurate, but the axial load is not, near shallow arcs.
  - A nearly straight segment's chord stiffness is huge: at curvature 0.06, about 25 load units
    separate neighbouring layers.
  - So g is off by up to a few % there, and that is the arm's normal state.
  - Hence the chain uses shooting. The table remains as a reference (accuracy in the test log).

### 4. Tendons: `bowstring/tendons.py` — DONE (2026-11-22)
- Tendon lengths from the disk poses, and their first and second derivatives.
- Taken from the tendon code in `bowstring/rod.py`, rewritten in terms of the disk poses.

### 5. Chain arm: `bowstring/chain.py` — DONE (2026-11-22)
- The arm as n segments joined at disks. Its energy is the segment energies plus Σ T·tendon length.
- **The unknowns are the segment base loads**, not the disk poses.
  - Each segment is integrated forward from its load, and the disk poses follow.
  - The equations are the disk equilibria. Poses as unknowns stalled on nearly straight segments.
- **Forward map** (X, T) → w: the payload wrench, with the intermediate disks at equilibrium.
- **Inverse map** (X, w) → T, which is what the controller uses:
  - n = 3: a square Newton solve with fixed pair totals, about 0.03 s.
- **Stability:** λ_min of the Hessian with the payload held, plus each segment's clamped stability.
- **Rest:** starts from a coarse `rod.py` solution.

### 6. Online controller: `bowstring/controller.py`, `bowstring/reference.py` — DONE (2026-11-22)
- **Interface:**
  - `X0 = Controller.reset(D0)`: the arm at rest; every reference starts at X0.
  - Then `T, info = Controller.step(t, X*, Ẋ*, Ẍ*)`, or `Controller.choose(t, candidates)`.
  - It never sees the plant.
- **Each step:**
  1. Required wrench w = M·Ẍ*. Quasi-static mode sets w = 0.
  2. Inverse map at X*, warm-started from a linear extrapolation of the last two solutions. A
     solve that needs more than 8 Newton iterations counts as failed.
  3. Send T if the solve converged and the tensions are within limits. On report steps (every
     10th, or every step once the margin is small), the margin must also be > 0 with every
     segment stable.
- **When the request is infeasible:** the reference generators offer candidates in order:
  - path: speed up (adaptive trial acceleration), coast, brake;
  - live: as commanded, brake, hold still.
  The controller sends the first feasible one in the same step, and holds the last T only if
  none is. Live input uses `safe=True`: it moves only to poses where the payload could also be
  held at rest, so stopping is always possible.
- **Reference generators:**
  - `Trajectory.circle`, `Trajectory.line` (quintic start and stop);
  - `PathFollower.line`, `PathFollower.circle` (speed chosen online);
  - `Live` (velocity commands with speed and acceleration limits, deadzone, workspace box).
- **Speed:** 2.4 ms mean, 9.7 ms worst (fallback steps solve several candidates), against the
  10 ms of 100 Hz. The C integrator (`bowstring/_rk4.c`, built with the env's gcc on first use)
  is 43× faster than Python.
- **Found:**
  - With C = (3, 2.5, 2) and Tmin = 0.1, the payload can be held at fixed orientation only within
    about 0.03–0.2 of the rest pose (depending on direction) and ±0.2 rad.
  - At top speed on a tight curve, the path follower still holds T on a few steps: one step of
    lookahead cannot foresee centripetal demand. A curvature-based speed cap would fix it.
- **No tension limits (2026-11-22, the user's decision):**
  - Minimal co-contraction (`ChainArm(scheme="minimal")`): the slack side of each pair sits at
    Tmin = 0.1, and there is no Tmax. Fixed pair totals had capped segment 3's bending at 0.45.
  - Workspace: 13.5 reachable continuously, about 15 over every posture (3.0 before). It is a C
    round the base, not a ring.
  - Demos run end to end of the C. See the README.
- **Large motions (added 2026-11-22):**
  - Orientation-free mode, with a soft angle target from the triggers (relative rotation).
  - A posture switch (bulge up ↔ down) near the arm's axis.
  - An open-loop stability check: payload stiffness K_X > 0.
  - Workspace map: `scripts/workspace_map.py`, a band about 2.2–2.95 from the base.
  - Co-contraction kept at C = (3, 2.5, 2): higher C widens the band but softens the arm, and at
    1.5× it loses payload stiffness.
  - Demos and figures: `scripts/run_demos.py`, `scripts/make_controller_figures.py`
    (`figures/ctrl_*`). See the README.
- Checks: `tests/test_controller.py`.

### 7. Plant and closed loop: `bowstring/closed_loop.py`, `scripts/run_closed_loop.py` — DONE (2026-11-22)
- **Scope change (the user's decision):** a closed loop whose only sensor is an IMU at the tip
  (position, velocity and acceleration estimated from it).
- **Plant:** payload dynamics with the quasi-static arm (forward map) and light damping; RK4 at 1 kHz.
- **Estimator:** IMU strapdown with a model-aided position correction.
- **Controller:** feedforward from the online controller, plus feedback as a damped linear tension
  correction.
- **Results:** open loop diverges on the circle (8.7 s) and the tip-up line (9.8 s). Closed loop
  tracks both with position error below 0.008 and angle error below 0.032 rad.
- Details in the README.
- **Showcase** (`scripts/run_showcase.py`, `scripts/make_showcase.py`): demos for the wrist,
  waiter, drawing, pushes and a speed sweep, plus `figures/showreel.mp4`. Position error is at most
  0.016 (during the kicks) and at most 0.008 otherwise.

### 8. Live demo: `scripts/live_demo.py` with the gamepad
- A window shows the arm. The gamepad drives the payload, and the controller and plant run in real time.
- **Right stick: position, as a velocity command.**
  - Stick direction is the direction of motion in x, y; stick deflection is the speed, up to v_max.
  - The target position is the integral of this velocity.
  - Release the stick and the target stops where it is.
- **Triggers: payload orientation, as a rotation rate.**
  - Right trigger (RT) turns the payload clockwise, left trigger (LT) counterclockwise.
  - How far a trigger is pressed sets the rate, up to ω_max. Pressing both gives their difference.
  - The angle is the integral of this rate, clamped to the reachable range.
  - Release the triggers and the angle holds.
- **The stick and triggers** have a small deadzone. Their commands pass through the step-6 filter, so a
  sudden input never asks for an infeasible acceleration.
- The left stick is unused.
- At the workspace edge, or when the tensions saturate, the target stops moving that way, and the pad
  rumbles if supported.
- **Buttons:**
  - A: back to rest;
  - B: quasi-static or dynamic mode;
  - shoulder buttons (LB, RB): v_max and ω_max down or up;
  - Start: pause.
- **On screen:** the arm, the target, the reference and the payload, tension bars with their limits,
  and λ_min.
- Without a gamepad, the mouse sets the target position, and the scroll wheel the angle.
- Later the same demo drives SOFA instead of the Python plant.

### 9. SOFA test plant: `sofa/` — first pass DONE (2026-11-23)
- **Built:**
  - beams with SoftRobots cables (built from source) and the payload;
  - a static check, converging to the model at first order in mesh size;
  - the closed-loop showcase demos on SOFA.
- **Result:** every run completes. The error is 0.03–0.11 mean against 0.002 on the model plant.
  It comes from IMU-estimate bias caused by SOFA's leftover model mismatch, and halves with each
  mesh halving. Details in the README.
- **Added (2026-11-23):**
  - a SOFA Cosserat plant;
  - tendon-length encoders and a navigation-grade IMU;
  - PID (integral term);
  - the web console (`console/run.sh`), where robustness tests (robot ≠ model, rod mass,
    damping, payload) are now run interactively and as sweeps.
- **Not done yet:** a systematic robustness study (the console makes it a few sweeps).
- **Scene:**
  - a Cosserat (or BeamAdapter) rod of n segments, nearly inextensible, kept planar;
  - rigid disks with tendon holes;
  - one SoftRobots cable per tendon, routed hole to hole (straight chords), driven by force;
  - a rigid payload of mass m and inertia I;
  - light rod mass and damping.
- **Bridge:** a SOFA controller hook calls `Controller.step` at the control rate and writes T to the
  cables. It also logs the payload pose.
- **First:** constant tensions. The settled shape must match `chain.py`/`rod.py`, so that the scene
  and the model agree before any control is run.
- **Then:** the trajectory, path and live runs of steps 7–8, driven open loop by the online controller.
  Compare tracking error with the Python plant.
- **Robustness:** rod mass, damping, payload error.

## Checks (only these)
1. **Derivatives:** g and K from shooting, the table and the chain match finite differences.
2. **Table against shooting:** at random poses, on agreement in V, g and K.
3. **Chain against `rod.py`:** rest shapes and forward-map wrenches agree.
4. **Inverse then forward** gives back the requested wrench.
5. **Speed:** the controller step stays within the time budget.
6. **SOFA statics** match the model before any SOFA tracking run.

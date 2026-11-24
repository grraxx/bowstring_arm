# Bowstring tendon-driven elastica arm

ME246 project (`robot_math_28786.pdf`): trajectory tracking of a payload carried by an
n-segment elastica arm whose tendons run as straight strings between disks. This folder is
separate from `~/elastica` (the clamped-elastica solution space), whose solver and data it
will reuse.

## Text trial: the tip writes ME246 and underlines it, aiming a laser (2026-11-23)

Console trajectory **text** (`traj.preset = "text"`, `bowstring/experiment.py`). Closed loop on the
model plant, default sensing. `figures/trial_text.png` is the last replay frame.

- **Path:** slanted, handwriting-style letters (smooth splines, a few straight strokes for the 4 and
  the E's bar), then a swash from the 6's tail into a wavy underline drawn right to left (5 waves,
  tapered at the ends). The first version used squared block letters; its 6 read as an 8.
- **Pen lifts:** the tip cannot lift, so the hops between strokes are drawn too. Each letter is
  entered on its left edge and left on its right, so the hops stay in the gaps, short slashes.
- **Aiming** (`traj.aim`, `traj.aim_x`, `traj.aim_y`; also offered for circle, figure-eight,
  rectangle and line when the orientation is fixed): the payload angle is
  ψ = atan2(2.5 − y, 1.5 − x), so it always points at (1.5, 2.5). Its first two time derivatives
  are analytic (checked against finite differences to 3e-7), so the controller gets a smooth
  reference. The replay draws a target and a laser as a ray from the payload's front face along
  its *actual* heading. It stops, with a bright spot, only if it passes within 0.02 of the target;
  otherwise it runs on off the view, like a real laser. With the 0.2° angle error it hits (the
  miss at 0.5–0.9 away is under 3e-3).
- **Placement:** the binding constraint is the angle. With the text centred at x = 1.4 the run
  stopped at t = 33 s, tip (1.77, 1.97), ψ = 117.5°: there the arm holds ψ up to 118° but not 125°
  (quasi-static check). Centring at x = 1.2 (text x 0.7–1.76) keeps ψ between 35° and 111°.
- **Size:** `size` is the width of the whole drawing (default 1.0, letters 0.28 tall), centred with
  the underline on (1.2, 1.8).
- **Timing:** every stroke is rest-to-rest, so corners and retraces need no smoothing; stroke time
  follows its length (floor 0.5 s). Default 45 s gives 47.8 s with the two 1 s holds.
- **Result** (one run, ~20 s wall): position error mean 1.0e-3, max 3.5e-3 L; payload angle error
  max 0.21°; no infeasible steps; max tension 7.9; controller 0.96 ms mean.

## Live keyboard drive in the console (2026-11-23)

```bash
console/run.sh            # open http://localhost:8765 → Live tab → Start
```

- **Controls:**
  - W A S D or the arrow keys move the payload.
  - Q and E turn it (fixed orientation only).
  - Esc stops.
  - Speed, turn rate and acceleration are under *Live input* in the panel.
  - **Free orientation** is a switch next to Start. Only x and y are commanded, and the
    controller picks the least-tension angle.
- **Gamepad:** uses the browser's standard Gamepad API, so the Windows browser reads the 8BitDo
  directly (XInput mode). It replaces the planned pygame bridge.
  - Press any button on the pad once so the page sees it.
  - Either stick moves the payload, proportionally. LT turns it counterclockwise and RT
    clockwise. Start starts and stops the session.
  - While the pad is past its deadzone (0.08) it overrides the keys.
  - Commands go out at most every 40 ms when they change, one request at a time so none
    arrives stale.
- **What runs:** the same closed loop as a console run, on the model plant, built from the panel's
  parameters (sensing, controller, robot vs model).
  - `experiment.Session` now holds that loop. `run()` drives it along a preset, and
    `bowstring/live.py` drives it from the keys.
  - It runs on a thread in the server, paced to the wall clock at 100 Hz.
  - The page polls frames at 25 Hz and sends the key state whenever it changes, plus a
    keep-alive every 150 ms.
- **Start pose:** the trajectory centre x, y and angle ψ, held at rest.
- **The reference:** `reference.Live`, the gamepad generator. It limits velocity,
  acceleration and jerk.
  - Each step the feedforward's `choose` (safe mode) takes the commanded move, else braking,
    else holding still.
  - The HUD says when the arm refused a command and why.
- **Safety:** a command older than 0.5 s counts as released. Keys are also released when the tab
  loses focus.
- **Load:** the model plant needs 35–40% of real time on this machine. About 2% of steps finish
  late, and the loop catches up on the next steps.
- **SOFA:** refused in live mode, because it is slower than real time.

## Real-time solver, after Till et al. (2019) (2026-11-23)

```bash
python tests/test_realtime.py     # C = Python evaluate, implicit vs fine RK4, real-time check (~30 s)
```

The paper (`till-et-al-2019-...pdf`, IJRR 38(6)) makes Cosserat-rod dynamics real time in two ways:
- **Implicit time stepping** (its sec. 2.2). Time derivatives are replaced by an implicit
  formula (BDF-α), so each time step is one boundary-value problem in arc length, solved by
  shooting. Large steps stay stable. A rigid tip body enters as the distal boundary condition
  (its sec. 5.2).
- **Compiled residuals** (its sec. 2.7). The per-step cost is the ODE integration, so it is
  written in C++.

**Where the time went before** (figure-eight, model plant, 1 ms plant step, 100 Hz control):
- 0.29× real time.
- 87% of it was the plant. Explicit RK4 takes 5 equilibrium solves per 1 ms step, so 50 per
  control step.
- Each `ChainArm.evaluate` took about 210 µs, mostly Python overhead.
- The controller already fitted its budget, at about 1.5 ms per 10 ms step.

**What changed**
- `bowstring/_chain.c`: `ChainArm.evaluate` (without the Hessian) in C, built on first use like
  `_rk4.c`.
  - It agrees with the Python assembly to 3e-15 and takes 56 µs. Three RK4 segment
    integrations are about 40 µs of that.
  - `arm.use_c = False` goes back to the Python assembly.
- `closed_loop.ImplicitPlant`: the same plant (payload dynamics, quasi-static arm), stepped
  by BDF-α (α = −0.2) as in the paper.
  - The unknowns are the segment base loads.
  - The interior disk equilibria and the payload's M Ẍ + c Ẋ = w + w_ext, with Ẋ and Ẍ from the
    BDF-α formula, form one Newton solve per step.
  - The Jacobian is exact, and L is extrapolated from the last two steps. It takes 1.25 Newton
    iterations per step on average and 3 at most.
- **Gotcha (the paper's sec. 2.4, catastrophic cancellation):**
  - At dt = 1 ms the payload rows contain c₀²M·X ≈ 10⁵, which has to cancel.
  - Nearly straight segments make the first full Newton step overshoot there.
  - So the merit function measures those rows as a position error, ×1e3/(c₀²M + c₀c), as
    `ChainArm.forward` does.
  - With the raw residual, the line search crawled at about 19 iterations per tension update and
    finally stalled.
- The console has *Simulator → Model plant stepping*. It defaults to implicit; "explicit RK4"
  is kept as the reference.

**Speed** (full laps, run one at a time, IMU + tendon encoders; ×real time = simulated / wall)

| Preset | Python + RK4 | C + RK4 | C + implicit | Controller side, mean / max (ms per 10 ms step) |
|---|---|---|---|---|
| Figure-eight | ≈0.3 | 0.91 | **2.8** | 1.5 / 3.1 |
| Circle (free angle) | – | – | **2.1** | 2.5 / 4.2 |
| Rectangle | – | – | **2.7** | 1.5 / 4.9 |
| Wrist | – | – | **2.8** | 1.5 / 3.0 |

"Controller side" is feedforward, feedback and the encoder correction: what a real arm runs at
100 Hz. Running it alone (implicit plant, 1 ms) costs 0.22 ms per plant step.

**Accuracy against RK4**

| Test | Result |
|---|---|
| Payload released at constant tensions at an open-loop-unstable pose (K_X eigenvalue −0.21), 2 s, against RK4 at 0.25 ms | BDF-α at 1 ms: within 1.4e-4 in position and 4.4e-4 rad in angle, over a 0.15 divergence. At 5 ms: 7e-4. At 10 ms: 1.6e-3 |
| Closed loop with perfect sensing | Figure-eight paths within 5e-5 of RK4 (tracking error 0.00124 vs 0.00126). Wrist within 8e-5 |
| Closed loop with IMU + encoders | Paths within 5.5e-4 |

The plants differ more under IMU sensing than with perfect sensing. My explanation, not yet tested:
the estimator's trapezoidal strapdown integration matches BDF-α more closely than RK4. The
tracking errors under IMU sensing go either way (implicit vs RK4): figure-eight 0.00129 vs
0.00146, wrist 0.00044 vs 0.00029. Treat differences of about 3e-4 between the two plants as
numerical.

**Not done** (scope kept to the model plant)
- The SOFA plants are still slower than real time (Cosserat with 20 sections takes about 4 ms per
  1 ms step).
- The paper's method would also give a real-time dynamic plant with rod mass. That would be a
  planar Kirchhoff rod per segment, with the tendons' point loads at the disks as jumps. It
  would be a SOFA alternative written from the same equations.

## Gaps in the scan, and where the model is physically valid (2026-11-23)

> **Superseded (2026-11-23):** a tendon crossing the backbone is physical by construction. The
> backbone is two stacked strips with a gap between them, and the tendons pass through that gap.
> So the "contact" limits below are not validity limits of the arm. The scan results themselves
> stand.

```bash
python scripts/scan_failures.py --n-fail 4000 --workers 8   # classify the gaps -> data/scan_failures.npz
for k in 0 1 2 3; do python scripts/sensitivity_scan.py --Dmax 5 --out data/sensitivity_scan_hi.npz --part $k/4; done
python scripts/sensitivity_scan.py --out data/sensitivity_scan_hi.npz --merge 4
python scripts/plot_sensitivity.py --scan data/sensitivity_scan_hi.npz --tag _taut --clear taut   # or --clear all
```

**Workflow notes**
- `scan_failures.py` loads the elastica table in every worker, about 1 GB each. With 16
  workers it peaked at 15.2 of 15 GB, so use 8.
- Indexing a lazy `.npz` row by row ran out of memory earlier. The scripts now load every array
  once.

**The numerics are not the cause**
- Doubling the RK4 steps moves the payload by about 2e-9.
- Against the clamped-elastica dataset (`~/elastica`, FEniCS, 200 elements), shooting matches to
  8e-6 in energy and 1.2e-4 in end wrench (medians).
- The only larger differences are in almost-straight segments (d > 0.999), where the axial force
  is ill-conditioned for the table's interpolation.

**The 30% of tension sets with no equilibrium** (|D| ≤ 8 scan; 4,000 followed from a straight arm)
- **42%: the scan missed them.** The branch reaches them, the arm is stable at 97% of them, and
  only 10% keep the rod clear of its tendons.
- **57%: a tendon chord collapses.** A segment bends about 173° until two holes of a pair
  coincide. The model's tendon direction is singular there, and physically the disks would have
  collided long before.
- **0.9%:** a true fold (snap-through).
- **0%:** unexplained.

**The model's validity limit is contact, not existence**
- The straight-chord (bowstring) model lets the rod pass through a tendon. A real arm cannot.
- Only 18% of the |D| ≤ 8 configurations keep the rod clear of every tendon. The largest segment
  bend among them is 74°. For d₁ = 0.15, contact starts at about 56°.
- Rescanned at |D| ≤ 5 (2.5× finer, 95% solved), with two rules:

  | Rule | Configurations | Where they reach |
  |---|---|---|
  | Strict: rod clears every tendon | 22,570 | Only the outer ring r ≈ 2–3, ahead of and beside the base (`figures/sensitivity_tasks_valid.png`) |
  | Lenient: rod clears the taut tendons; a slack one (T = Tmin) may rest on it | 38,713 | Also r ≲ 1 around and behind the base, where ψ can turn through about 200° (`figures/sensitivity_tasks_taut.png`) |

- **The hollow ahead of the base (x ≈ 0.5–2.4, r ≈ 1–2):**
  - Under the strict rule it is physically unreachable.
  - Under the lenient rule, only about 20% of the configurations there clear the taut tendons, and
    they need |D| ≈ 7.
- **Inside the valid sets:**
  - The payload is stable without feedback almost everywhere (99.9%).
  - The PD error from a 10% wrong model is roughly a third of the full-scan value. B₁ and B₂ still
    dominate: 0.009 and 0.006 L (strict), 0.016 and 0.013 L (lenient).
- **The console's sample trajectories lie in invalid configurations.** The figure-eight, hold,
  line, rectangle and the wrist start all press the rod through a taut tendon by 0.04–0.22, with
  90–125° segment bends.
  - SOFA routes its cables in straight segments between disks too, so it does not catch this.
  - A physically valid demo needs poses on the outer ring or near the base, or more spacer disks
    per segment (a design change).

## Sensitivity and task capability (2026-11-23)

```bash
python scripts/sensitivity_scan.py     # 40³ tension sets -> data/sensitivity_scan.npz (~10 min, 16 workers)
python scripts/plot_sensitivity.py     # figures/sensitivity_{tasks,params,configs}.png, data/sensitivity_report.md
```

**Method**
- Each tension set D gives one rest equilibrium, so each sample is one configuration holding the
  payload. The sets are a grid of 40 values per tendon pair, spaced more finely near D = 0, with
  |D| up to 8.
- Of 64,000 sets, 44,862 have the arm stable and 39,016 also hold the payload with no feedback.
  The misses are mostly tightly curled arms (|D| > 6) that warm-started continuation does not
  reach.
- At each equilibrium, in closed form from the Hessian:
  - the payload stiffness K_X;
  - the wrench per unit tension change W_D;
  - the encoder Jacobian.
- By finite differences: the wrench error when one of the 7 static parameters is 10% off.
- These become task numbers for the console's default payload and controller (PD, ωn 5 and 15).
- Each map cell shows the best value over the configurations and angles that reach it. The
  parameter maps show the median.

**Check of the linearisation**
- dX/dD matches finite differences to 6 digits.
- The PD error predicted from a 10% wrong B₁ or d₂ matches closed-loop console runs within about
  15% (0.017 vs 0.014, 0.034 vs 0.035, 0.009 vs 0.008, 0.032 vs 0.029) where the scan has a sample
  within 0.1 of the test pose.

**Findings** (medians; `data/sensitivity_report.md` has them by region)
- **Holding without feedback:** the payload is stable open loop at 87% of usable equilibria. It
  is unstable mainly ahead of the base at 1 < x < 2.5. The bandwidth needed there is at most about
  4 rad/s for the default payload, below the default ωn of 5.
- **Reorienting** the payload is best close to the base (x ≈ 0.5) and behind it, up to about 250°
  of ψ. It falls to a few tens of degrees near full reach, where the arm is nearly straight.
- **Pushing** (force per unit tension, about 0.1–0.3) and **resisting a push** are best near the
  base. Both are worst ahead at mid-to-far reach, where the closed-loop stiffness is little more
  than the feedback's own M·ωn² = 2.5.
- **Holding cost:** curled poses need tensions of about 6. Poses on the ring near full reach need
  about 1–3.
- **Precision depends on knowing the stiffness.**
  - With PD, a model 10% off in all parameters at once gives 0.03–0.08 L of position error.
  - B₁ and B₂ dominate, at 0.044 and 0.037 L per 10%. Next come B₃ (0.009), the hole offsets
    (0.003–0.006) and Tmin (0.002).
  - The integral term removes this static error, within its clamp. Calibrating B₁ and B₂ is what
    matters for transients and for PD.
- **Tendon-length sensing:** the pose error per unit length error is about 4–6. It is worst at
  full reach and ahead.
- **Configurations:** mostly one shape per pose within ±15° of ψ. Two or three shapes appear near
  the base and along the outer rim.

## Console, Cosserat plant, integral action, tendon encoders (2026-11-23)

```bash
console/run.sh            # then open http://localhost:8765 (Windows browser works too)
```

**Console** (`console/`: `server.py` uses the standard library only, plus a static page)
- **Parameters:** every robot, model, controller, sensing and simulator parameter, grouped,
  each with a reset.
- **Robot vs model:** the true robot and the controller's model can differ (untick "Same as
  robot").
- **Sample trajectories:** circle, figure-eight, rounded rectangle, wrist, line, hold, each with
  optional pushes.
- **Runs:** queued and run in parallel as background processes, kept on disk under
  `data/console/runs/`.
- **Views:**
  - Replay: arm animation, error, angle, tensions and integral wrench, all on a shared time
    cursor.
  - Live: selecting a running run shows it as it computes, from about 3 chunks a second that
    the engine appends to `live.jsonl`. It follows the latest frame. Scrubbing back pauses
    following, and "Back to live" resumes it.
  - Failed runs replay too, with the arm captured at the moment of failure. They open 3 s
    before it.
  - Compare: tick runs to overlay their paths and errors; a table lists the parameters that
    differ.
  - Sweeps: one parameter over a range, plotting a summary metric against it.
- **Engine:** `bowstring/experiment.py`, which is also usable alone as
  `python -m bowstring.experiment <dir>` with a `config.json` in `<dir>`.

**Tendon-length encoders** (motor encoders; the control stays force-driven)
- The estimate is a complementary filter: the IMU gives fast motion, and the encoders give the
  absolute pose through the model's interior equilibrium.
- On SOFA beams (160 elements per segment) the mean error fell from 0.08 (IMU only) to 0.005.
- The IMU is a navigation-grade unit, with specs in physical units at a segment length of 0.2 m.
  Other grades can be chosen in the console.

**Units in the console** (non-dimensional; every field shows its unit, and the key sits at the
top of the panel)
- Lengths are in the segment length L. Stiffness is in the reference EI₀ (B = EI/EI₀).
- Forces and tensions are in EI₀/L², moments in EI₀/L.
- Time stays in seconds, since the massless arm has no time scale of its own. So mass is in
  EI₀s²/L³ and payload inertia in EI₀s²/L.

**Tendon encoders only** (sensing mode, no IMU)
- Each control step:
  - The pose is predicted at constant velocity.
  - The tendon-length pose fix corrects it through an alpha-beta filter of bandwidth ω (default
    60 rad/s).
- Figure-eight on the model plant:

  | Filter ω (rad/s) | Mean error |
  |---|---|
  | 20 | 0.21 (nearly lost) |
  | 40 | 0.0026 |
  | 60 | 0.0018 |
  | 80 | 0.0014 |

  IMU + encoders gives 0.0015. Below about 40 rad/s the filter lag destabilises the 15 rad/s
  angle loop. Above about 80 rad/s the filter itself is unstable at 100 Hz.
- **On SOFA beams with the default rod mass (0.015) it loses tracking.**
  - The disks vibrate, but the pose fix assumes they are at equilibrium, so it reads the
    vibration as tip motion. The 60 rad/s filter passes that into the loop.
  - With rod mass 0.0005 it tracks at 0.0036 (IMU + encoders: 0.0030).
  - At ω = 20 with ωₙ 3 (x, y) and 5 (ψ) it tracks at 0.015.
  - The IMU is what makes the encoder estimate robust to the arm's own dynamics.

**Integral action** (PID in wrench space)
- K_i = K_p·ω_i with ω_i = 0.2 ωₙ, clamped against wind-up, and it stops accumulating while the
  tension correction is saturated.
- Figure-eight on SOFA beams (160 elements): mean error 0.0072 → 0.0030, max 0.016 → 0.007.
- On the model plant it makes no difference (0.0014–0.0015), as expected.

**SOFA Cosserat plant** (`sofa/cosserat_scene.py`, plugin v25.12 built into `~/opt/Cosserat`)
- **Model:** constant-curvature sections with only in-plane bending, so the rod is exactly
  inextensible and unshearable, like the elastica. Per-segment stiffness is supported.
- **Tip moment:** gives the exact arc.
- **Constant tensions:** it converges to the model only at first order in section size, like the
  beam elements.

  | Sections per segment | Tip drift | Model force at SOFA's pose | Time per step |
  |---|---|---|---|
  | 10 | 0.026 | 0.011 | 1.3 ms |
  | 20 | 0.011 | 0.006 | 4.0 ms |
  | 40 | 0.005 | 0.003 | 17.7 ms |

- **Interior error:** the interior disks are off by 0.5° at 10 sections. Through the encoders
  this becomes a pose-estimate error of 0.017 (0.009 at 20 sections), so on Cosserat the
  encoder estimate is worse than on beams with 160 elements.
- **Velocity bug:** the plugin's mapped velocities disagree with its positions, so the tip
  kinematics (IMU) are taken from positions.
- **Start:** SOFA runs start by holding the payload at the start pose while the arm settles
  (1.5 s), then release it.
- **Figure-eight (16 s), 10 sections:**

  | Sensing | Mean error |
  |---|---|
  | Perfect state | 0.0066 |
  | IMU + encoders | 0.025 |
  | IMU only | 0.145 |

- **Verdict:** the beam scene is currently the more accurate SOFA plant per unit of compute.
  Both are offered in the console.

## SOFA plant (2026-11-23, PLAN step 9)

```bash
# setup (once): conda env sofa-env (python 3.12, sofa-python3 25.12 from conda-forge);
# SoftRobots v25.12 built from source into ~/opt/SoftRobots/install (cmake + conda compilers)
python sofa/check_static.py      # constant tensions: SOFA vs the model, mesh convergence
python sofa/run_sofa.py          # the showcase demos, closed loop, on SOFA -> data/runs/sofa_*.npz
python sofa/plot_sofa.py         # figures/sofa_demos.png, figures/sofa_mesh.png
```

**Scene (`sofa/scene.py`)**
- Corotational beams (`BeamFEMForceField`) with EI = 1 and EA = 10⁴, clamped at the base and kept
  in the plane.
- One SoftRobots `CableConstraint` per tendon, in force mode, routed hole to hole.
- The payload is a rigid mass on the tip. The rod's total mass is 0.015, and Rayleigh mass damping
  0.1 equals the payload damping the controller assumes.
- dt = 1 ms.
- **Gotcha:** SoftRobots' force mode imposes the constraint *impulse*, so a tension T must be sent
  as T·dt. Without this the cables act 1000× too strongly.

**Checks**
- The beam alone under a tip moment matches the exact arc to 10⁻⁴.
- At constant tensions, SOFA converges to the model at first order in the element size. Settling
  from the model's rest shape, the model's payload wrench at SOFA's pose is:

  | Elements per segment | Upright rest | Circle top |
  |---|---|---|
  | 20 | 0.039 | 0.049 |
  | 40 | 0.021 | 0.027 |
  | 80 | 0.011 | 0.015 |
  | 160 | 0.006 | 0.008 |

  The tip drifts much more (0.04–0.07 at 160 elements), because these poses are soft in one
  direction (K_X ≈ 0.08).

**Closed loop on SOFA** (160 elements per segment, 1.7 ms per step). All runs completed.

| Demo | SOFA error mean / max | Model plant mean / max |
|---|---|---|
| Wrist | 0.095 / 0.26 | 0.003 / 0.007 |
| Waiter | 0.091 / 0.28 | 0.002 / 0.006 |
| Figure-eight | 0.081 / 0.28 | 0.002 / 0.005 |
| Pushes | 0.109 / 0.31 | 0.002 / 0.016 |
| Circle, 24 s lap | 0.072 / 0.31 | 0.001 / 0.004 |
| Circle, 16 s lap | 0.065 / 0.25 | 0.002 / 0.005 |
| Circle, 10 s lap | 0.066 / 0.21 | 0.003 / 0.007 |
| Circle, 6 s lap | 0.026 / 0.08 | 0.004 / 0.008 |

- Almost all of the error is the IMU position estimate's error: the payload follows its estimate
  closely.
- **Why the estimate is off:** the model-aided correction reads any mismatch between the measured
  and the model-predicted acceleration as a position error. SOFA's leftover force error of about
  0.006, divided by the soft (or open-loop unstable) payload stiffness, biases the estimate by
  ~0.1.
- **The error follows SOFA's mesh error.** On the figure-eight the mean error halves with each
  mesh halving: 0.184 (80 elements), 0.081 (160), 0.040 (320). See `figures/sofa_mesh.png`.
- The controller and estimator are consistent with the continuum. But IMU-only sensing amplifies
  model error a lot: 0.3% of the tension scale in force gives about 3% of the arm length in
  position. A real arm's model errors (friction, disk thickness, material) will be larger than
  that.

## Showcase demos (2026-11-22)

```bash
python scripts/run_showcase.py      # simulations -> data/runs/show_*.npz  (~25 min)
python scripts/make_showcase.py     # figures/show_*.png, show_*.mp4, showreel.mp4
```

All demos run in closed loop on the simulated plant, and the only sensor is an IMU on the tip.
Each demo starts at rest exactly on its path, and the payload block is drawn on the tip.
The reel is `figures/showreel.mp4`: title cards, the four animations, then the speed plot.

| Demo | What it shows | Position error mean / max (× 10⁻³) | Angle error max |
|---|---|---|---|
| Wrist | position held at (1.6, 1.6) while the payload turns 20° → 135° → 20° | 3.2 / 7.2 | 0.40° |
| Waiter | payload kept pointing +y round a 1.0 × 0.6 rounded rectangle | 2.2 / 5.9 | 0.15° |
| Drawing | figure-eight (lemniscate a = 0.45) at constant orientation | 1.6 / 5.3 | 0.10° |
| Pushes | four 0.05 s kicks (three while holding, one while moving) | 1.8 / 15.6 | 0.34° |

- **Speed** (circle of radius 0.55 about (1.45, 1.5), free orientation, one lap each):

  | Lap | Peak speed | Mean / max error (× 10⁻³) |
  |---|---|---|
  | 24 s | 0.27 | 1.3 / 3.6 |
  | 16 s | 0.40 | 1.7 / 4.9 |
  | 10 s | 0.65 | 2.6 / 6.5 |
  | 6 s | 1.08 | 3.9 / 8.3 |

  See `figures/show_speeds.png`.
- **Pushes:** each kick peaks at about 0.015 and recovers to below 0.002 within about 1 s.
- **Controller step time:** about 2 ms on the fixed-orientation demos and about 4.5 ms with free
  orientation.
- No step was refused in any demo.
- **Limits:**
  - The plant is the same model the controller uses (the SOFA plant, step 9, is the independent
    test).
  - With an IMU as the only sensor, a steady unmodelled force or mass can't be told apart from a
    position error. A mass change therefore biases the estimate, so it is not demoed.
  - The simulation runs about 4× slower than real time.

## Closed loop with a tip IMU (2026-11-22)

```bash
python scripts/run_closed_loop.py      # circle and x-line, open and closed loop, on the plant -> data/runs/loop_*.npz (~3.5 min)
python scripts/plot_closed_loop.py     # -> figures/loop_circle|xline.png and .mp4
```

`bowstring/closed_loop.py`:
- **Plant.** The payload (m = 0.1, I = 0.009) moves under the wrench the arm exerts at the sent
  tensions, M Ẍ = w(X, T) − c Ẋ. The arm is massless and quasi-static, with light damping; RK4 at
  1 kHz.
- **IMU at the tip, 1 kHz, the only sensor.** An accelerometer in the payload frame and a rate gyro,
  each with white noise of 2e-3 per sample.
- **Estimator.** Trapezoidal strapdown integration from the known rest state, plus a model-aided
  correction at 100 Hz. The model's predicted acceleration at the estimated pose is compared with the
  measured one; the mismatch, divided through the payload stiffness, corrects the position (a
  damped solve, sign-safe for negative stiffness). Pure strapdown drifted up to 0.2 in 24 s; aided,
  the drift stays below 0.004.
- **Controller** (`LinearizedClosedLoop`, 100 Hz). The open-loop online controller at the reference
  gives the feedforward posture. The feedback wrench M (K_d ė + K_p e) is added as a linear correction
  of its tension differences, ΔD = (∂w/∂D)⁺ w_fb, Tikhonov-damped and capped.
  - The posture never jumps branch.
  - The error obeys M ë + M K_d ė + (M K_p + K_X) e = 0, stable while M K_p + K_X is positive
    definite. Natural frequencies are 5, 5 (x, y) and 15 (angle): M K_p = 2.5 against the most
    negative payload stiffness on the line, −1.04.

**Results** (`figures/loop_*.png`, `.mp4`):

| | Open loop | Closed loop, tip IMU only |
|---|---|---|
| Circle | diverges at t = 8.7 s | completes: position error mean 0.0018, max 0.0049; angle max 0.032 rad |
| Line at y = 1.8, pointing +y | diverges at t = 9.8 s | completes: position error mean 0.0036, max 0.0075; angle max 0.0027 rad |

The controller step takes 4.3 ms (circle) and 1.6 ms (line). No step was refused.

**What did not work, and why:**
- **Solving the inverse map at the estimated pose.** Near folds of the fixed-angle problem the
  tensions swing wildly for small angle changes; tension jumped to 12 and the arm snapped.
- **Re-solving at the reference with the feedback wrench.** It jumps to another posture branch at
  those folds.
- **Cancelling K_X in the feedback.** K_X changes with the tensions, and the mismatch, divided by the
  payload's small inertia, destabilised the loop.
- **Leaving the angle free in the loop.** The least-tension posture is sensitive to the wrench, so the
  payload spun.
- **A pseudo-inverse cutoff of 1e-3.** It discarded the useful directions; the Jacobian's condition
  number is about 7e4.
- **Pure strapdown estimation.** The IMU at 500 Hz under-resolves the payload's fast rotation, and the
  drift reached 0.2.

**Limits:**
- **Perfect model.** The plant is the same model as the controller; SOFA (PLAN step 9) will test that.
- **IMU idealised.** Noise is white; no bias drift or misalignment is modelled.
- **Model-aided position needs payload stiffness.** It is weak where K_X ≈ 0.

## Circle and x-line demos (2026-11-22)

```bash
python scripts/run_demos.py circle xline       # -> data/runs/demo_circle.npz, demo_xline.npz (~10 s)
python scripts/make_demo_figures.py            # -> figures/demo_circle|xline.png and .mp4 (~2 min)
```

The arm uses minimal co-contraction with no tension limits. It starts at rest bent upright
(D = (0, 0, 2.1), tip near (1.44, 2.11)) and carries a payload of mass 0.1, drawn on the tip.

- **Circle:** radius 0.55 about (1.45, 1.5), one lap counter-clockwise in 16 s, orientation free.
  - The commanded pose follows the reference exactly (error 1e-11), at 4.0 ms per step.
  - The peak tension is 5.0.
- **Line parallel to x** at y = 1.8, from x = 2.1 to 0.5 and back, payload angle held at 90°
  (pointing +y). This uses full x, y, θ control.
  - The angle error is 8e-12, at 1.2 ms per step.
  - The peak tension is 7.4.

**Open-loop stability.** Both paths cross regions where the payload sits in an *unstable*
equilibrium under tension control: the payload stiffness K_X has a negative eigenvalue (red in
the figures and animations).

| Demo | Share of the run unstable | Worst k_min |
|---|---|---|
| Circle (the half nearest the base) | 30% | −0.16 |
| Line (the part nearer the base) | 55% | −1.0 |

- **No posture helps there.** On the circle, a scan over the payload angle found no stable posture
  at those points.
- **What it means:** with the tensions held, the payload would drift away from these equilibria,
  with a time constant √(m/|k|) of about 1 s (circle) and 0.3 s (line).
- **How it is run:** the demos use `Controller(payload_stability="report")`, which records K_X
  instead of refusing those states. The default ("block") refuses them.

## Controller demos: no tension limits, whole workspace (2026-11-22)

```bash
python scripts/workspace_map.py        # continuous workspace, least-tension posture -> data/workspace.npz (~7 min)
python scripts/workspace_scan.py       # every posture: rest states over a D grid  -> data/workspace_scan.npz (~25 min, 10 workers)
python scripts/run_demos.py            # sweep, path, live -> data/runs/demo_*.npz (~2 min)
python scripts/make_controller_figures.py   # -> figures/ctrl_*.png, ctrl_*.mp4 (~5 min)
```

**Why segments 2 and 3 bent so little before.** With fixed pair totals C = (3, 2.5, 2), each pair
can only shift tension between its sides, |D_k| ≤ C_k − 2 Tmin = (2.8, 2.3, 1.8):
- Segment 3 is bent by pair 3 alone, so its moment was capped at 0.45 (about 26°); segment 2 at 0.91.
- The fixed totals also compressed the arm by 7.5 all the time, putting it close to buckling: soft,
  and split into bulge-up and bulge-down postures.

**Now: minimal co-contraction, no tension limits** (`ChainArm(scheme="minimal")`, Tmax = None).
- In each pair the slack side keeps the pretension Tmin = 0.1 and the pulling side carries the
  difference. Tendons only pull; there is no upper limit.
- The arm is compressed only as much as bending needs.
- It is still 3 unknowns for n = 3, so the inverse map stays square.

**Workspace** (`figures/ctrl_workspace.png`, before and after):

| | Area | Payload stiffness (median) | Closest the payload gets to the base |
|---|---|---|---|
| Fixed totals | 3.0 | 0.049 | 1.63 |
| No limits, reachable continuously (least-tension posture) | 13.5 | 0.087 | |
| No limits, every posture (tension scan, \|D\| ≤ 8) | about 15 | | 0.04 |

What is left out is behind the base beyond x ≈ −2, and a pocket in front of it.
- **Tension is no longer the limit.** With Tmax = 8 the area was already 13.2.
- **What limits it now:** stability with the payload free (the payload stiffness K_X must stay
  positive), and the geometry.
- **The workspace is a C, not a ring.** The payload cannot go all the way round the base: each lap
  adds 2π to the arm's total turning. The region behind the base is reached over the top (arm
  curled one way) or underneath (curled the other way), two postures with no path between them.
  A lap attempt wound the arm up to D₁ = 11 before failing.

**Demos** (orientation free, 100 Hz, m = 0.1, routes 0.3 inside the band's edges, end to end):

| Demo | Route | Held T | Posture switches | Mean step | Notes |
|---|---|---|---|---|---|
| Sweep, fixed timing | rest → top-back end → bottom-back end → start, 27 units | 0 | 2 | 4.1 ms | 99th pct 7.4 ms; peak tension 5.3 |
| Path, speed online, safe mode | rest → top-back → bottom-back, 17 units | 6 | 3 | 8.9 ms | 54 fallbacks |
| Live, scripted gamepad, safe mode | the same journey | 0 | 3 | 7.4 ms | 154 refused pushes at the edges |

**Controller changes that came with this:**
- **Stability on every step.** The arm's margin and the payload stiffness are checked every step
  (cheap); each segment's own test runs every 10th step.
- **Safe mode** (live input and the path follower). A step is accepted only if the payload could
  also be held at rest there, checked from the step's own solution. The path follower stops at its
  last accepted pose when nothing is feasible.
- **Posture switch (bulge up ↔ down).** Only in front of the base (|y| < 0.8, x > 0). It triggers
  for a 5% saving or when the current posture loses payload stiffness, with a 0.3 s dwell. Behind
  the base a "switch" would be a jump across the C.
- **Live input is jerk-limited.** The force asked of the soft arm ramps up instead of jumping.
- **Faster evaluations.** The tendon code is vectorised (about 2×), and solves give up early under
  the deadline.

**Limits:**
- **Refused or held steps take 20–35 ms**, against the 10 ms budget; normal steps are within it.
- **Distal segments bend less than they could.** The least-tension posture still bends segment 1
  most, since it carries every pair's moment. Another posture rule (for example, spreading the
  curvature) would use segments 2 and 3 more.
- **Payload rotational inertia is neglected** in free-orientation mode.

## Online controller (2026-11-22, PLAN step 6)

```bash
python tests/test_controller.py     # without a plant, about 10 s  -> logs/test_controller.log
```

```python
from bowstring.chain import ChainArm
from bowstring.controller import Controller, Payload
from bowstring.reference import Trajectory, PathFollower, Live
ctrl = Controller(ChainArm(), C=[3.0, 2.5, 2.0], payload=Payload(m=0.05, I=0.0045))
X0 = ctrl.reset(D0=[0.8, 0.4, 0.2])                     # arm at rest; references start at X0
traj = Trajectory.circle(X0, radius=0.02, duration=2.0)
T, info = ctrl.step(t, *traj(t))                       # fixed timing
live = Live(X0, lo=X0 - [0.08, 0.08, 0.3], hi=X0 + [0.08, 0.08, 0.3])
T, info = ctrl.choose(t, live.candidates(dt, cmd), safe=True); live.accept(info["choice"])
```

Each step: w = M·Ẍ*, inverse map at X* (warm-started), send T if feasible. Otherwise the
reference generator's next candidate is tried in the same step (path: speed up / coast / brake;
live: as commanded / brake / hold still), and the last T is held only if none is feasible.

Results (all checks passed):
- **Circle (radius 0.02, 2 s):** every step feasible. The forward map at the sent tensions
  reproduces M·Ẍ* to 4e-11.
- **Path with online speed:** reaches the end.
- **Live input pushing past the workspace:** brakes or holds and never sends an infeasible T.
- **Too-fast circle:** the infeasible steps are flagged and hold T.
- **Step time:** 2.4 ms mean, 9.7 ms worst (fallback steps).
- **Workspace:** with C = (3, 2.5, 2) and Tmin = 0.1 the arm holds the payload, at fixed
  orientation, only within about 0.03–0.2 of the rest pose and ±0.2 rad. Larger motions need
  more co-contraction or free pair totals.

## Segment, table, tendons and chain arm (2026-11-22, PLAN steps 1, 3–5)

```bash
python tests/test_chain.py      # PLAN checks 1–4, about 45 s  -> logs/test_chain.log
```

```python
from bowstring.chain import ChainArm
arm = ChainArm(n=3, offsets=(0.15, 0.2, 0.25), Tmin=0.1)
C, D = [3.0, 2.5, 2.0], [0.8, 0.4, 0.2]
rest = arm.rest(arm.tensions(C, D))                      # payload free
fw = arm.forward(X, arm.tensions(C, D), rest["L"])       # (X, T) -> w, K_X, margin
inv = arm.inverse(X, w, C, rest["L"], D)                 # (X, w) -> D, T
```

- **`bowstring/segment.py`.** One segment by RK4 shooting with its variational equations. Given a
  base load, it returns the end pose, the energy V, the end wrench g and the stiffness K.
  `shoot(r)` also solves for a given end pose, with continuation when Newton fails.
- **`bowstring/table.py`.** Reads `data/segment_table.npz` at any pose: bicubic within a layer,
  and Hermite across layers using the exact slopes (∂U/∂d = −fx, ∂M/∂d from symmetry).
- **`bowstring/tendons.py`.** Tendon lengths in the disk poses, with gradients and Hessians.
- **`bowstring/chain.py`.** The arm, with the segment base loads as unknowns.
  - Forward-integrating each segment gives the disk poses, and the equations are the disk
    equilibria.
  - Rest, forward map and inverse map (Newton; n = 3 only) are provided.
  - Stability is λ_min of the Hessian with the payload held, plus each segment's clamped stability.
  - Timing (pure Python): one evaluation takes about 10 ms. A forward step takes 0.05 s and an
    n = 3 inverse 0.03 s.

**Why the chain does not use the table.** The table gives V to about 5e-5, but the axial load of a
nearly straight segment is poorly resolved:
- That segment is very stiff along its chord. At curvature 0.06, about 25 load units separate
  neighbouring layers.
- The table's g is therefore off by up to 2% in the worst 1% of poses at d > 0.99, and those shallow
  arcs are the arm's usual state.
- The same stiffness made a chain with disk poses as unknowns stall on nearly straight segments.

So the chain integrates the segments directly, and the table stays as a reference.

**Checks (`logs/test_chain.log`), all passed:**
1. Derivatives match finite differences: 1e-8 for shooting, the chain and the envelope theorem,
   1e-10 for the tendons, and 6e-3 for the table's K.
2. Table against exact states, up to the energies the arm reaches: median V ≤ 5e-5 and g ≤ 1.2e-4;
   the worst 1% have g ≤ 2e-2.
3. Chain against `rod.py` (160 elements per segment, stable descent): rest disk poses agree to
   2.5e-4 and forward wrenches to 5e-5. `rod.py`'s plain Newton can land on unstable states, so the
   check uses its stable descent; the chain agrees with that.
4. The forward map of the inverse returns the requested wrench to 1e-13.

## FEniCSx refinement at high d (2026-11-22, PLAN step 2)

```bash
cd fenics
python pilot.py                                   # beam resolution study  -> logs/pilot_resolution.log
python layers.py ../data/high_d/config.json       # 39 layers on 18 workers, ~36 min; reruns skip finished layers
python validate_layers.py ../data/high_d/layers   # -> data/high_d/validation.log
```

`fenics/` holds copies of `~/elastica/solver/{solver,branch,stability,validate_results}.py`,
unchanged; `~/elastica` is no longer read. `data/production/fields.npz` is the production grid
(d 0.70–0.99, φ ±π/2, step 0.01) without θ(s), copied by `scripts/copy_production.py`.

`layers.py` is `main.py` cut down to one d layer per task, so a pool of workers solves layers
in parallel (each worker does serial solves). Each layer cold-starts from the exact first mode
at (d, 0, 0) and grows a 2D front with the same jump, stability and promotion rules. Changes:
- **Energy cutoff.** The front stops growing past U > 10, keeping 2 rings beyond for
  interpolation stencils. The arm's segments reach U ≤ 3.5 at d 0.97–0.99, ≤ 0.6 at
  0.99–0.999 and ≤ 0.05 above (every segment state in `data/maps`).
- **φ box and step follow the energy disk.** U ≈ (φ₁² + φ₂²)² / (16(1 − d)), so the disk
  radius is (16(1 − d)·U_max)^¼, which matches production to 0.01 at d 0.97–0.99. The step is
  0.01·((1 − d)/0.03)^¼: 0.010 at 0.97, 0.0043 at 0.999 and 0.0024 at 0.9999. That gives about
  72k nodes per layer, and no layer's box clipped the disk.
- **Beam: 1000 elements, grading 0.9** (end elements 1e-4, mid-span 1.5e-3). Against an
  8000-element reference at the disk edge, the worst relative error is 1.4e-5 at d = 0.97,
  8.7e-5 at 0.999 and 4.2e-4 at 0.9999. With 500 elements it is 4.8e-3 at 0.9999. Grading 0.95
  lands on wrong states.

Layers: d 0.970–0.999 step 0.001, and 0.9991–0.9999 step 1e-4. Output is
`data/high_d/layers/layer_<d>.npz`, holding fields on the (φ₁, φ₂) grid, `solved`, `beyond`
(rings past the cutoff), and θ for the solved half φ₁ + φ₂ ≥ 0.

Run: 1.43 M solves, 2.65 M nodes with U ≤ 10, 9.2 CPU-hours, 36 min on 18 workers (25 ms a
solve under full load, 9 ms alone). Every layer's front closed: nothing left waiting, every extra
cold start succeeded.

Validation, all passed:
- φ = 0 matches the exact first mode to 5e-6.
- 40 `solve_bvp` re-solves per layer: worst 1.4e-5 (d = 0.97) rising to 5.0e-4 (0.9999).
- No states on the S-shaped family, no unstable nodes, and the energy disk is closed.
- Against production at d = 0.97/0.98/0.99: medians 1e-4 to 3e-4, p99 up to 7e-3 in the moment.
  This is production's own 200-element error plus bicubic interpolation of it.

### Segment table (PLAN 2c): `python fenics/merge_table.py` → `data/segment_table.npz`

The table has 66 layers: production d 0.70–0.96 (27 layers) and the high-d layers
0.970–0.9999 (39). Each layer keeps its own φ grid, padded to 327 × 327:
- `phi[l, :n_phi[l]]` is layer l's grid, and the fields are `[l, j (φ₁), k (φ₂)]`.
- Fields of the up branch: `energy`, `fx`, `fy`, `M1`, `M2`, `p`.
- `computed` marks nodes the solver reached (False outside a high-d energy disk).
- `exists` marks nodes holding a stable up state; `computed & ~exists` lies past the up fold.
- The up state at a node is the stored one if p > 0, else the reflection of the down state
  stored at −φ.
- The down branch is the reflection of the table: down(d, φ₁, φ₂) = R up(d, −φ₁, −φ₂).
- Also stored: `amplitude` a(d), `source` and `energy_max`; 140 MB.

Overlap (`data/segment_table.log`): on the up branch at d = 0.97/0.98/0.99, away from the
fold, production and the high-d layers agree to median 1–3e-4 and p99 2–7e-4 in U, fx, M1 and
M2. That is production's own 200-element error. (1 − d)U is smooth in d across the switch at
0.96 → 0.97.

## So far: the forward and inverse maps (2026-11-10)

```bash
conda activate elastica-env                        # numpy, scipy, matplotlib
python tests/test_rod.py                           # finite-difference and round-trip checks
python scripts/compute_maps.py all                 # -> data/maps/*.npz   (about 7 min in all)
python scripts/plot_maps.py                        # -> figures/maps_*.png
```

The model is `bowstring/rod.py`: the whole arm as one discretised inextensible rod (N = 40
elements per segment, tangent angles as unknowns), with tendon chords between disk holes. It is
the reference ("direct") model. The stored segment table is not used yet.

- **Forward map** (X, T) → w: stationary energy with the payload pose held by a multiplier μ;
  the wrench on the payload is w = −μ (checked against −∂U*/∂X). `held` is a descent that only
  ends on stable states; plain Newton (`forward`) also converges to unstable ones.
- **Inverse map** (X, w) → D at fixed pair totals C (square for n = 3).
- **Stability margin**: smallest eigenvalue of the Hessian of the Lagrangian on the null space
  of the payload constraint, divided by h.
- `bowstring/continuation.py`: pseudo-arclength continuation in the payload angle or one D_k,
  through folds.

Parameters: n = 3, ℓ = B = 1, hole offsets d = (0.15, 0.20, 0.25), C = (3, 2.5, 2),
T_min = 0.1 (so |D_k| ≤ C_k − 0.2), reference D = (0.8, 0.4, 0.2).

| Figure | Shows |
|---|---|
| `maps_toy.png` | one-variable toy: one curve read as both maps; the cusp surface |
| `maps_scurves.png` | real arm, payload held at (2.9, 0): S-curve in the payload angle (folds at ±0.43) and in D₁ (snap at D₁ = 0.76) |
| `maps_cusp.png` | fold curves over (payload angle, D₁) in 3D and 2D; the bistable strip |
| `maps_forward.png` | forward map over the payload position at fixed T: force field, moment, contact-free region |
| `maps_inverse.png` | inverse map over the force (f_x, f_y) at fixed X: D₁…D₃, margin, the feasible lens |
| `maps_inverse_3d.png` | the force plane carried into (D₁, D₂, D₃) space, with the box of tension limits |
| `maps_workspace.png` | rest map D → X over the tension box |

Findings:
- Every fold of the forward map sits where the margin crosses zero. The inverse map is
  smooth across that curve. There, det ∂D/∂w changes sign: its sign matched the margin's at all
  7,353 solved points. So the controller must check the margin; a fold of the inverse map
  will not warn it.
- The fold curve of the downward bow from 15 angle sweeps passes through the snap found
  independently by the D₁ sweep (φ = −0.40, D₁ = 0.76).
- At the reference pose the tension limits allow only |f| ≲ 0.25–1, long before stability is
  lost (|f| ≈ 3–8). In the force sweep, the first limit hit is feasibility, not stability.
- The straight arm under co-contraction alone buckles between C = 3 and 4 per pair.
- The no-contact condition binds early: at the compressed pose (2.5, 0) most stable states
  have the rod crossing a tendon; at (2.9, 0) every stable state drawn is clear.

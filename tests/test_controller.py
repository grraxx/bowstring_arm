"""
test_controller.py
The online controller (PLAN step 6) on its own, without a plant: every step's tensions must
give the requested wrench at the reference pose, within the time budget.

Arm: minimal co-contraction, no tension limits. The fixed-orientation references are small
(a fixed payload angle confines the payload to a thin slice of the workspace).

1. Trajectory: a circle at 100 Hz. All steps feasible; forward(X*, T) = M Xdd* on spot checks.
2. Path: a line whose speed is chosen online; it reaches the end.
3. Live: scripted stick and trigger commands; the reference respects its limits, and steps at
   the workspace edge hold T.
4. Out of reach (path follower in safe mode): a line into the base region, outside the stable workspace (towards full
   extension would test the singular straight arm instead). Infeasible steps hold T and are
   flagged; the path follower stops at the limit instead.
5. Speed: mean and worst step time against the 10 ms budget (100 Hz).

usage: python tests/test_controller.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bowstring.chain import ChainArm                                   # noqa: E402
from bowstring.controller import Controller, Payload                   # noqa: E402
from bowstring.reference import Live, PathFollower, Trajectory         # noqa: E402

failed = []


def report(name, ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {text}", flush=True)
    if not ok:
        failed.append(name)


arm = ChainArm(n=3, offsets=(0.15, 0.2, 0.25), Tmin=0.1, scheme="minimal")   # no tension limits
C, D0 = [3.0, 2.5, 2.0], [0.8, 0.4, 0.2]
payload = Payload()
ctrl = Controller(arm, C, payload)
dt = 0.01
times = []

print("1. Circle, radius 0.02, 2 s, 100 Hz")
X0 = ctrl.reset(D0)
traj = Trajectory.circle(X0, 0.02, 2.0)
errs, feasible = [], []
for k in range(1, int(traj.duration / dt) + 1):
    X, Xd, Xdd = traj(k * dt)
    T, info = ctrl.step(k * dt, X, Xd, Xdd)
    times.append(info["time"])
    feasible.append(info["feasible"])
    if k % 20 == 0:
        fw = arm.forward(X, T, ctrl.history[-1][0], report=False)
        errs.append(np.abs(fw["w"] - info["w_req"]).max())
report("all steps feasible", all(feasible), f"{sum(feasible)}/{len(feasible)}")
report("forward(X*, T) = M Xdd*", max(errs) < 1e-7, f"max error {max(errs):.1e} on {len(errs)} checks")

print("2. Path: line of length 0.14 along the arm's easy direction, online speed (v_max 0.2, a_max 1)")
X0 = ctrl.reset(D0)
path = PathFollower.line(X0, X0 + np.array([-0.1, 0.1, 0.1]), v_max=0.2, a_max=1.0)
n_inf, steps, vmax = 0, 0, 0.0
while not path.done and steps < 1000:
    steps += 1
    T, info = ctrl.choose(steps * dt, path.candidates(dt))
    path.accept(info["choice"])
    n_inf += not info["feasible"]
    vmax = max(vmax, path.speed)
    times.append(info["time"])
report("path completed", path.done, f"{steps} steps, top speed {vmax:.3f}, {n_inf} infeasible steps")

tracking_times = list(times)     # steps of tests 1-2: plain tracking
print("3. Live: 3 s of scripted stick and trigger input")
X0 = ctrl.reset(D0)
live = Live(X0, v_max=0.1, w_max=0.5, a_max=0.5, alpha_max=2.0,
            lo=X0 - [0.08, 0.08, 0.3], hi=X0 + [0.08, 0.08, 0.3])
worst_a, n_inf, n_alt = 0.0, 0, 0
for k in range(1, 301):
    t = k * dt
    cmd = (np.cos(2 * t), np.sin(3 * t), 1.0 if t < 1 else (-1.0 if t < 2 else 0.0))
    refs = live.candidates(dt, cmd)
    T, info = ctrl.choose(t, refs, safe=True)
    live.accept(info["choice"])
    n_inf += not info["feasible"]
    n_alt += bool(info["choice"])
    Xdd = refs[info["choice"] or 0][2]
    worst_a = max(worst_a, np.hypot(*Xdd[:2]) / live.a_max, abs(Xdd[2]) / live.alpha_max)
    times.append(info["time"])
report("reference within its limits", worst_a <= 1 + 1e-9, f"max |acc| / limit {worst_a:.3f}")
report("workspace edge: the target brakes or holds instead", n_inf == 0,
       f"{n_alt} of 300 steps took the fallback (the commands keep pushing outwards), "
       f"{n_inf} held T")

normal_times = list(times)       # steps 1-3: normal operation
print("4. Out of reach: a line towards a payload pose 0.8 from the base, at fixed orientation")
X0 = ctrl.reset(D0)
far = np.r_[0.8 * X0[:2] / np.hypot(*X0[:2]), X0[2]]
traj = Trajectory.line(X0, far, 1.0)
reasons, held = {}, True
for k in range(1, int(traj.duration / dt) + 1):
    T_before = ctrl.T.copy()
    X, Xd, Xdd = traj(k * dt)
    T, info = ctrl.step(k * dt, X, Xd, Xdd)
    if not info["feasible"]:
        reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
        held &= np.array_equal(T, T_before)
        times.append(info["time"])
report("infeasible steps flagged and T held", bool(reasons) and held,
       f"{sum(reasons.values())} infeasible steps: {reasons}")
X0 = ctrl.reset(D0)
path = PathFollower.line(X0, far, v_max=0.5, a_max=1.0)
n_inf = 0
for k in range(1, 401):
    T, info = ctrl.choose(k * dt, path.candidates(dt), safe=True)
    path.accept(info["choice"])
    n_inf += not info["feasible"]
report("path follower stops at the reach limit", not path.done and n_inf < 20,
       f"stopped {path.length - path.sigma:.3f} short of the end, {n_inf} of 400 steps held T")

print("5. Speed")
nt = np.array(tracking_times)
lt = np.array(normal_times[len(tracking_times):])
ft = np.array(times[len(normal_times):])
report("step time within 10 ms while tracking (tests 1-2)", np.percentile(nt, 99) < 0.010,
       f"mean {nt.mean() * 1e3:.2f} ms, 99th pct {np.percentile(nt, 99) * 1e3:.2f} ms, "
       f"max {nt.max() * 1e3:.2f} ms over {len(nt)} steps")
print(f"  (live input, safe mode, pushing into the edge: mean {lt.mean() * 1e3:.1f} ms, "
      f"99th pct {np.percentile(lt, 99) * 1e3:.1f} ms over {len(lt)} steps)")
print(f"  (infeasible steps of test 4, which try every candidate: mean {ft.mean() * 1e3:.1f} ms, "
      f"max {ft.max() * 1e3:.1f} ms over {len(ft)} steps)")

print("FAILED: " + ", ".join(failed) if failed else "All checks passed")
sys.exit(1 if failed else 0)

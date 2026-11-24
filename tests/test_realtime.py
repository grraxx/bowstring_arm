"""
test_realtime.py
The real-time solver (Till et al. 2019 applied to the model plant):
1. The compiled ChainArm.evaluate matches the Python assembly (random loads and tensions).
2. The implicit (BDF-alpha) plant matches a fine-step explicit RK4 plant: payload released off
   equilibrium at constant tensions, 2 s, at an open-loop-unstable pose (errors grow there).
3. The closed loop (figure-eight, IMU + tendon encoders, 100 Hz control, 1 ms plant) runs faster
   than real time, and the controller side stays inside its 10 ms budget.

usage: python tests/test_realtime.py      (about 30 s)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bowstring import experiment as E                        # noqa: E402
from bowstring.chain import _CHAIN, ChainArm                 # noqa: E402
from bowstring.closed_loop import ImplicitPlant, Plant       # noqa: E402
from bowstring.controller import Payload                     # noqa: E402

failed = []


def report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        failed.append(name)


print("1. Compiled evaluate")
rng = np.random.default_rng(0)
arm = ChainArm(scheme="minimal", stiffness=(1.0, 0.8, 1.3))
worst = 0.0
for _ in range(200):
    L, T = rng.normal(scale=[3, 3, 2], size=(3, 3)), rng.uniform(0.1, 5, size=(3, 2))
    arm.use_c = True
    a = arm.evaluate(L, T)
    arm.use_c = False
    b = arm.evaluate(L, T)
    for k in ("P", "G", "dGdL", "dPdL", "lam_grad", "U"):
        worst = max(worst, float(np.max(np.abs(np.asarray(a[k]) - b[k])) / (1 + np.max(np.abs(b[k])))))
report("C library built", _CHAIN is not None, "bowstring/_build/_chain.so")
report("C evaluate = Python evaluate", worst < 1e-12, f"max relative difference {worst:.1e}")

print("2. Implicit plant against fine RK4")
P = E.resolve({"traj.preset": "figure8"})
arm = E.make_arm(P, "robot")
pay = Payload(m=P["robot.m"], I=P["robot.I"])
M = np.array([pay.m, pay.m, pay.I])
X0, L0, D0 = E.start_state(arm, np.array([1.6, 1.5, np.pi / 2]), False)
T = arm.tension_map(E.C, D0)[0]
Xs = X0 + np.array([0.02, -0.015, 0.03])
Ls = np.ravel(arm.forward(Xs, T, L0, report=False)["L"])


def release(cls, dt, dur=2.0):
    p = cls(arm, pay, Xs, Ls, damping=P["robot.alpha"] * M)
    out = []
    for _ in range(int(round(dur / dt))):
        p.step(T, dt)
        out.append(p.X.copy())
    return np.array(out)


ref = release(Plant, 2.5e-4)[3::4]
imp = release(ImplicitPlant, 1e-3)
e = float(np.abs(imp - ref).max())
report("BDF-alpha 1 ms vs RK4 0.25 ms", e < 1e-3,
       f"max |dX| {e:.1e} over a motion of {np.ptp(ref, 0).max():.2f}")

print("3. Closed loop in real time")
orig = E.build_reference
E.build_reference = lambda Pr: (lambda r: (r[0], 4.0, r[2], r[3], r[4]))(orig(Pr))
r = E.run({"traj.preset": "figure8", "sim.plant": "model"})
E.build_reference = orig
s = r["summary"]
rt = s["simulated"] / s["wall_s"]
report("completed", s["state"] == "done", s["status"])
report("faster than real time", rt > 1.0, f"{s['simulated']:.1f} s simulated in {s['wall_s']:.1f} s "
       f"(x{rt:.2f})")
report("controller step within 10 ms", s["ctrl_ms_p99"] < 10.0,
       f"mean {s['ctrl_ms_mean']:.2f} ms, 99th pct {s['ctrl_ms_p99']:.2f} ms")

print("FAILED: " + ", ".join(failed) if failed else "All checks passed")
sys.exit(1 if failed else 0)

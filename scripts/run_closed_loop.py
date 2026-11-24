"""
run_closed_loop.py
The circle and x-line demos against a simulated plant (bowstring/closed_loop.py), two ways:

  open    the open-loop controller's tensions (computed at the reference, nothing measured)
  closed  feedback from the tip IMU only: strapdown estimate of the payload pose and velocity with
          a model-aided correction; PD feedback wrench added as a damped linear correction of the
          feedforward tensions (bowstring/closed_loop.py, LinearizedClosedLoop)

Control at 100 Hz; plant, IMU and estimator at 1 kHz. IMU noise: accelerometer 2e-3, gyro 2e-3
(white, per sample). Gains: natural frequencies 5, 5 (x, y) and 15 (angle), critically damped. The arm starts at rest, bent upright (D = (0, 0, 2.1)), payload m = 0.1.

usage: python scripts/run_closed_loop.py [circle xline] [open closed]
       -> data/runs/loop_<demo>_<mode>.npz
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm                                        # noqa: E402
from bowstring.closed_loop import (IMU, Estimator, LinearizedClosedLoop,    # noqa: E402
                                   Plant, PlantError, wrap)
from bowstring.controller import Controller, Payload                        # noqa: E402
from bowstring.reference import Trajectory, quintic                         # noqa: E402

DT, SUB = 0.01, 10                      # control period 100 Hz; plant and IMU at 1 kHz
C = np.array([3.0, 2.5, 2.0])
D_UP = np.array([0.0, 0.0, 2.1])
PAYLOAD = Payload(m=0.1, I=0.1 * 0.3 ** 2)
OUT = ROOT / "data/runs"


def circle_reference(X0, centre=(1.45, 1.5), radius=0.55, duration=16.0):
    """Position of the circle demo (radius blending in from the rest pose X0)."""
    c = np.asarray(centre, float)
    a0 = np.arctan2(X0[1] - c[1], X0[0] - c[0])
    dr0 = np.hypot(*(X0[:2] - c)) - radius

    def ref(t):
        s_, ds, dds = quintic(t / duration)
        a, da, dda = a0 + 2 * np.pi * s_, 2 * np.pi * ds / duration, 2 * np.pi * dds / duration ** 2
        e = np.exp(-((a - a0) / 0.6) ** 2)
        r, r1 = radius + dr0 * e, dr0 * e * (-2 * (a - a0) / 0.36)
        r2 = dr0 * e * ((2 * (a - a0) / 0.36) ** 2 - 2 / 0.36)
        ca, sa = np.cos(a), np.sin(a)
        p1 = np.array([r1 * ca - r * sa, r1 * sa + r * ca])
        p2 = np.array([r2 * ca - 2 * r1 * sa - r * ca, r2 * sa + 2 * r1 * ca - r * sa])
        return c + r * np.array([ca, sa]), p1 * da, p2 * da ** 2 + p1 * dda
    return ref, duration


def xline_reference(X0, y=1.8, x_right=2.1, x_left=0.5):
    A, B = np.array([x_right, y, np.pi / 2]), np.array([x_left, y, np.pi / 2])
    legs = [Trajectory.line(X0, A, 4.0), Trajectory.line(A, B, 10.0), Trajectory.line(B, A, 10.0)]
    ends = np.cumsum([leg.duration for leg in legs])

    def ref(t):
        i = int(np.searchsorted(ends, t - 1e-12))
        i = min(i, len(legs) - 1)
        return legs[i](t - (ends[i - 1] if i else 0.0))
    return ref, ends[-1]


def run(demo, mode, imu_noise=2e-3, wn=(5.0, 5.0, 15.0), tag="", dt=None, model_aided=True):
    """wn: closed-loop natural frequencies (x, y, angle); M Kp must outweigh the arm's negative
    payload stiffness along the path (down to -1.04 on the line: 0.1 * 25 = 2.5)."""
    global DT, SUB
    if dt is not None:                          # control period; the plant stays at 500 Hz
        DT, SUB = dt, max(1, int(round(dt / 0.002)))
    arm = ChainArm(scheme="minimal")
    rest = arm.rest(arm.tension_map(C, D_UP)[0])
    X0, L0 = rest["P"][-1].copy(), rest["L"].copy()
    plant = Plant(arm, PAYLOAD, X0, L0)
    imu = IMU(accel_noise=imu_noise, gyro_noise=imu_noise)
    est = Estimator(X0)
    # Feedforward / reference posture: the open-loop controller run at the reference.
    free = demo == "circle"
    ol = Controller(ChainArm(scheme="minimal"), C, PAYLOAD, mode="dynamic",
                    orientation="free" if free else "track", payload_stability="report")
    ol.reset(D_UP)
    lin = LinearizedClosedLoop(arm, C, PAYLOAD, wn=wn)
    ref_fn, duration = (circle_reference if free else xline_reference)(X0)
    rows, psi_prev, Xh, Vh = [], X0[2], X0.copy(), np.zeros(3)
    L_est = L0.copy()
    psid_prev = 0.0
    start, status = time.time(), "completed"
    for k in range(1, int(round(duration / DT)) + 1):
        t = k * DT
        r = ref_fn(t)
        if free:                                   # position reference; the angle follows the
            Xr = np.r_[r[0], 0.0]                  # least-tension posture at the reference
            Xdr, Xddr = np.r_[r[1], 0.0], np.r_[r[2], 0.0]
        else:
            Xr, Xdr, Xddr = r
        T_ol, info_ol = ol.step(t, Xr, Xdr, Xddr)
        if free:
            psi_ref = info_ol["P"][-1][2]
            Xr = Xr.copy()
            Xr[2], Xdr, Xddr = psi_ref, Xdr.copy(), Xddr.copy()
            Xdr[2] = wrap(psi_ref - psi_prev) / DT
            Xddr[2] = (Xdr[2] - psid_prev) / DT if k > 1 else 0.0   # backward differences
            psi_prev, psid_prev = psi_ref, Xdr[2]
        if mode == "closed":
            w_ff = np.r_[PAYLOAD.m * Xddr[:2], 0.0] if free else PAYLOAD.M @ Xddr
            T, info = lin.step((Xr, Xdr, Xddr), (Xh, Vh), ol.history[-1][0], ol.history[-1][1],
                               w_ff)
            ctrl_time, ok = info["time"] + info_ol["time"], info_ol["feasible"]
        else:
            T, ctrl_time, ok = T_ol, info_ol["time"], info_ol["feasible"]
        try:
            for _ in range(SUB):
                acc = plant.step(T, DT / SUB)
                f, g = imu.read(plant.X[2], plant.V[2], acc[:2])
                Xh, Vh = est.update(f, g, DT / SUB)
            if model_aided and mode != "open":
                L_est = est.model_correct(arm, T, L_est, plant.M, plant.c)
                Xh = est.X.copy()
        except PlantError as err:
            status = f"plant failed at t = {t:.2f}: {err}"
            break
        err_pos = np.hypot(*(plant.X[:2] - Xr[:2]))
        rows.append({"t": t, "X_ref": Xr, "X": plant.X.copy(), "X_est": Xh.copy(), "T": T,
                     "L": plant.L.copy(), "ok": ok, "time": ctrl_time,
                     "err": err_pos, "err_psi": wrap(plant.X[2] - Xr[2]),
                     "est_err": np.hypot(*(Xh[:2] - plant.X[:2]))})
        if err_pos > 1.0:
            status = f"diverged at t = {t:.2f} (position error {err_pos:.2f})"
            break
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(OUT / f"loop_{demo}_{mode}{tag}.npz", status=status, m=PAYLOAD.m, I=PAYLOAD.I, dt=DT,
             imu_noise=imu_noise, **arrays)
    e = arrays["err"]
    print(f"{demo} {mode}{tag}: {status}; {len(rows)} steps; position error mean {e.mean():.4f}, max "
          f"{e.max():.4f}; angle error max {np.abs(arrays['err_psi']).max():.4f}; estimate drift max "
          f"{arrays['est_err'].max():.4f}; refused steps {np.sum(~arrays['ok'])}; controller "
          f"{1e3 * arrays['time'].mean():.1f} ms mean ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    demos = [a for a in args if a in ("circle", "xline")] or ["circle", "xline"]
    modes = [a for a in args if a in ("open", "closed")] or ["open", "closed"]
    for d in demos:
        for m in modes:
            run(d, m)

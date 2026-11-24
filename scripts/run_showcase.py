"""
run_showcase.py
Showcase demos of the closed-loop controller (tip IMU only) on the simulated arm:

  wrist     payload held at (1.6, 1.6) while it rotates 20 -> 135 -> 20 degrees
  waiter    payload kept level (pointing +y) round a 1.0 x 0.6 rounded rectangle
  figure8   the tip draws a figure-eight (lemniscate, half-width 0.45) at constant orientation
  disturb   pushes the controller does not know about: three kicks (x, y, a twist) while the
            payload holds a pose where it is open-loop unstable, then one more while it moves
  speeds    the circle of radius 0.55 about (1.45, 1.5) at four speeds (orientation free)

Every demo starts at rest exactly on its path: the arm is brought to the start pose by the
zero-wrench inverse map (tensions that hold the payload there at rest).

Control 100 Hz; plant, IMU and estimator 1 kHz; gains wn = (5, 5, 15), critically damped.

usage: python scripts/run_showcase.py [wrist waiter figure8 disturb speeds]
       -> data/runs/show_<name>.npz
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
                                   Plant, PlantError, TendonEncoders, wrap)
from bowstring.tendons import tendons                                       # noqa: E402
from bowstring.controller import Controller, Payload                        # noqa: E402
from bowstring.reference import quintic                                     # noqa: E402

DT, SUB = 0.01, 10
C = np.array([3.0, 2.5, 2.0])
D_UP = np.array([0.0, 0.0, 2.1])
PAYLOAD = Payload(m=0.1, I=0.1 * 0.3 ** 2)
OUT = ROOT / "data/runs"
UP = np.pi / 2


# ------------------------------------------------------------------ start pose
def start_state(X_start, free=False):
    """Loads and tension differences holding the payload at rest at X_start (track) or at the
    position X_start[:2] with the least-tension angle (free), reached from the upright rest
    pose by quasi-static continuation."""
    arm = ChainArm(scheme="minimal")
    r = arm.rest(arm.tension_map(C, D_UP)[0])
    X0, L, D = r["P"][-1], r["L"], D_UP.copy()
    for f in np.linspace(0, 1, 100)[1:]:
        if free:
            o = arm.inverse_free(X0[:2] + f * (X_start[:2] - X0[:2]), np.zeros(3), C, L, D,
                                 max_it=15, report=False, restore_its=20, alpha_min=1e-3)
        else:
            o = arm.inverse(X0 + f * (np.asarray(X_start) - X0), np.zeros(3), C, L, D,
                            report=False, max_it=40)
        if not o["converged"]:
            raise RuntimeError(f"start pose {X_start} not reached")
        L, D = o["L"], o["D"]
    return o["P"][-1].copy(), np.ravel(L), D


# ------------------------------------------------------------------ simulation
SENSING = os.environ.get("SENSING", "lengths")   # "lengths": high-end IMU + tendon encoders;
                                                 # "imu": the earlier IMU-only (noise 2e-3)


def simulate(name, ref, duration, X_start, free=False, w_ext=None, mass=None, wn=(5, 5, 15),
             imu_noise=2e-3, meta=None):
    """Closed loop on the plant. ref(t) -> (X*, Xd*, Xdd*); with free=True only the position
    is commanded and the angle follows the least-tension posture (from the feedforward)."""
    arm = ChainArm(scheme="minimal")
    X0, L0, D0 = start_state(X_start, free)
    plant = Plant(arm, PAYLOAD, X0, L0, w_ext=w_ext, mass=mass)
    if SENSING == "lengths":
        imu, enc = IMU.from_spec(), TendonEncoders()
    else:
        imu, enc = IMU(accel_noise=imu_noise, gyro_noise=imu_noise), None
    est = Estimator(X0)
    ff = Controller(ChainArm(scheme="minimal"), C, PAYLOAD, mode="dynamic",
                    orientation="free" if free else "track", payload_stability="report")
    ff.reset(D0, L0)
    fb = LinearizedClosedLoop(arm, C, PAYLOAD, wn=wn)
    L_est, Xh, Vh = L0.copy(), X0.copy(), np.zeros(3)
    psi_prev, psid_prev = X0[2], 0.0
    rows, status, start = [], "completed", time.time()
    for k in range(1, int(round(duration / DT)) + 1):
        t = k * DT
        Xr, Xdr, Xddr = (np.asarray(v, float).copy() for v in ref(t))
        T_ff, info_ff = ff.step(t, Xr, Xdr, Xddr)
        if free:                       # angle reference: the feedforward's least-tension posture
            psi = info_ff["P"][-1][2]
            Xr[2], Xdr[2] = psi, wrap(psi - psi_prev) / DT
            Xddr[2] = (Xdr[2] - psid_prev) / DT if k > 1 else 0.0
            psi_prev, psid_prev = psi, Xdr[2]
            w_ff = np.r_[PAYLOAD.m * Xddr[:2], 0.0]
        else:
            w_ff = PAYLOAD.M @ Xddr
        T, info = fb.step((Xr, Xdr, Xddr), (Xh, Vh), ff.history[-1][0], ff.history[-1][1], w_ff)
        try:
            for _ in range(SUB):
                acc = plant.step(T, DT / SUB)
                f_, g_ = imu.read(plant.X[2], plant.V[2], acc[:2])
                Xh, Vh = est.update(f_, g_, DT / SUB)
            if enc is not None:
                lam = enc.read(tendons(arm.evaluate(plant.L, T)["P"], arm.offsets)[0])
                L_est = est.length_correct(arm, T, L_est, lam, DT)
            else:
                L_est = est.model_correct(arm, T, L_est, plant.M, plant.c)
            Xh, Vh = est.X.copy(), est.V.copy()
        except PlantError as err:
            status = f"plant failed at t = {t:.2f}: {err}"
            break
        rows.append({"t": t, "X_ref": Xr, "X": plant.X.copy(), "X_est": Xh.copy(), "T": T,
                     "L": plant.L.copy(), "ok": info_ff["feasible"],
                     "time": info["time"] + info_ff["time"],
                     "err": np.hypot(*(plant.X[:2] - Xr[:2])), "err_psi": wrap(plant.X[2] - Xr[2]),
                     "est_err": np.hypot(*(plant.X[:2] - Xh[:2])),
                     "w_ext": np.zeros(3) if w_ext is None else np.asarray(w_ext(t), float),
                     "m": PAYLOAD.m if mass is None else mass(t)[0]})
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(OUT / f"show_{name}.npz", status=status, dt=DT, free=free, meta=str(meta or {}),
             sensing=SENSING,
             **arrays)
    e, ep = arrays["err"], np.abs(arrays["err_psi"])
    print(f"{name}: {status}; {len(rows)} steps; position error mean {e.mean():.4f} max {e.max():.4f}; "
          f"angle error mean {np.degrees(ep.mean()):.2f} max {np.degrees(ep.max()):.2f} deg; "
          f"estimate error max {arrays['est_err'].max():.4f}; "
          f"max tension {arrays['T'].max():.1f}; controller {1e3 * arrays['time'].mean():.1f} ms "
          f"({time.time() - start:.0f} s)", flush=True)
    return arrays


# ------------------------------------------------------------------ references
def seq(*legs):
    """Concatenate legs (duration, f(tau) -> (X, dX/dtau, d2X/dtau2)) timed by quintics."""
    ends = np.cumsum([d for d, _ in legs])

    def ref(t):
        i = min(int(np.searchsorted(ends, t - 1e-12)), len(legs) - 1)
        d, f = legs[i]
        s, ds, dds = quintic((t - (ends[i - 1] if i else 0.0)) / d)
        X, dX, ddX = f(s)
        return X, dX * ds / d, ddX * ds ** 2 / d ** 2 + dX * dds / d ** 2
    return ref, ends[-1]


def line(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return lambda s: (A + (B - A) * s, B - A, np.zeros(3))


def hold(X):
    X = np.asarray(X, float)
    return lambda s: (X, np.zeros(3), np.zeros(3))


def rounded_rect(cx, cy, w, h, rad, psi):
    """Closed path parametrised by s in [0, 1] (arc length), counter-clockwise from the middle
    of the bottom edge; numerical derivatives of an exact geometric curve."""
    straight = [(w / 2 - rad, (cx, cy - h / 2), (1, 0)), (h - 2 * rad, (cx + w / 2, cy - h / 2 + rad), (0, 1)),
                (w - 2 * rad, (cx + w / 2 - rad, cy + h / 2), (-1, 0)), (h - 2 * rad, (cx - w / 2, cy + h / 2 - rad), (0, -1)),
                (w / 2 - rad, (cx - w / 2 + rad, cy - h / 2), (1, 0))]
    arcs = [((cx + w / 2 - rad, cy - h / 2 + rad), -np.pi / 2), ((cx + w / 2 - rad, cy + h / 2 - rad), 0.0),
            ((cx - w / 2 + rad, cy + h / 2 - rad), np.pi / 2), ((cx - w / 2 + rad, cy - h / 2 + rad), np.pi)]
    pieces = []
    for i in range(5):
        pieces.append(("L",) + straight[i])
        if i < 4:
            pieces.append(("A", np.pi / 2 * rad) + arcs[i])
    total = sum(p[1] for p in pieces)

    def point(s):
        d = (s % 1.0) * total
        for p in pieces:
            if d <= p[1] + 1e-12:
                if p[0] == "L":
                    return np.array(p[2]) + d * np.array(p[3])
                a = p[3] + d / rad
                return np.array(p[2]) + rad * np.array([np.cos(a), np.sin(a)])
            d -= p[1]
        return np.array(pieces[-1][2]) + pieces[-1][1] * np.array(pieces[-1][3])

    def f(s, h_=1e-4):
        p0, pp, pm = point(s), point(s + h_), point(s - h_)
        return (np.r_[p0, psi], np.r_[(pp - pm) / (2 * h_), 0.0],
                np.r_[(pp - 2 * p0 + pm) / h_ ** 2, 0.0])
    return f


def lemniscate(cx, cy, a, psi):
    """Figure-eight x = a cos u / (1 + sin^2 u), y = a sin u cos u / (1 + sin^2 u), u = 2 pi s."""
    def p(u):
        q = 1 + np.sin(u) ** 2
        return np.array([cx + a * np.cos(u) / q, cy + a * np.sin(u) * np.cos(u) / q])

    def f(s, h_=1e-4):
        u = 2 * np.pi * s
        p0, pp, pm = p(u), p(u + h_), p(u - h_)
        d1 = (pp - pm) / (2 * h_) * 2 * np.pi
        d2 = (pp - 2 * p0 + pm) / h_ ** 2 * (2 * np.pi) ** 2
        return np.r_[p0, psi], np.r_[d1, 0.0], np.r_[d2, 0.0]
    return f


def circle(cx, cy, R, a0):
    def f(s):
        a = a0 + 2 * np.pi * s
        ca, sa = np.cos(a), np.sin(a)
        return (np.r_[cx + R * ca, cy + R * sa, 0.0], np.r_[-R * sa, R * ca, 0.0] * 2 * np.pi,
                np.r_[-R * ca, -R * sa, 0.0] * (2 * np.pi) ** 2)
    return f


# ------------------------------------------------------------------ demos
def demo_wrist():
    P = (1.6, 1.6)
    a, b = np.r_[P, np.radians(20)], np.r_[P, np.radians(135)]
    ref, dur = seq((1.0, hold(a)), (6.0, line(a, b)), (1.0, hold(b)), (6.0, line(b, a)),
                   (1.0, hold(a)))
    simulate("wrist", ref, dur, a, meta={"position": P, "angles_deg": (20, 135)})


def demo_waiter():
    f = rounded_rect(1.4, 1.8, 1.0, 0.6, 0.12, UP)
    ref, dur = seq((16.0, f))
    simulate("waiter", ref, dur, f(0.0)[0], meta={"rect": (1.4, 1.8, 1.0, 0.6), "psi_deg": 90})


def demo_figure8():
    f = lemniscate(1.6, 1.5, 0.45, UP)
    ref, dur = seq((16.0, f))
    simulate("figure8", ref, dur, f(0.0)[0], meta={"centre": (1.6, 1.5), "a": 0.45, "psi_deg": 90})


def demo_disturb():
    H = np.array([1.1, 1.8, UP])
    G = np.array([1.7, 1.8, UP])
    kicks = [(2.0, np.array([0.4, 0.0, 0.0])), (4.0, np.array([0.0, -0.4, 0.0])),
             (6.0, np.array([0.0, 0.0, 0.05])), (11.0, np.array([-0.3, 0.3, 0.0]))]

    def w_ext(t):                           # 0.05 s pushes
        for t0, w in kicks:
            if t0 <= t < t0 + 0.05:
                return w
        return np.zeros(3)

    ref, dur = seq((8.0, hold(H)), (5.0, line(H, G)), (5.0, line(G, H)), (1.0, hold(H)))
    simulate("disturb", ref, dur, H, w_ext=w_ext,
             meta={"hold": tuple(H), "kicks": [(t0, tuple(w)) for t0, w in kicks]})


def demo_speeds():
    c, R = (1.45, 1.5), 0.55
    top = np.array([c[0], c[1] + R, 0.0])
    out = {}
    for lap in (24.0, 16.0, 10.0, 6.0):
        ref, dur = seq((lap, circle(c[0], c[1], R, np.pi / 2)))
        z = simulate(f"speed_{lap:g}", ref, dur, top, free=True,
                     meta={"centre": c, "radius": R, "lap_s": lap})
        out[lap] = (z["err"].mean(), z["err"].max())
    print("speed sweep (lap s: mean, max error):", {k: tuple(round(x, 4) for x in v) for k, v in out.items()})


if __name__ == "__main__":
    demos = {"wrist": demo_wrist, "waiter": demo_waiter, "figure8": demo_figure8,
             "disturb": demo_disturb, "speeds": demo_speeds}
    for name in sys.argv[1:] or list(demos):
        demos[name]()

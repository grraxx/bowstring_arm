"""
run_sofa.py
The closed-loop controller (tip IMU only) on the SOFA arm, for the showcase demos.

Same controller, references, gains and start as scripts/run_showcase.py; only the plant
changes. SOFA (sofa/scene.py: corotational beams, SoftRobots cables, a rigid payload, light rod
mass) is started on the model's equilibrium shape at the start pose; the controller then sees
only the IMU on the tip (accelerometer from SOFA's tip motion, gyro, noise 2e-3) and knows the
start pose. Control 100 Hz, SOFA and the IMU 1 kHz.

usage: python sofa/run_sofa.py [wrist waiter figure8 disturb speed_24 speed_16 speed_10 speed_6]
       -> data/runs/sofa_<name>.npz    (sofa-env)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sofa.scene import SofaArm, model_nodes                                 # noqa: E402
from bowstring.chain import ChainArm                                        # noqa: E402
from bowstring.closed_loop import (IMU, Estimator, LinearizedClosedLoop,  # noqa: E402
                                   TendonEncoders, wrap)
from bowstring.controller import Controller                                 # noqa: E402
import run_showcase as rs                                                   # noqa: E402

NE = int(os.environ.get("SOFA_NE", 160))     # beam elements per segment (check_static.py)
TAG = os.environ.get("SOFA_TAG", "")
SENSING = os.environ.get("SENSING", "lengths")  # "lengths": high-end IMU + tendon encoders;
                                                # "imu": the earlier IMU-only (noise 2e-3)
ALPHA = 0.1                   # SOFA mass damping; the controller assumes c = ALPHA M


def simulate(name, ref, duration, X_start, free=False, w_ext=None, wn=(5, 5, 15),
             imu_noise=2e-3):
    arm = ChainArm(scheme="minimal")
    X0, L0, D0 = rs.start_state(X_start, free)
    T0 = arm.tension_map(rs.C, D0)[0]
    sim = SofaArm(model_nodes(L0, arm.n, NE), T0, rs.PAYLOAD, NE=NE, alpha=ALPHA)
    sub = int(round(rs.DT / sim.dt))
    Mv = np.array([rs.PAYLOAD.m, rs.PAYLOAD.m, rs.PAYLOAD.I])
    c = ALPHA * Mv
    if SENSING == "lengths":
        imu, enc = IMU.from_spec(), TendonEncoders()
    else:
        imu, enc = IMU(accel_noise=imu_noise, gyro_noise=imu_noise), None
    est = Estimator(X0)
    ff = Controller(ChainArm(scheme="minimal"), rs.C, rs.PAYLOAD, mode="dynamic",
                    orientation="free" if free else "track", payload_stability="report")
    ff.reset(D0, L0)
    fb = LinearizedClosedLoop(arm, rs.C, rs.PAYLOAD, wn=wn)
    L_est, Xh, Vh = L0.copy(), X0.copy(), np.zeros(3)
    psi_prev, psid_prev = X0[2], 0.0
    rows, status, start = [], "completed", time.time()
    for k in range(1, int(round(duration / rs.DT)) + 1):
        t = k * rs.DT
        Xr, Xdr, Xddr = (np.asarray(v, float).copy() for v in ref(t))
        T_ff, info_ff = ff.step(t, Xr, Xdr, Xddr)
        if free:
            psi = info_ff["P"][-1][2]
            Xr[2], Xdr[2] = psi, wrap(psi - psi_prev) / rs.DT
            Xddr[2] = (Xdr[2] - psid_prev) / rs.DT if k > 1 else 0.0
            psi_prev, psid_prev = psi, Xdr[2]
            w_ff = np.r_[rs.PAYLOAD.m * Xddr[:2], 0.0]
        else:
            w_ff = rs.PAYLOAD.M @ Xddr
        T, info = fb.step((Xr, Xdr, Xddr), (Xh, Vh), ff.history[-1][0], ff.history[-1][1], w_ff)
        sim.set_tensions(T)
        for _ in range(sub):
            sim.set_push(np.zeros(3) if w_ext is None else w_ext(sim.t))
            acc = sim.step()
            X, V = sim.tip()
            f_, g_ = imu.read(X[2], V[2], acc[:2])
            Xh, Vh = est.update(f_, g_, sim.dt)
        if enc is not None:
            L_est = est.length_correct(arm, T, L_est, enc.read(sim.tendon_lengths()), rs.DT)
        else:
            L_est = est.model_correct(arm, T, L_est, Mv, c)
        Xh, Vh = est.X.copy(), est.V.copy()
        X, V = sim.tip()
        if not np.all(np.isfinite(X)) or np.hypot(*(X[:2] - Xr[:2])) > 1.0:
            status = f"diverged at t = {t:.2f}"
            break
        rows.append({"t": t, "X_ref": Xr, "X": X.copy(), "X_est": Xh.copy(), "T": T,
                     "nodes": sim.nodes()[:: NE // 20].astype(np.float32),
                     "ok": info_ff["feasible"], "time": info["time"] + info_ff["time"],
                     "err": np.hypot(*(X[:2] - Xr[:2])), "err_psi": wrap(X[2] - Xr[2]),
                     "est_err": np.hypot(*(X[:2] - Xh[:2])),
                     "w_ext": np.zeros(3) if w_ext is None else np.asarray(w_ext(t), float)})
    rs.OUT.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    np.savez(rs.OUT / f"sofa_{name}{TAG}.npz", status=status, dt=rs.DT, free=free, NE=NE, sensing=SENSING, **arrays)
    e, ep, ee = arrays["err"], np.abs(arrays["err_psi"]), arrays["est_err"]
    print(f"{name}: {status}; position error mean {e.mean():.4f} max {e.max():.4f}; angle error "
          f"max {np.degrees(ep.max()):.2f} deg; IMU estimate error mean {ee.mean():.4f} max "
          f"{ee.max():.4f}; max tension {arrays['T'].max():.1f} ({time.time() - start:.0f} s)",
          flush=True)
    return arrays


def demos():
    """The showcase's demos, rebuilt with this module's simulate."""
    UP = rs.UP
    f_rect = rs.rounded_rect(1.4, 1.8, 1.0, 0.6, 0.12, UP)
    f_8 = rs.lemniscate(1.6, 1.5, 0.45, UP)
    a, b = np.r_[1.6, 1.6, np.radians(20)], np.r_[1.6, 1.6, np.radians(135)]
    H, G = np.array([1.1, 1.8, UP]), np.array([1.7, 1.8, UP])
    kicks = [(2.0, np.array([0.4, 0.0, 0.0])), (4.0, np.array([0.0, -0.4, 0.0])),
             (6.0, np.array([0.0, 0.0, 0.05])), (11.0, np.array([-0.3, 0.3, 0.0]))]

    def w_ext(t):
        for t0, w in kicks:
            if t0 <= t < t0 + 0.05:
                return w
        return np.zeros(3)

    out = {
        "wrist": lambda: simulate("wrist", *rs.seq(
            (1.0, rs.hold(a)), (6.0, rs.line(a, b)), (1.0, rs.hold(b)), (6.0, rs.line(b, a)),
            (1.0, rs.hold(a))), a),
        "waiter": lambda: simulate("waiter", *rs.seq((16.0, f_rect)), f_rect(0.0)[0]),
        "figure8": lambda: simulate("figure8", *rs.seq((16.0, f_8)), f_8(0.0)[0]),
        "disturb": lambda: simulate("disturb", *rs.seq(
            (8.0, rs.hold(H)), (5.0, rs.line(H, G)), (5.0, rs.line(G, H)), (1.0, rs.hold(H))), H,
            w_ext=w_ext),
    }
    c, R = (1.45, 1.5), 0.55
    for lap in (24, 16, 10, 6):
        out[f"speed_{lap}"] = (lambda lap=lap: simulate(
            f"speed_{lap}", *rs.seq((float(lap), rs.circle(c[0], c[1], R, np.pi / 2))),
            np.array([c[0], c[1] + R, 0.0]), free=True))
    return out


if __name__ == "__main__":
    table = demos()
    for name in sys.argv[1:] or list(table):
        table[name]()

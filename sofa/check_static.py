"""
check_static.py
Does the SOFA scene agree with the model at constant tensions?

At a pose where the payload is open-loop stable, the model gives the tensions that hold it at
rest and the arm's equilibrium shape. SOFA starts on that shape with those tensions, heavily
damped, and settles. Reported per mesh (beam elements per segment):
  - the tip's drift from the model's rest pose;
  - the model's payload wrench at SOFA's settled pose (zero if the two mechanics agree). This is
    the better measure: at these poses the payload is very soft in one direction (K_X ~ 0.08),
    so a small force difference shows up as a large drift.

usage: python sofa/check_static.py     (sofa-env)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from sofa.scene import SofaArm, model_nodes                     # noqa: E402
from bowstring.chain import ChainArm                            # noqa: E402
from run_showcase import C, D_UP, PAYLOAD, start_state          # noqa: E402

MESHES = (20, 40, 80, 160)


def settle(L, T, NE, seconds=30.0):
    s = SofaArm(model_nodes(L, 3, NE), T, PAYLOAD, alpha=2.0, NE=NE)
    for i in range(int(seconds / s.dt)):
        s.step()
        if i > 2000 and np.abs(s.tip()[1]).max() < 1e-8:
            break
    return s.poses()


def main():
    arm = ChainArm(scheme="minimal")
    T_up = arm.tension_map(C, D_UP)[0]
    _, L_c, D_c = start_state(np.array([1.45, 2.05, 0.0]), free=True)
    cases = {"upright rest": (T_up, arm.rest(T_up)["L"]),
             "circle top (1.45, 2.05)": (arm.tension_map(C, D_c)[0], L_c)}
    for name, (T, L) in cases.items():
        P = arm.evaluate(L, T)["P"]
        K = arm.forward(P[-1], T, L, report=True)["K_X"]
        print(f"{name}: model tip {np.round(P[-1], 4)}, K_X eigenvalues "
              f"{np.round(np.linalg.eigvalsh(0.5 * (K + K.T)), 3)}", flush=True)
        for NE in MESHES:
            Ps = settle(L, T, NE)
            w = arm.forward(Ps[-1], T, L, report=False)["w"]
            print(f"  {NE:4d} elements/segment: tip drift {np.hypot(*(Ps[-1, :2] - P[-1, :2])):.4f}, "
                  f"model wrench at SOFA's pose |w_xy| {np.hypot(*w[:2]):.4f}, "
                  f"moment {abs(w[2]):.4f}", flush=True)


if __name__ == "__main__":
    main()

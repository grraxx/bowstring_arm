"""
sensitivity_scan.py
Task-capability and parameter-sensitivity data for the n = 3 bowstring arm (ChainArm, minimal
tension scheme, no tension cap), over every posture the tensions can make.

Each sample is one set of tension differences D on a grid (denser near D = 0); the arm's rest
equilibrium under it (payload free, w = 0) is found warm-started along the grid, so each sample
is one configuration of the arm holding the payload. At each equilibrium, in closed form from the
Hessian H (in the disk poses) and the tendon gradients:
    margin_I   smallest eigenvalue of H over the inner disks: > 0 means the arm holds its shape
               for a held payload pose (what the controller relies on)
    K_X        payload stiffness, Schur complement of H (open loop stable if positive definite)
    W_D        wrench on the payload per unit change of D at a fixed payload pose (inner disks
               re-equilibrating): dD/dw = -W_D^-1, and open loop dX/dD = -K_X^-1 W_D
    J_lam      tendon lengths per unit payload motion, inner disks following: encoder observability
and by finite differences, the parameter sensitivity: the wrench error dw_p at the same payload
pose and tensions when one parameter (d1, d2, d3, B1, B2, B3, Tmin) is 10 % larger. A controller
whose model is off by that much applies that wrench error. plot_sensitivity.py turns all this
into task maps.

usage: python scripts/sensitivity_scan.py [--Dmax 8] [--n 40] [--workers 16]
       -> data/sensitivity_scan.npz
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOMINAL = {"offsets": (0.15, 0.20, 0.25), "stiffness": (1.0, 1.0, 1.0), "Tmin": 0.1}
PARAMS = ["d1", "d2", "d3", "B1", "B2", "B3", "Tmin"]
REL = 0.10                                     # parameter perturbation, relative


def perturbed(name):
    from bowstring.chain import ChainArm
    off, B, Tmin = list(NOMINAL["offsets"]), list(NOMINAL["stiffness"]), NOMINAL["Tmin"]
    if name[0] == "d":
        off[int(name[1]) - 1] *= 1 + REL
    elif name[0] == "B":
        B[int(name[1]) - 1] *= 1 + REL
    elif name == "Tmin":
        Tmin *= 1 + REL
    return ChainArm(offsets=off, stiffness=B, Tmin=Tmin, scheme="minimal")


def metrics(arm, arms_p, D, r):
    C = np.zeros(3)
    T, dT = arm.tension_map(C, D)
    ev = arm.evaluate(r["L"], T, hessian=True)
    H, lg = ev["H"], ev["lam_grad"]
    if not np.isfinite(H).all():
        return None
    iI, iX = arm.iI, arm.iX
    Gd = (lg[:, 0] * dT[:, :1] + lg[:, 1] * dT[:, 1:]).T            # dG/dD, (9, 3)
    HII, HIX = H[iI, iI], H[iI, iX]
    KX = H[iX, iX] - HIX.T @ np.linalg.solve(HII, HIX)
    WD = Gd[iX] - HIX.T @ np.linalg.solve(HII, Gd[iI])
    dPi = -np.linalg.solve(HII, HIX)
    n = arm.n
    lgm = lg.reshape(2 * n, 3 * n)
    Jl = lgm[:, iI] @ dPi + lgm[:, iX]
    X = r["P"][-1]
    dw = np.full((len(PARAMS), 3), np.nan)
    for k, a in enumerate(arms_p):
        try:
            f = a.forward(X, a.tension_map(C, D)[0], r["L"], report=False)
            if f["converged"]:
                dw[k] = f["w"]
        except (ValueError, np.linalg.LinAlgError):
            pass
    con = arm.contact(r["L"], r["P"])
    return {"contact": con["margin"], "chord": con["chord"], "bend": con["bend"],
            "KX": 0.5 * (KX + KX.T), "WD": WD, "margin_I": float(np.linalg.eigvalsh(HII)[0]),
            "margin": float(np.linalg.eigvalsh(H)[0]), "sig_enc": np.linalg.svd(Jl, compute_uv=False),
            "T_max": float(T.max()), "dw": dw}


def scan_slice(args):
    """All (D2, D3) for one D1, warm-started along the grid (snake order)."""
    d1, grid = args
    from bowstring.chain import ChainArm
    arm = ChainArm(scheme="minimal", **NOMINAL)
    arms_p = [perturbed(p) for p in PARAMS]
    C = np.zeros(3)
    def attempt(T, L0):
        try:
            r = arm.rest(T, L0)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return r if r["converged"] else None

    def solve(D, starts):
        """Warm start from each neighbour (L0, D0); if that fails, walk the tensions from D0 to D
        in 4 steps. Cold start (direct rod, ~3 s) only when there is no neighbour."""
        for L0, D0 in starts:
            r = attempt(arm.tension_map(C, D)[0], L0)
            if r is not None:
                return r
            Lw = L0
            for s in (0.25, 0.5, 0.75, 1.0):
                r = attempt(arm.tension_map(C, D0 + s * (D - D0))[0], Lw)
                if r is None:
                    break
                Lw = r["L"]
            if r is not None:
                return r
        return None if starts else attempt(arm.tension_map(C, D)[0], None)

    out, last, prev = [], None, {}
    for a, d2 in enumerate(grid):
        order = range(len(grid)) if a % 2 == 0 else range(len(grid) - 1, -1, -1)
        row = {}
        for j in order:
            D = np.array([d1, d2, grid[j]])
            starts = [x for x in (last, prev.get(j)) if x is not None]
            r = solve(D, starts)
            if r is None:
                out.append((D, None, None))
                continue
            last = row[j] = (r["L"], D)
            try:
                m = metrics(arm, arms_p, D, r)
            except (ValueError, np.linalg.LinAlgError):
                m = None
            if m is not None:
                m["segments_stable"] = bool(r["segments_stable"])
            out.append((D, r["P"], m))
        prev = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Dmax", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=40,
                    help="grid points per D (even, so D = 0, a straight singular arm, is skipped)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "data/sensitivity_scan.npz"))
    ap.add_argument("--part", default=None,
                    help="k/K: only every K-th D1 slice from k, written to <out>.part<k>.npz "
                         "(to split a long scan into short runs)")
    ap.add_argument("--merge", type=int, default=0, help="K: merge <out>.part0..K-1.npz into <out>")
    args = ap.parse_args()
    if args.merge:
        merge(args.out, args.merge)
        return
    u = np.linspace(-1, 1, args.n)
    grid = np.sign(u) * args.Dmax * np.abs(u) ** 1.5           # denser near D = 0
    d1s = grid
    if args.part:
        k, K = (int(v) for v in args.part.split("/"))
        d1s = grid[k::K]
        args.out = f"{args.out}.part{k}.npz"
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(scan_slice, [(d1, grid) for d1 in d1s], chunksize=1)
    rows = [r for p in parts for r in p]
    N = len(rows)
    out = {"D": np.array([r[0] for r in rows]), "P": np.full((N, 4, 3), np.nan),
           "ok": np.zeros(N, bool), "seg_ok": np.zeros(N, bool),
           "margin_I": np.full(N, np.nan), "margin": np.full(N, np.nan),
           "KX": np.full((N, 3, 3), np.nan), "WD": np.full((N, 3, 3), np.nan),
           "sig_enc": np.full((N, 3), np.nan), "T_max": np.full(N, np.nan),
           "dw": np.full((N, len(PARAMS), 3), np.nan), "contact": np.full(N, np.nan),
           "chord": np.full(N, np.nan), "bend": np.full(N, np.nan)}
    for i, (D, P, m) in enumerate(rows):
        if P is not None:
            out["P"][i] = P
        if m is None:
            continue
        out["ok"][i] = True
        out["seg_ok"][i] = m["segments_stable"]
        for k in ("margin_I", "margin", "KX", "WD", "sig_enc", "T_max", "dw", "contact", "chord", "bend"):
            out[k][i] = m[k]
    np.savez(args.out, grid=grid, params=np.array(PARAMS), rel=REL, offsets=NOMINAL["offsets"],
             Tmin=NOMINAL["Tmin"], **out)
    good = out["ok"] & out["seg_ok"] & (out["margin_I"] > 0)
    print(f"{N} equilibria in {time.time() - t0:.0f} s: {out['ok'].sum()} evaluated, "
          f"{good.sum()} with the arm stable (margin_I > 0, segments stable), "
          f"{(good & (out['margin'] > 0)).sum()} also open-loop stable; "
          f"{(good & (out['contact'] >= 0)).sum()} usable with the rod clear of its tendons. Wrote {args.out}")


def merge(out, K):
    zs = [dict(np.load(f"{out}.part{k}.npz")) for k in range(K)]
    per_row = [k for k, v in zs[0].items() if v.ndim and len(v) == len(zs[0]["D"])]
    merged = {k: (np.concatenate([z[k] for z in zs]) if k in per_row else zs[0][k]) for k in zs[0]}
    np.savez(out, **merged)
    ok = merged["ok"] & merged["seg_ok"] & (merged["margin_I"] > 0)
    print(f"merged {K} parts: {len(merged['D'])} samples, {ok.sum()} usable, "
          f"{(ok & (merged['contact'] >= 0)).sum()} usable and clear of the tendons -> {out}")


if __name__ == "__main__":
    main()

"""
workspace_scan.py
The static workspace over every posture: rest equilibria (payload free, w = 0) over a grid of
tension differences D, with no tension limits (minimal co-contraction, ChainArm(scheme=
"minimal")). A state counts if it is stable with the payload free (Hessian in all disk poses
positive definite, every segment stable). Unlike workspace_map.py, which follows the
least-tension posture, this finds every posture the tensions can make.

usage: python scripts/workspace_scan.py [--Dmax 8] [--n 25] [--workers 10]
       -> data/workspace_scan.npz (tip poses, D, stable) and data/workspace_scan_grid.npz
          (occupancy of the (x, y) grid, step 0.05)
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


def scan_slice(args):
    """All (D2, D3) for one D1, warm-started along the grid (snake order)."""
    d1, grid = args
    from bowstring.chain import ChainArm
    arm = ChainArm(scheme="minimal")
    C = np.zeros(3)
    out = []
    L_row = None
    for a, d2 in enumerate(grid):
        L = L_row
        d3s = grid if a % 2 == 0 else grid[::-1]
        for b, d3 in enumerate(d3s):
            D = np.array([d1, d2, d3])
            T = arm.tension_map(C, D)[0]
            r = None
            for L0 in ((L,) if L is not None else ()) + (None,):
                try:
                    r = arm.rest(T, L0)
                except (ValueError, np.linalg.LinAlgError):
                    r = None
                if r is not None and r["converged"]:
                    break
            if r is None or not r["converged"]:
                out.append((D, np.full(3, np.nan), False))
                continue
            L = r["L"]
            if b == 0:
                L_row = L
            stable = bool(r["margin"] > 0 and r["segments_stable"])
            out.append((D, r["P"][-1], stable))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Dmax", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    grid = np.linspace(-args.Dmax, args.Dmax, args.n)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(scan_slice, [(d1, grid) for d1 in grid])
    rows = [r for p in parts for r in p]
    D = np.array([r[0] for r in rows])
    tip = np.array([r[1] for r in rows])
    stable = np.array([r[2] for r in rows])
    np.savez(ROOT / "data/workspace_scan.npz", D=D, tip=tip, stable=stable, grid=grid)
    h = 0.05
    xs = np.arange(-3.05, 3.05 + 1e-9, h)
    occ = np.zeros((len(xs), len(xs)), dtype=bool)
    good = stable & np.isfinite(tip[:, 0])
    i = np.clip(np.round((tip[good, 0] - xs[0]) / h).astype(int), 0, len(xs) - 1)
    j = np.clip(np.round((tip[good, 1] - xs[0]) / h).astype(int), 0, len(xs) - 1)
    occ[i, j] = True
    np.savez(ROOT / "data/workspace_scan_grid.npz", xs=xs, ys=xs, occ=occ)
    r = np.hypot(tip[good, 0], tip[good, 1])
    print(f"{len(rows)} rest solves in {time.time() - t0:.0f} s: {np.isfinite(tip[:, 0]).sum()} "
          f"converged, {good.sum()} stable; tip distance {r.min():.2f}..{r.max():.2f}; "
          f"x {tip[good, 0].min():.2f}..{tip[good, 0].max():.2f}, "
          f"y {tip[good, 1].min():.2f}..{tip[good, 1].max():.2f}")


if __name__ == "__main__":
    main()

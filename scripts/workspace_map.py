"""
workspace_map.py
Where the n = 3 arm can hold the payload at rest (w = 0) with its orientation free: a flood fill
over a grid of payload positions (x, y), starting at the rest pose of D0. At each node the
orientation-free inverse map (least normalised tension differences) is solved, warm-started from
a solved neighbour; the node is reachable if the tensions are within the limits and the
equilibrium is stable (margin > 0, every segment stable) with the payload free too: the payload
stiffness K_X is positive definite, since in open loop nothing else holds the payload.

usage: python scripts/workspace_map.py [--scheme minimal] [--Tmax none] [--h 0.05]
                                                                   -> data/workspace.npz
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=1.0, help="co-contraction: C = scale * (3, 2.5, 2)")
ap.add_argument("--h", type=float, default=0.05, help="grid step in x and y")
ap.add_argument("--scheme", default="minimal", choices=("fixed", "minimal"),
                help="tension scheme (see ChainArm); default minimal co-contraction")
ap.add_argument("--Tmax", type=float, default=None, help="tension limit (default: none)")
ap.add_argument("--out", default=str(ROOT / "data/workspace.npz"))
args = ap.parse_args()

arm = ChainArm(Tmax=args.Tmax, scheme=args.scheme)
C = args.scale * np.array([3.0, 2.5, 2.0])
D0 = np.array([0.8, 0.4, 0.2])
rest = arm.rest(arm.tensions(C, D0))
X0 = rest["P"][-1]
xs = np.arange(-3.05, 3.05 + 1e-9, args.h)
ys = np.arange(-3.05, 3.05 + 1e-9, args.h)
nx, ny = len(xs), len(ys)
status = np.zeros((nx, ny), dtype=np.int8)        # 0 untried, 1 reachable, 2 tried and not
D = np.full((nx, ny, 3), np.nan)
psi = np.full((nx, ny), np.nan)
margin = np.full((nx, ny), np.nan)
kmin = np.full((nx, ny), np.nan)          # smallest eigenvalue of the payload stiffness K_X
state = {}

# Seeds: the grid nodes nearest the rest pose of D0 and of its mirror image -D0 (the arm is
# symmetric about its axis, so the two halves mirror each other).
queue = deque()
for Ds in (D0, -D0):
    r = arm.rest(arm.tension_map(C, Ds)[0])
    i0, j0 = int(np.argmin(abs(xs - r["P"][-1][0]))), int(np.argmin(abs(ys - r["P"][-1][1])))
    queue.append(((i0, j0), (r["L"], Ds)))
t0 = time.time()
tried = 0
while queue:
    (i, j), (L, Dw) = queue.popleft()
    if status[i, j]:
        continue
    tried += 1
    try:
        # Offline: a generous solver budget (the controller's defaults fail fast instead).
        o = arm.inverse_free([xs[i], ys[j]], np.zeros(3), C, L, Dw, max_it=30, restore_its=40,
                             alpha_min=1e-4)
        ok = (o["converged"] and o["within_limits"] and o["margin"] > 0
              and o["segments_stable"] and np.linalg.eigvalsh(o["K_X"])[0] > 0)
    except (ValueError, np.linalg.LinAlgError):
        ok = False
    if not ok:
        status[i, j] = 2
        continue
    status[i, j] = 1
    D[i, j], psi[i, j], margin[i, j] = o["D"], o["P"][-1, 2], o["margin"]
    kmin[i, j] = np.linalg.eigvalsh(o["K_X"])[0]
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a, b = i + di, j + dj
        if 0 <= a < nx and 0 <= b < ny and not status[a, b]:
            queue.append(((a, b), (o["L"], o["D"])))
    if tried % 500 == 0:
        print(f"  {tried} nodes tried, {np.sum(status == 1)} reachable, {time.time() - t0:.0f} s",
              flush=True)

reach = status == 1
print(f"C = {np.round(C, 3)}: {reach.sum()} reachable nodes of {tried} tried "
      f"(area {reach.sum() * args.h ** 2:.2f}), {time.time() - t0:.0f} s")
print(f"  x {xs[reach.any(1)].min():.2f}..{xs[reach.any(1)].max():.2f}, "
      f"y {ys[reach.any(0)].min():.2f}..{ys[reach.any(0)].max():.2f}; "
      f"angle {np.nanmin(psi):.2f}..{np.nanmax(psi):.2f}")
print(f"  payload stiffness (smallest eigenvalue of K_X): median {np.nanmedian(kmin):.4f}, "
      f"5th percentile {np.nanpercentile(kmin, 5):.4f}")
np.savez(args.out, xs=xs, ys=ys, reach=reach, tried=status > 0, D=D, psi=psi, margin=margin,
         kmin=kmin, scheme=args.scheme, Tmax=np.nan if args.Tmax is None else args.Tmax,
         C=C, D0=D0, X0=X0, Tmin=arm.Tmin, offsets=arm.offsets)
print(f"Wrote {args.out}")

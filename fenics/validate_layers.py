"""
validate_layers.py
Checks of the layer files written by layers.py, against independent references
(the checks of validate_results.py, per layer):

1. analytic   phi1 = phi2 = 0 is the exact first clamped mode: fx = 16 K^2, U, |p|
2. bvp        random nodes with U <= energy_max re-solved by scipy solve_bvp (tol 1e-8)
              from the stored state: same branch, fx, fy, M1, M2, U agree
3. branches   in-layer grid edges: snaps, smooth crossings, same-sign jumps (rare),
              no node on the unstable S-shaped family
4. stability  every snap-edge node and a random sample are stable
5. closure    the energy disk is closed: no unsolved node beside a node with U <= energy_max
6. overlap    (layers at d = 0.97, 0.98, 0.99) against data/production/fields.npz
              (200 elements, grading 0.8), interpolated bicubically to the layer nodes

usage: python validate_layers.py <layers dir> [--samples 40] [--production ../data/production/fields.npz]
Exits with status 1 if a check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from branch import first_mode_projection, is_jump
from stability import unstable_directions
from validate_results import bvp_check, exact_first_mode

TOL = {"analytic": 2e-4, "bvp_rel": 2e-3, "same_sign_jump_edges": 1e-3}


def load_layer(path):
    z = np.load(path)
    L = {k: z[k] for k in z.files if k not in ("summary", "settings")}
    L["summary"], L["settings"] = json.loads(str(z["summary"])), json.loads(str(z["settings"]))
    M = len(L["phi"])
    row = np.full(M * M, -1)
    row[L["theta_nodes"]] = np.arange(len(L["theta_nodes"]))
    L["row"] = row
    return L


def theta_at(L, j, k):
    M = len(L["phi"])
    node = j * M + k
    if L["row"][node] >= 0:
        return L["theta"][L["row"][node]].astype(float)
    mirror = (M - 1 - k) * M + (M - 1 - j)
    return -L["theta"][L["row"][mirror]][::-1].astype(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("layers")
    ap.add_argument("--samples", type=int, default=40, help="BVP solves per layer")
    ap.add_argument("--stability-samples", type=int, default=2000)
    ap.add_argument("--production", default=str(Path(__file__).resolve().parents[1]
                                                / "data/production/fields.npz"))
    ap.add_argument("--json")
    args = ap.parse_args()
    files = sorted(Path(args.layers).glob("layer_*.npz"))
    rng = np.random.default_rng(0)
    prod = np.load(args.production) if Path(args.production).exists() else None
    report, failed = {"layers": []}, set()
    worst = {"analytic": 0.0, "bvp": 0.0}
    print(f"{'d':>8} {'nodes':>7} {'analytic':>9} {'bvp max':>9} {'bvp med':>9} {'snaps':>6} "
          f"{'smooth':>6} {'same':>5} {'S-fam':>5} {'unstab':>6} {'open':>5}  overlap vs production")
    for path in files:
        L = load_layer(path)
        d, phi, s = float(L["d"]), L["phi"], L["beam_x"]
        u_max = L["settings"]["energy_max"]
        tol = L["settings"]["jump_tol"]
        solved = L["solved"]
        M = len(phi)
        res = {"d": d, "nodes": int(solved.sum())}

        # 1. analytic
        c = M // 2
        fx, U, p = exact_first_mode(d)
        res["analytic"] = max(abs(L["fx"][c, c] / fx - 1), abs(L["energy"][c, c] / U - 1),
                              abs(abs(L["p"][c, c]) / p - 1))

        # 2. bvp on random nodes inside the energy disk
        cand = np.argwhere(solved & (L["energy"] <= u_max))
        picks = cand[rng.choice(len(cand), min(args.samples, len(cand)), replace=False)]
        errs, conv, same = [], 0, 0
        for j, k in picks:
            th = theta_at(L, j, k)
            out = bvp_check(d, phi[j], phi[k], th, s, L["fx"][j, k], L["fy"][j, k])
            if out is None:
                continue
            conv += 1
            bfx, bfy, m0, m1, bU, bp = out
            same += np.sign(bp) == np.sign(L["p"][j, k]) or abs(bp - L["p"][j, k]) < 1e-3 * abs(p)
            sf = max(abs(bfx), abs(bfy), 1.0)
            sm = max(abs(m0), abs(m1), 1.0)
            errs.append(max(abs(L["fx"][j, k] - bfx) / sf, abs(L["fy"][j, k] - bfy) / sf,
                            abs(L["moment"][j, k] - m0) / sm,
                            abs(L["moment_right"][j, k] - m1) / sm, abs(L["energy"][j, k] / bU - 1)))
        errs = np.array(errs) if errs else np.array([np.inf])
        res.update(bvp_max=float(errs.max()), bvp_median=float(np.median(errs)),
                   bvp_converged=f"{conv}/{len(picks)}", bvp_same_branch=int(same))

        # 3. branches on in-layer edges
        P, Q, E = L["p"], L["q"], np.where(solved, L["energy"], np.inf)
        snaps = smooth = same_sign = edges = 0
        near_snap = np.zeros_like(solved)
        for axis in (0, 1):
            a, b = [np.take(X, range(M - 1), axis=axis) for X in (P, Q)], \
                   [np.take(X, range(1, M), axis=axis) for X in (P, Q)]
            ok = np.take(solved, range(M - 1), axis=axis) & np.take(solved, range(1, M), axis=axis)
            change = (np.sign(a[0]) != np.sign(b[0])) & ok
            jump = is_jump(a[1], b[1], tol) & ok
            edges += int(ok.sum())
            snaps += int((change & jump).sum())
            smooth += int((change & ~jump).sum())
            same_sign += int((jump & ~change).sum())
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis], hi[axis] = slice(0, M - 1), slice(1, M)
            near_snap[tuple(lo)] |= change & jump
            near_snap[tuple(hi)] |= change & jump
        lowest = np.full(E.shape, np.inf)
        for axis in (0, 1):
            for sh in (1, -1):
                r = np.roll(E, sh, axis=axis)
                edge = [slice(None)] * 2
                edge[axis] = 0 if sh == 1 else -1
                r[tuple(edge)] = np.inf
                lowest = np.minimum(lowest, r)
        s_family = int((solved & (np.abs(Q) < tol) & (E > 2.0 * lowest)).sum())
        res.update(edges=edges, snaps=snaps, smooth=smooth, same_sign=same_sign, s_family=s_family)

        # 4. stability: snap-edge nodes and a sample
        flat = np.flatnonzero(solved.ravel())
        nodes = np.union1d(np.flatnonzero(near_snap.ravel()),
                           rng.choice(flat, min(args.stability_samples, len(flat)), replace=False))
        unstable = sum(bool(unstable_directions(theta_at(L, *divmod(int(n), M)), s,
                                                L["fx"].ravel()[n], L["fy"].ravel()[n]))
                       for n in nodes)
        res.update(stability_checked=len(nodes), unstable=int(unstable))

        # 5. closure of the energy disk
        inside = solved & (L["energy"] <= u_max)
        padded = np.pad(inside, 1)
        beside = np.zeros_like(inside)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                beside |= padded[1 + a:1 + a + M, 1 + b:1 + b + M]
        res["open"] = int((beside & ~solved).sum())

        # 6. overlap with production
        overlap = ""
        if prod is not None:
            i = np.flatnonzero(np.isclose(prod["d_vals"], d, atol=1e-9))
            if len(i):
                i = int(i[0])
                pp = prod["phi1_vals"]
                both = inside & (np.abs(phi)[:, None] <= pp[-1]) & (np.abs(phi)[None, :] <= pp[-1])
                jj, kk = np.nonzero(both)
                diffs = {}
                same_branch = np.sign(RectBivariateSpline(pp, pp, prod["p"][i])(phi[jj], phi[kk],
                                                                               grid=False)) \
                    == np.sign(P[jj, kk])
                for key in ("energy", "fx", "moment"):
                    interp = RectBivariateSpline(pp, pp, prod[key][i], kx=3, ky=3)(phi[jj], phi[kk],
                                                                                   grid=False)
                    scale = np.maximum(np.abs(L[key][jj, kk]), 1.0)
                    rel = np.abs(interp - L[key][jj, kk])[same_branch] / scale[same_branch]
                    diffs[key] = (float(np.median(rel)), float(np.percentile(rel, 99)))
                res["overlap"] = {k: {"median": v[0], "p99": v[1]} for k, v in diffs.items()}
                res["overlap"]["same_branch"] = float(same_branch.mean())
                overlap = ", ".join(f"{k} med {v[0]:.1e} p99 {v[1]:.1e}" for k, v in diffs.items()) \
                    + f"; same branch {same_branch.mean():.2%}"
        print(f"{d:8.5f} {res['nodes']:7d} {res['analytic']:9.1e} {res['bvp_max']:9.1e} "
              f"{res['bvp_median']:9.1e} {snaps:6d} {smooth:6d} {same_sign:5d} {s_family:5d} "
              f"{unstable:6d} {res['open']:5d}  {overlap}", flush=True)
        worst["analytic"] = max(worst["analytic"], res["analytic"])
        worst["bvp"] = max(worst["bvp"], res["bvp_max"])
        if res["analytic"] > TOL["analytic"]:
            failed.add("analytic")
        if res["bvp_max"] > TOL["bvp_rel"] or same != conv or conv < 0.95 * len(picks):
            failed.add("bvp")
        if same_sign > TOL["same_sign_jump_edges"] * max(edges, 1) or s_family:
            failed.add("branches")
        if unstable:
            failed.add("stability")
        if res["open"]:
            failed.add("closure")
        report["layers"].append(res)
    report["failed"] = sorted(failed)
    report["worst"] = worst
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    print(f"{len(files)} layers; worst analytic {worst['analytic']:.1e}, worst bvp {worst['bvp']:.1e}")
    print("FAILED: " + ", ".join(sorted(failed)) if failed else "All checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

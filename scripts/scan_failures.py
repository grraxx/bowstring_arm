"""
scan_failures.py
Are the gaps in sensitivity_scan.py numerical artifacts or physical limits of the arm?

A. Failed tension sets (no equilibrium found by the scan). Each is followed from a nearly
   straight arm along the ray s D, s: 0.02 -> 1, adaptive steps (halved down to 1e-4 on a failed
   Newton solve), tracking the arm's stability margin (smallest eigenvalue of the Hessian in all
   disk poses, payload free) and the contact margin (below).
     reached s = 1            the equilibrium exists on the branch from straight; the scan missed it
     stalls, margin -> 0      a fold of the rest map (saddle-node): past it the arm snaps through to
                              another branch, physically real (singularity of the tension -> pose map)
     stalls, margin healthy   a solver failure (numerical)
B. Every solved configuration:
     contact   least distance of the backbone from each straight tendon chord, on its own side;
               negative = the rod passes through a tendon, which the straight-chord (bowstring)
               model allows and a real arm cannot (the slack tendon would wrap on the rod)
     dataset   each segment checked against the clamped-elastica dataset (~/elastica, FEniCS,
               200 elements; data/segment_table.npz): inside its range (chord d >= 0.7)? energy
               and end wrench from shooting vs the table; table valid (stable, >= 2 grid cells
               from the snap fold)?
     fidelity  RK4 steps doubled (subsample)
C. Empty map cells (h = 0.1) inside reach: the orientation-free inverse map at the cell centre,
   warm-started from the nearest solved configurations: stable configuration found (sampling
   gap), only unstable ones (no stable configuration: physical), or none.

usage: python scripts/scan_failures.py [--workers 16] -> data/scan_failures.npz, summary
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_W = {}


def _arm():
    if "arm" not in _W:
        from bowstring.chain import ChainArm
        from bowstring.table import SegmentTable
        _W["arm"] = ChainArm(scheme="minimal")
        _W["table"] = SegmentTable()
    return _W["arm"], _W["table"]


def contact_margin(arm, L, P):
    """Least signed distance of the backbone from the tendon chords, on the side away from the
    chord (negative: the rod crosses a tendon). Pair k runs through segments 1..k+1."""
    from bowstring.tendons import hole
    shape = arm.shape(L)
    worst = np.inf
    for k in range(arm.n):
        for s in (1.0, -1.0):
            for j in range(1, k + 2):
                ga, _, _ = hole(P[j - 1], arm.offsets[k], s)
                gb, _, _ = hole(P[j], arm.offsets[k], s)
                u = (gb - ga) / np.linalg.norm(gb - ga)
                r = shape[j - 1] - ga
                cross = u[0] * r[:, 1] - u[1] * r[:, 0]
                worst = min(worst, float(np.min(-s * cross)))
    return worst


def dataset_check(arm, table, L, P):
    """Per segment: chord d, |bend|, table status (0 valid, 1 d < 0.7 beyond the dataset,
    2 inside its range but not valid: near or past the fold or outside the phi grid),
    relative energy error and wrench error (shooting vs table) where valid."""
    from bowstring.chain import relative_pose
    from bowstring.segment import integrate
    L = np.asarray(L).reshape(arm.n, 3)
    rep = arm.segment_report(L, P)
    out = []
    for i in range(arm.n):
        pose, M1, V, J, th = integrate(L[i] / arm.B[i], record=True)
        d = rep[i]["chart"][0]
        st = None
        if d >= table.d[0]:
            st = table.state(pose, 1 if rep[i]["p"] >= 0 else -1)
        if st is None:
            out.append((d, abs(th[-1] - th[0]), 1 if d < table.d[0] else 2, np.nan, np.nan))
            continue
        # shooting end wrench in pose coordinates, unit stiffness: g = (nx, ny, M(1)) at the end
        g = np.array([L[i, 0] / arm.B[i], L[i, 1] / arm.B[i], M1])
        eU = abs(V - st.V) / max(abs(st.V), 1e-3)
        eg = float(np.abs(g - st.g).max()) / max(float(np.abs(st.g).max()), 1e-2)
        out.append((d, abs(th[-1] - th[0]), 0, eU, eg))
    return np.array(out)


def shortest_chord(arm, P):
    """Shortest tendon chord (its two holes nearly coinciding: the bowstring model's tendon
    length |b - a| has a cone singularity at zero, its line of action flips)."""
    from bowstring.tendons import hole
    best = np.inf
    for k in range(arm.n):
        for s in (1.0, -1.0):
            for j in range(1, k + 2):
                a, b = hole(P[j - 1], arm.offsets[k], s)[0], hole(P[j], arm.offsets[k], s)[0]
                best = min(best, float(np.linalg.norm(b - a)))
    return best


def ray(D):
    """Part A: follow s D from s = 0.02."""
    arm, _ = _arm()
    C = np.zeros(3)
    D = np.asarray(D, float)
    P0 = np.array([[i, 0.0, 0.0] for i in range(arm.n + 1)])
    s, ds = 0.02, 0.05
    T = arm.tension_map(C, s * D)[0]
    try:
        r = arm.rest(T, arm.loads_from_poses(P0, T))
    except (ValueError, np.linalg.LinAlgError):
        r = None
    if r is None or not r["converged"]:
        try:
            r = arm.rest(T)                                  # cold (direct rod)
        except (ValueError, np.linalg.LinAlgError):
            return {"s_end": 0.0, "margin_end": np.nan, "margin0": np.nan, "s_contact": np.nan,
                    "X": np.full(3, np.nan), "seg_ok": False, "contact": np.nan, "chord": np.nan,
                    "bend": np.nan}
    margin0 = r["margin"]
    s_contact = np.nan
    while s < 1.0 - 1e-12:
        step = min(ds, 1.0 - s)
        T = arm.tension_map(C, (s + step) * D)[0]
        try:
            r2 = arm.rest(T, r["L"])
        except (ValueError, np.linalg.LinAlgError):
            r2 = None
        if r2 is None or not r2["converged"] or not np.isfinite(r2["margin"]):
            ds = step / 2
            if ds < 1e-4:
                break
            continue
        s, r = s + step, r2
        ds = min(0.1, ds * 1.5)
        if np.isnan(s_contact) and contact_margin(arm, r["L"], r["P"]) < 0:
            s_contact = s
    from bowstring.segment import integrate
    bend = max(abs(np.ptp(integrate(r["L"][i] / arm.B[i], record=True)[4][[0, -1]])) for i in range(arm.n))
    return {"s_end": s, "margin_end": r["margin"], "margin0": margin0, "s_contact": s_contact,
            "X": r["P"][-1], "seg_ok": bool(r["segments_stable"]),
            "contact": contact_margin(arm, r["L"], r["P"]), "chord": shortest_chord(arm, r["P"]),
            "bend": float(bend)}


def solved(args):
    """Part B on one solved configuration."""
    D, P, fid = args
    arm, table = _arm()
    T = arm.tension_map(np.zeros(3), D)[0]
    try:
        r = arm.rest(T, arm.loads_from_poses(P, T))
    except (ValueError, np.linalg.LinAlgError):
        r = None
    out = {"contact": np.nan, "seg": np.full((3, 5), np.nan), "dX_rk4": np.nan}
    if r is None or not r["converged"]:
        return out
    out["contact"] = contact_margin(arm, r["L"], r["P"])
    out["seg"] = dataset_check(arm, table, r["L"], r["P"])
    if fid:
        import bowstring.segment as seg
        steps = seg._steps
        seg._steps = lambda load: 2 * steps(load)
        try:
            r2 = arm.rest(T, r["L"])
            out["dX_rk4"] = float(np.abs(r2["P"][-1] - r["P"][-1]).max()) if r2["converged"] else np.nan
        finally:
            seg._steps = steps
    return out


def cell(args):
    """Part C: the orientation-free inverse map at (x, y), from nearby configurations."""
    xy, starts = args
    arm, _ = _arm()
    C = np.zeros(3)
    best = {"found": 0, "stable": False, "margin_I": np.nan, "contact": np.nan}
    for D0, P0 in starts:
        T0 = arm.tension_map(C, D0)[0]
        try:
            o = arm.inverse_free(xy, np.zeros(3), C, arm.loads_from_poses(P0, T0), D0, max_it=30,
                                 restore_its=40, alpha_min=1e-4)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not o["converged"]:
            continue
        best["found"] = 1
        ok = o["margin"] > 0 and o["segments_stable"]
        c = contact_margin(arm, o["L"], o["P"])
        if ok and (not best["stable"] or c > best["contact"]):
            best.update(stable=True, margin_I=o["margin"], contact=c)
        elif not best["stable"]:
            best.update(margin_I=o["margin"], contact=c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=str(ROOT / "data/sensitivity_scan.npz"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-fail", type=int, default=0, help="0: all failed sets")
    ap.add_argument("--n-fid", type=int, default=1000)
    args = ap.parse_args()
    # load every array once: indexing a lazy NpzFile (z["D"][i]) re-reads the whole array per call
    # and the row view pins that copy, so the job lists below grew to tens of GB (OOM-killed)
    z = dict(np.load(args.scan))
    conv = np.isfinite(z["P"][:, -1, 0])
    rng = np.random.default_rng(0)
    fail = np.nonzero(~conv)[0]
    if args.n_fail:
        fail = rng.choice(fail, args.n_fail, replace=False)
    good = np.nonzero(conv)[0]
    fid = set(rng.choice(good, min(args.n_fid, len(good)), replace=False).tolist())

    # empty cells inside reach (r < 2.95), next to filled ones
    use = z["ok"] & z["seg_ok"] & (z["margin_I"] > 0)
    X = z["P"][use, -1]
    h = 0.1
    ij = np.floor(X[:, :2] / h).astype(int)
    filled = set(map(tuple, ij))
    empty = []
    for i in range(-30, 31):
        for j in range(-30, 31):
            c = ((i + 0.5) * h, (j + 0.5) * h)
            if (i, j) in filled or math.hypot(*c) > 2.95:
                continue
            if any((i + a, j + b) in filled for a in (-1, 0, 1) for b in (-1, 0, 1)):
                empty.append(c)
    Duse, Puse = z["D"][use], z["P"][use]
    cell_args = []
    for c in empty:
        dist = np.hypot(X[:, 0] - c[0], X[:, 1] - c[1])
        near = np.argsort(dist)[:6]
        cell_args.append((np.array(c), [(Duse[k], Puse[k]) for k in near]))

    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        A = pool.map(ray, [z["D"][i] for i in fail], chunksize=8)
        print(f"A: {len(fail)} failed sets followed in {time.time() - t0:.0f} s", flush=True)
        B = pool.map(solved, [(z["D"][i], z["P"][i], i in fid) for i in good], chunksize=16)
        print(f"B: {len(good)} solved configurations checked, {time.time() - t0:.0f} s", flush=True)
        Cc = pool.map(cell, cell_args, chunksize=2)
        print(f"C: {len(empty)} empty cells, {time.time() - t0:.0f} s", flush=True)

    out = {"fail_idx": fail, "ok_idx": good, "cells": np.array(empty)}
    for pre, rows in (("a_", A), ("b_", B), ("c_", Cc)):
        for k in rows[0]:
            out[pre + k] = np.array([r[k] for r in rows])
    np.savez(ROOT / "data/scan_failures.npz", **out)
    summarize(z, out)


def summarize(z, o):
    s_end, m_end, m0 = o["a_s_end"], o["a_margin_end"], o["a_margin0"]
    reached = s_end >= 1 - 1e-9
    rel = m_end / np.abs(m0)
    fold = ~reached & (rel < 0.02)
    collapse = ~reached & ~fold & (o["a_chord"] < 0.02)
    other = ~reached & ~fold & ~collapse
    n = len(s_end)
    print(f"\nA. Failed tension sets ({n}), followed from a straight arm:")
    print(f"   reached (scan missed them):        {reached.mean():6.1%}"
          f"  - arm stable {np.mean(o['a_seg_ok'][reached] & (o['a_margin_end'][reached] > 0)):.0%},"
          f" clear of tendons {np.mean(o['a_contact'][reached] >= 0):.0%}")
    print(f"   stopped at a fold (margin -> 0):   {fold.mean():6.1%}  - at s = {np.median(s_end[fold]):.2f} (median)")
    print(f"   stopped, a tendon chord collapses: {collapse.mean():6.1%}  - shortest chord median "
          f"{np.nanmedian(o['a_chord'][collapse]):.4f}, largest segment bend median "
          f"{np.degrees(np.nanmedian(o['a_bend'][collapse])):.0f} deg")
    print(f"   stopped, unexplained:              {other.mean():6.1%}  - margin/initial median "
          f"{np.nanmedian(rel[other]) if other.any() else np.nan:.2f}, shortest chord median "
          f"{np.nanmedian(o['a_chord'][other]) if other.any() else np.nan:.3f}")
    sc = o["a_s_contact"]
    print(f"   rod touched a tendon before the end or stop: {np.mean(np.isfinite(sc)):.0%} (median s {np.nanmedian(sc):.2f})")
    c = o["b_contact"]
    seg = o["b_seg"]
    print(f"\nB. Solved configurations ({len(c)}):")
    print(f"   rod clear of every tendon:         {np.mean(c >= 0):.1%}")
    use = z["ok"][o["ok_idx"]] & z["seg_ok"][o["ok_idx"]] & (z["margin_I"][o["ok_idx"]] > 0)
    print(f"   usable (arm stable) and clear:     {np.mean(use & (c >= 0)):.1%} of solved, "
          f"{np.sum(use & (c >= 0))} configurations")
    st = seg[:, :, 2]
    print(f"   segments vs the elastica dataset:  valid {np.mean(st == 0):.1%}, beyond it (d < 0.7) "
          f"{np.mean(st == 1):.1%}, near/past the fold or off-grid {np.mean(st == 2):.1%}")
    eU, eg = seg[:, :, 3][st == 0], seg[:, :, 4][st == 0]
    print(f"   shooting vs FEniCS, where valid:   energy rel. err. median {np.nanmedian(eU):.1e}, 99th {np.nanpercentile(eU, 99):.1e};"
          f" wrench median {np.nanmedian(eg):.1e}, 99th {np.nanpercentile(eg, 99):.1e}")
    bend = np.degrees(seg[:, :, 1])
    clear = c >= 0
    print(f"   largest segment bend: clear configurations max {np.nanmax(bend[clear]):.0f} deg; "
          f"crossing ones median {np.nanmedian(np.nanmax(bend[~clear], 1)):.0f} deg")
    print(f"   RK4 steps doubled: payload moves median {np.nanmedian(o['b_dX_rk4']):.1e}, max {np.nanmax(o['b_dX_rk4']):.1e}")
    f, s_, cc = o["c_found"], o["c_stable"], o["c_contact"]
    print(f"\nC. Empty cells inside reach next to filled ones ({len(f)}):")
    print(f"   stable configuration found (sampling gap):   {np.mean(s_):.1%}"
          f"  - clear of tendons {np.mean(cc[s_] >= 0):.0%}")
    print(f"   only unstable configurations:                {np.mean(f.astype(bool) & ~s_):.1%}")
    print(f"   none found:                                  {np.mean(~f.astype(bool)):.1%}")


if __name__ == "__main__":
    main()

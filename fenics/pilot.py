"""
pilot.py
Beam resolution study before the layer runs: at points on the edge of the energy disk
(U ~ energy_max, the sharpest boundary layers) and inside it, solve on graded beams of
200 ... 4000 elements and compare fx, fy, M1, M2 and U with an 8000-element reference.
Also times a warm start from a neighbouring grid node at each resolution.

usage: python pilot.py [--grading 0.8]
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time

import numpy as np

from layers import Beam, DEFAULTS, layer_axis


def points():
    """(d, phi1, phi2): on the U = 10 circle in three directions, and half-way in."""
    out = []
    for d in (0.97, 0.99, 0.999, 0.9999):
        r = (16.0 * (1.0 - d) * 10.0) ** 0.25          # disk radius along an axis
        for a in (0.0, np.pi / 4, -np.pi / 4):          # axis, phi1 = phi2, phi1 = -phi2
            # U ~ (phi1^2 + phi2^2)^2: the same radius in every direction
            out.append((d, 0.95 * r * np.cos(a), 0.95 * r * np.sin(a)))
        out.append((d, 0.5 * r, -0.2 * r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grading", type=float, default=0.8)
    ap.add_argument("--elements", default="200,500,1000,2000,4000")
    ap.add_argument("--reference", type=int, default=8000)
    args = ap.parse_args()
    Ns = [int(n) for n in args.elements.split(",")]

    ref_beam = Beam(args.reference, args.grading)
    refs = []
    for d, p1, p2 in points():
        st = ref_beam.cold(d, p1, p2)
        refs.append(st)
        if st is None:
            print(f"reference failed at {(d, p1, p2)}")
        else:
            print(f"ref d={d} phi=({p1:+.3f},{p2:+.3f}): U={st['energy']:.4f} fx={st['fx']:.1f} "
                  f"fy={st['fy']:.2f} M1={st['moment']:.3f} M2={st['moment_right']:.3f} "
                  f"p={st['p']:+.4f} bl=1/sqrt(fx)={1 / np.sqrt(abs(st['fx'])):.4f}", flush=True)

    keys = ("fx", "fy", "moment", "moment_right", "energy")
    print(f"\nRelative error against N = {args.reference} (forces scaled by max(|fx|,|fy|,1), "
          "moments by max(|M1|,|M2|,1)); grading", args.grading)
    for N in Ns:
        beam = Beam(N, args.grading)
        errs, times = [], []
        for (d, p1, p2), ref in zip(points(), refs):
            if ref is None:
                continue
            guess = np.interp(beam.x, ref_beam.x, ref["theta"])
            st = beam.warm(d, p1, p2, guess, ref["fx"], ref["fy"])
            if st is None:
                errs.append({k: np.nan for k in keys})
                continue
            sf = max(abs(ref["fx"]), abs(ref["fy"]), 1.0)
            sm = max(abs(ref["moment"]), abs(ref["moment_right"]), 1.0)
            errs.append({"fx": abs(st["fx"] - ref["fx"]) / sf, "fy": abs(st["fy"] - ref["fy"]) / sf,
                         "moment": abs(st["moment"] - ref["moment"]) / sm,
                         "moment_right": abs(st["moment_right"] - ref["moment_right"]) / sm,
                         "energy": abs(st["energy"] / ref["energy"] - 1)})
            # Warm start from the neighbouring grid node of that layer's phi grid.
            phi, _ = layer_axis(d, DEFAULTS)
            step = phi[1] - phi[0]
            t = time.perf_counter()
            for _ in range(3):
                beam.warm(d, p1 + step, p2, st["theta"], st["fx"], st["fy"], max_it=50)
            times.append((time.perf_counter() - t) / 3)
        worst = {k: np.nanmax([e[k] for e in errs]) for k in keys}
        by_d = {}
        for (d, _, _), e in zip([p for p, r in zip(points(), refs) if r is not None], errs):
            by_d[d] = max(by_d.get(d, 0.0), max(e.values()))
        print(f"N={N:5d}: worst " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
              + "; worst by d " + ", ".join(f"{d}: {v:.1e}" for d, v in by_d.items())
              + f"; warm start {np.mean(times) * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()

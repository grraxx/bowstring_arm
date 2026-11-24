"""
merge_table.py
The stored segment space: production layers (d 0.70-0.96) and the high-d layers
(d 0.970-0.9999) merged into one table of the up branch, data/segment_table.npz.

usage: python merge_table.py [--production ../data/production/fields.npz]
                             [--layers ../data/high_d/layers] [--out ../data/segment_table.npz]

Each layer keeps its own phi grid (symmetric about 0, phi1 and phi2 alike). Arrays are
padded to the largest grid: phi[l, :n_phi[l]] holds layer l's grid, and the fields
[l, :n_phi[l], :n_phi[l]] hold its values (NaN in the padding).

Fields of the up branch (p > 0), indexed [layer, j (phi1), k (phi2)]:
    energy U, fx, fy, M1 = theta'(0), M2 = theta'(L), p
    computed   the solver reached the node or its reflection (False outside the energy disk
               of a high-d layer)
    exists     a stable up state is stored there; computed & ~exists is past the up fold
The solver stores one state per node: up wherever it is stable (promotion pass), else down.
The up state at x is the stored one if p > 0, else the reflection theta -> -theta of the
stored state at -x if that is down: same U and fx, opposite fy, M1, M2 and p.
The down branch is the reflection of this table: down(d, phi1, phi2) = R up(d, -phi1, -phi2).

Units: l = B = 1; forces in B / l^2, moments in B / l. Forces are the solver's Lagrange
multipliers, with dU = -fx dd - M1 dphi1 + M2 dphi2.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_erosion

from branch import first_mode_amplitude

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("energy", "fx", "fy", "M1", "M2", "p")


def up_branch(f, computed):
    """Up-branch fields on one layer from the stored states f (2D arrays)."""
    R = (slice(None, None, -1), slice(None, None, -1))      # (j, k) -> (n-1-j, n-1-k)
    stored_up = computed & (f["p"] > 0)
    reflected_up = computed[R] & (f["p"][R] < 0)
    out = {}
    for key in KEYS:
        sign = 1.0 if key in ("energy", "fx") else -1.0
        out[key] = np.where(stored_up, f[key], np.where(reflected_up, sign * f[key][R], np.nan))
    return out, stored_up | reflected_up, computed | computed[R]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--production", default=str(ROOT / "data/production/fields.npz"))
    ap.add_argument("--layers", default=str(ROOT / "data/high_d/layers"))
    ap.add_argument("--out", default=str(ROOT / "data/segment_table.npz"))
    ap.add_argument("--production-max-d", type=float, default=0.965,
                    help="production layers up to this d; the high-d layers take over above")
    args = ap.parse_args()

    layers = []                     # (d, phi, stored fields, computed, source)
    with np.load(args.production) as z:
        prod = {k: z[k] for k in z.files}
    for i, d in enumerate(prod["d_vals"]):
        if d <= args.production_max_d + 1e-9:
            f = {"energy": prod["energy"][i], "fx": prod["fx"][i], "fy": prod["fy"][i],
                 "M1": prod["moment"][i], "M2": prod["moment_right"][i], "p": prod["p"][i]}
            layers.append((float(d), prod["phi1_vals"], f, prod["solved"][i] > 0.5, "production"))
    high = {}
    for path in sorted(glob.glob(str(Path(args.layers) / "layer_*.npz"))):
        with np.load(path) as z:
            f = {"energy": z["energy"], "fx": z["fx"], "fy": z["fy"], "M1": z["moment"],
                 "M2": z["moment_right"], "p": z["p"]}
            d = float(z["d"])
            high[d] = (z["phi"], f, z["solved"])
            settings = json.loads(str(z["settings"]))
            layers.append((d, z["phi"], f, z["solved"].astype(bool), "high_d"))
    layers.sort(key=lambda layer: layer[0])
    d_vals = np.array([layer[0] for layer in layers])
    if np.any(np.diff(d_vals) <= 0):
        raise SystemExit("duplicate d layers")

    n_layers, n_max = len(layers), max(len(layer[1]) for layer in layers)
    phi = np.full((n_layers, n_max), np.nan)
    n_phi = np.zeros(n_layers, dtype=np.int64)
    table = {k: np.full((n_layers, n_max, n_max), np.nan) for k in KEYS}
    exists = np.zeros((n_layers, n_max, n_max), dtype=bool)
    computed = np.zeros((n_layers, n_max, n_max), dtype=bool)
    source = []
    for l, (d, grid, f, solved, src) in enumerate(layers):
        n = len(grid)
        if not np.allclose(grid, -grid[::-1]):
            raise SystemExit(f"layer {d}: phi grid not symmetric about 0")
        up, ex, comp = up_branch(f, solved)
        phi[l, :n], n_phi[l] = grid, n
        for k in KEYS:
            table[k][l, :n, :n] = up[k]
        exists[l, :n, :n], computed[l, :n, :n] = ex, comp
        source.append(src)

    # Checks: the up branch is stable-only (p > 0 where it exists); production and the
    # high-d layers agree on the overlap layers d = 0.97, 0.98, 0.99.
    report = {}
    assert np.all(table["p"][exists] > 0) and np.all(np.isfinite(table["energy"][exists]))
    for d in (0.97, 0.98, 0.99):
        i = np.flatnonzero(np.isclose(prod["d_vals"], d))
        if not len(i) or d not in high:
            continue
        i = int(i[0])
        pf = {"energy": prod["energy"][i], "fx": prod["fx"][i], "fy": prod["fy"][i],
              "M1": prod["moment"][i], "M2": prod["moment_right"][i], "p": prod["p"][i]}
        p_up, p_ex, _ = up_branch(pf, prod["solved"][i] > 0.5)
        grid, f, solved = high[d]
        h_up, h_ex, _ = up_branch(f, solved.astype(bool))
        pp = prod["phi1_vals"]
        ok = h_ex & (h_up["energy"] <= settings["energy_max"])
        jj, kk = np.nonzero(ok)
        # Compare only away from production's up fold: every production node within
        # 3 cells holds the up branch (the spline is filled with 0 where it does not).
        inner = binary_erosion(p_ex, np.ones((7, 7), dtype=bool))
        ex_interp = RectBivariateSpline(pp, pp, inner.astype(float), kx=1, ky=1)(
            grid[jj], grid[kk], grid=False) > 1 - 1e-9
        rows = {}
        for key in ("energy", "fx", "M1", "M2"):
            vals = RectBivariateSpline(pp, pp, np.where(p_ex, p_up[key], 0.0))(
                grid[jj], grid[kk], grid=False)
            scale = np.maximum(np.abs(h_up[key][jj, kk]), 1.0)
            rel = (np.abs(vals - h_up[key][jj, kk]) / scale)[ex_interp]
            rows[key] = {"median": float(np.median(rel)), "p99": float(np.percentile(rel, 99))}
        rows["compared"] = float(ex_interp.mean())
        report[f"{d:.2f}"] = rows

    out = Path(args.out)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, d_vals=d_vals, phi=phi, n_phi=n_phi,
                        amplitude=first_mode_amplitude(d_vals), source=np.array(source),
                        exists=exists, computed=computed, energy_max=settings["energy_max"],
                        notes=__doc__, overlap_report=json.dumps(report), **table)
    tmp.replace(out)

    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB): {n_layers} layers, "
          f"d {d_vals[0]:.4f} ... {d_vals[-1]:.4f}, grids up to {n_max} x {n_max}")
    print(f"  production: {source.count('production')} layers (d <= {args.production_max_d}), "
          f"high_d: {source.count('high_d')} layers")
    print(f"  up branch exists at {exists.sum():,} of {computed.sum():,} computed nodes "
          f"({exists.sum() / computed.sum():.1%})")
    for l in (0, source.index("high_d") - 1, source.index("high_d"), n_layers - 1):
        n = n_phi[l]
        print(f"  d = {d_vals[l]:.4f} ({source[l]}): phi +-{phi[l, n - 1]:.4f} step "
              f"{phi[l, 1] - phi[l, 0]:.5f}, computed {computed[l].sum():,}, up {exists[l].sum():,}")
    print("Overlap, up branch, high_d nodes with U <= energy_max against production "
          "(bicubic), relative to max(|value|, 1):")
    for d, rows in report.items():
        print(f"  d = {d}: " + ", ".join(f"{k} med {v['median']:.1e} p99 {v['p99']:.1e}"
                                          for k, v in rows.items() if isinstance(v, dict))
              + f"; compared (away from the fold) at {rows['compared']:.2%}")


if __name__ == "__main__":
    main()

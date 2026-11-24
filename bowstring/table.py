"""
table.py
Reading the stored segment space (data/segment_table.npz, the up branch on 66 d layers)
at any pose.

The fields are scaled by (1 - d) = sigma^2: W = sigma^2 U and Y1, Y2 = sigma^2 M1, M2.
- In each layer: bicubic Catmull-Rom (C1) on the layer's uniform (phi1, phi2) grid.
- Across layers: cubic Hermite in sigma = sqrt(1 - d) between the two bracketing layers,
  with the exact slopes the solver gives at every node (envelope theorem and symmetry of
  second derivatives):
      dU/dd = -fx,   dM1/dd = dfx/dphi1,   dM2/dd = -dfx/dphi2
  (dfx/dphi from central differences in the layer). So no slope is estimated across the
  unevenly spaced layers.
The table is valid at a point when every node of the 4 x 4 stencil in both layers holds a
stable up state.

Per segment (SegmentState): V from U; g from (dU/dd, -M1, M2), carried to the pose r by the
chord chart; K from the derivatives of the same interpolants (symmetrised). The down branch
is the reflection down(d, phi1, phi2) = R up(d, -phi1, -phi2): same U and fx, opposite fy,
M1, M2 and p.
"""
import math
from pathlib import Path

import numpy as np

from .segment import SegmentState, chart, to_pose_coords

DEFAULT = Path(__file__).resolve().parents[1] / "data/segment_table.npz"
# Per-layer fields: values W, Y1, Y2, their sigma-slopes, and (for seeds only) fy, p.
W, Y1, Y2, SW, SY1, SY2, FY, P = range(8)


def _catmull_rom(t):
    t2, t3 = t * t, t * t * t
    w = np.array([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2]) / 2
    dw = np.array([-3 * t2 + 4 * t - 1, 9 * t2 - 10 * t, -9 * t2 + 8 * t + 1, 3 * t2 - 2 * t]) / 2
    return w, dw


def _central(f, h, axis):
    """df/dphi along axis by central differences, one-sided at gaps; NaN where neither."""
    fp, fm = np.roll(f, -1, axis), np.roll(f, 1, axis)
    edge_p = [slice(None)] * f.ndim
    edge_m = [slice(None)] * f.ndim
    edge_p[axis], edge_m[axis] = -1, 0
    fp[tuple(edge_p)], fm[tuple(edge_m)] = np.nan, np.nan
    c = (fp - fm) / (2 * h)
    c = np.where(np.isnan(c), (fp - f) / h, c)
    return np.where(np.isnan(c), (f - fm) / h, c)


class SegmentTable:
    def __init__(self, path=DEFAULT):
        with np.load(path) as z:
            self.d = z["d_vals"]
            self.phi = z["phi"]
            self.n_phi = z["n_phi"]
            ok = z["exists"] & z["computed"]
            f = {k: np.where(ok, z[k], np.nan) for k in ("energy", "fx", "fy", "M1", "M2", "p")}
        self.sigma = np.sqrt(1.0 - self.d)
        self.h = self.phi[:, 1] - self.phi[:, 0]
        L, M = len(self.d), self.phi.shape[1]
        h = self.h[:, None, None]
        fx1, fx2 = _central(f["fx"], h, 1), _central(f["fx"], h, 2)
        s, s2 = self.sigma[:, None, None], self.sigma[:, None, None] ** 2
        # d(sigma^2 f)/dsigma = 2 sigma f + sigma^2 df/dd * dd/dsigma, dd/dsigma = -2 sigma
        F = np.empty((L, M, M, 8))
        F[..., W] = s2 * f["energy"]
        F[..., Y1] = s2 * f["M1"]
        F[..., Y2] = s2 * f["M2"]
        F[..., SW] = 2 * s * f["energy"] + 2 * s ** 3 * f["fx"]
        F[..., SY1] = 2 * s * f["M1"] - 2 * s ** 3 * fx1
        F[..., SY2] = 2 * s * f["M2"] + 2 * s ** 3 * fx2
        F[..., FY] = s2 * f["fy"]
        F[..., P] = s2 * f["p"]
        self.ok = np.isfinite(F).all(axis=-1)
        F[~self.ok] = 0.0
        self.F = F

    def _layer(self, l, p1, p2):
        """Values, d/dphi1, d/dphi2 of the 8 per-layer fields at (p1, p2), or None."""
        n, h, origin = self.n_phi[l], self.h[l], self.phi[l, 0]
        u1, u2 = (p1 - origin) / h, (p2 - origin) / h
        j, k = int(math.floor(u1)), int(math.floor(u2))
        if j < 1 or k < 1 or j + 2 >= n or k + 2 >= n:
            return None
        if not self.ok[l, j - 1:j + 3, k - 1:k + 3].all():
            return None
        S = self.F[l, j - 1:j + 3, k - 1:k + 3]                        # (4, 4, 8)
        w1, dw1 = _catmull_rom(u1 - j)
        w2, dw2 = _catmull_rom(u2 - k)
        return np.stack([np.einsum("j,k,jkf->f", w1, w2, S),
                         np.einsum("j,k,jkf->f", dw1, w2, S) / h,
                         np.einsum("j,k,jkf->f", w1, dw2, S) / h], axis=1)   # (8, 3)

    def lookup(self, d, p1, p2):
        """Up branch at (d, p1, p2): U, its gradient Gq and Hessian Hq in (d, phi1, phi2),
        and the seed fields (fx, fy, M1, p); None if the table is not valid there."""
        if not (self.d[0] <= d <= self.d[-1]):
            return None
        l = min(int(np.searchsorted(self.d, d, side="right")) - 1, len(self.d) - 2)
        a, b = self._layer(l, p1, p2), self._layer(l + 1, p1, p2)
        if a is None or b is None:
            return None
        x = math.sqrt(1.0 - d)
        x0, x1 = self.sigma[l], self.sigma[l + 1]
        hx = x1 - x0
        t = (x - x0) / hx
        t2, t3 = t * t, t * t * t
        H = np.array([2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2])
        dH = np.array([6 * t2 - 6 * t, 3 * t2 - 4 * t + 1, -6 * t2 + 6 * t, 3 * t2 - 2 * t]) / hx
        ddH = np.array([12 * t - 6, 6 * t - 4, -12 * t + 6, 6 * t - 2]) / hx ** 2

        def hermite(iv, islope, basis):
            # rows: value, d/dphi1, d/dphi2 of the field, combined across the two layers
            return (basis[0] * a[iv] + basis[1] * hx * a[islope]
                    + basis[2] * b[iv] + basis[3] * hx * b[islope])

        Wv, Ws, Wss = (hermite(W, SW, B) for B in (H, dH, ddH))
        Y1v, Y1s = hermite(Y1, SY1, H), hermite(Y1, SY1, dH)
        Y2v, Y2s = hermite(Y2, SY2, H), hermite(Y2, SY2, dH)
        lin = (1 - t) * a + t * b                                      # seeds only

        s2 = x * x
        dsd = -0.5 / x                      # dsigma/dd
        d2sd = -0.25 / x ** 3               # d2sigma/dd2
        U = Wv[0] / s2
        U_s = Ws[0] / s2 - 2 * Wv[0] / x ** 3
        U_ss = Wss[0] / s2 - 4 * Ws[0] / x ** 3 + 6 * Wv[0] / x ** 4
        U_d = U_s * dsd
        U_dd = U_ss * dsd ** 2 + U_s * d2sd
        U_dphi = (Ws[1:] / s2 - 2 * Wv[1:] / x ** 3) * dsd             # (2,)
        M1, M2 = Y1v[0] / s2, Y2v[0] / s2
        M1_d = (Y1s[0] / s2 - 2 * Y1v[0] / x ** 3) * dsd
        M2_d = (Y2s[0] / s2 - 2 * Y2v[0] / x ** 3) * dsd
        Gq = np.array([U_d, -M1, M2])
        Hq = np.array([[U_dd, U_dphi[0], U_dphi[1]],
                       np.concatenate([[-M1_d], -Y1v[1:] / s2]),
                       np.concatenate([[M2_d], Y2v[1:] / s2])])
        seeds = {"fx": -U_d, "fy": lin[FY, 0] / s2, "M1": M1, "p": lin[P, 0] / s2}
        return U, Gq, 0.5 * (Hq + Hq.T), seeds

    def state(self, r, branch):
        """SegmentState at pose r on branch +1 (up) / -1 (down), or None if not valid."""
        (d, phi1, phi2), Q, Hd, Ha = chart(r)
        b = 1 if branch >= 0 else -1
        out = self.lookup(d, b * phi1, b * phi2)
        if out is None:
            return None
        U, Gq, Hq, seeds = out
        S = np.array([1.0, b, b])                    # U_b(d, phi) = U_up(d, b phi)
        g, K = to_pose_coords(S * Gq, S[:, None] * Hq * S[None, :], Q, Hd, Ha)
        # Shooting warm start: n = -R(alpha) (fx, fy), M0 = M1, on this branch.
        alpha = -phi1
        ca, sa = math.cos(alpha), math.sin(alpha)
        fx, fy, M1 = seeds["fx"], b * seeds["fy"], b * seeds["M1"]
        load = np.array([-(ca * fx - sa * fy), -(sa * fx + ca * fy), M1])
        return SegmentState(U, g, K, load, b, b * seeds["p"], True, "table", (d, phi1, phi2))

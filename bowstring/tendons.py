"""
tendons.py
Tendon lengths of the bowstring arm in terms of the disk poses P_j = (x_j, y_j, psi_j),
j = 0..n (disk 0 is the base), with first and second derivatives.

Tendon k+ / k- (k = 0..n-1) has holes at +-d_k along each disk's normal (-sin psi, cos psi),
passes disks 0..k and is anchored at disk k + 1; between disks it is a straight chord.
The same geometry as bowstring/rod.py, written in the disk poses. All chords are handled at
once (vectorised), since this runs inside every evaluation of the controller.
"""
import math
from functools import lru_cache

import numpy as np


def hole(P, d, s):
    """Hole of side s (+1 / -1) at offset d on disk pose P, and its Jacobian (2, 3)
    and second derivative in psi (2,)."""
    x, y, psi = P
    c, sn = math.cos(psi), math.sin(psi)
    g = np.array([x - s * d * sn, y + s * d * c])
    J = np.array([[1.0, 0.0, -s * d * c], [0.0, 1.0, -s * d * sn]])
    return g, J, np.array([s * d * sn, -s * d * c])


@lru_cache(maxsize=None)
def _chords(n):
    """Chord list for n pairs: arrays k, side index, side sign, lower disk, and the scatter
    indices of each chord's 6 pose variables (lower disk, upper disk)."""
    rows = [(k, side, s, j) for k in range(n) for side, s in enumerate((1.0, -1.0))
            for j in range(k + 1)]
    k, side, s, ja = (np.array(c) for c in zip(*rows))
    idx = np.stack([3 * ja + a for a in range(6)], axis=1)       # (m, 6): Pa then Pb
    return k, side, s.astype(float), ja, idx


def tendons(P, offsets, T=None):
    """Lengths lam (n, 2), gradients (n, 2, 3(n+1)) in all disk poses and, with tensions
    T (n, 2), the Hessian of sum T lam (3(n+1), 3(n+1)). Side index 0 is +, 1 is -."""
    n = len(offsets)
    nv = 3 * (n + 1)
    k, side, s, ja, idx = _chords(n)
    P = np.asarray(P, float)
    d = np.asarray(offsets, float)[k] * s                       # signed offset per chord
    pa, pb = P[ja], P[ja + 1]
    ca, sa, cb, sb = np.cos(pa[:, 2]), np.sin(pa[:, 2]), np.cos(pb[:, 2]), np.sin(pb[:, 2])
    ga = np.column_stack([pa[:, 0] - d * sa, pa[:, 1] + d * ca])
    gb = np.column_stack([pb[:, 0] - d * sb, pb[:, 1] + d * cb])
    v = gb - ga
    Lc = np.hypot(v[:, 0], v[:, 1])
    u = v / Lc[:, None]
    lam = np.zeros((n, 2))
    np.add.at(lam, (k, side), Lc)
    # d(hole)/d(psi) columns and second derivatives
    dpa = np.column_stack([-d * ca, -d * sa])
    dpb = np.column_stack([-d * cb, -d * sb])
    # chord length gradient w.r.t. (Pa, Pb): (-u, -u.dpa, u, u.dpb)
    gloc = np.column_stack([-u, -(u * dpa).sum(1), u, (u * dpb).sum(1)])     # (m, 6)
    grad = np.zeros((n, 2, nv))
    np.add.at(grad, (k[:, None], side[:, None], idx), gloc)
    H = None
    if T is not None:
        t = np.asarray(T, float)[k, side]
        # D = dv/d(Pa, Pb) (m, 2, 6)
        D = np.zeros((len(k), 2, 6))
        D[:, 0, 0] = D[:, 1, 1] = -1.0
        D[:, :, 2] = -dpa
        D[:, 0, 3] = D[:, 1, 4] = 1.0
        D[:, :, 5] = dpb
        Pm = (np.eye(2)[None] - u[:, :, None] * u[:, None, :]) / Lc[:, None, None]
        blk = t[:, None, None] * np.einsum("mai,mab,mbj->mij", D, Pm, D)
        # curvature of the holes: d2g/dpsi2 = (d sin, -d cos), contracted with +-t u
        blk[:, 5, 5] += t * (u[:, 0] * d * sb - u[:, 1] * d * cb)
        blk[:, 2, 2] -= t * (u[:, 0] * d * sa - u[:, 1] * d * ca)
        H = np.zeros((nv, nv))
        np.add.at(H, (idx[:, :, None], idx[:, None, :]), blk)
    return lam, grad, H

"""
continuation.py
Pseudo-arclength continuation of the equilibria of BowstringRod with the payload
held, in one parameter p: the payload angle (p = X[2]) or one tension difference
(p = D_k at fixed pair totals C). Unknowns y = (theta, mu, p), with the wrench on
the payload w = -mu. Folds of the forward map are the turning points in p, where
the stability margin passes through zero.
"""

import numpy as np

from .rod import BowstringRod


class Branch:
    def __init__(self, rod: BowstringRod, X, C, D, param):
        """param: "angle" (X[2] free) or ("D", k)."""
        self.rod, self.X, self.C, self.D = rod, np.asarray(X, float), np.asarray(C, float), np.asarray(D, float)
        self.param = param

    def unpack(self, y):
        m = self.rod.m
        theta, mu, p = y[:m], y[m:m + 3], y[-1]
        X, D = self.X.copy(), self.D.copy()
        if self.param == "angle":
            X[2] = p
        else:
            D[self.param[1]] = p
        return theta, mu, X, D

    def residual(self, y):
        """F(y), dF/dy (m + 3, m + 4), the Lagrangian Hessian and A."""
        rod = self.rod
        theta, mu, X, D = self.unpack(y)
        T = rod.tensions(self.C, D)
        _, lg, Ht = rod.tendons(theta, T)
        g = rod.Kb @ theta + np.einsum("ks,ksm->m", T, lg)
        c, A, curv = rod.constraint(theta)
        F = np.r_[g - A.T @ mu, c - X]
        HL = rod.Kb + Ht - curv(mu)
        dp = np.zeros(rod.m + 3)
        if self.param == "angle":
            dp[-1] = -1.0
        else:
            k = self.param[1]
            dp[:rod.m] = 0.5 * (lg[k, 0] - lg[k, 1])
        J = np.block([[HL, -A.T, dp[:rod.m, None]], [A, np.zeros((3, 3)), dp[rod.m:, None]]])
        return F, J, HL, A

    def tangent(self, J, previous=None):
        _, _, vt = np.linalg.svd(J)
        t = vt[-1]
        if previous is not None and t @ previous < 0:
            t = -t
        return t / np.linalg.norm(t)

    def trace(self, y0, ds, steps, p_bounds, tol=1e-9, direction=1.0):
        """Follow the branch from the equilibrium y0 for `steps` steps of arclength
        ds (in the scaled norm), or until p leaves p_bounds. Returns a list of
        dicts: y, p, w, margin, contact."""
        rod = self.rod
        F, J, HL, A = self.residual(y0)
        t = self.tangent(J)
        if t[-1] * direction < 0:
            t = -t
        y = y0.copy()
        out = []

        def record(y, HL, A):
            theta, mu, X, D = self.unpack(y)
            out.append({"y": y.copy(), "p": float(y[-1]), "w": -mu.copy(), "X": X, "D": D,
                        "margin": rod.margin(HL, A), "contact": rod.contact_margin(theta)})

        record(y, HL, A)
        h = ds
        for _ in range(steps):
            for _attempt in range(8):
                z = y + h * t
                ok = False
                for _it in range(20):
                    F, J, HL, A = self.residual(z)
                    G = np.r_[F, t @ (z - y) - h]
                    if np.abs(G).max() < tol:
                        ok = True
                        break
                    JJ = np.vstack([J, t])
                    z = z - np.linalg.solve(JJ, G)
                if ok and np.abs(z - y).max() < 0.5:
                    break
                h /= 2
            else:
                break
            _, J, HL, A = self.residual(z)
            t = self.tangent(J, t)
            y = z
            record(y, HL, A)
            h = min(ds, h * 1.5)
            if not (p_bounds[0] <= y[-1] <= p_bounds[1]):
                break
        return out

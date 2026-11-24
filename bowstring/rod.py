"""
rod.py
The bowstring arm as one discretised inextensible rod: the reference model for the
forward and inverse maps (PDF Sections 3, 5 and 7), exact as N grows.

n segments of length 1 (bending stiffness B = 1), N elements each, m = n N.
Unknowns: tangent angles theta[1..m] at the nodes; theta[0] = 0 is the clamp.
Element e joins nodes e - 1 and e at the mean angle (theta[e-1] + theta[e]) / 2,
so the rod is inextensible by construction. Disk j sits at node j N, turned to
theta[j N]. Tendon k+ / k- (k = 0..n-1) has holes at +-d_k along the disk normal
(-sin, cos), passes disks 0..k and is anchored at disk k + 1; between disks it is
a straight chord of tension T.

Potential U(theta; T) = 1/2 sum (theta_e - theta_{e-1})^2 / h + sum T lambda.
The payload pose X = (x, y, angle) of the last disk is c(theta).

Forward map (X, T) -> w: stationary U subject to c(theta) = X, with multiplier mu.
    The envelope theorem gives dU*/dX = mu, so the wrench the arm applies to the
    payload is w = -mu.
Inverse map (X, w) -> T: the same equations with mu = -w known, the tension
    differences D_k unknown and the pair totals C_k fixed (T+- = (C +- D) / 2);
    square for n = 3.
Stability with the payload held: the Hessian of the Lagrangian on the null space
    of dc/dtheta is positive definite. Its smallest eigenvalue / h approximates
    the smallest eigenvalue of the rod's second variation (per unit length), the
    margin plotted in the figures.
"""

import math

import numpy as np
from scipy.linalg import null_space


class BowstringRod:
    def __init__(self, n=3, offsets=(0.15, 0.2, 0.25), N=40):
        self.n, self.N = n, N
        self.m = n * N
        self.h = 1.0 / N
        self.d = np.asarray(offsets, dtype=float)
        assert len(self.d) == n
        m, h = self.m, self.h
        k = (2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)) / h
        k[-1, -1] = 1.0 / h
        self.Kb = k
        self.B = 0.5 * (np.eye(m) + np.eye(m, k=-1))        # mean angles a = B theta
        # Chords: (pair k, side s, lower disk, upper disk).
        self.chords = [(kk, s, j - 1, j) for kk in range(n) for s in (1.0, -1.0)
                       for j in range(1, kk + 2)]

    # ------------------------------------------------------------ geometry
    def nodes(self, theta):
        """Node positions (m + 1, 2) and element mean angles (m,)."""
        a = self.B @ theta
        pos = np.vstack([np.zeros(2), self.h * np.cumsum(np.column_stack((np.cos(a), np.sin(a))), 0)])
        return pos, a

    def disks(self, theta):
        """Disk poses (n + 1, 3): x, y, angle; disk 0 is the base."""
        pos, _ = self.nodes(theta)
        idx = np.arange(self.n + 1) * self.N
        return np.column_stack((pos[idx], np.r_[0.0, theta][idx]))

    def hole(self, disk_pose, k, s):
        psi = disk_pose[2]
        return disk_pose[:2] + s * self.d[k] * np.array([-math.sin(psi), math.cos(psi)])

    def tip(self, theta):
        return self.disks(theta)[-1]

    # ------------------------------------------------------------ derivatives
    def _hole_jac(self, theta, a, j, k, s):
        """d(hole)/d(theta), (2, m); zero for the base disk."""
        J = np.zeros((2, self.m))
        if j == 0:
            return J
        node = j * self.N
        J[0] = -self.h * np.sin(a[:node]) @ self.B[:node]
        J[1] = self.h * np.cos(a[:node]) @ self.B[:node]
        psi = theta[node - 1]
        J[0, node - 1] += -s * self.d[k] * math.cos(psi)
        J[1, node - 1] += -s * self.d[k] * math.sin(psi)
        return J

    def tendons(self, theta, T=None, hessian=True):
        """Tendon lengths lam (n, 2), their gradients (n, 2, m) and, with weights
        T (n, 2), the Hessian of sum T lam (m, m)."""
        n, m, h = self.n, self.m, self.h
        disks = self.disks(theta)
        _, a = self.nodes(theta)
        lam = np.zeros((n, 2))
        grad = np.zeros((n, 2, m))
        H = np.zeros((m, m))
        curve = np.zeros(m)                    # diagonal of the mean-angle curvature term
        cache = {}
        for (k, s, ja, jb) in self.chords:
            side = 0 if s > 0 else 1
            ga, gb = self.hole(disks[ja], k, s), self.hole(disks[jb], k, s)
            for j in (ja, jb):
                if (j, k, s) not in cache:
                    cache[(j, k, s)] = self._hole_jac(theta, a, j, k, s)
            Ja, Jb = cache[(ja, k, s)], cache[(jb, k, s)]
            L = float(np.linalg.norm(gb - ga))
            u = (gb - ga) / L
            lam[k, side] += L
            grad[k, side] += u @ (Jb - Ja)
            if T is None or not hessian:
                continue
            t = T[k, side]
            if t == 0.0:
                continue
            Jd = Jb - Ja
            H += t * Jd.T @ ((np.eye(2) - np.outer(u, u)) / L) @ Jd
            # Second derivatives of the holes, contracted with the pull on each.
            for j, G in ((jb, t * u), (ja, -t * u)):
                if j == 0:
                    continue
                node = j * self.N
                curve[:node] += -h * (G[0] * np.cos(a[:node]) + G[1] * np.sin(a[:node]))
                psi = theta[node - 1]
                nu = np.array([-math.sin(psi), math.cos(psi)])
                H[node - 1, node - 1] += -s * self.d[k] * (G @ nu)
        if T is not None and hessian:
            H += (self.B.T * curve) @ self.B
        return lam, grad, H

    def constraint(self, theta):
        """Tip pose c (3,), its Jacobian A (3, m), and the curvature matrices of
        c_x and c_y as functions g -> sum_a g_a d2 c_a (for a = x, y)."""
        pos, a = self.nodes(theta)
        A = np.zeros((3, self.m))
        A[0] = -self.h * np.sin(a) @ self.B
        A[1] = self.h * np.cos(a) @ self.B
        A[2, -1] = 1.0
        c = np.r_[pos[-1], theta[-1]]

        def curvature(g):
            diag = -self.h * (g[0] * np.cos(a) + g[1] * np.sin(a))
            return (self.B.T * diag) @ self.B
        return c, A, curvature

    def energy(self, theta, T):
        lam, _, _ = self.tendons(theta, None, hessian=False)
        return 0.5 * theta @ self.Kb @ theta + float(np.sum(T * lam))

    # ------------------------------------------------------------ tensions
    @staticmethod
    def tensions(C, D):
        C, D = np.asarray(C, float), np.asarray(D, float)
        return np.column_stack(((C + D) / 2, (C - D) / 2))

    # ------------------------------------------------------------ stability
    def margin(self, H, A=None):
        """Smallest eigenvalue / h of H, on the null space of A if given."""
        if A is not None:
            Z = null_space(A)
            H = Z.T @ H @ Z
        return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0]) / self.h

    def contact_margin(self, theta):
        """Exact no-contact margin: for every chord, the least distance of the rod
        between its two disks from the chord, on the side away from it (negative:
        the rod crosses a tendon)."""
        pos, _ = self.nodes(theta)
        disks = self.disks(theta)
        worst = np.inf
        for (k, s, ja, jb) in self.chords:
            ga, gb = self.hole(disks[ja], k, s), self.hole(disks[jb], k, s)
            u = (gb - ga) / np.linalg.norm(gb - ga)
            seg = pos[ja * self.N: jb * self.N + 1]
            r = seg - ga
            cross = u[0] * r[:, 1] - u[1] * r[:, 0]       # > 0: left of the chord
            worst = min(worst, float(np.min(-s * cross)))  # the rod must be on the -s side
        return worst

    # ------------------------------------------------------------ solves
    def forward(self, X, T, theta0, w0=None, tol=1e-10, max_it=40):
        """Forward map: theta, w (3,), margin, converged."""
        theta = theta0.copy()
        mu = np.zeros(3) if w0 is None else -np.asarray(w0, float)
        m = self.m
        for _ in range(max_it):
            _, lg, Ht = self.tendons(theta, T)
            g = self.Kb @ theta + np.einsum("ks,ksm->m", T, lg)
            c, A, curv = self.constraint(theta)
            F = np.r_[g - A.T @ mu, c - X]
            if np.abs(F).max() < tol:
                HL = self.Kb + Ht - curv(mu)
                return theta, -mu, self.margin(HL, A), True
            HL = self.Kb + Ht - curv(mu)
            J = np.block([[HL, -A.T], [A, np.zeros((3, 3))]])
            step = np.linalg.solve(J, -F)
            scale = min(1.0, 0.3 / max(np.abs(step[:m]).max(), 1e-300))
            theta = theta + scale * step[:m]
            mu = mu + scale * step[m:]
        return theta, -mu, np.nan, False

    def inverse(self, X, w, C, theta0, D0, tol=1e-10, max_it=40, details=False):
        """Inverse map (n = 3): theta, D (n,), margin, converged [, dD/dw]."""
        assert self.n == 3
        theta, D = theta0.copy(), np.asarray(D0, float).copy()
        w = np.asarray(w, float)
        m = self.m
        for _ in range(max_it):
            T = self.tensions(C, D)
            _, lg, Ht = self.tendons(theta, T)
            g = self.Kb @ theta + np.einsum("ks,ksm->m", T, lg)
            c, A, curv = self.constraint(theta)
            F = np.r_[g + A.T @ w, c - X]
            HL = self.Kb + Ht + curv(w)
            G = 0.5 * (lg[:, 0] - lg[:, 1]).T                 # d(grad U)/dD, (m, n)
            J = np.block([[HL, G], [A, np.zeros((3, 3))]])
            if np.abs(F).max() < tol:
                out = (theta, D, self.margin(HL, A), True)
                if details:
                    rhs = np.vstack([-A.T, np.zeros((3, 3))])
                    sens = np.linalg.solve(J, rhs)
                    out = out + (sens[m:],)
                return out
            step = np.linalg.solve(J, -F)
            scale = min(1.0, 0.3 / max(np.abs(step[:m]).max(), 1e-300))
            theta = theta + scale * step[:m]
            D = D + scale * step[m:]
        return (theta, D, np.nan, False) + ((None,) if details else ())

    def restore(self, theta, X, tol=1e-12, max_it=20):
        """Put the payload back at X by minimal-norm Gauss-Newton steps."""
        for _ in range(max_it):
            c, A, _ = self.constraint(theta)
            r = X - c
            if np.abs(r).max() < tol:
                break
            theta = theta + A.T @ np.linalg.solve(A @ A.T, r)
        return theta

    def held(self, X, T, theta0, lean=1.0, tol=1e-9, max_it=300):
        """Stable equilibrium with the payload held at X: a descent of U on the
        constraint manifold (modified Newton in the null space, |eigenvalues|, and
        along the softest mode at a saddle, towards `lean`). Returns theta, w,
        margin, converged. Unlike `forward`, it cannot end on an unstable state."""
        theta = self.restore(theta0.copy(), X)
        for _ in range(max_it):
            _, lg, Ht = self.tendons(theta, T)
            g = self.Kb @ theta + np.einsum("ks,ksm->m", T, lg)
            _, A, curv = self.constraint(theta)
            mu = np.linalg.solve(A @ A.T, A @ g)
            HL = self.Kb + Ht - curv(mu)
            Z = null_space(A)
            gr = Z.T @ g
            eigs, vecs = np.linalg.eigh(Z.T @ HL @ Z)
            if np.abs(gr).max() < tol:
                if eigs[0] > 0:
                    theta, w, marg, ok = self.forward(X, T, theta, -mu)   # polish
                    return theta, w, marg, ok
                v = Z @ vecs[:, 0]
                direction = v * np.sign(lean * (v.sum() or 1.0))
            else:
                direction = -Z @ (vecs @ ((vecs.T @ gr) / np.maximum(np.abs(eigs), 1e-8)))
            alpha = min(1.0, 0.3 / np.abs(direction).max())
            e0 = self.energy(theta, T)
            while alpha > 1e-12:
                trial = self.restore(theta + alpha * direction, X)
                if self.energy(trial, T) < e0 - 1e-12 or (np.abs(gr).max() < 1e-6 and self.energy(trial, T) <= e0):
                    break
                alpha /= 2
            theta = trial
        return theta, -mu, np.nan, False

    def rest(self, T, theta0, tol=1e-10, max_it=200):
        """Payload free and at rest (w = 0): descend U to a stable state. Returns
        theta, margin, converged."""
        theta = theta0.copy()
        for _ in range(max_it):
            _, lg, Ht = self.tendons(theta, T)
            g = self.Kb @ theta + np.einsum("ks,ksm->m", T, lg)
            H = self.Kb + Ht
            eigs, vecs = np.linalg.eigh(H)
            if np.abs(g).max() < tol and eigs[0] > 0:
                return theta, float(eigs[0]) / self.h, True
            if np.abs(g).max() < tol:                          # saddle: leave along the soft mode
                direction = vecs[:, 0]
            else:
                direction = -vecs @ ((vecs.T @ g) / np.maximum(np.abs(eigs), 1e-10))
            alpha = min(1.0, 0.3 / np.abs(direction).max())
            e0 = self.energy(theta, T)
            while alpha > 1e-12 and self.energy(theta + alpha * direction, T) > e0 + 1e-4 * alpha * min(g @ direction, 0):
                alpha /= 2
            theta = theta + alpha * direction
        return theta, np.nan, False

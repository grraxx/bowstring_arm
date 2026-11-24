"""
chain.py
The bowstring arm as a chain of n elastica segments joined at disks.

Energy U(P; T) = sum_i V_i(r_i) + sum_k,s T_ks lam_ks(P), with P_1..P_n the disk poses
(P_0 = 0 is the base clamp, X = P_n the payload) and r_i the pose of disk i in the frame of
disk i-1. Equilibrium of an intermediate disk: dU/dP_j = sum_i J_ij^T g_i + dU_t/dP_j = 0,
with g_i = dV_i/dr_i the segment's end wrench. The wrench on the payload is w = -dU/dX.

Unknowns are the segment base loads L_i = (nx, ny, M0), not the disk poses. Each segment is
integrated forward from its load (segment.integrate), which gives r_i, g_i = (nx, ny, M(1))
and their derivatives without any per-segment solve; the disk poses follow by composition.
A nearly straight segment is very stiff along its chord (1e-4 of chord length is tens of load
units), so a solve in the poses is badly conditioned, while in the loads the axial force is
set directly by the disk equilibria and the system stays well conditioned, also when straight.

Forward map (X, T) -> w: equilibria of disks 1..n-1 and P_n = X; w = -dU/dX, stiffness
    K_X = d(-w)/dX by the Schur complement, margin = lambda_min of the Hessian in P_1..P_{n-1}.
Inverse map (X, w) -> D at fixed pair totals C (T+- = (C +- D) / 2): for n = 3 a square
    Newton solve in (L, D). (n = 4 would be under-determined; it is out of scope for now.)
Rest: the payload free, w = 0.
"""
import math

import numpy as np

from .segment import build_shared, first_mode_projection, integrate, unstable_directions
from .tendons import tendons


def _load_chain_c():
    """The compiled evaluate (_chain.c, which includes _rk4.c); None if it cannot be built
    (the Python assembly is used then)."""
    import ctypes
    from pathlib import Path
    here = Path(__file__).parent
    c = build_shared(here / "_chain.c", "_chain.so", deps=[here / "_rk4.c"])
    if c is None:
        return None
    # raw addresses (arr.ctypes.data): data_as() cost ~2 us per array, a third of the call
    c.chain_evaluate.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 11
    c.chain_evaluate.restype = ctypes.c_int
    return c.chain_evaluate


_CHAIN = _load_chain_c()


def relative_pose(Pa, Pb):
    """r = pose of Pb in the frame of Pa, dr/d(Pa, Pb) (3, 6), and g -> sum_a g_a d2r_a (6, 6)."""
    xa, ya, pa = Pa
    c, s = math.cos(pa), math.sin(pa)
    dx, dy = Pb[0] - xa, Pb[1] - ya
    rx, ry = c * dx + s * dy, -s * dx + c * dy
    J = np.array([[-c, -s, ry, c, s, 0.0],
                  [s, -c, -rx, -s, c, 0.0],
                  [0.0, 0.0, -1.0, 0.0, 0.0, 1.0]])

    def curvature(g):
        C = np.zeros((6, 6))
        row = np.array([g[0] * s + g[1] * c, -g[0] * c + g[1] * s, 0.0,
                        -g[0] * s - g[1] * c, g[0] * c - g[1] * s, 0.0])
        C[2, :] += row
        C[:, 2] += row
        C[2, 2] = -(g[0] * rx + g[1] * ry)
        return C
    return np.array([rx, ry, Pb[2] - pa]), J, curvature


def compose(Pa, r):
    """Pose of a disk at relative pose r from Pa."""
    c, s = math.cos(Pa[2]), math.sin(Pa[2])
    return np.array([Pa[0] + c * r[0] - s * r[1], Pa[1] + s * r[0] + c * r[1], Pa[2] + r[2]])


class ChainArm:
    def __init__(self, n=3, offsets=(0.15, 0.2, 0.25), Tmin=0.1, Tmax=None, scheme="fixed",
                 eps=0.05, stiffness=None):
        """scheme "fixed": each pair's total is fixed, T+- = (C_k +- D_k) / 2, so
            |D_k| <= C_k - 2 Tmin and the co-contraction C always compresses the arm.
        scheme "minimal": the slack side of each pair sits at Tmin and the pulling side carries
            the difference, T+- = Tmin + (sqrt(D^2 + eps^2) +- D) / 2 (smoothed at D = 0); the
            bending is limited by Tmax only and the arm is compressed only as much as bending
            needs. C is then ignored. With Tmax = None there is no tension limit at all: the
            tensions are whatever equilibrium needs (tendons only pull, the slack side keeps Tmin).
        stiffness: bending stiffness B_i of each segment (default 1). A segment of stiffness B
            under loads L has the shape of the unit segment under L / B; its end moment and
            energy scale by B."""
        self.n = n
        self.B = np.ones(n) if stiffness is None else np.asarray(stiffness, float)
        assert len(self.B) == n
        self.offsets = np.asarray(offsets, dtype=float)
        assert len(self.offsets) == n
        self.Tmin, self.Tmax = Tmin, Tmax
        self.scheme, self.eps = scheme, eps

        self.iX = slice(3 * n - 3, 3 * n)            # payload pose among P_1..P_n
        # Compiled evaluate (hessian=False); the Python assembly stays as the reference.
        self.use_c = n <= 6
        self._off_c = np.ascontiguousarray(self.offsets, float)
        self._B_c = np.ascontiguousarray(self.B, float)
        self.iI = slice(0, 3 * n - 3)                # intermediate disks

    @staticmethod
    def tensions(C, D):
        """Fixed-total tensions T+- = (C +- D) / 2."""
        C, D = np.asarray(C, float), np.asarray(D, float)
        return np.column_stack(((C + D) / 2, (C - D) / 2))

    def tension_map(self, C, D):
        """Tensions (n, 2) from the differences D under this arm's scheme, and dT/dD (n, 2)."""
        D = np.asarray(D, float)
        if self.scheme == "minimal":
            r = np.sqrt(D * D + self.eps ** 2)
            T = self.Tmin + 0.5 * np.column_stack((r + D, r - D))
            dT = 0.5 * np.column_stack((D / r + 1.0, D / r - 1.0))
            return T, dT
        return self.tensions(C, D), np.tile([0.5, -0.5], (self.n, 1))

    def d_scale(self, C):
        """Range of each D_k, used to normalise the tension differences."""
        if self.scheme == "minimal":
            return np.full(self.n, 1.0 if self.Tmax is None else self.Tmax - self.Tmin)
        return np.asarray(C, float) - 2.0 * self.Tmin

    # ------------------------------------------------------------ assembly
    def evaluate(self, L, T, hessian=False):
        """Everything at segment loads L (n, 3) and tensions T (n, 2):
        P (n+1, 3), G = dU/dP (3n), dG/dL (3n, 3n), dP/dL (3n, 3n), tendon gradients
        (n, 2, 3n), segment energies and, with hessian, H = d2U/dP2 (3n, 3n)."""
        n = self.n
        L = np.asarray(L, float).reshape(n, 3)
        if not hessian and self.use_c and _CHAIN is not None:
            return self._evaluate_c(L, T)
        P = [np.zeros(3)]
        segs = []
        for i in range(n):
            B = self.B[i]
            pose, M1, V, J, _ = integrate(L[i] / B)
            segs.append((pose, np.array([L[i, 0], L[i, 1], B * M1]), B * V, J[:3] / B,
                         np.vstack([np.eye(3)[:2], J[3]])))
            P.append(compose(P[-1], pose))
        P = np.array(P)
        nv = 3 * (n + 1)
        G = np.zeros(nv)
        Hgeo = np.zeros((nv, nv))
        A = np.zeros((3 * n, nv))                    # dr/dP, all disks
        Bdirect = np.zeros((nv, 3 * n))              # dG/dL at fixed P: J^T dg
        Js = []
        for i in range(n):
            _, J, curv = relative_pose(P[i], P[i + 1])
            g, dg = segs[i][1], segs[i][4]
            sl = slice(3 * i, 3 * i + 6)
            G[sl] += J.T @ g
            Hgeo[sl, sl] += curv(g)
            A[3 * i:3 * i + 3, sl] = J
            Bdirect[sl, 3 * i:3 * i + 3] = J.T @ dg
            Js.append(J)
        lam, lg, Ht = tendons(P, self.offsets, T)
        G += np.einsum("ks,ksv->v", T, lg)
        Hgeo += Ht
        A, G, Hgeo, Bdirect, lg = A[:, 3:], G[3:], Hgeo[3:, 3:], Bdirect[3:], lg[:, :, 3:]
        Bd = np.zeros((3 * n, 3 * n))
        for i in range(n):
            Bd[3 * i:3 * i + 3, 3 * i:3 * i + 3] = segs[i][3]
        dPdL = np.linalg.solve(A, Bd)
        out = {"P": P, "G": G, "dGdL": Hgeo @ dPdL + Bdirect, "dPdL": dPdL, "lam_grad": lg,
               "U": sum(s[2] for s in segs) + float(np.sum(T * lam)), "L": L}
        if hessian:
            # An exactly straight segment has no finite stiffness along its chord (dpose/dload
            # is singular): H is then undefined, reported as NaN.
            H = Hgeo.copy()
            for i in range(n):
                try:
                    K = segs[i][4] @ np.linalg.inv(segs[i][3])    # dg/dr
                except np.linalg.LinAlgError:
                    H[:] = np.nan
                    break
                K = 0.5 * (K + K.T)
                cols = slice(3 * i - 3, 3 * i + 3) if i else slice(0, 3)
                Jc = Js[i] if i else Js[i][:, 3:]
                H[cols, cols] += Jc.T @ K @ Jc
            out["H"] = H
        return out

    def _evaluate_c(self, L, T):
        """evaluate(hessian=False) through the compiled assembly (same outputs)."""
        f = _CHAIN
        n, m = self.n, 3 * self.n
        L = np.ascontiguousarray(L, float)
        T = np.ascontiguousarray(T, float)
        P, G = np.empty((n + 1, 3)), np.empty(m)
        dGdL, dPdL = np.empty((m, m)), np.empty((m, m))
        lg, lam, U = np.empty((n, 2, m)), np.empty((n, 2)), np.empty(1)
        if f(n, L.ctypes.data, T.ctypes.data, self._off_c.ctypes.data, self._B_c.ctypes.data,
             P.ctypes.data, G.ctypes.data, dGdL.ctypes.data, dPdL.ctypes.data, lg.ctypes.data,
             lam.ctypes.data, U.ctypes.data):
            raise np.linalg.LinAlgError("singular dr/dP")
        return {"P": P, "G": G, "dGdL": dGdL, "dPdL": dPdL, "lam_grad": lg, "U": float(U[0]),
                "L": L}

    def shape(self, L):
        """Centre line of the whole arm for drawing: (n, N + 1, 2) points per segment."""
        L = np.asarray(L, float).reshape(self.n, 3)
        out, base = [], np.zeros(3)
        for i in range(self.n):
            pose, _, _, _, th = integrate(L[i] / self.B[i], record=True)
            ds = 1.0 / (len(th) - 1)
            x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
            y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
            c, sn = math.cos(base[2]), math.sin(base[2])
            out.append(np.column_stack([base[0] + c * x - sn * y, base[1] + sn * x + c * y]))
            base = compose(base, pose)
        return out

    def contact(self, L, P):
        """Validity of the straight-chord (bowstring) tendons at this configuration:
        margin   least signed distance of the backbone from each tendon chord, on its own side
                 (negative: the rod passes through a tendon, which this model allows and a real
                 arm cannot; the slack tendon would wrap on the rod instead);
        chord    shortest tendon chord (at zero, two holes coincide: the disks have collided and
                 the tendon direction is singular);
        bend     largest total turning of a segment (rad)."""
        from .tendons import hole
        shape = self.shape(L)
        margin, chord = np.inf, np.inf
        for k in range(self.n):
            for s in (1.0, -1.0):
                for j in range(1, k + 2):
                    ga, gb = hole(P[j - 1], self.offsets[k], s)[0], hole(P[j], self.offsets[k], s)[0]
                    c = float(np.linalg.norm(gb - ga))
                    chord = min(chord, c)
                    u = (gb - ga) / max(c, 1e-12)
                    r = shape[j - 1] - ga
                    margin = min(margin, float(np.min(-s * (u[0] * r[:, 1] - u[1] * r[:, 0]))))
        bend = max(abs(float(P[i + 1, 2] - P[i, 2])) for i in range(self.n))
        return {"margin": margin, "chord": chord, "bend": bend}

    def segment_report(self, L, P):
        """Per segment: chord chart (d, phi1, phi2), p, branch sign and clamped stability."""
        out = []
        for i in range(self.n):
            pose, _, _, _, thetas = integrate(L[i] / self.B[i], record=True)
            d = math.hypot(pose[0], pose[1])
            alpha = math.atan2(pose[1], pose[0])
            s = np.linspace(0.0, 1.0, len(thetas))
            tc = thetas - alpha
            ca, sa = math.cos(alpha), math.sin(alpha)
            fx = -(ca * L[i, 0] + sa * L[i, 1]) / self.B[i]       # unit-stiffness loads
            fy = -(-sa * L[i, 0] + ca * L[i, 1]) / self.B[i]
            p = float(first_mode_projection(tc, s))
            out.append({"chart": (d, -alpha, pose[2] - alpha), "p": p,
                        "stable": unstable_directions(tc, s, fx, fy) == 0})
        return out

    # ------------------------------------------------------------ statics guess
    def loads_from_poses(self, P, T, w=np.zeros(3)):
        """Segment loads that balance every disk at the given poses (statics, back from the
        tip): a starting point for the solves."""
        n = self.n
        P = np.asarray(P, float)
        _, lg, _ = tendons(P, self.offsets, None)
        tg = np.einsum("ks,ksv->v", T, lg)
        g = [None] * n
        Jr = [relative_pose(P[i], P[i + 1]) for i in range(n)]
        for i in range(n - 1, -1, -1):
            rhs = tg[3 * i + 3:3 * i + 6].copy()
            if i == n - 1:
                rhs += w
            else:
                rhs += Jr[i + 1][1][:, :3].T @ g[i + 1]
            g[i] = -np.linalg.solve(Jr[i][1][:, 3:].T, rhs)
        L = np.empty((n, 3))
        for i in range(n):
            r = Jr[i][0]
            L[i] = [g[i][0], g[i][1], g[i][2] - g[i][0] * r[1] + g[i][1] * r[0]]
        return L

    # ------------------------------------------------------------ Newton
    def _newton(self, residual, x, tol, max_it=50, alpha_min=1e-6):
        """Newton with backtracking on |F|; residual(x) -> (F, J, info) or None."""
        res = residual(x)
        if res is None:
            raise ValueError("cannot evaluate at the starting point")
        F, J, info = res
        for _ in range(max_it):
            norm = np.abs(F).max()
            if norm < tol:
                return x, info, True, norm
            step = np.linalg.lstsq(J, -F, rcond=None)[0]
            alpha = 1.0
            while alpha > alpha_min:
                trial = residual(x + alpha * step)
                if trial is not None and np.abs(trial[0]).max() < (1 - 1e-4 * alpha) * norm:
                    break
                alpha *= 0.5
            else:
                return x, info, False, norm
            x = x + alpha * step
            F, J, info = trial
        return x, info, bool(np.abs(F).max() < tol), float(np.abs(F).max())

    def _finish(self, L, tensions, ok, resid, ev=None, report=True, **extra):
        """The solution; with report, also K_X, the stability margin and the segment checks
        (6 more integrations, so the controller runs it only every few steps)."""
        if not report:
            return {"w": -ev["G"][self.iX], "L": ev["L"], "P": ev["P"], "U": ev["U"],
                    "converged": ok, "residual": resid, **extra}
        ev = self.evaluate(L, tensions, hessian=True)
        if report == "light":
            # The arm's margin and the payload stiffness only (one evaluation); the segment
            # checks (report=True) record theta along every segment and cost as much again.
            H = ev["H"]
            HII, HIX = H[self.iI, self.iI], H[self.iI, self.iX]
            if np.isfinite(H).all():
                K_X = H[self.iX, self.iX] - HIX.T @ np.linalg.solve(HII, HIX)
                margin = float(np.linalg.eigvalsh(HII)[0])
            else:
                K_X, margin = np.full((3, 3), np.nan), np.nan
            return {"w": -ev["G"][self.iX], "K_X": 0.5 * (K_X + K_X.T), "margin": margin,
                    "segments_stable": True, "L": ev["L"], "P": ev["P"], "U": ev["U"],
                    "converged": ok, "residual": resid, **extra}
        H = ev["H"]
        HII, HIX = H[self.iI, self.iI], H[self.iI, self.iX]
        if np.isfinite(H).all():
            K_X = H[self.iX, self.iX] - HIX.T @ np.linalg.solve(HII, HIX)
            margin = float(np.linalg.eigvalsh(HII)[0])
        else:
            K_X, margin = np.full((3, 3), np.nan), np.nan
        segs = self.segment_report(ev["L"], ev["P"])
        return {"w": -ev["G"][self.iX], "K_X": 0.5 * (K_X + K_X.T),
                "margin": margin,
                "segments_stable": all(s["stable"] for s in segs), "segments": segs,
                "L": ev["L"], "P": ev["P"], "U": ev["U"], "converged": ok, "residual": resid,
                **extra}

    # ------------------------------------------------------------ maps
    def rest(self, T, L0=None, tol=1e-9):
        """Payload free (w = 0). Starts from the coarse direct rod if no loads are given."""
        T = np.asarray(T, float)
        if L0 is None:
            from .rod import BowstringRod
            rod = BowstringRod(self.n, tuple(self.offsets), N=20)
            theta, _, _ = rod.rest(T, np.zeros(rod.m))
            L0 = self.loads_from_poses(rod.disks(theta), T)

        def residual(x):
            ev = self.evaluate(x, T)
            return ev["G"], ev["dGdL"], ev

        L, _, ok, resid = self._newton(residual, np.ravel(L0), tol)
        out = self._finish(L, T, ok, resid)
        H = self.evaluate(L, T, True)["H"]
        out["margin"] = float(np.linalg.eigvalsh(H)[0]) if np.isfinite(H).all() else np.nan
        return out

    def forward(self, X, T, L0, tol=1e-9, report=True):
        """Forward map at payload pose X, from loads L0 (a nearby solution)."""
        T, X = np.asarray(T, float), np.asarray(X, float)

        def residual(x):
            ev = self.evaluate(x, T)
            F = np.concatenate([ev["G"][self.iI], 1e3 * (ev["P"][-1] - X)])
            J = np.vstack([ev["dGdL"][self.iI], 1e3 * ev["dPdL"][self.iX]])
            return F, J, ev

        L, ev, ok, resid = self._newton(residual, np.ravel(L0), tol)
        return self._finish(L, T, ok, resid, ev, report)

    def within_limits(self, T):
        T = np.asarray(T)
        return bool(np.all(T >= self.Tmin - 1e-12)
                    and (self.Tmax is None or np.all(T <= self.Tmax + 1e-12)))

    def _inverse_residual(self, X, w, C):
        n = self.n

        def residual(x):
            L, D = x[:3 * n], x[3 * n:]
            T, dT = self.tension_map(C, D)
            ev = self.evaluate(L, T)
            G = ev["G"].copy()
            G[self.iX] += w                                     # -dU/dX = w
            dGdD = (ev["lam_grad"][:, 0] * dT[:, :1] + ev["lam_grad"][:, 1] * dT[:, 1:]).T
            F = np.concatenate([G, 1e3 * (ev["P"][-1] - X)])
            J = np.vstack([np.hstack([ev["dGdL"], dGdD]),
                           np.hstack([1e3 * ev["dPdL"][self.iX], np.zeros((3, n))])])
            return F, J, ev
        return residual

    def inverse(self, X, w, C, L0, D0, tol=1e-9, report=True, max_it=50, alpha_min=1e-6):
        """Inverse map (n = 3): tension differences D giving wrench w at payload pose X,
        from loads L0 and differences D0 (a nearby solution)."""
        if self.n != 3:
            raise NotImplementedError("the inverse map is square only for n = 3")
        X, w, C = (np.asarray(a, float) for a in (X, w, C))
        residual = self._inverse_residual(X, w, C)
        x0 = np.concatenate([np.ravel(L0), np.asarray(D0, float)])
        x, ev, ok, resid = self._newton(residual, x0, tol, max_it, alpha_min)
        D = x[3 * self.n:]
        T = self.tension_map(C, D)[0]
        return self._finish(x[:3 * self.n], T, ok, resid, ev, report, D=D, T=T,
                            within_limits=self.within_limits(T))

    def inverse_free(self, Xxy, w, C, L0, D0, psi_ref=None, k_psi=0.0, tol=1e-9, max_it=50,
                     report=True, step_max=0.2, restore_its=8, alpha_min=0.03):
        """Inverse map with the payload angle left free (n = 3): payload position Xxy (2,) and
        wrench w. The solutions form a one-parameter family; this picks the one with the
        smallest normalised tension differences f = sum (D_k / s_k)^2 / 2, s_k = C_k - 2 Tmin
        (the most room before the limits), plus k_psi (psi - psi_ref)^2 / 2 to pull the angle
        towards psi_ref.

        Reduced gradient: (1) restore the equilibrium, wrench and position equations exactly
        by minimum-norm Newton steps; (2) step along the family (the null vector of their
        Jacobian) against the gradient of f, at most step_max in the normalised D, with a
        line search on f after restoring again. Stops when the gradient along the family
        vanishes. Returns like inverse; "converged" means the equations hold (the optimum may
        still be approached over the next calls, as the controller warm-starts)."""
        if self.n != 3:
            raise NotImplementedError("the inverse map is for n = 3")
        n, nL = self.n, 3 * self.n
        Xxy, w, C = (np.asarray(a, float) for a in (Xxy, w, C))
        sc = self.d_scale(C)

        def parts(x):
            T, dT = self.tension_map(C, x[nL:])
            ev = self.evaluate(x[:nL], T)
            G = ev["G"].copy()
            G[self.iX] += w
            dGdD = (ev["lam_grad"][:, 0] * dT[:, :1] + ev["lam_grad"][:, 1] * dT[:, 1:]).T
            c = np.concatenate([G, 1e3 * (ev["P"][-1, :2] - Xxy)])
            Jc = np.vstack([np.hstack([ev["dGdL"], dGdD]),
                            np.hstack([1e3 * ev["dPdL"][self.iX][:2], np.zeros((2, n))])])
            psi = ev["P"][-1, 2]
            e = 0.0 if psi_ref is None else psi - psi_ref
            f = 0.5 * np.sum((x[nL:] / sc) ** 2) + 0.5 * k_psi * e * e
            gf = k_psi * e * np.concatenate([ev["dPdL"][self.iX][2], np.zeros(n)])
            gf[nL:] += x[nL:] / sc ** 2
            return c, Jc, f, gf, ev

        def restore(x, its=restore_its):
            """Minimum-norm Newton onto the equations; (x, parts) or None."""
            try:
                pr = parts(x)
            except (ValueError, np.linalg.LinAlgError):
                return None
            for _ in range(its):
                norm = np.abs(pr[0]).max()
                if norm < tol:
                    return x, pr
                dx = np.linalg.lstsq(pr[1], -pr[0], rcond=None)[0]
                alpha = 1.0
                while alpha > alpha_min:            # default fails fast: the controller has a deadline
                    try:
                        trial = parts(x + alpha * dx)
                    except (ValueError, np.linalg.LinAlgError):
                        trial = None
                    if trial is not None and np.abs(trial[0]).max() < (1 - 1e-4 * alpha) * norm:
                        break
                    alpha *= 0.5
                else:
                    return None
                x, pr = x + alpha * dx, trial
            return (x, pr) if np.abs(pr[0]).max() < tol else None

        x0 = np.concatenate([np.ravel(L0), np.asarray(D0, float)])
        res = restore(x0)
        if res is None:
            ev = parts(x0)[4] if True else None
            T = self.tension_map(C, x0[nL:])[0]
            return self._finish(x0[:nL], T, False, np.inf, ev, report, D=x0[nL:].copy(), T=T,
                                within_limits=False)
        x, (c, Jc, f, gf, ev) = res
        for _ in range(max_it):
            t = np.linalg.svd(Jc)[2][-1]                     # null vector of the equations
            slope = gf @ t
            if abs(slope) < 1e-8:
                break
            # Newton-like length from the D-part curvature, capped in normalised D.
            curv = np.sum((t[nL:] / sc) ** 2) + 1e-12
            lam = -slope / curv
            lam *= min(1.0, step_max / max(np.abs(lam * t[nL:] / sc).max(), 1e-300))
            moved = False
            while abs(lam) * np.abs(t[nL:] / sc).max() > 1e-10:
                trial = restore(x + lam * t)
                if trial is not None and trial[1][2] < f - 1e-14:
                    x, (c, Jc, f, gf, ev) = trial
                    moved = True
                    break
                lam *= 0.5
            if not moved:
                break
        D = x[nL:]
        T = self.tension_map(C, D)[0]
        return self._finish(x[:nL], T, True, float(np.abs(c).max()), ev, report, D=D, T=T,
                            within_limits=self.within_limits(T))

"""
controller.py
Online, open-loop controller for the n = 3 bowstring arm (PLAN step 6).

Each control step takes the reference payload pose X* with its first and second time
derivatives and returns the tendon tensions T. It never sees the plant.

    1. Required wrench on the payload: w = M Xdd* with M = diag(m, m, I) ("dynamic" mode),
       or w = 0 ("quasi-static": the arm only holds the payload where the reference is).
    2. Inverse map at X*, warm-started from a linear extrapolation of the last two solutions.
    3. Feasible (converged, tensions within limits, and, on report steps, a positive stability
       margin with every segment stable and a positive-definite payload stiffness K_X, since in
       open loop nothing else holds the payload)? Send T. Otherwise hold the last T and say why.

`choose` takes several candidate references in order of preference (the reference generators
offer, e.g., speed up / coast / brake) and sends the first feasible one, so the generator adapts
within the same control step instead of after a held step.

Stability: every step checks the arm's margin and the payload stiffness K_X (one extra
evaluation); every `report_every` steps (and every step once the margin is small) also each
segment's own stability, which costs as much as a solve.
"""
import time
from dataclasses import dataclass

import numpy as np

from .chain import ChainArm


@dataclass
class Payload:
    m: float = 0.05            # mass
    I: float = 0.05 * 0.3 ** 2  # rotational inertia about its centre (the last disk)

    @property
    def M(self):
        return np.diag([self.m, self.m, self.I])


class Controller:
    def __init__(self, arm: ChainArm, C, payload=Payload(), mode="dynamic", report_every=10,
                 margin_min=1e-3, tol=1e-8, max_it=8, orientation="track", k_psi=0.0,
                 payload_stability="block"):
        """orientation "track": the payload angle follows the reference X*[2].
        orientation "free": only the position is commanded; the angle is the one with the
        least normalised tension differences (the most room before the limits), pulled
        towards X*[2] with weight k_psi (0: not at all). Moving the payload far necessarily
        turns it, so large motions need "free"."""
        if arm.n != 3:
            raise NotImplementedError("the controller is for n = 3")
        self.arm, self.C, self.payload = arm, np.asarray(C, float), payload
        self.mode, self.report_every = mode, report_every
        self.orientation, self.k_psi = orientation, k_psi
        # payload_stability "block": refuse states whose payload stiffness K_X is not positive
        # definite (in open loop the payload would drift away from them); "report": accept
        # them and record K_X (info["kx"]), to show where open-loop tension control falls short.
        self.payload_stability = payload_stability
        # Free orientation: optimiser iterations per step towards the least-tension posture.
        # Where the posture changes fast along the path, too few lets it lag onto a posture
        # that loses payload stiffness.
        self.free_its = 4
        # Free orientation: above this normalised tension use f = sum (D/s)^2 / 2, also try the
        # mirror-image posture (the arm bulging the other way), and switch if it needs less.
        self.switch_f = 0.25
        self.dwell = 0.3          # s after a switch before switching back (unless infeasible)
        self.switch_band = 0.8    # try the mirror posture for targets within |y| < this
        self.switch_gain = 0.95   # switch when it needs at least 5% less (the dwell stops flipping)
        self.last_switch = -np.inf
        self.margin_min, self.tol = margin_min, tol
        self.max_it = max_it          # a warm-started solve needs 2-4; give up early otherwise

    # ------------------------------------------------------------ state
    def reset(self, D0, L0=None):
        """Arm at rest with tension differences D0 (payload free). Returns the payload pose X0,
        where every reference should start. L0: segment loads of that rest state, if known
        (otherwise the rest solve starts from the coarse rod and may find another posture)."""
        D0 = np.asarray(D0, float)
        rest = self.arm.rest(self.arm.tension_map(self.C, D0)[0], L0)
        if not rest["converged"]:
            raise RuntimeError("rest solve failed")
        self.history = [(rest["L"].ravel(), D0)]       # accepted solutions, newest last
        self.T = self.arm.tension_map(self.C, D0)[0]    # last command sent
        self.margin = rest["margin"]
        self.steps = 0
        self.psi_hist = [(None, rest["P"][-1][2])]     # (t, payload angle) of accepted steps
        self.kx = np.inf                               # smallest payload stiffness seen last
        return rest["P"][-1].copy()

    def _guess(self):
        if len(self.history) >= 2:
            (L0, D0), (L1, D1) = self.history[-2:]
            return 2 * L1 - L0, 2 * D1 - D0
        return self.history[-1]

    # ------------------------------------------------------------ one step
    def _solve(self, X, Xdd, report, t=0.0, mirror=True, warm=None, optimise=True, w_fb=None):
        """Inverse map for one reference; (solution or None, reason or None). No state change."""
        X, Xdd = np.asarray(X, float), np.asarray(Xdd, float)
        if self.mode != "dynamic":
            w = np.zeros(3)
        elif self.orientation == "free":
            # The angle is an output here; estimating its acceleration from past steps feeds
            # back into the angle chosen and diverges, so the payload's rotational inertia is
            # neglected (moment demand 0). The plant (PLAN step 7) measures what that costs.
            w = np.array([self.payload.m * Xdd[0], self.payload.m * Xdd[1], 0.0])
        else:
            w = self.payload.M @ Xdd
        if w_fb is not None:                 # closed loop: feedback wrench on top
            w = w + np.asarray(w_fb, float)
        sol = None
        starts = [warm] if warm is not None else [self._guess(), self.history[-1]]
        for L0, D0 in starts:          # given warm start, else extrapolated, then the last one
            try:
                if self.orientation == "free":
                    sol = self.arm.inverse_free(
                        X[:2], w, self.C, L0, D0, psi_ref=X[2] if self.k_psi > 0 else None,
                        k_psi=self.k_psi, tol=self.tol, max_it=self.free_its if optimise else 0,
                        report=report,
                        restore_its=12, alpha_min=0.01)
                else:
                    sol = self.arm.inverse(X, w, self.C, L0, D0, tol=self.tol,
                                           report=report, max_it=self.max_it, alpha_min=0.03)
            except (ValueError, np.linalg.LinAlgError):
                sol = None
            if sol is not None and sol["converged"]:
                break
        if self.orientation == "free" and mirror:
            sol = self._mirror_switch(X, w, sol, report, t)
        if sol is None or not sol["converged"]:
            return None, "no solution", w
        if report:
            self.margin = sol["margin"]
            self.kx = float(np.linalg.eigvalsh(sol["K_X"])[0])
            self.K_X = sol["K_X"]
        if not sol["within_limits"]:
            return sol, "tension limits", w
        if report and not sol["margin"] > self.margin_min:
            return sol, "unstable", w
        if report is True and not sol["segments_stable"]:
            return sol, "a segment unstable", w
        # Open loop: the payload is free, so it must also sit in a restoring well, K_X > 0
        # (with constant T, M dX'' = -K_X dX).
        if (report and self.payload_stability == "block"
                and not np.linalg.eigvalsh(sol["K_X"])[0] > 0):
            return sol, "payload not held (K_X)", w
        return sol, None, w

    def _f(self, sol):
        return 0.5 * float(np.sum((sol["D"] / self.arm.d_scale(self.C)) ** 2))

    def _mirror_switch(self, X, w, sol, report, t):
        """Near the straight pose the arm can hold the payload bulging up or bulging down
        (mirror images). Following one branch across the axis needs ever larger tension
        differences; when the solution is poor (no solution, outside the limits, or f above
        switch_f), try the mirror image of the last solution (loads (nx, -ny, -M0), D -> -D)
        as the warm start, and keep it if it is feasible and needs less."""
        def ok(x):          # converged, within limits and, when checked, payload stiffness > 0
            return (x is not None and x["converged"] and x["within_limits"]
                    and ("K_X" not in x or np.linalg.eigvalsh(x["K_X"])[0] > 0))

        good = ok(sol)
        if (good and self._f(sol) <= self.switch_f) or abs(X[1]) > self.switch_band or X[0] < 0:
            # The mirror posture is only near for targets near the axis in front of the base.
            # Behind it the two "mirror" postures are the two ends of the workspace's C (arm
            # curled one way or the other), and switching would be a jump across it.
            return sol
        if good and self.steps % 10:
            return sol          # a good solution: look at the mirror only every 10th step
        if good and t - self.last_switch < self.dwell:
            return sol          # dwell: no flipping back and forth
        L, D = self.history[-1]
        Lm = (np.asarray(L).reshape(3, 3) * [1.0, -1.0, -1.0]).ravel()
        try:
            alt = self.arm.inverse_free(X[:2], w, self.C, Lm, -np.asarray(D), tol=self.tol,
                                        max_it=10, report=report)
        except (ValueError, np.linalg.LinAlgError):
            return sol
        # Hysteresis: switch only for a clear gain, so the posture does not flip back and forth.
        if ok(alt) and (not good or self._f(alt) < self.switch_gain * self._f(sol)):
            alt["switched"] = True
            return alt
        return sol

    def step(self, t, X, Xd, Xdd, w_fb=None):
        """Tensions (3, 2) for the reference (X, Xd, Xdd) at time t, and a diagnostics dict.
        w_fb: an extra wrench on the payload (closed loop), added to M Xdd."""
        return self.choose(t, [(X, Xd, Xdd)], w_fb=w_fb)

    def choose(self, t, candidates, safe=False, w_fb=None):
        """Like step, for references in order of preference (e.g. speed up, coast, brake):
        sends the first feasible one; info["choice"] is its index, or None if none was (then
        the last feasible T is held). With safe, a candidate must also leave the payload
        holdable at rest at its pose (Xdd = 0), so that stopping is always possible: used for
        live input, which may stop at any moment."""
        start = time.perf_counter()
        self.steps += 1
        # Every step: the arm's margin and the payload stiffness (cheap, "light"); every
        # report_every steps, or near the margin boundary, also each segment's own stability.
        report = (True if (self.steps % self.report_every == 0
                           or self.margin < 10 * self.margin_min) else "light")
        choice, reason, sol, w = None, None, None, np.zeros(3)
        for i, (X, Xd, Xdd) in enumerate(candidates):
            # Only the preferred candidate may switch posture; braking and holding stay put.
            sol, reason, w = self._solve(X, Xdd, report, t, mirror=(i == 0), w_fb=w_fb)
            # Safe: the payload must also be holdable at rest there. Only the small inertial
            # wrench differs, so start from the solution just found (one or two iterations).
            if reason is None and safe and np.any(Xdd):
                _, still, _ = self._solve(X, np.zeros(3), "light", mirror=False,
                                          warm=(np.ravel(sol["L"]), sol["D"]), optimise=False)
                reason = None if still is None else f"cannot stop there ({still})"
            if reason is None:
                choice = i
                break
        if choice is not None:
            new = (np.ravel(sol["L"]), sol["D"])
            if sol.get("switched"):
                self.last_switch = t
            # After a branch switch, extrapolating across the two branches would be nonsense.
            self.history = [new] if sol.get("switched") else (self.history + [new])[-2:]
            self.T = sol["T"]
            self.P = sol["P"]
            self.psi_hist = (self.psi_hist + [(t, sol["P"][-1][2])])[-3:]
        info = {"t": t, "w_req": w, "feasible": choice is not None, "choice": choice,
                "reason": reason, "T": self.T.copy(), "D": self.history[-1][1].copy(),
                "L": self.history[-1][0].copy(), "margin": self.margin, "kx": self.kx,
                "report": report,
                "residual": None if sol is None else sol["residual"],
                "switched": bool(choice is not None and sol.get("switched", False)),
                "P": getattr(self, "P", None), "time": time.perf_counter() - start}
        return self.T.copy(), info

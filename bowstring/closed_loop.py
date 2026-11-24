"""
closed_loop.py
Closed-loop control of the payload with a tip IMU as the only sensor (n = 3).

Plant     The payload (mass m, inertia I) moves under the wrench the arm applies at the commanded
          tensions: M Xdd = w(X, T) - c Xd. The arm itself is massless (quasi-static): its disks
          follow the equilibrium for the current payload pose and tensions (forward map,
          warm-started). Light viscous damping c. RK4.
IMU       At the tip, in the payload frame: specific force f = R(-psi) [xdd, ydd] (the arm moves in
          a horizontal plane, no gravity) and rate gyro psi_d; white noise and constant bias.
          IMU.from_spec: a high-end (navigation-grade) unit, specs in physical units converted
          with a length scale (the segment length, default 0.2 m; time is in seconds).
Encoders  The six tendon lengths, from the motor encoders (the control stays force-driven: the
          tensions are commanded, the lengths only measured). Tendons inextensible.
Estimator Strapdown integration of the IMU (trapezoidal) from the known initial state, corrected
          at the control rate by either
          - length_correct (with encoders): the pose whose model tendon lengths match the
            measured ones (interior disks at equilibrium for the commanded tensions), blended
            in by a second-order complementary filter; or
          - model_correct (IMU only): the arm model's predicted acceleration at the estimated
            pose against the measured one, through the payload stiffness. Sensitive to model
            error (on SOFA: 0.006 of force error -> 0.1 of position bias).
Controller (LinearizedClosedLoop)
          The open-loop online controller at the reference gives the feedforward posture; the
          feedback wrench M (Kd (Xd* - Xd^) + Kp (X* - X^)) is added as a damped linear correction
          of its tension differences. The error obeys M e'' + M Kd e' + (M Kp + K_X) e = 0, stable
          while M Kp + K_X is positive definite, also where the payload is open-loop unstable.
"""
import math
import time

import numpy as np

from .chain import ChainArm
from .tendons import tendons


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PlantError(RuntimeError):
    pass


class Plant:
    """Payload dynamics with the quasi-static arm. Optional, unknown to the controller:
    w_ext(t), an external wrench on the payload (pushes, kicks), and mass(t) -> (m, I), the
    payload's actual mass and inertia (e.g. picking something up)."""

    def __init__(self, arm: ChainArm, payload, X0, L0, damping=(0.01, 0.01, 0.001), w_ext=None,
                 mass=None):
        self.arm, self.M = arm, np.array([payload.m, payload.m, payload.I])
        self.c = np.asarray(damping, float)
        self.X, self.V = np.asarray(X0, float).copy(), np.zeros(3)
        self.L = np.asarray(L0, float).copy()
        self.w_ext, self.mass, self.t = w_ext, mass, 0.0

    def _accel(self, X, V, T, t):
        fw = self.arm.forward(X, T, self.L, report=False)
        if not fw["converged"]:
            raise PlantError(f"no arm equilibrium for the payload at {np.round(X, 3)}")
        M = self.M if self.mass is None else np.array([self.mass(t)[0], self.mass(t)[0],
                                                        self.mass(t)[1]])
        w = fw["w"] if self.w_ext is None else fw["w"] + np.asarray(self.w_ext(t), float)
        return (w - self.c * V) / M, fw["L"]

    def step(self, T, dt):
        """Advance by dt with tensions T; returns the payload acceleration at the new state."""
        X, V, t = self.X, self.V, self.t
        a1, L1 = self._accel(X, V, T, t)
        self.L = L1
        a2, _ = self._accel(X + 0.5 * dt * V, V + 0.5 * dt * a1, T, t + 0.5 * dt)
        a3, _ = self._accel(X + 0.5 * dt * (V + 0.5 * dt * a1), V + 0.5 * dt * a2, T, t + 0.5 * dt)
        a4, L4 = self._accel(X + dt * (V + 0.5 * dt * a2), V + dt * a3, T, t + dt)
        self.X = X + dt * (V + dt / 6 * (a1 + a2 + a3))
        self.V = V + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        self.L, self.t = L4, t + dt
        a, self.L = self._accel(self.X, self.V, T, self.t)
        return a

    def margin(self, T):
        """Stability of the arm's interior at the current state (payload held)."""
        return self.arm.forward(self.X, T, self.L, report="light")["margin"]


def bdf_alpha(dt, alpha):
    """BDF-alpha coefficients (Celaya et al. 2013, as used by Till et al. 2019, sec. 2.2):
    y_t ~ c0 y_i + c1 y_{i-1} + c2 y_{i-2} + d1 y_t,{i-1}. alpha = 0 is BDF2, -0.5 trapezoidal."""
    return ((1.5 + alpha) / (dt * (1 + alpha)), -2.0 / dt,
            (0.5 + alpha) / (dt * (1 + alpha)), alpha / (1 + alpha))


class ImplicitPlant(Plant):
    """The same plant (payload dynamics, quasi-static arm), stepped implicitly after Till et al.
    (2019): the time derivatives are replaced by the BDF-alpha formula, so each step is one
    boundary-value problem in arc length, solved by shooting on the segment base loads L.
    The payload's equation of motion is the distal boundary condition (their sec. 5.2):

        interior disks   dU/dP_I(L) = 0
        payload          dU/dX(L) + M Xdd + c Xd - w_ext = 0,  X = P_n(L),
                         Xd = c0 X + Xh,  Xdd = c0 Xd + Vh   (history terms Xh, Vh)

    One Newton solve per step (with the exact Jacobian dU/dX_L + (c0^2 M + c0 c) dX/dL),
    warm-started by extrapolating L, replaces the four forward solves of explicit RK4 plus
    one for the output acceleration."""

    def __init__(self, *args, alpha=-0.2, tol=1e-9, max_it=20, **kw):
        super().__init__(*args, **kw)
        self.alpha, self.tol, self.max_it = alpha, tol, max_it
        self.hist = None                  # (X_{i-1}, X_{i-2}, V_{i-1}, V_{i-2}, A_{i-1})
        self.L_prev = None
        self.iterations = 0               # Newton iterations in the last step

    def step(self, T, dt):
        arm, n = self.arm, self.arm.n
        if self.hist is None:             # starts at rest: constant history
            self.hist = (self.X.copy(), self.X.copy(), self.V.copy(), self.V.copy(), np.zeros(3))
        X1, X2, V1, V2, A1 = self.hist
        c0, c1, c2, d1 = bdf_alpha(dt, self.alpha)
        Xh = c1 * X1 + c2 * X2 + d1 * V1
        Vh = c1 * V1 + c2 * V2 + d1 * A1
        t = self.t + dt
        M = self.M if self.mass is None else np.array([self.mass(t)[0], self.mass(t)[0],
                                                        self.mass(t)[1]])
        w_ext = np.zeros(3) if self.w_ext is None else np.asarray(self.w_ext(t), float)
        k = c0 * c0 * M + c0 * self.c
        iI, iX = arm.iI, arm.iX

        def residual(L):
            ev = arm.evaluate(L, T)
            X = ev["P"][-1]
            V = c0 * X + Xh
            A = c0 * V + Vh
            F = ev["G"].copy()
            F[iX] += M * A + self.c * V - w_ext
            J = ev["dGdL"].copy()
            J[iX] += k[:, None] * ev["dPdL"][iX]
            return F, J, (X, V, A)

        L0 = np.ravel(self.L)
        L = L0 if self.L_prev is None else 2 * L0 - self.L_prev
        try:
            F, J, state = residual(L)
        except np.linalg.LinAlgError:
            L = L0
            F, J, state = residual(L)
        # Merit: the payload rows hold c0^2 M X ~ 1e5 at dt = 1 ms, which must cancel (the
        # "catastrophic cancellation" of implicit differences at small dt, Till et al. 2019,
        # sec. 2.4), and a nearly straight segment makes the first full Newton step overshoot
        # there. Measured as a position error, 1e3 F / k (the scaling ChainArm.forward uses),
        # they are on the interior rows' footing. Row scaling leaves the Newton step unchanged.
        S = np.ones(3 * n)
        S[iX] = 1e3 / k

        def merit(F):
            return float(np.abs(S * F).max())

        norm = merit(F)
        it = 0
        while norm > self.tol:
            if it >= self.max_it:
                raise PlantError(f"implicit step did not converge (|F| = {norm:.1e})")
            step = np.linalg.solve(J, -F)
            a = 1.0
            while True:
                try:
                    trial = residual(L + a * step)
                    if merit(trial[0]) < (1 - 1e-4 * a) * norm:
                        break
                except np.linalg.LinAlgError:
                    pass
                a *= 0.5
                if a < 1e-4:
                    if norm < 1e2 * self.tol:          # round-off floor
                        trial = (F, J, state)
                        a = 0.0
                        break
                    raise PlantError(f"implicit step stalled (|F| = {norm:.1e}) at "
                                     f"{np.round(state[0], 3)}")
            L = L + a * step
            F, J, state = trial
            norm = merit(F) if a > 0 else 0.0
            it += 1
        X, V, A = state
        self.iterations = it
        self.hist = (X.copy(), X1, V.copy(), V1, A.copy())
        self.L_prev, self.L = L0, np.ravel(L).copy()
        self.X, self.V, self.t = X.copy(), V.copy(), t
        return A.copy()


G0 = 9.80665
# A high-end (navigation-grade) IMU: accelerometer noise density (m/s^2/sqrt(Hz)) and bias
# (m/s^2), gyro angle random walk (deg/sqrt(h)) and bias (deg/h).
HIGH_END_IMU = {"accel_noise": 10e-6 * G0, "accel_bias": 25e-6 * G0,
                "gyro_arw": 0.002, "gyro_bias": 0.005}


class IMU:
    @classmethod
    def from_spec(cls, spec=HIGH_END_IMU, length_scale=0.2, rate=1000.0, seed=0):
        """An IMU from datasheet specs, in the arm's units (length = segment length, given in
        metres by length_scale; seconds). Per-sample white noise sigma = density sqrt(rate);
        each bias is a constant for the run, drawn with the spec as its standard deviation."""
        rng = np.random.default_rng(seed + 1000)
        sa = spec["accel_noise"] * math.sqrt(rate) / length_scale
        sg = math.radians(spec["gyro_arw"]) / 60.0 * math.sqrt(rate)
        ba = rng.standard_normal(2) * spec["accel_bias"] / length_scale
        bg = float(rng.standard_normal()) * math.radians(spec["gyro_bias"]) / 3600.0
        return cls(sa, sg, ba, bg, seed)

    def __init__(self, accel_noise=0.0, gyro_noise=0.0, accel_bias=(0.0, 0.0), gyro_bias=0.0,
                 seed=0):
        self.sa, self.sg = accel_noise, gyro_noise
        self.ba, self.bg = np.asarray(accel_bias, float), gyro_bias
        self.rng = np.random.default_rng(seed)

    def read(self, psi, psi_d, acc_xy):
        c, s = math.cos(psi), math.sin(psi)
        f = np.array([c * acc_xy[0] + s * acc_xy[1], -s * acc_xy[0] + c * acc_xy[1]])
        return (f + self.ba + self.sa * self.rng.standard_normal(2),
                psi_d + self.bg + self.sg * self.rng.standard_normal())


class TendonEncoders:
    """Tendon lengths (n, 2) from the motor encoders: the true lengths plus white noise of the
    encoder resolution (in segment lengths; 1e-5 = 2 um at 0.2 m)."""

    def __init__(self, noise=1e-5, seed=0):
        self.noise, self.rng = noise, np.random.default_rng(seed + 2000)

    def read(self, lam):
        lam = np.asarray(lam, float)
        return lam + self.noise * self.rng.standard_normal(lam.shape)


class Estimator:
    """Strapdown integration of the tip IMU, trapezoidal: the angle from the mean of consecutive
    gyro samples, the world-frame acceleration from consecutive accelerometer samples rotated by
    their angles, then velocity and position. (Treating each sample as constant over the step,
    first order, left a velocity bias that drifted the position by 0.18 in 14 s on the line.)"""

    def __init__(self, X0, V0=np.zeros(3)):
        self.X, self.V = np.asarray(X0, float).copy(), np.asarray(V0, float).copy()
        self.prev = None                     # (world-frame acceleration, gyro) of the last sample

    @staticmethod
    def _rotate(psi, f):
        c, s = math.cos(psi), math.sin(psi)
        return np.array([c * f[0] - s * f[1], s * f[0] + c * f[1]])

    def update(self, f, gyro, dt):
        if self.prev is None:
            self.prev = (self._rotate(self.X[2], f), gyro)
        a0, g0 = self.prev
        self.X[2] += 0.5 * (g0 + gyro) * dt
        a1 = self._rotate(self.X[2], f)
        v0 = self.V[:2].copy()
        self.V[:2] += 0.5 * (a0 + a1) * dt
        self.X[:2] += 0.5 * (v0 + self.V[:2]) * dt
        self.V[2] = gyro
        self.prev = (a1, gyro)
        return self.X.copy(), self.V.copy()

    def length_correct(self, arm, T, L, lam_meas, dt, wn=3.0, zeta=1.0, reg=1e-8):
        """Tendon-length correction, at the control rate (dt). The model's tendon lengths at the
        estimated pose (interior disks at equilibrium for the commanded tensions T; the
        interior is stiff, so force errors move it little) against the measured ones give a
        pose fix dX = (J^T J)^-1 J^T (lam_meas - lam(X^)), J = d lam / dX (6 x 3) from the
        interior's linearisation. Blended with the IMU by a second-order complementary filter
        (crossover wn rad/s): X^ += 2 zeta wn dt dX, V^ += wn^2 dt dX.
        Returns the arm loads at the estimate (warm start for the next call)."""
        try:
            fw = arm.forward(self.X, T, L, report=False)
        except (ValueError, np.linalg.LinAlgError):
            return L
        if not fw["converged"]:
            return L
        ev = arm.evaluate(fw["L"], T, hessian=True)
        H = ev["H"]
        if not np.isfinite(H).all():
            return fw["L"]
        n = arm.n
        lg = ev["lam_grad"].reshape(2 * n, 3 * n)                    # d lam / d(P_1..P_n)
        dPi = -np.linalg.solve(H[arm.iI, arm.iI], H[arm.iI, arm.iX])  # interior follows X
        J = lg[:, arm.iI] @ dPi + lg[:, arm.iX]
        r = np.ravel(lam_meas) - tendons(ev["P"], arm.offsets)[0].ravel()
        JtJ = J.T @ J
        dX = np.linalg.solve(JtJ + reg * np.trace(JtJ) * np.eye(3), J.T @ r)
        self.X += 2 * zeta * wn * dt * dX
        self.V += wn ** 2 * dt * dX
        return fw["L"]

    def model_correct(self, arm, T, L, M, c, gain=0.05, reg=0.05, gate=2.0, cap=0.002):
        """Model-aided correction, at the control rate. The arm model predicts the payload's
        acceleration at the estimated pose from the tensions sent; it depends on the pose
        through the payload stiffness K_X, so the mismatch with the measured (world-frame, latest
        sample) acceleration tells where the estimate is off:
            a_meas - a_model(X^) ~ (K_X / M) (X^ - X)  ->  dX = -(K_X/M)^+ (a_meas - a_model), damped.
        Applied with a small gain (a low-pass), position only (x, y), at most `cap` per call.
        A mismatch above `gate` (an external push) is skipped: the estimate coasts on the IMU
        alone until it is over. A steady unmodelled force or mass, on the other hand, cannot be
        told apart from a position error with an IMU alone: it biases the estimate.
        Returns the arm loads at the estimate (warm start for the next call), or L if the model
        cannot be evaluated."""
        if self.prev is None:
            return L
        a_meas = self.prev[0]            # the latest sample: the same instant as the model below
        try:
            fw = arm.forward(self.X, T, L, report="light")
        except (ValueError, np.linalg.LinAlgError):
            return L
        if not fw["converged"]:
            return L
        nu = a_meas - (fw["w"][:2] - c[:2] * self.V[:2]) / M[:2]
        if np.hypot(*nu) > gate:
            return fw["L"]
        A = fw["K_X"][:2, :2] / M[:2, None]
        scale = max(np.abs(A).max(), 1e-9)
        # damped least squares: well posed whatever the sign of K_X's eigenvalues (adding
        # reg I directly could cancel a negative eigenvalue and make the system singular)
        dX = gain * -np.linalg.solve(A.T @ A + (reg * scale) ** 2 * np.eye(2), A.T @ nu)
        if np.hypot(*dX) > cap:
            dX *= cap / np.hypot(*dX)
        self.X[:2] += dX
        return fw["L"]


class LinearizedClosedLoop:
    """Feedback as a small linear correction of the feedforward tensions.

    The feedforward (the online Controller at the reference) gives a posture (loads L, tension
    differences D) whose wrench at the reference pose is M Xdd*. The feedback wrench
    w_fb = M (Kd (Xd* - Xd^) + Kp (X* - X^)) is then realised by dD = S w_fb, where S = dD/dw at
    the fixed reference pose comes from the inverse map's Jacobian (the posture's interior
    stays at equilibrium). S is applied through a Tikhonov-damped solve (and the change of D is
    capped): near a fold of the fixed-pose problem, where the tensions have little authority over
    some wrench direction, the correction gives up that direction instead of asking for huge
    tensions. No new nonlinear solve, so no jump to another posture branch."""

    def __init__(self, arm: ChainArm, C, payload, wn=(3.0, 3.0, 6.0), zeta=1.0, damping=1e-6,
                 dD_max=2.0, wi=0.0, i_max=0.5, dt=0.01):
        """PID in wrench space: w_fb = M (Kd de + Kp e + Ki int e), Kp = wn^2, Kd = 2 zeta wn,
        Ki = Kp wi (wi: integral corner, rad/s, per axis; 0 = PD). Per axis the error obeys
        s^3 + Kd s^2 + (Kp + K_X/M) s + Ki = 0, stable (Routh) while Kd (Kp + K_X/M) > Ki.
        Anti-windup: the integral wrench M Ki int e is clamped to +-i_max per axis and does not
        accumulate while the tension correction is saturated (dD_max)."""
        self.arm, self.C = arm, np.asarray(C, float)
        self.M = np.array([payload.m, payload.m, payload.I])
        wn = np.asarray(wn, float)
        self.Kp, self.Kd = wn ** 2, 2 * zeta * wn
        self.Ki = self.Kp * np.broadcast_to(np.asarray(wi, float), (3,))
        self.i_max, self.dt = i_max, dt
        self.z, self.saturated = np.zeros(3), False
        self.damping, self.dD_max = damping, dD_max

    def step(self, ref, est, L_ff, D_ff, w_ff):
        start = time.perf_counter()
        (X, Xd, Xdd), (Xh, Vh) = ref, est
        X, Xd = np.asarray(X, float), np.asarray(Xd, float)
        e = X - Xh
        e[2] = wrap(e[2])
        if self.Ki.any() and not self.saturated:
            self.z += e * self.dt
            lim = self.i_max / np.maximum(self.M * self.Ki, 1e-12)
            self.z = np.clip(self.z, -lim, lim)
        w_i = self.M * self.Ki * self.z
        w_fb = self.M * (self.Kd * (Xd - Vh) + self.Kp * e) + w_i
        n = self.arm.n
        F, J, _ = self.arm._inverse_residual(X, np.asarray(w_ff, float), self.C)(
            np.concatenate([np.ravel(L_ff), np.asarray(D_ff, float)]))
        rhs = np.zeros(len(F))
        rhs[3 * n - 3:3 * n] = -w_fb                 # residual rows of the payload wrench
        # Tikhonov-damped solve: directions with singular values far below the largest (a fold
        # of the fixed-pose problem, no authority over that wrench) are attenuated, the rest
        # (relative singular values down to ~1e-5 here) pass almost unchanged.
        U, sv, Vt = np.linalg.svd(J)
        lam = self.damping * sv[0]
        dx = Vt.T @ ((sv / (sv ** 2 + lam ** 2)) * (U.T @ rhs))
        dD = dx[3 * n:]
        self.saturated = bool(np.abs(dD).max() > self.dD_max)
        if self.saturated:                           # safety net
            dD = dD * self.dD_max / np.abs(dD).max()
        dx[3 * n:] = dD
        D = np.asarray(D_ff, float) + dD
        T = self.arm.tension_map(self.C, D)[0]
        return T, {"w_fb": w_fb, "w_i": w_i, "dD": dx[3 * n:],
                   "time": time.perf_counter() - start}

"""
reference.py
Online reference generators for the payload pose X = (x, y, angle) (PLAN step 6).
Each gives a smooth X*, Xd*, Xdd* one control step at a time.

Trajectory   a circle or a line with given timing, started and stopped smoothly (quintic).
PathFollower a geometric path; its speed along the path is chosen online: each step it offers
             speed up / coast / brake, the controller sends the first feasible one, and it brakes
             in time to stop at the end.
Live         velocity commands (gamepad sticks and triggers, in [-1, 1]) through a filter with
             speed and acceleration limits, kept in a workspace box; each step it offers the
             commanded motion / braking / holding still.

Use: refs = gen.candidates(dt[, cmd]); T, info = controller.choose(t, refs[, safe=True]);
     gen.accept(info["choice"]).  (safe=True for Live: move only where stopping is possible.)
"""
import numpy as np


def quintic(tau):
    """s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 on [0, 1] (clamped), with ds/dtau, d2s/dtau2."""
    tau = min(max(tau, 0.0), 1.0)
    return (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5,
            30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4,
            60 * tau - 180 * tau ** 2 + 120 * tau ** 3)


class Trajectory:
    """X*(t) for t in [0, duration], at rest at both ends."""

    def __init__(self, sample, duration):
        self._sample, self.duration = sample, duration

    def __call__(self, t):
        return self._sample(t)

    @classmethod
    def circle(cls, X0, radius, duration, turns=1.0, centre_dir=np.pi):
        """A circle through the start pose X0 (angle held), its centre at `radius` from X0 in
        direction centre_dir; `turns` laps in `duration`, quintic in the polar angle."""
        X0 = np.asarray(X0, float)
        c = X0[:2] + radius * np.array([np.cos(centre_dir), np.sin(centre_dir)])
        a0, total = centre_dir + np.pi, 2 * np.pi * turns

        def sample(t):
            s, ds, dds = quintic(t / duration)
            a, da, dda = a0 + total * s, total * ds / duration, total * dds / duration ** 2
            ca, sa = np.cos(a), np.sin(a)
            X = np.array([c[0] + radius * ca, c[1] + radius * sa, X0[2]])
            Xd = np.array([-radius * sa * da, radius * ca * da, 0.0])
            Xdd = np.array([-radius * (ca * da ** 2 + sa * dda),
                            radius * (-sa * da ** 2 + ca * dda), 0.0])
            return X, Xd, Xdd
        return cls(sample, duration)

    @classmethod
    def line(cls, X0, X1, duration):
        """Straight line from pose X0 to pose X1 (angle interpolated), quintic in time."""
        X0, X1 = np.asarray(X0, float), np.asarray(X1, float)

        def sample(t):
            s, ds, dds = quintic(t / duration)
            return X0 + (X1 - X0) * s, (X1 - X0) * ds / duration, (X1 - X0) * dds / duration ** 2
        return cls(sample, duration)


class PathFollower:
    """A geometric path X(sigma), sigma in [0, length], with its speed chosen online.

    path(sigma) -> (X, dX/dsigma, d2X/dsigma2). Each update: accelerate at a_max up to v_max
    while the last step was feasible, decelerate at a_max after an infeasible one, and brake in
    time to stop at the end. Xd = X' sd, Xdd = X'' sd^2 + X' sdd."""

    def __init__(self, path, length, v_max, a_max):
        self.path, self.length, self.v_max, self.a_max = path, length, v_max, a_max
        self.sigma, self.speed = 0.0, 0.0
        self.a_try = a_max        # acceleration offered first; adapts to what is feasible

    @property
    def done(self):
        return self.sigma >= self.length - 1e-12 and self.speed == 0.0

    def _advance(self, dt, acc):
        """State and reference after one step at acceleration acc (speed kept in [0, v_max])."""
        speed = min(max(self.speed + acc * dt, 0.0), self.v_max)
        acc = (speed - self.speed) / dt
        sigma = min(self.sigma + 0.5 * (self.speed + speed) * dt, self.length)
        if sigma >= self.length:
            speed = 0.0
        X, dX, ddX = self.path(sigma)
        return (sigma, speed), (X, dX * speed, ddX * speed ** 2 + dX * acc)

    def speed_cap(self):
        """Highest speed whose centripetal acceleration, kappa v^2, stays within the
        acceleration found feasible so far (a_try), over the next stopping distance."""
        reach = self.speed ** 2 / (2 * self.a_max) + 0.05 * self.length
        sig = np.linspace(self.sigma, min(self.sigma + reach, self.length), 12)
        kappa = max(np.hypot(*self.path(x)[2][:2]) for x in sig)
        return np.sqrt(self.a_try / kappa) if kappa > 1e-9 else np.inf

    def candidates(self, dt):
        """References for speeding up (at the adaptive a_try), coasting and braking, in that
        order. Only braking once it is time to stop at the end, or when the speed is above
        the curvature cap ahead."""
        remaining = self.length - self.sigma
        if self.speed ** 2 / (2 * self.a_max) >= remaining:
            accs = [-self.a_max]                         # stop at the end
        else:
            # Track the allowed speed smoothly (time constant 0.2 s) rather than bang-bang
            # between speeding up and braking at the cap, which makes Xdd chatter.
            v_goal = min(self.v_max, self.speed_cap())
            acc = float(np.clip((v_goal - self.speed) / 0.2, -self.a_max, self.a_try))
            accs = [acc, 0.0, -self.a_max] if acc > 0 else [acc, -self.a_max]
        self._pending = [self._advance(dt, a) for a in accs]
        return [ref for _, ref in self._pending]

    def accept(self, choice):
        """Move to the chosen candidate. With none feasible, stay at the last accepted pose and
        stop (going on, even braking, could carry the reference further into a region where
        the arm cannot hold the payload). Speeding up accepted: offer 1.1x more next time (up
        to a_max); refused: offer half."""
        if choice is None:
            self.speed = 0.0
        else:
            self.sigma, self.speed = self._pending[choice][0]
        if len(self._pending) > 1:
            self.a_try = (min(self.a_max, 1.1 * self.a_try) if choice == 0
                          else max(0.5 * self.a_try, 1e-3 * self.a_max))

    def update(self, dt, feasible=True):
        """One step without lookahead: speed up while feasible, else brake."""
        refs = self.candidates(dt)
        i = 0 if feasible else len(refs) - 1
        self.accept(i)
        return refs[i]

    @classmethod
    def line(cls, X0, X1, v_max, a_max):
        """Straight path between poses X0 and X1; sigma is the distance travelled in (x, y)
        (or in the angle, for a pure rotation)."""
        X0, X1 = np.asarray(X0, float), np.asarray(X1, float)
        length = float(np.hypot(*(X1 - X0)[:2])) or abs(X1[2] - X0[2])
        u = (X1 - X0) / length
        return cls(lambda s: (X0 + u * s, u, np.zeros(3)), length, v_max, a_max)

    @classmethod
    def circle(cls, X0, radius, v_max, a_max, turns=1.0, centre_dir=np.pi):
        """Circular path through X0 (angle held); sigma is arc length."""
        X0 = np.asarray(X0, float)
        c = X0[:2] + radius * np.array([np.cos(centre_dir), np.sin(centre_dir)])
        a0 = centre_dir + np.pi

        def path(s):
            a = a0 + s / radius
            ca, sa = np.cos(a), np.sin(a)
            return (np.array([c[0] + radius * ca, c[1] + radius * sa, X0[2]]),
                    np.array([-sa, ca, 0.0]), np.array([-ca, -sa, 0.0]) / radius)
        return cls(path, 2 * np.pi * radius * turns, v_max, a_max)


class Live:
    """Target pose driven by velocity commands, e.g. the gamepad.

    cmd = (ux, uy, ur) in [-1, 1]: the stick gives the x, y velocity (ux, uy) * v_max; the
    triggers give the rotation rate ur * w_max with ur = LT - RT (counterclockwise positive, so
    RT turns the payload clockwise). The velocity
    follows the command with acceleration at most a_max (x, y) and alpha_max (angle), so Xdd
    stays bounded; the pose stays inside the box [lo, hi]. With the controller's `choose`, a
    command that cannot be met is replaced, in the same step, by braking or holding still."""

    def __init__(self, X0, v_max=0.1, w_max=0.5, a_max=0.5, alpha_max=2.0, lo=None, hi=None,
                 deadzone=0.08, jerk=None):
        self.X = np.asarray(X0, float).copy()
        self.V = np.zeros(3)
        self.A = np.zeros(3)
        # Jerk limit: the acceleration (so the force demanded of the arm) ramps up over
        # a_max / jerk seconds instead of jumping, which matters for a soft open-loop arm.
        self.jerk = 4.0 * a_max if jerk is None else jerk
        self.v_max, self.w_max, self.a_max, self.alpha_max = v_max, w_max, a_max, alpha_max
        self.lo = np.full(3, -np.inf) if lo is None else np.asarray(lo, float)
        self.hi = np.full(3, np.inf) if hi is None else np.asarray(hi, float)
        self.deadzone = deadzone

    def _shape(self, u):
        """Deadzone, then rescaled so the output still reaches 1."""
        u = float(np.clip(u, -1.0, 1.0))
        return 0.0 if abs(u) < self.deadzone else np.sign(u) * (abs(u) - self.deadzone) / (1 - self.deadzone)

    def _advance(self, dt, target, X=None, V=None):
        """Pose, velocity and acceleration after one step towards the velocity `target`."""
        X = self.X if X is None else X
        V0 = self.V if V is None else V
        acc = (target - V0) / dt
        a_xy = np.hypot(*acc[:2])
        if a_xy > self.a_max:
            acc[:2] *= self.a_max / a_xy
        acc[2] = np.clip(acc[2], -self.alpha_max, self.alpha_max)
        dA = acc - self.A                                   # jerk limit on x, y
        if np.hypot(*dA[:2]) > self.jerk * dt:
            dA[:2] *= self.jerk * dt / np.hypot(*dA[:2])
        acc[:2] = self.A[:2] + dA[:2]
        V1 = V0 + acc * dt
        X1 = X + 0.5 * (V0 + V1) * dt
        out = (X1 < self.lo) | (X1 > self.hi)      # workspace box: stop at a face
        X1 = np.clip(X1, self.lo, self.hi)
        V1[out], acc[out] = 0.0, 0.0
        return X1, V1, acc

    def candidates(self, dt, cmd):
        """References in order: as commanded; braking (command released); holding still."""
        ux, uy, ur = (self._shape(u) for u in cmd)
        v_xy = np.array([ux, uy])
        if np.hypot(*v_xy) > 1:
            v_xy /= np.hypot(*v_xy)
        target = np.array([v_xy[0] * self.v_max, v_xy[1] * self.v_max, ur * self.w_max])
        self._pending = [self._advance(dt, target), self._advance(dt, np.zeros(3)),
                         (self.X.copy(), np.zeros(3), np.zeros(3))]
        return [(X.copy(), V.copy(), a.copy()) for X, V, a in self._pending]

    def accept(self, choice):
        """Move to the chosen candidate; with none feasible, hold still."""
        X, V, A = self._pending[-1 if choice is None else choice]
        self.X, self.V, self.A = X.copy(), V.copy(), A.copy()

    def sync_angle(self, psi):
        """With the orientation left free, the controller chooses the payload angle. While
        the triggers are idle, call this with the angle achieved, so that the angle target
        follows it (no pull) and the triggers turn the payload from where it is."""
        self.X[2], self.V[2] = psi, 0.0

    def update(self, dt, cmd, feasible=True):
        """One step without lookahead: as commanded while feasible, else hold still."""
        refs = self.candidates(dt, cmd)
        i = 0 if feasible else len(refs) - 1
        self.accept(i)
        return refs[i]

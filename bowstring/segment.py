"""
segment.py
One elastica segment (length 1, B = 1) between two disks, by shooting.

A segment starts clamped at its lower disk, at angle 0 in that disk's frame, and ends at the
relative pose r = (rx, ry, rt) of its upper disk: position and angle in the lower disk's frame.

Shooting: unknown base load (nx, ny, M0), constant internal force n = (nx, ny),
    x' = cos t,  y' = sin t,  t' = M,  M' = nx sin t - ny cos t,  M(0) = M0,
integrated by RK4 together with its variational equations. Newton matches the end pose to r.
By the envelope theorem the end wrench is g = dV/dr = (nx, ny, M(1)), and the stiffness is
K = dg/dr = (dg/dload) (dpose/dload)^-1.

Chord chart (the FEniCSx table's coordinates): d = |(rx, ry)|, alpha = atan2(ry, rx),
phi1 = -alpha, phi2 = rt - alpha. The solver's multipliers in the chord frame are
(fx, fy) = -R(-alpha) n, so that dU = -fx dd - M1 dphi1 + M2 dphi2.
"""
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FENICS = Path(__file__).resolve().parents[1] / "fenics"
if str(_FENICS) not in sys.path:
    sys.path.append(str(_FENICS))
from branch import first_mode_amplitude, first_mode_projection  # noqa: E402
from stability import unstable_directions  # noqa: E402


@dataclass
class SegmentState:
    V: float                    # bending energy
    g: np.ndarray               # dV/dr, (3,)
    K: np.ndarray               # d2V/dr2, (3, 3)
    load: np.ndarray            # base load (nx, ny, M0): the warm start for shooting
    branch: int                 # +1 up / -1 down (sign of p), as in the FEniCSx table
    p: float                    # first-mode projection
    stable: bool                # clamped stability with both ends held
    backend: str                # "table" or "shoot"
    chart: tuple = field(default=())   # (d, phi1, phi2)


# ------------------------------------------------------------------ chord chart

def chart(r):
    """(d, phi1, phi2), the Jacobian Q = dq/dr (3, 3) and the second derivatives of
    d and alpha in (rx, ry) (2, 2 each)."""
    rx, ry, rt = r
    d2 = rx * rx + ry * ry
    d = math.sqrt(d2)
    a = math.atan2(ry, rx)
    dd = np.array([rx / d, ry / d, 0.0])
    da = np.array([-ry / d2, rx / d2, 0.0])
    Q = np.array([dd, -da, np.array([0.0, 0.0, 1.0]) - da])
    d3, d4 = d2 * d, d2 * d2
    Hd = np.array([[ry * ry / d3, -rx * ry / d3], [-rx * ry / d3, rx * rx / d3]])
    Ha = np.array([[2 * rx * ry / d4, (ry * ry - rx * rx) / d4],
                   [(ry * ry - rx * rx) / d4, -2 * rx * ry / d4]])
    return (d, -a, rt - a), Q, Hd, Ha


def to_pose_coords(Gq, Hq, Q, Hd, Ha):
    """g and K in r from the gradient Gq and Hessian Hq of U(d, phi1, phi2)."""
    g = Q.T @ Gq
    K = Q.T @ Hq @ Q
    # sum_a dU/dq_a d2q_a/dr2: d2phi1 = d2phi2 = -d2alpha
    K[:2, :2] += Gq[0] * Hd - (Gq[1] + Gq[2]) * Ha
    return g, 0.5 * (K + K.T)


# ------------------------------------------------------------------ shooting

def _steps(load):
    """RK4 steps: enough to resolve boundary layers of width 1/sqrt(|n|)."""
    n = max(abs(load[0]), abs(load[1]), 1.0)
    return int(min(4000, max(200, 25.0 * math.sqrt(n))))


def build_shared(src, name, deps=()):
    """ctypes library compiled from the C file src into bowstring/_build/<name>, rebuilt when src or
    a dep is newer. Tries $CC, the environment's gcc, then gcc, cc and clang. Returns the
    CDLL, or None with a warning (the callers then fall back to pure Python, which is far
    slower than real time)."""
    import ctypes
    import os
    import shutil
    import subprocess
    import warnings
    lib = Path(__file__).parent / "_build" / name
    newest = max(Path(f).stat().st_mtime for f in (src, *deps))
    if not lib.exists() or lib.stat().st_mtime < newest:
        lib.parent.mkdir(exist_ok=True)
        cands = [os.environ.get("CC"), str(Path(sys.executable).with_name("gcc")), "gcc", "cc",
                 "clang"]
        errors = []
        for cc in filter(None, cands):
            if not (Path(cc).exists() or shutil.which(cc)):
                continue
            r = subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", str(lib), str(src), "-lm"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            errors.append(f"{cc}: {r.stderr.strip()[:300]}")
        else:
            warnings.warn(f"could not compile {Path(src).name} (no working C compiler; "
                          f"{'; '.join(errors) or 'none found'}): using pure Python, "
                          "which is much slower than real time", RuntimeWarning)
            return None
    try:
        return ctypes.CDLL(str(lib))
    except OSError as err:
        warnings.warn(f"could not load {lib}: {err}: using pure Python", RuntimeWarning)
        return None


def _load_c():
    """The compiled integrator (_rk4.c); None if it cannot be built (pure Python then)."""
    import ctypes
    c = build_shared(Path(__file__).with_name("_rk4.c"), "_rk4.so")
    if c is None:
        return None
    dp = ctypes.POINTER(ctypes.c_double)
    c.integrate.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                            dp, dp]
    c.integrate.restype = None
    return c, dp


_C = _load_c()


def _pack(s, thetas):
    J = np.array([[s[5], s[9], s[13]], [s[6], s[10], s[14]],
                  [s[7], s[11], s[15]], [s[8], s[12], s[16]]])
    return np.array(s[:3]), float(s[3]), float(s[4]), J, thetas


def integrate(load, record=False):
    """End state of the shooting ODE and its derivatives in the load.

    Returns pose (3,), M(1), V, J (4, 3) = d(x, y, t, M)(1)/d(nx, ny, M0) and, with record,
    the angles t at the N + 1 nodes. Uses the compiled integrator when available."""
    if _C is None:
        return integrate_py(load, record)
    lib, dp = _C
    nx, ny, M0 = (float(v) for v in load)
    N = _steps(load)
    out = np.empty(17)
    thetas = np.empty(N + 1) if record else None
    lib.integrate(nx, ny, M0, N, out.ctypes.data_as(dp),
                  thetas.ctypes.data_as(dp) if record else None)
    return _pack(out, thetas)


def integrate_py(load, record=False):
    """integrate() in pure Python (fallback, and the reference for the C version)."""
    nx, ny, M0 = (float(v) for v in load)
    N = _steps(load)
    h = 1.0 / N
    sin, cos = math.sin, math.cos
    # state: x, y, t, M, E, then (dx, dy, dt, dM) for nx, ny, M0
    s = [0.0, 0.0, 0.0, M0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
    thetas = [0.0] if record else None

    def f(s):
        t, M = s[2], s[3]
        c, sn = cos(t), sin(t)
        k = nx * c + ny * sn
        return [c, sn, M, nx * sn - ny * c, 0.5 * M * M,
                -sn * s[7], c * s[7], s[8], k * s[7] + sn,
                -sn * s[11], c * s[11], s[12], k * s[11] - c,
                -sn * s[15], c * s[15], s[16], k * s[15]]

    for _ in range(N):
        k1 = f(s)
        k2 = f([a + 0.5 * h * b for a, b in zip(s, k1)])
        k3 = f([a + 0.5 * h * b for a, b in zip(s, k2)])
        k4 = f([a + h * b for a, b in zip(s, k3)])
        s = [a + h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
             for a, b1, b2, b3, b4 in zip(s, k1, k2, k3, k4)]
        if record:
            thetas.append(s[2])
    return _pack(s, np.array(thetas) if record else None)


_AMP = {}


def amplitude(d):
    """|p| of the exact first clamped mode at (d, 0, 0); a(d) / sqrt(1 - d) is smooth,
    so it is interpolated from 120 exact values."""
    if "d" not in _AMP:
        ds = 1.0 - np.geomspace(0.8, 1e-6, 120)
        _AMP["d"], _AMP["a"] = ds, first_mode_amplitude(ds) / np.sqrt(1.0 - ds)
    return float(np.interp(d, _AMP["d"], _AMP["a"]) * math.sqrt(max(1.0 - d, 0.0)))


def branch_of(p, d, previous):
    """Sign of p when it is clearly on a branch; near p = 0 keep the previous label."""
    if abs(p) > 0.3 * amplitude(d) or previous is None:
        return 1 if p >= 0 else -1
    return previous


def _newton(r, load, tol=1e-13, accept=1e-11, max_it=30):
    """Newton on the base load so that the end pose is r; (load, integrate output) or None.
    Converged at |pose - r| < tol, or < accept once the residual stops falling (round-off).
    The pose residual matters: g is off by about K * residual, and K reaches ~1e6 along
    the chord of a nearly straight segment."""
    out = integrate(load)
    F = out[0] - r
    for _ in range(max_it):
        if np.abs(F).max() < tol:
            return load, out
        try:
            step = np.linalg.solve(out[3][:3], -F)
        except np.linalg.LinAlgError:
            return None
        alpha, norm = 1.0, np.abs(F).max()
        while alpha > 1e-3:                     # backtrack on the residual
            trial = load + alpha * step
            out_t = integrate(trial)
            Ft = out_t[0] - r
            if np.abs(Ft).max() < norm:
                break
            if norm < accept:                   # round-off floor: stop here
                return load, out
            alpha *= 0.5
        else:
            return None
        load, out, F = trial, out_t, Ft
    return (load, out) if np.abs(F).max() < accept else None


def _continuation(r, r0, load0, min_step=1e-3):
    """March the target from r0 (solved by load0) to r, halving steps that fail."""
    t, step, load = 0.0, 1.0, np.asarray(load0, float)
    while t < 1.0:
        t_next = min(1.0, t + step)
        res = _newton(r0 + t_next * (r - r0), load)
        if res is None:
            step *= 0.5
            if step < min_step:
                return None
            continue
        load, t, step = res[0], t_next, min(1.0, 2.0 * step)
    return load


def arc_pose(rt):
    """End pose of the unloaded segment (a circular arc) with end angle rt."""
    if abs(rt) < 1e-9:
        return np.array([1.0, 0.0, rt])
    return np.array([math.sin(rt) / rt, (1.0 - math.cos(rt)) / rt, rt])


def shoot(r, load0=None, branch=None, r0=None, stability=True):
    """SegmentState at relative pose r by shooting, warm-started from load0 (which solves
    pose r0, if given). If Newton fails it continues from r0, or from the unloaded arc.
    Returns None if r is out of reach (d >= 1) or every attempt fails."""
    r = np.asarray(r, dtype=float)
    if math.hypot(r[0], r[1]) >= 1.0:
        return None
    res = _newton(r, np.array(load0, float)) if load0 is not None else None
    if res is None and load0 is not None and r0 is not None:
        load = _continuation(r, np.asarray(r0, float), load0)
        res = None if load is None else _newton(r, load)
    if res is None:
        load = _continuation(r, arc_pose(r[2]), np.array([0.0, 0.0, r[2]]))
        res = None if load is None else _newton(r, load)
    if res is None:
        return None
    load = res[0]
    pose, M1, V, J, thetas = integrate(load, record=True)
    g = np.array([load[0], load[1], M1])
    dg = np.vstack([np.eye(3)[:2], J[3]])
    try:
        K = dg @ np.linalg.inv(J[:3])
    except np.linalg.LinAlgError:
        return None
    (d, phi1, phi2), *_ = chart(r)
    s = np.linspace(0.0, 1.0, len(thetas))
    alpha = -phi1
    t_chord = thetas - alpha
    p = float(first_mode_projection(t_chord, s))
    stable = True
    if stability:
        ca, sa = math.cos(alpha), math.sin(alpha)
        fx = -(ca * load[0] + sa * load[1])          # (fx, fy) = -R(-alpha) n
        fy = -(-sa * load[0] + ca * load[1])
        stable = unstable_directions(t_chord, s, fx, fy) == 0
    return SegmentState(V, g, 0.5 * (K + K.T), load, branch_of(p, d, branch), p, stable,
                        "shoot", (d, phi1, phi2))


def pose_of(chart_q):
    """Relative pose r from (d, phi1, phi2)."""
    d, phi1, phi2 = chart_q
    return np.array([d * math.cos(phi1), -d * math.sin(phi1), phi2 - phi1])

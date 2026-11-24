"""
run_demos.py
Runs the online controller (orientation free, dynamic mode, 100 Hz) on three demos that move the
payload from one end of the arm's workspace to the other. Arm: minimal co-contraction, no tension
limits (ChainArm(scheme="minimal")). Routes run along loops 0.3 inside the band's outer and inner
boundaries (data/workspace.npz, from scripts/workspace_map.py).

The workspace is a C, not a ring: the payload cannot go all the way round the base, since each
lap would add 2 pi to the arm's total turning (the region behind the base is reached over the top
with the arm curled one way, or underneath with it curled the other, two postures with no path
between them). So the demos run end to end:

  sweep  fixed timing: rest pose -> top-back end (+160 deg) -> round the front to the bottom-back
         end (-160 deg) -> back to the start, stopping at each reversal
  path   inner loop: rest pose -> top-back end (+150 deg), then round to the bottom-back end
         (-150 deg), speed chosen online, safe mode
  live   a scripted gamepad session: steer to the top-back end, then right round to the
         bottom-back end, with pushes into the band's edges and the triggers turning the payload
         (soft angle target, relative rotation); safe mode

Each run records, per step: reference, commanded tensions, disk poses and segment loads (the arm
shape the controller computed), wrench, feasibility, the fallback taken, posture switches and the
step time. There is no plant yet (PLAN step 7): the arm drawn is the equilibrium the controller
commands.

usage: python scripts/run_demos.py [sweep path live]      -> data/runs/demo_<name>.npz
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
from scipy.interpolate import CubicSpline                               # noqa: E402
from scipy.ndimage import distance_transform_edt                        # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm                                    # noqa: E402
from bowstring.controller import Controller, Payload                    # noqa: E402
from bowstring.reference import Live, PathFollower, Trajectory, quintic  # noqa: E402

DT = 0.01
C = np.array([3.0, 2.5, 2.0])          # unused by the minimal scheme (kept for the interface)
D0 = np.array([0.8, 0.4, 0.2])
PAYLOAD = Payload(m=0.1, I=0.1 * 0.3 ** 2)
OUT = ROOT / "data/runs"


def band_loops(inset=0.3, harmonics=20, n=1500):
    """The two closed loops `inset` inside the workspace band (outer, inner): contours of the
    distance to the band's boundary, resampled and low-pass filtered (Fourier) to be smooth.
    Both run clockwise."""
    ws = np.load(ROOT / "data/workspace.npz")
    xs = ws["xs"]
    dist = distance_transform_edt(ws["reach"]) * (xs[1] - xs[0])
    X, Y = np.meshgrid(xs, ws["ys"], indexing="ij")
    fig = plt.figure()
    segs = plt.contour(X, Y, dist, [inset]).allsegs[0]
    plt.close(fig)
    segs = sorted(segs, key=len)[-2:]
    segs = sorted(segs, key=lambda s: -np.hypot(*s.T).mean())          # outer first
    loops = []
    for s in segs:
        s = np.vstack([s, s[:1]])
        arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(s, axis=0).T))])
        u = np.linspace(0, arc[-1], n, endpoint=False)
        z = np.interp(u, arc, s[:, 0]) + 1j * np.interp(u, arc, s[:, 1])
        Z = np.fft.fft(z)
        Z[harmonics + 1:-harmonics] = 0
        z = np.fft.ifft(Z)
        loop = np.column_stack([z.real, z.imag])
        if np.sum(loop[:-1, 0] * loop[1:, 1] - loop[1:, 0] * loop[:-1, 1]) > 0:
            loop = loop[::-1]                                          # make it clockwise
        loops.append(loop)
    return loops


def loop_arc(loop, start, polar_to, direction):
    """Points of a (clockwise) loop from its point nearest `start` to polar angle polar_to,
    going clockwise (direction +1, polar angle decreasing) or counter-clockwise (-1)."""
    n = len(loop)
    i = int(np.argmin(np.hypot(*(loop - start[:2]).T)))
    pts = [loop[i]]
    th = np.unwrap(np.arctan2(loop[:, 1], loop[:, 0]))      # continuous along the loop index
    th0 = np.arctan2(loop[i, 1], loop[i, 0])
    target = polar_to
    # unwrap the target relative to the start, in the direction of travel
    if direction > 0:
        while target > th0:
            target -= 2 * np.pi
    else:
        while target < th0:
            target += 2 * np.pi
    acc = th0
    for step in range(1, n):
        j = (i + direction * step) % n
        dth = np.angle(np.exp(1j * (np.arctan2(loop[j, 1], loop[j, 0])
                                    - np.arctan2(pts[-1][1], pts[-1][0]))))
        acc += dth
        pts.append(loop[j])
        if (direction > 0 and acc <= target) or (direction < 0 and acc >= target):
            break
    return np.array(pts)


def lead_in(X0, arc, n=30):
    """Points from the rest pose X0 to the start of an arc along a cubic Hermite curve that
    meets the arc tangentially, so the route has no corner (a corner is a spike in Xdd)."""
    p0, p1 = X0[:2], arc[0]
    gap = np.hypot(*(p1 - p0))
    t1 = arc[min(5, len(arc) - 1)] - arc[0]
    t1 = gap * t1 / np.hypot(*t1)
    m0 = p1 - p0
    s = np.linspace(0, 1, n, endpoint=False)[:, None]
    return ((2 * s ** 3 - 3 * s ** 2 + 1) * p0 + (s ** 3 - 2 * s ** 2 + s) * m0
            + (-2 * s ** 3 + 3 * s ** 2) * p1 + (s ** 3 - s ** 2) * t1)


def spline_path(points):
    """Arc-length cubic spline through route points: sigma -> (X, X', X'') with angle 0."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    points = points[np.concatenate([[True], seg > 1e-9])]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    cs = CubicSpline(s, points, axis=0)

    def path(sig):
        return (np.r_[cs(sig), 0.0], np.r_[cs(sig, 1), 0.0], np.r_[cs(sig, 2), 0.0])
    return path, s[-1]


def record(rows, t, ref, info, extra=None):
    rows.append({"t": t, "X_ref": np.asarray(ref[0], float), "Xdd_ref": np.asarray(ref[2], float),
                 "T": info["T"], "D": info["D"], "L": info["L"], "P": info["P"],
                 "w": info["w_req"], "feasible": info["feasible"],
                 "choice": -1 if info["choice"] is None else info["choice"],
                 "margin": info["margin"], "kx": info["kx"], "report": info["report"] is True,
                 "time": info["time"], "switched": info["switched"], **(extra or {})})


def save(name, rows, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    np.savez(OUT / f"demo_{name}.npz", C=C, D0=D0, m=PAYLOAD.m, I=PAYLOAD.I, dt=DT,
             offsets=np.array([0.15, 0.2, 0.25]), Tmin=0.1, scheme="minimal", meta=str(meta),
             **arrays)
    t = arrays["time"]
    print(f"{name}: {len(rows)} steps, held T on {np.sum(~arrays['feasible'])}, fallback on "
          f"{np.sum(arrays['choice'] > 0)}, posture switches {np.sum(arrays['switched'])}; "
          f"step time mean {t.mean() * 1e3:.2f} ms, 99th pct {np.percentile(t, 99) * 1e3:.1f} ms; "
          f"max tension {arrays['T'].max():.2f}; {meta} -> {OUT / f'demo_{name}.npz'}", flush=True)


def new_controller():
    ctrl = Controller(ChainArm(scheme="minimal"), C, PAYLOAD, mode="dynamic", orientation="free")
    return ctrl, ctrl.reset(D0)


def demo_sweep(speed=0.45):
    """Outer loop, fixed timing: rest pose -> top-back end -> bottom-back end -> start. Each leg
    is timed by a quintic (at rest at both ends); duration = length / speed * 1.5."""
    ctrl, X0 = new_controller()
    outer = band_loops()[0]
    up = loop_arc(outer, X0, np.radians(160), -1)
    across = loop_arc(outer, up[-1], np.radians(-160), +1)
    back = loop_arc(outer, across[-1], np.arctan2(X0[1], X0[0]), -1)
    legs = [spline_path(np.vstack([lead_in(X0, up), up])), spline_path(across),
            spline_path(np.vstack([back, X0[None, :2]]))]
    rows, t0 = [], 0.0
    for path, length in legs:
        duration = 1.5 * length / speed
        for k in range(1, int(duration / DT) + 1):
            tl = k * DT
            s, ds, dds = quintic(tl / duration)
            sig, sd, sdd = length * s, length * ds / duration, length * dds / duration ** 2
            X, dX, ddX = path(sig)
            ref = (X, dX * sd, ddX * sd ** 2 + dX * sdd)
            T, info = ctrl.step(t0 + tl, *ref)
            record(rows, t0 + tl, ref, info)
        t0 += duration
    save("sweep", rows, {"legs": [round(l, 2) for _, l in legs]})


def demo_path():
    """Inner loop: rest pose -> top-back end, then round to the bottom-back end; speed chosen
    online (v_max 1.5, a_max 1), safe mode."""
    ctrl, X0 = new_controller()
    inner = band_loops()[1]
    up = loop_arc(inner, X0, np.radians(150), -1)
    across = loop_arc(inner, up[-1], np.radians(-150), +1)
    legs = [spline_path(np.vstack([lead_in(X0, up), up])), spline_path(across)]
    rows, k, done = [], 0, []
    for path, length in legs:
        pf = PathFollower(path, length, v_max=1.5, a_max=1.0)
        n0 = k
        while not pf.done and k - n0 < 6000:
            k += 1
            refs = pf.candidates(DT)
            T, info = ctrl.choose(k * DT, refs, safe=True)
            pf.accept(info["choice"])
            ref = refs[info["choice"]] if info["choice"] is not None else (
                pf.path(pf.sigma)[0], np.zeros(3), np.zeros(3))
            record(rows, k * DT, ref, info, {"speed": pf.speed})
        done.append((round(pf.sigma, 2), round(length, 2)))
    save("path", rows, {"legs (travelled, length)": done})


def demo_live(duration=45.0):
    """Scripted gamepad session: steer counter-clockwise to the top-back end of the band, then
    clockwise right round to the bottom-back end (following the band's middle), with pushes into
    the outer and inner edges on the way and the triggers turning the payload; release at the end."""
    ctrl, X0 = new_controller()
    ctrl.k_psi = 0.3                                    # triggers set a soft angle target
    live = Live(X0, v_max=0.6, w_max=0.6, a_max=0.6, alpha_max=3.0)
    outer, inner = band_loops()
    near = np.argmin(np.hypot(*(inner[None, :, :] - outer[:, None, :]).transpose(2, 0, 1)), axis=1)
    mid = 0.5 * (outer + inner[near])
    rows, heading = [], -1                              # -1: counter-clockwise first
    for k in range(1, int(duration / DT) + 1):
        t = k * DT
        X = live.X
        polar = np.degrees(np.arctan2(X[1], X[0]))
        if heading < 0 and polar > 150:
            heading = +1                                # reached the top-back end: turn round
        j = int(np.argmin(np.hypot(*(mid - X[:2]).T)))
        tangent = mid[(j + 3) % len(mid)] - mid[(j - 3) % len(mid)]
        tangent *= heading / np.hypot(*tangent)
        keep = np.clip(3.0 * (mid[j] - X[:2]), -0.4, 0.4)   # the user corrects towards the middle
        out = np.array([tangent[1], -tangent[0]]) * heading  # outwards
        if heading > 0 and polar < -150:
            stick = np.zeros(2)                         # reached the bottom-back end: release
        elif heading > 0 and 0 < polar < 40 and t % 6 < 1.5:
            stick = 0.8 * out                           # pushes into the outer edge
        elif heading > 0 and -60 < polar < -20 and t % 6 < 1.5:
            stick = -0.8 * out                          # and the inner edge
        else:
            stick = 0.9 * tangent + keep
        lt_rt = 1.0 if 4 < t < 6 else (-1.0 if 8 < t < 10 else 0.0)
        cmd = (stick[0], stick[1], lt_rt)
        refs = live.candidates(DT, cmd)
        T, info = ctrl.choose(t, refs, safe=True)
        live.accept(info["choice"])
        if lt_rt == 0.0 and info["P"] is not None:
            live.sync_angle(info["P"][-1][2])          # rotation is relative to the angle held
        record(rows, t, refs[info["choice"]] if info["choice"] is not None else refs[-1], info,
               {"cmd": np.array(cmd)})
    save("live", rows, {"final polar": round(float(np.degrees(np.arctan2(live.X[1], live.X[0]))), 1)})


def demo_circle(centre=(1.45, 1.5), radius=0.55, duration=16.0):
    """A circle of radius 0.55 about (1.45, 1.5), one lap counter-clockwise in 16 s (quintic in
    the angle), orientation free. It starts at the upright rest pose of D = (0, 0, 2.1), which
    lies just outside the circle near its top; the radius blends in from there. Open-loop
    payload stability is reported, not enforced (see Controller(payload_stability=...))."""
    D_up = np.array([0.0, 0.0, 2.1])
    ctrl = Controller(ChainArm(scheme="minimal"), C, PAYLOAD, mode="dynamic", orientation="free",
                      payload_stability="report")
    X0 = ctrl.reset(D_up)
    c = np.asarray(centre, float)
    a0 = np.arctan2(X0[1] - c[1], X0[0] - c[0])
    dr0 = np.hypot(*(X0[:2] - c)) - radius

    def radius_at(a):                    # starts at the rest pose, settles onto the circle
        e = np.exp(-((a - a0) / 0.6) ** 2)
        return radius + dr0 * e, dr0 * e * (-2 * (a - a0) / 0.36), \
            dr0 * e * ((2 * (a - a0) / 0.36) ** 2 - 2 / 0.36)

    rows = []
    for k in range(1, int(duration / DT) + 1):
        t = k * DT
        s_, ds, dds = quintic(t / duration)
        a, da, dda = a0 + 2 * np.pi * s_, 2 * np.pi * ds / duration, 2 * np.pi * dds / duration ** 2
        r, r1, r2 = radius_at(a)
        ca, sa = np.cos(a), np.sin(a)
        # X = c + r(a) (cos a, sin a); derivatives by the chain rule
        p1 = np.array([r1 * ca - r * sa, r1 * sa + r * ca])
        p2 = np.array([r2 * ca - 2 * r1 * sa - r * ca, r2 * sa + 2 * r1 * ca - r * sa])
        X = np.r_[c + r * np.array([ca, sa]), 0.0]
        ref = (X, np.r_[p1 * da, 0.0], np.r_[p2 * da ** 2 + p1 * dda, 0.0])
        T, info = ctrl.step(t, *ref)
        record(rows, t, ref, info)
    save("circle", rows, {"centre": tuple(centre), "radius": radius})


def demo_xline(y=1.8, x_right=2.1, x_left=0.5):
    """Full pose control: the payload moves parallel to the x axis at y = 1.8 with its tip
    pointing towards +y (angle pi/2) throughout: from the upright rest pose of D = (0, 0, 2.1)
    to (2.1, 1.8, pi/2), then to (0.5, 1.8) and back. Open-loop payload stability reported."""
    D_up = np.array([0.0, 0.0, 2.1])
    ctrl = Controller(ChainArm(scheme="minimal"), C, PAYLOAD, mode="dynamic", orientation="track",
                      payload_stability="report")
    X0 = ctrl.reset(D_up)
    A, B = np.array([x_right, y, np.pi / 2]), np.array([x_left, y, np.pi / 2])
    legs = [Trajectory.line(X0, A, 4.0), Trajectory.line(A, B, 10.0), Trajectory.line(B, A, 10.0)]
    rows, t0 = [], 0.0
    for leg in legs:
        for k in range(1, int(leg.duration / DT) + 1):
            ref = leg(k * DT)
            T, info = ctrl.step(t0 + k * DT, *ref)
            record(rows, t0 + k * DT, ref, info)
        t0 += leg.duration
    save("xline", rows, {"y": y, "x": (x_left, x_right)})


if __name__ == "__main__":
    which = sys.argv[1:] or ["circle", "xline"]
    start = time.time()
    for name in which:
        {"sweep": demo_sweep, "path": demo_path, "live": demo_live, "circle": demo_circle,
         "xline": demo_xline}[name]()
    print(f"{time.time() - start:.0f} s")

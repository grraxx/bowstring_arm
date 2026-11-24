"""
experiment.py
One closed-loop simulation from a flat config: the engine behind the web console
(console/server.py) and usable on its own:

    python -m bowstring.experiment <run_dir>       # reads <run_dir>/config.json

<run_dir> gets status.json (progress, updated while running), result.json (for the browser) and
result.npz. Run it with the sofa-env interpreter when the plant is a SOFA one.

Config keys (SCHEMA below has every key with its default, range and meaning):
  traj.*    the reference: a sample trajectory and its size, place, speed, pushes
  robot.*   the true robot (the plant): hole offsets, segment stiffness, payload, damping, rod mass
  model.*   what the controller believes about the robot; model.link = true copies robot.*
  ctrl.*    controller: PID gains, anti-windup, tension map, feedforward mode
  sense.*   sensing: tip IMU grade, tendon encoders, estimator
  sim.*     plant: the model's own plant, SOFA Cosserat rods or SOFA beam elements
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .chain import ChainArm
from .closed_loop import (G0, IMU, Estimator, ImplicitPlant, LinearizedClosedLoop, Plant,
                          PlantError, TendonEncoders, wrap)
from .controller import Controller, Payload
from .reference import quintic
from .tendons import tendons

ROOT = Path(__file__).resolve().parents[1]
C = np.array([3.0, 2.5, 2.0])            # unused by the "minimal" tension map, kept for the API
D_UP = np.array([0.0, 0.0, 2.1])         # tension differences of the upright rest pose
DT = 0.01                                # control period (s); plant and sensors at 1 ms
SUB = 10

IMU_GRADES = {                           # representative datasheet values
    "navigation": {"accel_noise": 10e-6 * G0, "accel_bias": 25e-6 * G0,
                   "gyro_arw": 0.002, "gyro_bias": 0.005},
    "tactical": {"accel_noise": 50e-6 * G0, "accel_bias": 300e-6 * G0,
                 "gyro_arw": 0.1, "gyro_bias": 1.0},
    "consumer MEMS": {"accel_noise": 150e-6 * G0, "accel_bias": 5e-3 * G0,
                      "gyro_arw": 0.5, "gyro_bias": 20.0},
    "ideal": {"accel_noise": 0.0, "accel_bias": 0.0, "gyro_arw": 0.0, "gyro_bias": 0.0},
}

AIMABLE = ["circle", "figure8", "rectangle", "line", "text"]     # presets that can aim at a point

PRESETS = {
    # name: (description, trajectory defaults)
    "circle": ("Circle; orientation free (least tension) or fixed",
               {"traj.x": 1.45, "traj.y": 1.5, "traj.size": 0.55, "traj.duration": 10.0,
                "traj.free": True, "traj.psi": 90.0}),
    "figure8": ("Figure-eight (lemniscate) at fixed orientation",
                {"traj.x": 1.6, "traj.y": 1.5, "traj.size": 0.45, "traj.duration": 16.0,
                 "traj.free": False, "traj.psi": 90.0}),
    "rectangle": ("Rounded rectangle at fixed orientation (the waiter)",
                  {"traj.x": 1.4, "traj.y": 1.8, "traj.size": 1.0, "traj.height": 0.6,
                   "traj.duration": 16.0, "traj.free": False, "traj.psi": 90.0}),
    "text": ("The tip writes ME246, then a squiggly underline, the payload aiming a laser at a point",
             {"traj.x": 1.2, "traj.y": 1.8, "traj.size": 1.0, "traj.duration": 45.0,
              "traj.free": False, "traj.psi": 90.0, "traj.aim": True}),
    "wrist": ("Position held, the payload turns from psi to psi2 and back",
              {"traj.x": 1.6, "traj.y": 1.6, "traj.psi": 20.0, "traj.psi2": 135.0,
               "traj.duration": 15.0, "traj.free": False}),
    "line": ("Straight line along x and back at fixed orientation",
             {"traj.x": 1.1, "traj.y": 1.8, "traj.size": 0.6, "traj.duration": 12.0,
              "traj.free": False, "traj.psi": 90.0}),
    "hold": ("Hold one pose (try it with pushes)",
             {"traj.x": 1.1, "traj.y": 1.8, "traj.psi": 90.0, "traj.duration": 8.0,
              "traj.free": False, "traj.kicks": True}),
}


SENSE_MODES = ["IMU + tendon encoders", "IMU only (model-aided)", "tendon encoders only",
               "perfect (true state)"]
# Units (non-dimensional): lengths in the segment length L, stiffness in the reference bending
# stiffness EI0 (B = EI / EI0), forces and tensions in EI0/L^2, moments in EI0/L. Time stays in
# seconds (the massless arm has no time scale of its own), so masses are in EI0 s^2/L^3 and the
# payload inertia in EI0 s^2/L.
UNITS_NOTE = ("L segment length · EI₀ reference stiffness, B = 1 · EI₀/L² force, tension "
              "· EI₀/L moment · s time, kept dimensional · EI₀s²/L³ mass · EI₀s²/L inertia "
              "· – dimensionless")


def _f(key, group, label, default, lo=None, hi=None, step=None, unit="", help="", kind="number",
       options=None, show=None, active=None):
    return {"key": key, "group": group, "label": label, "default": default, "min": lo, "max": hi,
            "step": step, "unit": unit, "help": help, "kind": kind, "options": options,
            "show": show, "active": active}


SCHEMA = [
    # ---------------------------------------------------------------- trajectory
    _f("traj.preset", "Trajectory", "Trajectory", "figure8", kind="select",
       options=list(PRESETS), help="Picking one loads its defaults below."),
    _f("traj.x", "Trajectory", "Centre x", 1.6, -1, 3.2, 0.05, "L"),
    _f("traj.y", "Trajectory", "Centre y", 1.5, -1, 3.2, 0.05, "L"),
    _f("traj.size", "Trajectory", "Size", 0.45, 0.05, 1.5, 0.05, "L",
       "Radius (circle), half-width (figure-eight), width (rectangle), length (line), "
       "width of the text (text).",
       show=["circle", "figure8", "rectangle", "line", "text"]),
    _f("traj.height", "Trajectory", "Height", 0.6, 0.05, 1.5, 0.05, "L", show=["rectangle"]),
    _f("traj.psi", "Trajectory", "Angle ψ", 90.0, -180, 180, 5, "deg",
       "Fixed payload orientation (start angle for the wrist)."),
    _f("traj.psi2", "Trajectory", "Turn to ψ₂", 135.0, -180, 180, 5, "deg", show=["wrist"]),
    _f("traj.free", "Trajectory", "Free orientation", False, kind="bool",
       help="Only x, y commanded; the controller picks the least-tension angle.",
       show=["circle", "line"]),
    _f("traj.aim", "Trajectory", "Aim at a point", False, kind="bool",
       help="The payload always points at the target below (a laser is drawn in the replay); "
            "the angle above is ignored.", show=AIMABLE),
    _f("traj.aim_x", "Trajectory", "Target x", 1.5, -1, 3.2, 0.05, "L", show=AIMABLE),
    _f("traj.aim_y", "Trajectory", "Target y", 2.5, -1, 3.2, 0.05, "L", show=AIMABLE),
    _f("traj.duration", "Trajectory", "Lap time", 16.0, 2, 60, 1, "s",
       "Duration of one lap or pass (the whole drawing for text); shorter = faster."),
    _f("traj.laps", "Trajectory", "Laps", 1, 1, 5, 1, "–", kind="int",
       show=["circle", "figure8", "rectangle", "line"]),
    _f("traj.kicks", "Trajectory", "Pushes", False, kind="bool",
       help="Three 0.05 s pushes the controller is not told about (+x, -y, a twist) at 25, 50, "
            "75 % of the run."),
    _f("traj.kick_size", "Trajectory", "Push size", 0.4, 0, 2, 0.05, "EI₀/L²",
       "Force of the x/y pushes; the twist is 1/8 of it, in EI₀/L."),
    # ---------------------------------------------------------------- robot
    _f("robot.d1", "Robot (true plant)", "Hole offset d₁", 0.15, 0.05, 0.4, 0.01, "L"),
    _f("robot.d2", "Robot (true plant)", "Hole offset d₂", 0.20, 0.05, 0.4, 0.01, "L"),
    _f("robot.d3", "Robot (true plant)", "Hole offset d₃", 0.25, 0.05, 0.4, 0.01, "L"),
    _f("robot.B1", "Robot (true plant)", "Stiffness B₁", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("robot.B2", "Robot (true plant)", "Stiffness B₂", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("robot.B3", "Robot (true plant)", "Stiffness B₃", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("robot.m", "Robot (true plant)", "Payload mass m", 0.1, 0.01, 1, 0.01, "EI₀s²/L³"),
    _f("robot.I", "Robot (true plant)", "Payload inertia I", 0.009, 0.0005, 0.1, 0.0005, "EI₀s²/L"),
    _f("robot.alpha", "Robot (true plant)", "Damping α", 0.1, 0, 5, 0.05,
       "1/s", "Payload damping c = alpha (m, m, I)."),
    _f("robot.rod_mass", "Robot (true plant)", "Rod mass", 0.015, 0, 0.2, 0.0005, "EI₀s²/L³",
       help="SOFA plants only: whole arm, spread along it. The model plant treats the arm as "
            "massless.", active={"sim.plant": ["SOFA Cosserat", "SOFA beams"]}),
    # ---------------------------------------------------------------- model
    _f("model.link", "Controller's model", "Same as robot", True, kind="bool",
       help="Untick to give the controller a wrong model of the robot."),
    _f("model.d1", "Controller's model", "Hole offset d₁", 0.15, 0.05, 0.4, 0.01, "L"),
    _f("model.d2", "Controller's model", "Hole offset d₂", 0.20, 0.05, 0.4, 0.01, "L"),
    _f("model.d3", "Controller's model", "Hole offset d₃", 0.25, 0.05, 0.4, 0.01, "L"),
    _f("model.B1", "Controller's model", "Stiffness B₁", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("model.B2", "Controller's model", "Stiffness B₂", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("model.B3", "Controller's model", "Stiffness B₃", 1.0, 0.2, 5, 0.05, "EI₀",
       "Bending stiffness of the segment, relative to the reference EI₀."),
    _f("model.m", "Controller's model", "Payload mass m", 0.1, 0.01, 1, 0.01, "EI₀s²/L³"),
    _f("model.I", "Controller's model", "Payload inertia I", 0.009, 0.0005, 0.1, 0.0005, "EI₀s²/L"),
    _f("model.alpha", "Controller's model", "Damping α", 0.1, 0, 5, 0.05, "1/s"),
    # ---------------------------------------------------------------- controller
    _f("ctrl.wn_xy", "Controller", "ωn x, y", 5.0, 0.5, 20, 0.5, "rad/s",
       "Feedback bandwidth for x and y: Kp = ωn², Kd = 2ζωn."),
    _f("ctrl.wn_psi", "Controller", "ωn ψ", 15.0, 1, 40, 1, "rad/s"),
    _f("ctrl.zeta", "Controller", "Damping ratio ζ", 1.0, 0.2, 3, 0.05, "–"),
    _f("ctrl.wi_ratio", "Controller", "Integral ωi / ωn", 0.2, 0, 1, 0.05, "–",
       "Ki = Kp (ratio x wn); 0 = PD only."),
    _f("ctrl.i_max", "Controller", "Integral limit", 0.5, 0, 3, 0.05, "EI₀/L²",
       "Anti-windup clamp on the integral wrench, per axis (EI₀/L for ψ)."),
    _f("ctrl.dD_max", "Controller", "Max ΔD (feedback)", 2.0, 0.1, 10, 0.1, "EI₀/L²",
       "Largest feedback change of a tension difference."),
    _f("ctrl.tikhonov", "Controller", "Tikhonov damping", 1e-6, 1e-9, 1e-2, None, "–",
       "Relative damping of the wrench-to-tension solve (log scale).", kind="log"),
    _f("ctrl.Tmin", "Controller", "Slack tension Tmin", 0.1, 0.01, 1, 0.01, "EI₀/L²"),
    _f("ctrl.feedforward", "Controller", "Feedforward", "dynamic", kind="select",
       options=["dynamic", "quasi-static"], help="dynamic: w* = M Xdd*; quasi-static: w* = 0."),
    # ---------------------------------------------------------------- sensing
    _f("sense.mode", "Sensing", "Estimate from", "IMU + tendon encoders", kind="select",
       options=SENSE_MODES,
       help="tendon encoders only: the pose solved from the six tendon lengths (assuming the "
            "inner disks at equilibrium), velocity from an alpha-beta filter on it; no IMU."),
    _f("sense.imu", "Sensing", "IMU grade", "navigation", kind="select", options=list(IMU_GRADES),
       show={"sense.mode": SENSE_MODES[:2]}),
    _f("sense.scale", "Sensing", "Segment length", 0.2, 0.02, 2, 0.01, "m",
       "Physical L: converts the IMU datasheet values into the arm's units.",
       show={"sense.mode": SENSE_MODES[:2]}),
    _f("sense.enc_noise", "Sensing", "Encoder noise", 1e-5, 1e-7, 1e-2, None, "L",
       "Tendon-length measurement noise (log scale).", kind="log",
       show={"sense.mode": [SENSE_MODES[0], SENSE_MODES[2]]}),
    _f("sense.crossover", "Sensing", "Crossover ω", 3.0, 0.2, 30, 0.1, "rad/s",
       "Below it the encoders dominate the estimate, above it the IMU.",
       show={"sense.mode": [SENSE_MODES[0]]}),
    _f("sense.enc_bw", "Sensing", "Filter bandwidth ω", 60.0, 5, 80, 1, "rad/s",
       "Encoders only: bandwidth of the alpha-beta filter that gives the pose and velocity. "
       "Keep it well above the angle loop's ωn (below ~40 the lag destabilises it); above ~80 "
       "the filter itself goes unstable at the 100 Hz control rate. On SOFA with rod mass the "
       "disks' vibration is read as tip motion: try rod mass 0.0005, or ~20 with ωn 3 / 5.",
       show={"sense.mode": [SENSE_MODES[2]]}),
    # ---------------------------------------------------------------- live input
    _f("live.v_max", "Live input", "Top speed", 0.3, 0.02, 1.5, 0.02, "L/s",
       help="Live tab: payload speed with a movement key held or the stick fully over."),
    _f("live.w_max", "Live input", "Top turn rate", 0.6, 0.05, 3, 0.05, "rad/s",
       help="Live tab: payload turn rate with Q / E or a full trigger (fixed orientation only)."),
    _f("live.a_max", "Live input", "Acceleration", 0.8, 0.05, 5, 0.05, "L/s²",
       help="Live tab: how fast the payload's velocity follows the keys."),
    # ---------------------------------------------------------------- simulator
    _f("sim.plant", "Simulator", "Plant", "model", kind="select",
       options=["model", "SOFA Cosserat", "SOFA beams"],
       help="model: the controller's own equations, massless arm (fast). SOFA: independent "
            "simulator with rod mass (slower)."),
    _f("sim.integrator", "Simulator", "Model plant stepping", "implicit BDF-α", kind="select",
       options=["implicit BDF-α", "explicit RK4"],
       help="Model plant only. Implicit (Till et al. 2019): one shooting solve per 1 ms step, "
            "the payload's equation of motion as the tip boundary condition, real time. "
            "Explicit RK4: five solves per step, the earlier reference.",
       active={"sim.plant": ["model"]}),
    _f("sim.sections", "Simulator", "SOFA sections", 20, 2, 40, 1, "/seg.", kind="int",
       help="SOFA plants only. Cosserat: 10 is ~1.3 ms per step, 20 ~4 ms (more accurate). "
            "Beams use 8x this.", active={"sim.plant": ["SOFA Cosserat", "SOFA beams"]}),
]
DEFAULTS = {f["key"]: f["default"] for f in SCHEMA}


# ------------------------------------------------------------------ references
def seq(*legs):
    """Legs (duration, f) with f(s) -> (X, dX/ds, d2X/ds2), s in [0, 1], quintic timing."""
    starts = np.cumsum([0.0] + [d for d, _ in legs])

    def ref(t):
        i = int(np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(legs) - 1))
        d, f = legs[i]
        s, sd, sdd = quintic(np.clip((t - starts[i]) / d, 0, 1))
        X, Xs, Xss = f(s)
        return X, Xs * sd / d, Xss * (sd / d) ** 2 + Xs * sdd / d ** 2
    return ref, float(starts[-1])


def _const(X):
    X = np.asarray(X, float)
    return lambda s: (X.copy(), np.zeros(3), np.zeros(3))


def _line(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return lambda s: (A + s * (B - A), B - A, np.zeros(3))


def _closed(point, psi, laps):
    """A closed curve point(u) -> (x, y), u in [0, 1), traversed `laps` times at fixed psi, with
    derivatives by central differences."""
    def f(s, h=1e-4):
        u = laps * s
        p0, pp, pm = (np.asarray(point(v)) for v in (u, u + h, u - h))
        d1 = (pp - pm) / (2 * h) * laps
        d2 = (pp - 2 * p0 + pm) / h ** 2 * laps ** 2
        return np.r_[p0, psi], np.r_[d1, 0.0], np.r_[d2, 0.0]
    return f


def _rect_point(cx, cy, w, h, r):
    straight = [w - 2 * r, h - 2 * r, w - 2 * r, h - 2 * r]
    arc = 0.5 * np.pi * r
    total = sum(straight) + 4 * arc
    corners = [(cx + w / 2 - r, cy - h / 2 + r), (cx + w / 2 - r, cy + h / 2 - r),
               (cx - w / 2 + r, cy + h / 2 - r), (cx - w / 2 + r, cy - h / 2 + r)]
    starts = [(cx - w / 2 + r, cy - h / 2), (cx + w / 2, cy - h / 2 + r),
              (cx + w / 2 - r, cy + h / 2), (cx - w / 2, cy + h / 2 - r)]
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def point(u):
        L = (u % 1.0) * total
        for k in range(4):
            if L <= straight[k]:
                return (starts[k][0] + dirs[k][0] * L, starts[k][1] + dirs[k][1] * L)
            L -= straight[k]
            if L <= arc:
                a = -np.pi / 2 + k * np.pi / 2 + L / r
                return (corners[k][0] + r * np.cos(a), corners[k][1] + r * np.sin(a))
            L -= arc
        return starts[0]
    return point


# Handwriting-style glyphs in units of the letter height: (width, strokes), a stroke being a
# smooth spline ("s") through its points or a straight line ("l"). A letter is entered on its left
# edge and left on its right, so the pen-up hops between letters stay inside the gaps.
_GLYPHS = {
    "M": (1.0, [("s", [(x * .82, y) for x, y in [
        (0, 0), (.02, .6), (.14, .97), (.32, .6), (.5, .12), (.68, .6), (.86, .97), (.97, .6),
        (.99, .2), (1.05, .02), (1.14, .12), (1.2, .32)]])]),
    "E": (.5, [("s", [(.05, .5), (.05, .25), (.18, .06), (.33, 0), (.5, .08)]),
               ("s", [(.5, .08), (.33, 0), (.18, .06), (.05, .25), (.05, .5)]),
               ("l", [(.05, .5), (.36, .5)]), ("l", [(.36, .5), (.05, .5)]),
               ("s", [(.05, .5), (.06, .78), (.2, .96), (.35, 1), (.5, .92)])]),
    "2": (.55, [("s", [(.03, .78), (.12, .94), (.28, 1), (.45, .93), (.5, .75), (.4, .55),
                       (.2, .32), (.02, .04)]),
                ("l", [(.02, 0), (.55, 0)])]),
    "4": (.5, [("l", [(.34, 0), (.34, 1)]), ("l", [(.34, 1), (0, .3)]), ("l", [(0, .3), (.5, .3)])]),
    "6": (.55, [("s", [(.05, .32), (.12, .1), (.28, .01), (.45, .07), (.52, .26), (.44, .45),
                       (.27, .52), (.11, .45), (.05, .31), (.03, .55), (.1, .8), (.28, .96),
                       (.5, .99)])]),
}


def _spline(pts, psi):
    """Leg along a smooth curve through pts, parametrised by chord length: -> (length, f)."""
    P = np.asarray(pts, float)
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    cs = CubicSpline(d / d[-1], P)
    return float(d[-1]), lambda s: (np.r_[cs(s), psi], np.r_[cs(s, 1), 0.0], np.r_[cs(s, 2), 0.0])


def _text_legs(x, y, width, psi, dur, word="ME246", slant=0.2, gap=0.07, hold=1.0, min_leg=0.5):
    """Legs that write `word` in a slanted hand, then a wavy underline drawn right to left, centred
    on (x, y) with the whole drawing `width` wide. Each stroke is a rest-to-rest leg, so corners and
    retraced strokes need no smoothing, and the pen-up hops between strokes are straight legs. The
    time of each leg follows its length, with a floor, so the total is about `dur`."""
    H = width / (sum(_GLYPHS[c][0] for c in word) + gap * (len(word) - 1) + slant)   # letter height
    left, base = x - width / 2, y - 0.25 * H            # underline sits 0.4 H below the baseline
    legs, last, x0 = [], None, 0.0

    def go(to):                                          # hop from the last point to `to`
        nonlocal last
        if last is not None and (to[0] != last[0] or to[1] != last[1]):
            a, b = np.r_[last, psi], np.r_[to, psi]
            legs.append((float(np.hypot(*(b - a)[:2])), _line(a, b)))
        last = to

    for c in word:
        w, strokes = _GLYPHS[c]
        for kind, pts in strokes:
            P = [(left + (x0 + u + slant * v) * H, base + v * H) for u, v in pts]
            go(P[0])
            if kind == "l":
                a, b = np.r_[P[0], psi], np.r_[P[1], psi]
                legs.append((float(np.hypot(*(b - a)[:2])), _line(a, b)))
            else:
                legs.append(_spline(P, psi))
            last = P[-1]
        x0 += w + gap
    yu, amp = base - 0.4 * H, 0.1 * H
    squiggle = lambda u: (left + width * (1 - u),                          # noqa: E731
                          yu + amp * math.sin(2 * np.pi * 5 * u) * (0.4 + 0.6 * math.sin(np.pi * u)))
    (ex, ey), (sx, sy) = last, squiggle(0.0)               # swash from the last tail into the underline
    legs.append(_spline([(ex, ey), (ex + 0.1 * H, ey - 0.25 * H), (sx + 0.2 * H, 0.5 * (ey + sy)),
                         (sx + 0.15 * H, sy + 0.25 * H), (sx, sy)], psi))
    sq = np.array([squiggle(u) for u in np.linspace(0, 1, 401)])
    legs.append((float(np.hypot(*np.diff(sq, axis=0).T).sum()), _closed(squiggle, psi, 1)))
    total = sum(L for L, _ in legs)
    start = legs[0][1](0.0)[0]
    end = np.r_[squiggle(1.0), psi]
    return [(hold, _const(start))] + [(max(dur * L / total, min_leg), f) for L, f in legs] \
        + [(hold, _const(end))]


def _aimed(ref, target):
    """ref with the payload angle turned to point at `target` from the tip, psi = atan2(ty - y,
    tx - x), and its first two time derivatives from the tip's velocity and acceleration."""
    tx, ty = target

    def f(t):
        X, V, A = ref(t)
        dx, dy = tx - X[0], ty - X[1]
        r2 = dx * dx + dy * dy
        N = dy * V[0] - dx * V[1]
        return (np.r_[X[:2], math.atan2(dy, dx)], np.r_[V[:2], N / r2],
                np.r_[A[:2], (dy * A[0] - dx * A[1]) / r2 + 2 * N * (dx * V[0] + dy * V[1]) / r2 ** 2])
    return f


def build_reference(P):
    """-> ref(t), duration, start pose, orientation free?, pushes w_ext(t)."""
    name, x, y = P["traj.preset"], P["traj.x"], P["traj.y"]
    psi, dur, laps = math.radians(P["traj.psi"]), P["traj.duration"], int(P["traj.laps"])
    free = bool(P["traj.free"]) and name in ("circle", "line")
    size = P["traj.size"]
    if name == "circle":
        point = lambda u: (x + size * math.cos(np.pi / 2 + 2 * np.pi * u),        # noqa: E731
                           y + size * math.sin(np.pi / 2 + 2 * np.pi * u))
        ref, T = seq((dur * laps, _closed(point, psi, laps)))
    elif name == "figure8":
        point = lambda u: (x + size * math.sin(2 * np.pi * u),                     # noqa: E731
                           y + 0.5 * size * math.sin(4 * np.pi * u))
        ref, T = seq((dur * laps, _closed(point, psi, laps)))
    elif name == "rectangle":
        r = min(0.12, 0.45 * min(size, P["traj.height"]))
        ref, T = seq((dur * laps, _closed(_rect_point(x, y, size, P["traj.height"], r), psi,
                                          laps)))
    elif name == "text":
        ref, T = seq(*_text_legs(x, y, size, psi, dur))
    elif name == "wrist":
        a, b = np.r_[x, y, psi], np.r_[x, y, math.radians(P["traj.psi2"])]
        u = dur / 15.0
        ref, T = seq((1.0 * u, _const(a)), (6.0 * u, _line(a, b)), (1.0 * u, _const(b)),
                     (6.0 * u, _line(b, a)), (1.0 * u, _const(a)))
    elif name == "line":
        A, B = np.r_[x, y, psi], np.r_[x + size, y, psi]
        legs = []
        for _ in range(laps):
            legs += [(dur / 2, _line(A, B)), (dur / 2, _line(B, A))]
        ref, T = seq((1.0, _const(A)), *legs, (1.0, _const(A)))
    elif name == "hold":
        ref, T = seq((dur, _const(np.r_[x, y, psi])))
    else:
        raise ValueError(f"unknown trajectory {name}")
    if P["traj.aim"] and not free and name in AIMABLE:
        ref = _aimed(ref, (P["traj.aim_x"], P["traj.aim_y"]))
    X_start = ref(0.0)[0]
    w_ext = None
    if P["traj.kicks"]:
        k = P["traj.kick_size"]
        kicks = [(0.25 * T, np.r_[k, 0, 0]), (0.5 * T, np.r_[0, -k, 0]),
                 (0.75 * T, np.r_[0, 0, k / 8])]

        def w_ext(t):
            for t0, w in kicks:
                if t0 <= t < t0 + 0.05:
                    return w
            return np.zeros(3)
    return ref, T, X_start, free, w_ext


# ------------------------------------------------------------------ setup
def resolve(cfg):
    P = dict(DEFAULTS)
    if cfg.get("traj.preset") and cfg["traj.preset"] in PRESETS:
        P.update(PRESETS[cfg["traj.preset"]][1])
    P.update(cfg)
    if P["model.link"]:
        for k in ("d1", "d2", "d3", "B1", "B2", "B3", "m", "I", "alpha"):
            P[f"model.{k}"] = P[f"robot.{k}"]
    return P


def make_arm(P, who):
    return ChainArm(scheme="minimal", Tmin=P["ctrl.Tmin"],
                    offsets=(P[f"{who}.d1"], P[f"{who}.d2"], P[f"{who}.d3"]),
                    stiffness=(P[f"{who}.B1"], P[f"{who}.B2"], P[f"{who}.B3"]))


def start_state(arm, X_start, free):
    """Loads, tension differences and pose holding the payload at rest at X_start, reached from
    the upright rest pose by quasi-static continuation (the controller's model)."""
    r = arm.rest(arm.tension_map(C, D_UP)[0])
    if not r["converged"]:
        raise RuntimeError("the model has no upright rest pose for these parameters")
    X0, L, D = r["P"][-1], r["L"], D_UP.copy()
    target = np.array(X_start, float)
    for f in np.linspace(0, 1, 100)[1:]:
        if free:
            o = arm.inverse_free(X0[:2] + f * (target[:2] - X0[:2]), np.zeros(3), C, L, D,
                                 max_it=15, report=False, restore_its=20, alpha_min=1e-3)
        else:
            Xf = X0 + f * np.r_[target[:2] - X0[:2], wrap(target[2] - X0[2])]
            o = arm.inverse(Xf, np.zeros(3), C, L, D, report=False, max_it=40)
        if not o["converged"]:
            raise RuntimeError(f"start pose {np.round(target, 3).tolist()} not reachable at rest "
                               f"(stopped at {100 * f:.0f} % of the way from the upright pose)")
        L, D = o["L"], o["D"]
    return o["P"][-1].copy(), np.ravel(L), D


class ModelPlant:
    """The model's own plant (closed_loop.Plant): payload dynamics, quasi-static arm."""

    def __init__(self, arm, payload, X0, L0, alpha, dt=1e-3, integrator="implicit BDF-α"):
        M = np.array([payload.m, payload.m, payload.I])
        self.push = np.zeros(3)
        cls = Plant if integrator == "explicit RK4" else ImplicitPlant
        self.p = cls(arm, payload, X0, L0, damping=alpha * M, w_ext=lambda t: self.push)
        self.arm, self.dt, self.T = arm, dt, None

    def set_tensions(self, T):
        self.T = np.asarray(T, float)

    def set_push(self, w):
        self.push = np.asarray(w, float)

    def step(self):
        return self.p.step(self.T, self.dt)

    def tip(self):
        return self.p.X.copy(), self.p.V.copy()

    def tendon_lengths(self):
        return tendons(self.arm.evaluate(self.p.L, self.T)["P"], self.arm.offsets)[0]

    def drawing(self):
        segs = self.arm.shape(self.p.L)
        line = [segs[0][0]] + [s[np.linspace(0, len(s) - 1, 21).astype(int)[1:]] for s in segs]
        return np.vstack([np.atleast_2d(line[0])] + line[1:]), \
            self.arm.evaluate(self.p.L, self.T)["P"]


class SofaPlant:
    def __init__(self, kind, P, arm, payload, L0, T0, sections, X0=None, settle=1.5):
        sys.path.insert(0, str(ROOT))
        from sofa.scene import SofaArm, model_nodes
        B = (P["robot.B1"], P["robot.B2"], P["robot.B3"])
        if kind == "SOFA Cosserat":
            from sofa.cosserat_scene import SofaCosseratArm
            self.s = SofaCosseratArm(model_nodes(L0, 3, sections, B), T0, payload,
                                     offsets=tuple(arm.offsets), stiffness=B, NS=sections,
                                     rod_mass=P["robot.rod_mass"], alpha=P["robot.alpha"])
        else:
            if any(abs(b - 1.0) > 1e-12 for b in B):
                raise ValueError("SOFA beams support bending stiffness 1 only; use SOFA Cosserat")
            ne = 8 * sections
            self.s = SofaArm(model_nodes(L0, 3, ne, B), T0, payload, offsets=tuple(arm.offsets),
                             NE=ne, rod_mass=P["robot.rod_mass"], alpha=P["robot.alpha"])
        self.n_draw = None
        if X0 is not None and settle > 0:
            self.settle(np.asarray(X0, float), payload, settle, P["robot.alpha"])

    def settle(self, X0, payload, seconds, alpha):
        """Start at rest: SOFA starts on the model's shape, which is not quite SOFA's own
        equilibrium. Hold the payload at X0 with a stiff spring (as a fixture would), damp
        heavily and let the arm settle under the start tensions; then release."""
        M = np.array([payload.m, payload.m, payload.I])
        k = M * 400.0                                       # 20 rad/s
        c = 2 * np.sqrt(k * M)
        self.s.set_damping(5.0)
        for _ in range(int(seconds / self.s.dt)):
            X, V = self.s.tip()
            e = X - X0
            e[2] = (e[2] + math.pi) % (2 * math.pi) - math.pi
            self.s.set_push(-k * e - c * V)
            self.s.step()
        self.s.set_push(np.zeros(3))
        self.s.set_damping(alpha)
        if hasattr(self.s, "X_prev"):                       # Cosserat: positions-based kinematics
            self.s.X_prev, self.s.V = self.s.poses()[-1].copy(), np.zeros(3)
        else:
            self.s.v_prev = self.s.tip()[1].copy()
        self.s.t = 0.0

    def set_tensions(self, T):
        self.s.set_tensions(T)

    def set_push(self, w):
        self.s.set_push(w)

    def step(self):
        a = self.s.step()
        if not np.all(np.isfinite(a)):
            raise PlantError("SOFA diverged")
        return a

    def tip(self):
        return self.s.tip()

    def tendon_lengths(self):
        return self.s.tendon_lengths()

    def drawing(self):
        nodes = self.s.nodes()
        idx = np.unique(np.linspace(0, len(nodes) - 1, 61).round().astype(int))
        return nodes[idx, :2], self.s.poses()


# ------------------------------------------------------------------ run
class Session:
    """One closed loop built from a resolved config: plant, sensors, estimator, the online
    feedforward and the feedback. Stepped one control period at a time, so run() can drive it
    along a preset and the console's live mode from the keyboard."""

    def __init__(self, P, X_start, free, report=lambda **kw: None):
        self.P, self.free = P, free
        report(state="starting", message="finding the start pose")
        self.true_arm, self.model_arm = make_arm(P, "robot"), make_arm(P, "model")
        pay_true = Payload(m=P["robot.m"], I=P["robot.I"])
        self.pay_model = Payload(m=P["model.m"], I=P["model.I"])
        self.Mm = np.array([self.pay_model.m, self.pay_model.m, self.pay_model.I])
        X0, L0, D0 = start_state(self.model_arm, X_start, free)
        T0 = self.model_arm.tension_map(C, D0)[0]
        fw = self.true_arm.forward(X0, T0, L0, report=False)
        if not fw["converged"]:
            raise RuntimeError("the true robot has no arm equilibrium at the start pose")
        L0_true = np.ravel(fw["L"])

        report(message="building the plant")
        if P["sim.plant"] == "model":
            self.plant = ModelPlant(self.true_arm, pay_true, X0, L0_true, P["robot.alpha"],
                                    integrator=P["sim.integrator"])
        else:
            report(message="settling the SOFA arm at the start pose")
            self.plant = SofaPlant(P["sim.plant"], P, self.true_arm, pay_true, L0_true, T0,
                                   int(P["sim.sections"]), X0=X0)
        self.plant.set_tensions(T0)

        self.mode = P["sense.mode"]
        if self.mode not in SENSE_MODES:
            raise ValueError(f"unknown sensing mode {self.mode!r}")
        self.use_imu = self.mode.startswith("IMU")
        self.imu = IMU.from_spec(IMU_GRADES[P["sense.imu"]], length_scale=P["sense.scale"])
        self.enc = TendonEncoders(noise=P["sense.enc_noise"])
        self.est = Estimator(X0)
        self.ff = Controller(make_arm(P, "model"), C, self.pay_model,
                             mode="dynamic" if P["ctrl.feedforward"] == "dynamic" else "static",
                             orientation="free" if free else "track", payload_stability="report")
        self.ff.reset(D0, L0)
        wn = np.array([P["ctrl.wn_xy"], P["ctrl.wn_xy"], P["ctrl.wn_psi"]])
        self.fb = LinearizedClosedLoop(self.model_arm, C, self.pay_model, wn=wn,
                                       zeta=P["ctrl.zeta"], damping=P["ctrl.tikhonov"],
                                       dD_max=P["ctrl.dD_max"], wi=P["ctrl.wi_ratio"] * wn,
                                       i_max=P["ctrl.i_max"], dt=DT)
        self.X0 = X0
        self.L_est, self.Xh, self.Vh = L0.copy(), X0.copy(), np.zeros(3)
        self.psi_prev, self.psid_prev = X0[2], 0.0
        self.k = 0

    def advance(self, Xr, Xdr, Xddr, info_ff, w_ext=None):
        """The rest of control step k + 1 after the feedforward (info_ff, from self.ff.step or
        self.ff.choose at this step's reference): feedback, the plant over one control period
        (SUB sensor samples), the estimator. Xr, Xdr, Xddr are changed in place (free
        orientation: the angle the feedforward chose). Returns the step's row."""
        P, plant, est, model_arm = self.P, self.plant, self.est, self.model_arm
        self.k += 1
        k, t = self.k, self.k * DT
        if self.free:
            psi = info_ff["P"][-1][2]
            Xr[2], Xdr[2] = psi, wrap(psi - self.psi_prev) / DT
            Xddr[2] = (Xdr[2] - self.psid_prev) / DT if k > 1 else 0.0
            self.psi_prev, self.psid_prev = psi, Xdr[2]
            w_ff = np.r_[self.pay_model.m * Xddr[:2], 0.0]
        else:
            w_ff = self.Mm * Xddr if P["ctrl.feedforward"] == "dynamic" else np.zeros(3)
        T, info = self.fb.step((Xr, Xdr, Xddr), (self.Xh, self.Vh), self.ff.history[-1][0],
                               self.ff.history[-1][1], w_ff)
        plant.set_tensions(T)
        for j in range(SUB):
            tj = (k - 1) * DT + j * DT / SUB
            plant.set_push(np.zeros(3) if w_ext is None else w_ext(tj))
            acc = plant.step()
            X, V = plant.tip()
            if self.use_imu:
                f_, g_ = self.imu.read(X[2], V[2], acc[:2])
                est.update(f_, g_, DT / SUB)
        mode = self.mode
        if mode == "tendon encoders only":          # alpha-beta: predict at constant velocity,
            est.X += est.V * DT                     # correct from the tendon-length pose fix
            self.L_est = est.length_correct(model_arm, T, self.L_est,
                                            self.enc.read(plant.tendon_lengths()), DT,
                                            wn=P["sense.enc_bw"])
        elif mode.startswith("IMU + tendon"):
            self.L_est = est.length_correct(model_arm, T, self.L_est,
                                            self.enc.read(plant.tendon_lengths()), DT,
                                            wn=P["sense.crossover"])
        elif mode.startswith("IMU only"):
            self.L_est = est.model_correct(model_arm, T, self.L_est, self.Mm,
                                           P["model.alpha"] * self.Mm)
        else:
            est.X[:], est.V[:] = X, V
        self.Xh, self.Vh = est.X.copy(), est.V.copy()
        if not np.all(np.isfinite(X)) or np.hypot(*(X[:2] - Xr[:2])) > 1.5:
            raise PlantError(f"tracking lost (error above 1.5) at t = {t:.2f} s")
        Xh = self.Xh
        return {"t": t, "X_ref": Xr, "X": X.copy(), "X_est": Xh.copy(), "T": T,
                "w_i": info["w_i"], "ok": bool(info_ff["feasible"]),
                "ctrl_ms": 1e3 * (info["time"] + info_ff["time"]),
                "err": float(np.hypot(*(X[:2] - Xr[:2]))),
                "err_psi": float(wrap(X[2] - Xr[2])),
                "est_err": float(np.hypot(*(X[:2] - Xh[:2]))),
                "w_ext": np.zeros(3) if w_ext is None else np.asarray(w_ext(t), float)}

    def frame(self, k_row):
        """The arm drawing for row index k_row, or None if it is not finite."""
        line, disks = self.plant.drawing()
        if not (np.all(np.isfinite(line)) and np.all(np.isfinite(disks))):
            return None
        return {"k": k_row, "line": np.round(line[:, :2], 4).tolist(),
                "disks": np.round(disks, 4).tolist()}


def run(cfg, out_dir=None, progress=None):
    """One simulation. Returns the result dict (also written to out_dir if given)."""
    wall = time.time()
    P = resolve(cfg)
    status = {"state": "starting", "progress": 0.0, "message": "", "t": 0.0}

    def report(**kw):
        status.update(kw)
        if out_dir:
            tmp = Path(out_dir) / "status.json.tmp"
            tmp.write_text(json.dumps(status))
            tmp.replace(Path(out_dir) / "status.json")
        if progress:
            progress(status)

    ref, duration, X_start, free, w_ext = build_reference(P)
    s = Session(P, X_start, free, report)
    rows, frames = [], []
    msg, state = "completed", "done"
    n_steps = int(round(duration / DT))
    ref_path = np.round([ref(t)[0][:2] for t in np.linspace(0.0, duration, int(duration * 25) + 1)],
                        4).tolist()
    live = Live(out_dir, {"config": P, "offsets": list(s.true_arm.offsets), "dt": DT,
                          "duration": duration, "ref_path": ref_path})
    report(state="running", message="")
    try:
        for k in range(1, n_steps + 1):
            t = k * DT
            Xr, Xdr, Xddr = (np.asarray(v, float).copy() for v in ref(t))
            _, info_ff = s.ff.step(t, Xr, Xdr, Xddr)
            rows.append(s.advance(Xr, Xdr, Xddr, info_ff, w_ext))
            if k % 4 == 1 or k == n_steps:                       # 25 frames per second
                f = s.frame(len(rows) - 1)
                if f:
                    frames.append(f)
            if k % 25 == 0:
                live.push(rows, frames)
                report(progress=k / n_steps, t=t, wall_s=time.time() - wall)
    except (PlantError, RuntimeError, ValueError, np.linalg.LinAlgError) as err:
        msg, state = f"stopped at t = {len(rows) * DT:.2f} s: {err}", "failed"
        try:                                    # the arm as it was when the run stopped
            f = s.frame(len(rows) - 1) if rows else None
            if f:
                frames.append(f)
        except Exception:                                          # noqa: BLE001
            pass
    live.push(rows, frames, final=True)
    if not rows:
        raise RuntimeError(msg)
    A = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    e, ep, ee = A["err"], np.abs(A["err_psi"]), A["est_err"]
    summary = {"status": msg, "state": state, "duration": duration, "simulated": float(A["t"][-1]),
               "err_mean": float(e.mean()), "err_max": float(e.max()),
               "err_rms": float(np.sqrt((e ** 2).mean())),
               "psi_err_max_deg": float(np.degrees(ep.max())),
               "est_err_max": float(ee.max()), "T_max": float(A["T"].max()),
               "ctrl_ms_mean": float(A["ctrl_ms"].mean()),
               "ctrl_ms_p99": float(np.percentile(A["ctrl_ms"], 99)),
               "infeasible_steps": int((~A["ok"]).sum()), "wall_s": time.time() - wall}
    result = {"config": P, "summary": summary,
              "offsets": list(s.true_arm.offsets), "dt": DT, "ref_path": ref_path,
              "series": {k: np.round(v, 6).tolist() for k, v in A.items() if k != "ok"},
              "frames": frames}
    if out_dir:
        out = Path(out_dir)
        np.savez(out / "result.npz", **A)
        (out / "result.json").write_text(json.dumps(result))
    report(state=state, progress=1.0, message=msg, summary=summary)
    return result


class Live:
    """Appends the run as it goes to <out_dir>/live.jsonl for the console's live view: a head
    line (config, offsets, dt, duration, the whole reference path), then one line per chunk with
    the new rows (as series) and the new arm frames."""

    def __init__(self, out_dir, head):
        self.f = open(Path(out_dir) / "live.jsonl", "w") if out_dir else None
        self.nr = self.nf = 0
        self._write({"head": head})

    def _write(self, obj):
        if self.f:
            self.f.write(json.dumps(obj) + "\n")
            self.f.flush()

    def push(self, rows, frames, final=False):
        new = rows[self.nr:]
        if new or final:
            series = {k: np.round(np.array([r[k] for r in new]), 6).tolist()
                      for k in (new[0] if new else {}) if k != "ok"}
            self._write({"rows": series, "frames": frames[self.nf:], "final": final})
            self.nr, self.nf = len(rows), len(frames)
        if final and self.f:
            self.f.close()
            self.f = None


def main():
    out = Path(sys.argv[1])
    cfg = json.loads((out / "config.json").read_text())
    try:
        run(cfg, out)
    except Exception as err:                                       # noqa: BLE001
        (out / "status.json").write_text(json.dumps(
            {"state": "failed", "progress": 1.0, "message": f"{type(err).__name__}: {err}",
             "trace": traceback.format_exc()}))
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

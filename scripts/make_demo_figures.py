"""
make_demo_figures.py
Figures and animations of the circle and x-line demos (scripts/run_demos.py circle xline).

  figures/demo_<name>.png   the commanded arm through the run (with the payload mass), the
                            reference coloured where the payload is open-loop unstable, the
                            tensions, the payload stiffness and the controller step time
  figures/demo_<name>.mp4   the arm with its tendons (line width = tension) and payload, the
                            reference, and the six tensions as bars

The arm drawn is the equilibrium the controller commands at each step (no plant yet). Where the
payload stiffness K_X has a negative eigenvalue, that equilibrium is unstable with the tensions
held: in open loop the payload would drift away from it.

usage: python scripts/make_demo_figures.py [circle xline] [--no-video]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.collections import LineCollection             # noqa: E402
from matplotlib.lines import Line2D                           # noqa: E402
from matplotlib.patches import Polygon                        # noqa: E402
import numpy as np                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm      # noqa: E402
from bowstring.tendons import hole        # noqa: E402

FIG, RUNS = ROOT / "figures", ROOT / "data/runs"
# Reference palette (dataviz skill): categorical slots 1-3 for the tendon pairs, text and surface
# tokens, the critical status colour for open-loop instability.
PAIR = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
CRITICAL = "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})
ARM = ChainArm()
TITLES = {"circle": "Circle: radius 0.55 about (1.45, 1.5), orientation free",
          "xline": "Line parallel to x at y = 1.8, payload pointing towards +y (x, y, θ control)"}
SHORT = {"circle": "Circle about (1.45, 1.5), radius 0.55", "xline": "Line at y = 1.8, payload pointing +y"}
LIM = dict(x=(-0.7, 3.2), y=(-0.5, 3.0))
PAYLOAD_SIZE = 0.24


def base(ax):
    ax.add_patch(Polygon([[-0.12, -0.3], [0, -0.3], [0, 0.3], [-0.12, 0.3]], closed=True,
                         facecolor=GRID, edgecolor=INK2, hatch="////", lw=0.8, zorder=2))
    ax.set_aspect("equal")
    ax.set_xlim(*LIM["x"])
    ax.set_ylim(*LIM["y"])


def draw_arm(ax, L, P, T=None, color=INK, lw=2.0, alpha=1.0, payload=True, zorder=4):
    artists = []
    for seg in ARM.shape(L):
        artists += ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, alpha=alpha, zorder=zorder)
    if T is not None:
        for k in range(3):
            for side, s in enumerate((1.0, -1.0)):
                pts = np.array([hole(P[j], ARM.offsets[k], s)[0] for j in range(k + 2)])
                artists += ax.plot(pts[:, 0], pts[:, 1], color=PAIR[k], alpha=0.85 * alpha,
                                   lw=0.4 + 0.45 * T[k, side], ls="-" if s > 0 else (0, (3, 1.5)),
                                   zorder=zorder - 1)
    for j in range(1, 4):                                    # disks: bars across the arm
        x, y, psi = P[j]
        n = np.array([-np.sin(psi), np.cos(psi)])
        artists += ax.plot([x - 0.28 * n[0], x + 0.28 * n[0]], [y - 0.28 * n[1], y + 0.28 * n[1]],
                           color=color, lw=1.2, alpha=alpha, zorder=zorder)
    if payload:                                              # the payload: a block on the tip
        x, y, psi = P[3]
        t, n = np.array([np.cos(psi), np.sin(psi)]), np.array([-np.sin(psi), np.cos(psi)])
        h = PAYLOAD_SIZE / 2
        c = np.array([x, y]) + h * t
        corners = [c + a * h * t + b * h * n for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = Polygon(corners, closed=True, facecolor=color if alpha < 1 else INK2,
                       edgecolor=color, alpha=alpha, lw=1.0, zorder=zorder + 1)
        ax.add_patch(poly)
        artists.append(poly)
        if alpha == 1.0:
            artists.append(ax.text(*c, "m", color=SURFACE, ha="center", va="center", fontsize=8,
                                   fontweight="bold", zorder=zorder + 2))
    return artists


def reference(ax, X, unstable, lw=1.4):
    """Reference path of the payload: dashed where it is held stably, red where not."""
    seg = np.stack([X[:-1, :2], X[1:, :2]], axis=1)
    bad = unstable[1:] | unstable[:-1]
    ax.add_collection(LineCollection(seg[~bad], colors=INK2, linewidths=lw, linestyles=(0, (4, 2)),
                                     zorder=3))
    ax.add_collection(LineCollection(seg[bad], colors=CRITICAL, linewidths=lw + 1.2, zorder=3))


def legend_items():
    return [Line2D([], [], color=INK2, lw=1.4, ls=(0, (4, 2)), label="reference (payload)"),
            Line2D([], [], color=CRITICAL, lw=2.6, label="payload unstable in open loop (K_X ≤ 0)")]


def pair_legend(ax):
    handles = [Line2D([], [], color=PAIR[k], lw=2, label=f"pair {k + 1}") for k in range(3)]
    handles += [Line2D([], [], color=INK2, lw=1.5, label="+ side"),
                Line2D([], [], color=INK2, lw=1.5, ls=(0, (3, 1.5)), label="− side")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, fontsize=7.5,
              handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)


def fig_demo(name):
    z = np.load(RUNS / f"demo_{name}.npz")
    t, T, kx = z["t"], z["T"], z["kx"]
    unstable = kx <= 0
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.25), hspace=0.6, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    base(ax)
    reference(ax, z["X_ref"], unstable)
    n = len(t)
    snaps = np.linspace(0, n - 1, 9).astype(int)
    for i, k in enumerate(snaps[:-1]):
        draw_arm(ax, z["L"][k], z["P"][k], color=MUTED, lw=1.0, alpha=0.3 + 0.45 * i / len(snaps))
    draw_arm(ax, z["L"][-1], z["P"][-1], T[-1])
    ax.legend(handles=legend_items(), loc="upper left", fontsize=8)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title("Commanded arm through the run; tendons at the end")

    a1 = fig.add_subplot(gs[0, 1])
    for k in range(3):
        a1.plot(t, T[:, k, 0], color=PAIR[k], lw=1.6)
        a1.plot(t, T[:, k, 1], color=PAIR[k], lw=1.6, ls=(0, (3, 1.5)))
    a1.set_ylabel("tension")
    a1.set_ylim(0, None)
    a1.set_title("Tendon tensions sent (no limits; slack side at 0.1)", pad=14)
    pair_legend(a1)

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, kx, color=INK, lw=1.5)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.fill_between(t, kx, 0, where=unstable, color=CRITICAL, alpha=0.25, lw=0)
    a2.set_ylabel("k_min")
    a2.set_title(f"Payload stiffness, smallest eigenvalue of K_X: below 0 on "
                 f"{unstable.mean():.0%} of the run")

    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    if name == "xline":
        a3.plot(t, np.degrees(z["P"][:, -1, 2]), color=INK, lw=1.6, label="payload angle")
        a3.plot(t, np.degrees(z["X_ref"][:, 2]), color=INK2, lw=1.2, ls=(0, (4, 2)),
                label="reference")
        a3.set_ylabel("degrees")
        a3.set_title("Payload angle: 90° (pointing +y) once on the line")
        a3.legend(loc="lower right", fontsize=8)
    else:
        a3.plot(t, 1e3 * z["time"], color=INK, lw=0.8)
        a3.axhline(10, color=INK2, lw=1, ls=":")
        a3.text(t[0], 10, "10 ms budget (100 Hz)", color=INK2, va="bottom", fontsize=8,
                bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1))
        a3.set_ylim(0, max(12, 1.1e3 * np.percentile(z["time"], 99.5)))
        a3.set_ylabel("ms")
        a3.set_title(f"Controller step time: mean {1e3 * z['time'].mean():.1f} ms")
    a3.set_xlabel("time (s)")
    fig.suptitle(TITLES[name], x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.savefig(FIG / f"demo_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/demo_{name}.png")


def video_demo(name, fps=25):
    z = np.load(RUNS / f"demo_{name}.npz")
    t, T, kx = z["t"], z["T"], z["kx"]
    unstable = kx <= 0
    step = max(1, int(round(1 / (fps * float(z["dt"])))))
    frames = np.arange(0, len(t), step)
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.35, 1), wspace=0.18)
    ax = fig.add_subplot(gs[0])
    base(ax)
    reference(ax, z["X_ref"], unstable, lw=1.1)
    ax.legend(handles=legend_items(), loc="lower right", fontsize=8)
    ax.set_ylim(-0.9, 3.0)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title(SHORT[name])
    bx = fig.add_subplot(gs[1])
    pos = np.array([0, 1, 3, 4, 6, 7], float)
    bars = bx.bar(pos, np.zeros(6), width=0.8, color=[PAIR[k] for k in range(3) for _ in range(2)])
    for b, hatch in zip(bars, ["", "///"] * 3):
        b.set_hatch(hatch)
        b.set_edgecolor(SURFACE)
    bx.set_xticks(pos, ["1 +", "1 −", "2 +", "2 −", "3 +", "3 −"])
    bx.set_ylim(0, float(T.max()) * 1.1)
    bx.set_ylabel("tension")
    bx.set_title("Tensions sent (pair, side)")
    bx.grid(axis="x", visible=False)
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    event = ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top", fontsize=9, color=CRITICAL)
    dyn = []

    def update(i):
        k = frames[i]
        for a in dyn:
            a.remove()
        dyn.clear()
        dyn.extend(draw_arm(ax, z["L"][k], z["P"][k], T[k]))
        dyn.extend(ax.plot(*z["X_ref"][k, :2], marker="o", ms=5, color=INK2, zorder=7))
        for b, v in zip(bars, T[k].ravel()):
            b.set_height(v)
        clock.set_text(f"t = {t[k]:5.2f} s")
        event.set_text("payload unstable in open loop here (K_X ≤ 0)" if unstable[k] else "")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(FIG / f"demo_{name}.mp4", writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
    plt.close(fig)
    print(f"wrote figures/demo_{name}.mp4 ({len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["circle", "xline"])
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    for name in args.names:
        fig_demo(name)
        if not args.no_video:
            video_demo(name)

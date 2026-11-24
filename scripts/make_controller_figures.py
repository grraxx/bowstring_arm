"""
make_controller_figures.py
Figures and animations of the online controller (from scripts/workspace_map.py and
scripts/run_demos.py outputs).

  figures/ctrl_workspace.png     where the payload can be held, for four co-contraction levels
  figures/ctrl_<demo>.png        per demo: the arm moving along the band, tensions, wrench or
                                 speed, step time against the 10 ms budget
  figures/ctrl_<demo>.mp4        per demo: the arm with its tendons (line width = tension), the
                                 reference, and the six tensions as bars

The arm drawn is the equilibrium the controller commands at each step (no plant yet).

usage: python scripts/make_controller_figures.py [--no-video]
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
from matplotlib.lines import Line2D                           # noqa: E402
import numpy as np                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm      # noqa: E402
from bowstring.tendons import hole        # noqa: E402

FIG = ROOT / "figures"
RUNS = ROOT / "data/runs"
# Reference palette (dataviz skill): categorical slots 1-3 for the three tendon pairs,
# text and surface tokens, sequential blue for the workspace, status colours for events.
PAIR = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BAND, BAND_EDGE = "#cde2fb", "#3987e5"
SERIOUS, CRITICAL = "#ec835a", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})
ARM = ChainArm()


def band(ax, ws, alpha=1.0):
    X, Y = np.meshgrid(ws["xs"], ws["ys"], indexing="ij")
    ax.contourf(X, Y, ws["reach"].astype(float), [0.5, 1.5], colors=[BAND], alpha=alpha)
    ax.contour(X, Y, ws["reach"].astype(float), [0.5], colors=[BAND_EDGE], linewidths=0.8)
    ax.plot(0, 0, marker="s", color=INK, ms=6, zorder=5)
    ax.set_aspect("equal")


def draw_arm(ax, L, P, T=None, color=INK, lw=2.0, alpha=1.0, tendons=True, zorder=4):
    artists = []
    for seg in ARM.shape(L):
        artists += ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, alpha=alpha, zorder=zorder)
    if tendons and T is not None:
        for k in range(3):
            for side, s in enumerate((1.0, -1.0)):
                pts = np.array([hole(P[j], ARM.offsets[k], s)[0] for j in range(k + 2)])
                artists += ax.plot(pts[:, 0], pts[:, 1], color=PAIR[k], alpha=0.85 * alpha,
                                   lw=0.4 + 0.5 * T[k, side], ls="-" if s > 0 else (0, (3, 1.5)),
                                   zorder=zorder - 1)
    for j in range(1, 4):                                    # disks: short bars across the arm
        x, y, psi = P[j]
        nx, ny = -np.sin(psi), np.cos(psi)
        r = 0.28
        artists += ax.plot([x - r * nx, x + r * nx], [y - r * ny, y + r * ny], color=color,
                           lw=1.2, alpha=alpha, zorder=zorder)
    return artists


def pair_legend(ax, loc="lower right"):
    handles = [Line2D([], [], color=PAIR[k], lw=2, label=f"pair {k + 1}") for k in range(3)]
    handles += [Line2D([], [], color=INK2, lw=1.5, label="+ side"),
                Line2D([], [], color=INK2, lw=1.5, ls=(0, (3, 1.5)), label="− side")]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=(1.0, 1.0), ncol=5, fontsize=7.5,
              handlelength=1.8, columnspacing=1.0, borderaxespad=0.2)


# ------------------------------------------------------------------ workspace
def fig_workspace():
    """Before (fixed pair totals) and now (minimal co-contraction, no tension limits)."""
    old = np.load(ROOT / "data/workspace_C1.0.npz")
    new = np.load(ROOT / "data/workspace.npz")
    scan = np.load(ROOT / "data/workspace_scan.npz")
    sw = np.load(RUNS / "demo_sweep.npz")
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.4), sharex=True, sharey=True)
    # before
    ax = axs[0]
    band(ax, old)
    fixed = ChainArm()
    rest = fixed.rest(fixed.tensions(old["C"], old["D0"]))
    draw_arm(ax, rest["L"], rest["P"], fixed.tensions(old["C"], old["D0"]), lw=1.6)
    ax.set_title("Before: fixed pair totals C = (3, 2.5, 2)", pad=18)
    ax.text(0.0, 1.0, f"area {old['reach'].sum() * 0.05 ** 2:.1f}; |D| ≤ C − 2 Tmin caps the "
            "bending moment of segment 3 at 0.45", transform=ax.transAxes, fontsize=8, color=INK2,
            va="bottom")
    # now
    ax = axs[1]
    good = scan["stable"] & np.isfinite(scan["tip"][:, 0])
    ax.plot(scan["tip"][good, 0], scan["tip"][good, 1], ".", ms=1.2, color=MUTED, alpha=0.5,
            label="stable rest poses, every posture (tension scan)", zorder=1)
    band(ax, new, alpha=0.85)
    pol = np.unwrap(np.arctan2(sw["X_ref"][:, 1], sw["X_ref"][:, 0]))
    for k in (int(np.argmax(pol)), int(np.argmin(pol))):
        draw_arm(ax, sw["L"][k], sw["P"][k], sw["T"][k], lw=1.8)
    ax.plot(sw["X_ref"][:, 0], sw["X_ref"][:, 1], color=INK2, lw=1.0, ls=(0, (4, 2)),
            label="sweep demo route", zorder=3)
    ax.annotate("the two ends meet here but do not connect:\nover the top the arm is curled one way,\n"
                "underneath the other", xy=(-2.0, 0.0), xytext=(-3.05, -2.95), fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_title("Now: minimal co-contraction, no tension limits", pad=18)
    ax.text(0.0, 1.0, f"area {new['reach'].sum() * 0.05 ** 2:.1f} reachable continuously "
            f"(shaded); payload stiffness median {np.nanmedian(new['kmin']):.3f} "
            f"(before {np.nanmedian(old['kmin']):.3f})", transform=ax.transAxes, fontsize=8,
            color=INK2, va="bottom")
    ax.legend(loc="upper left", fontsize=8, markerscale=6)
    for ax in axs:
        ax.add_patch(plt.Circle((0, 0), 3.0, fill=False, ls=":", lw=0.8, color=MUTED))
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("x  (segment lengths)")
    axs[0].set_ylabel("y")
    fig.suptitle("Where the arm can hold the payload (orientation free, stable with the payload "
                 "free); dotted circle: full reach", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "ctrl_workspace.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/ctrl_workspace.png")


# ------------------------------------------------------------------ demo dashboards
TITLES = {"sweep": "Sweep: fixed timing, end to end of the workspace and back",
          "path": "Path: end to end, speed chosen online (safe mode)",
          "live": "Live: scripted gamepad session (safe mode)"}


def fig_demo(name, ws):
    z = np.load(RUNS / f"demo_{name}.npz")
    t, T, w = z["t"], z["T"], z["w"]
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.05, 1.25), hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    band(ax, ws, alpha=0.7)
    ax.plot(z["X_ref"][:, 0], z["X_ref"][:, 1], color=INK2, lw=1.2, ls=(0, (4, 2)),
            label="reference (payload)")
    n = len(t)
    snaps = np.linspace(0, n - 1, 9).astype(int)
    for i, k in enumerate(snaps[:-1]):
        draw_arm(ax, z["L"][k], z["P"][k], tendons=False, color=MUTED, lw=1.0,
                 alpha=0.35 + 0.5 * i / len(snaps))
    draw_arm(ax, z["L"][-1], z["P"][-1], T[-1])
    sw = np.flatnonzero(z["switched"])
    if len(sw):
        ax.plot(z["X_ref"][sw, 0], z["X_ref"][sw, 1], "o", ms=8, mfc="none", mec=INK, mew=1.5,
                label="posture switch (bulge up ↔ down)", zorder=6)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title("Commanded arm, every 1/8 of the run; tendons at the end")
    ax.legend(loc="lower left", fontsize=8)

    a1 = fig.add_subplot(gs[0, 1])
    for k in range(3):
        a1.plot(t, T[:, k, 0], color=PAIR[k], lw=1.6)
        a1.plot(t, T[:, k, 1], color=PAIR[k], lw=1.6, ls=(0, (3, 1.5)))
    a1.axhline(float(z["Tmin"]), color=CRITICAL, lw=1, ls=":")
    a1.text(t[0], float(z["Tmin"]), "T min", color=CRITICAL, va="bottom", fontsize=8)
    a1.set_ylabel("tension")
    a1.set_title("Tendon tensions sent", pad=14)
    pair_legend(a1)
    a1.set_ylim(0, None)

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    if name == "path":
        a2.plot(t, z["speed"], color=INK, lw=1.6)
        a2.set_ylabel("speed")
        a2.set_title("Speed along the path (chosen online; curvature-capped)")
    else:
        a2.plot(t, np.hypot(w[:, 0], w[:, 1]), color=INK, lw=1.6)
        a2.set_ylabel("|w|")
        a2.set_title(f"Force demanded on the payload, |w| = m |Ẍ*|  (m = {float(z['m'])})")
    a2.set_ylim(0, None)

    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    a3.plot(t, 1e3 * z["time"], color=INK, lw=0.8)
    a3.axhline(10, color=INK2, lw=1, ls=":")
    a3.text(t[0], 10, "10 ms budget (100 Hz)", color=INK2, va="bottom", fontsize=8,
            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=1), zorder=5)
    fb = z["choice"] > 0
    held = ~z["feasible"]
    top = max(12, np.percentile(1e3 * z["time"], 99.5) * 1.15)
    if fb.any():
        a3.plot(t[fb], np.full(fb.sum(), top * 0.97), "|", color=SERIOUS, ms=6,
                label=f"fallback (brake / hold): {fb.sum()} steps")
    if held.any():
        a3.plot(t[held], np.full(held.sum(), top * 0.9), "|", color=CRITICAL, ms=6,
                label=f"no feasible step, T held: {held.sum()}")
    if fb.any() or held.any():
        a3.legend(loc="upper right", fontsize=8)
    a3.set_ylim(0, top)
    a3.set_ylabel("ms")
    a3.set_xlabel("time (s)")
    a3.set_title(f"Controller step time: mean {1e3 * z['time'].mean():.1f} ms, "
                 f"99th pct {np.percentile(1e3 * z['time'], 99):.1f} ms")
    fig.suptitle(TITLES[name], x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.savefig(FIG / f"ctrl_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/ctrl_{name}.png")


# ------------------------------------------------------------------ animations
def video_demo(name, ws, fps=25):
    z = np.load(RUNS / f"demo_{name}.npz")
    t, T = z["t"], z["T"]
    step = max(1, int(round(1 / (fps * float(z["dt"])))))
    frames = np.arange(0, len(t), step)
    tmax = float(np.nanmax(T)) * 1.1
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.5, 1), wspace=0.15)
    ax = fig.add_subplot(gs[0])
    band(ax, ws, alpha=0.7)
    ax.plot(z["X_ref"][:, 0], z["X_ref"][:, 1], color=INK2, lw=1.0, ls=(0, (4, 2)))
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title(TITLES[name])
    bx = fig.add_subplot(gs[1])
    pos = np.array([0, 1, 3, 4, 6, 7], float)
    colors = [PAIR[k] for k in range(3) for _ in range(2)]
    bars = bx.bar(pos, np.zeros(6), width=0.8, color=colors)
    for b, hatch in zip(bars, ["", "///"] * 3):
        b.set_hatch(hatch)
        b.set_edgecolor(SURFACE)
    bx.set_xticks(pos, ["1 +", "1 −", "2 +", "2 −", "3 +", "3 −"])
    bx.axhline(float(z["Tmin"]), color=CRITICAL, lw=1, ls=":")
    bx.text(7.6, float(z["Tmin"]), " T min", color=CRITICAL, va="center", fontsize=8)
    bx.set_ylim(0, tmax)
    bx.set_ylabel("tension")
    bx.set_title("Tensions sent (pair, side)")
    bx.grid(axis="x", visible=False)
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9, color=INK)
    event = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top", fontsize=9)
    dyn = []

    def update(i):
        k = frames[i]
        for a in dyn:
            a.remove()
        dyn.clear()
        dyn.extend(draw_arm(ax, z["L"][k], z["P"][k], T[k]))
        dyn.extend(ax.plot(*z["X_ref"][k, :2], marker="o", ms=7, mfc="none", mec=INK, mew=1.5,
                           zorder=6))
        for b, v in zip(bars, T[k].ravel()):
            b.set_height(v)
        clock.set_text(f"t = {t[k]:5.2f} s   step {1e3 * z['time'][k]:4.1f} ms")
        lo, hi = max(0, k - step), k + 1
        if (~z["feasible"][lo:hi]).any():
            event.set_text("no feasible step: tensions held")
            event.set_color(CRITICAL)
        elif (z["choice"][lo:hi] > 0).any():
            event.set_text("command refused: braking / holding")
            event.set_color(SERIOUS)
        elif z["switched"][lo:hi].any():
            event.set_text("posture switch (bulge up ↔ down)")
            event.set_color(INK)
        else:
            event.set_text("")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(FIG / f"ctrl_{name}.mp4", writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
    plt.close(fig)
    print(f"wrote figures/ctrl_{name}.mp4 ({len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    ws = np.load(ROOT / "data/workspace.npz")
    fig_workspace()
    for name in ("sweep", "path", "live"):
        fig_demo(name, ws)
    if not args.no_video:
        for name in ("sweep", "path", "live"):
            video_demo(name, ws)

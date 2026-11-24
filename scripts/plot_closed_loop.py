"""
plot_closed_loop.py
Open loop against closed loop (tip IMU only) on the simulated plant, for the circle and the
x-line: the payload paths and the tracking errors over time.

usage: python scripts/plot_closed_loop.py [--no-video]   -> figures/loop_<demo>.png and .mp4
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_demo_figures import (CRITICAL, INK, INK2, MUTED, SURFACE, base, draw_arm)  # noqa: E402
sys.path.insert(0, str(ROOT))
from bowstring.chain import ChainArm  # noqa: E402

ARM = ChainArm(scheme="minimal")

OPEN, CLOSED = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
TITLES = {"circle": "Circle, radius 0.55 about (1.45, 1.5)",
          "xline": "Line at y = 1.8, payload pointing +y"}


def main():
    for demo in ("circle", "xline"):
        runs = {m: np.load(ROOT / f"data/runs/loop_{demo}_{m}.npz") for m in ("open", "closed")}
        ref = np.load(ROOT / f"data/runs/demo_{demo}.npz")        # the full reference
        fig = plt.figure(figsize=(13, 5.6))
        gs = fig.add_gridspec(2, 2, width_ratios=(1, 1.3), hspace=0.5, wspace=0.25)
        ax = fig.add_subplot(gs[:, 0])
        base(ax)
        ax.plot(ref["X_ref"][:, 0], ref["X_ref"][:, 1], color=INK2, lw=1.2, ls=(0, (4, 2)),
                label="reference")
        for m, col in (("open", OPEN), ("closed", CLOSED)):
            z = runs[m]
            ax.plot(z["X"][:, 0], z["X"][:, 1], color=col, lw=1.8,
                    label=f"{m} loop: {str(z['status'])}")
            ax.plot(*z["X"][-1, :2], "x", color=col, ms=9, mew=2)
        zc = runs["closed"]
        k = max(0, len(zc["t"]) - 60)                  # the closed-loop arm shortly before the end
        P = ARM.evaluate(zc["L"][k], zc["T"][k])["P"]
        draw_arm(ax, zc["L"][k], P, zc["T"][k])
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_xlabel("x  (segment lengths)")
        ax.set_ylabel("y")
        ax.set_title("Payload paths on the simulated arm")
        a1 = fig.add_subplot(gs[0, 1])
        a2 = fig.add_subplot(gs[1, 1], sharex=a1)
        for m, col in (("open", OPEN), ("closed", CLOSED)):
            z = runs[m]
            a1.semilogy(z["t"], np.maximum(z["err"], 1e-5), color=col, lw=1.5, label=f"{m} loop")
            a2.plot(z["t"], np.degrees(z["err_psi"]), color=col, lw=1.5)
        a1.semilogy(zc["t"], np.maximum(zc["est_err"], 1e-5), color=MUTED, lw=1.0,
                    label="closed loop: IMU position-estimate error")
        unstable = ref["kx"] <= 0
        for a in (a1, a2):
            ylim = a.get_ylim()
            a.fill_between(ref["t"], ylim[0], ylim[1], where=unstable, color=CRITICAL, alpha=0.08,
                           lw=0, label="payload open-loop unstable" if a is a1 else None)
            a.set_ylim(ylim)
        a1.set_ylabel("position error")
        a1.set_title("Tracking error of the payload position (log scale)")
        a1.legend(loc="lower right", fontsize=7.5)
        a2.set_ylabel("degrees")
        a2.set_xlabel("time (s)")
        a2.set_title("Payload angle error")
        fig.suptitle(f"{TITLES[demo]}: open loop vs closed loop with a tip IMU only", x=0.01,
                     ha="left", fontweight="bold", fontsize=11)
        fig.savefig(ROOT / f"figures/loop_{demo}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote figures/loop_{demo}.png")




def video(demo, fps=25):
    """The simulated arm under closed loop, with the open-loop run's payload path as a ghost."""
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from make_demo_figures import PAIR, SHORT
    zc = np.load(ROOT / f"data/runs/loop_{demo}_closed.npz")
    zo = np.load(ROOT / f"data/runs/loop_{demo}_open.npz")
    ref = np.load(ROOT / f"data/runs/demo_{demo}.npz")
    t, T = zc["t"], zc["T"]
    step = max(1, int(round(1 / (fps * float(zc["dt"])))))
    frames = np.arange(0, len(t), step)
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.35, 1), wspace=0.18)
    ax = fig.add_subplot(gs[0])
    base(ax)
    ax.set_ylim(-0.9, 3.0)
    ax.plot(ref["X_ref"][:, 0], ref["X_ref"][:, 1], color=INK2, lw=1.1, ls=(0, (4, 2)),
            label="reference")
    ax.plot(zo["X"][:, 0], zo["X"][:, 1], color=OPEN, lw=1.2, alpha=0.6,
            label="open loop (diverges)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title(f"{SHORT[demo]}: closed loop, tip IMU only")
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
    trail, = ax.plot([], [], color=CLOSED, lw=1.8, label="closed loop")
    dyn = []

    def update(i):
        k = frames[i]
        for a in dyn:
            a.remove()
        dyn.clear()
        P = ARM.evaluate(zc["L"][k], T[k])["P"]
        dyn.extend(draw_arm(ax, zc["L"][k], P, T[k]))
        dyn.extend(ax.plot(*zc["X_ref"][k, :2], marker="o", ms=5, color=INK2, zorder=7))
        dyn.extend(ax.plot(*zc["X_est"][k, :2], marker="+", ms=9, mew=1.6, color=CLOSED, zorder=8))
        trail.set_data(zc["X"][:k + 1, 0], zc["X"][:k + 1, 1])
        for b, v in zip(bars, T[k].ravel()):
            b.set_height(v)
        clock.set_text(f"t = {t[k]:5.2f} s   error {zc['err'][k]:.4f}   (+: IMU estimate)")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(ROOT / f"figures/loop_{demo}.mp4", writer=FFMpegWriter(fps=fps, bitrate=1800), dpi=110)
    plt.close(fig)
    print(f"wrote figures/loop_{demo}.mp4 ({len(frames)} frames)")


if __name__ == "__main__":
    main()
    if "--no-video" not in sys.argv:
        for demo in ("circle", "xline"):
            video(demo)

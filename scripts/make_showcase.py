"""
make_showcase.py
Figures, animations and a showreel of the showcase demos (scripts/run_showcase.py).

  figures/show_<demo>.png      wrist, waiter, figure8, disturb: the arm and the payload's path,
                               tracking error, payload angle, tensions
  figures/show_<demo>.mp4      the simulated arm with its tendons and payload, the "ink" trail
  figures/show_speeds.png      tracking error against speed round the circle
  figures/showreel.mp4         title cards, the four animations and the speed result

usage: python scripts/make_showcase.py [--no-video]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
import numpy as np                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from make_demo_figures import (INK, INK2, MUTED, PAIR, SURFACE, base, draw_arm,  # noqa: E402
                               pair_legend)
from bowstring.chain import ChainArm                                              # noqa: E402

RUNS, FIG = ROOT / "data/runs", ROOT / "figures"
TRAIL, PUSH = "#2a78d6", "#d03b3b"
ARM = ChainArm(scheme="minimal")
FFMPEG = str(Path(sys.executable).with_name("ffmpeg"))
TITLES = {
    "wrist": ("Wrist", "Payload held still at (1.6, 1.6) while it turns 20° → 135° → 20°"),
    "waiter": ("Waiter", "Payload kept level (pointing +y) round a 1.0 × 0.6 rectangle"),
    "figure8": ("Drawing", "The tip draws a figure-eight at constant orientation"),
    "disturb": ("Pushes", "Four unannounced kicks, three while holding and one while moving"),
}
SUBTITLE = "Closed loop · only sensor: an IMU on the tip · simulated tendon-driven elastica arm"


def load(name):
    return np.load(RUNS / f"show_{name}.npz")


def poses(z, k):
    return ARM.evaluate(z["L"][k], z["T"][k])["P"]


def kick_times(z):
    w = np.abs(z["w_ext"]).max(axis=1) > 0
    return z["t"][np.flatnonzero(w & ~np.r_[False, w[:-1]])]


# ------------------------------------------------------------------ static figures
def fig_demo(name):
    z = load(name)
    t, T = z["t"], z["T"]
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.25), hspace=0.62, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    base(ax)
    ax.plot(z["X_ref"][:, 0], z["X_ref"][:, 1], color=INK2, lw=1.2, ls=(0, (4, 2)),
            label="reference")
    ax.plot(z["X"][:, 0], z["X"][:, 1], color=TRAIL, lw=1.8, label="payload")
    snaps = np.linspace(0, len(t) - 1, 7).astype(int)
    for i, k in enumerate(snaps[:-1]):
        draw_arm(ax, z["L"][k], poses(z, k), color=MUTED, lw=1.0, alpha=0.3 + 0.4 * i / len(snaps))
    draw_arm(ax, z["L"][-1], poses(z, len(t) - 1), T[-1])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title("Simulated arm through the run")

    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, 1e3 * z["err"], color=INK, lw=1.3)
    a1.set_ylabel("× 10⁻³")
    a1.set_title(f"Payload position error: mean {1e3 * z['err'].mean():.1f}, "
                 f"max {1e3 * z['err'].max():.1f} × 10⁻³")
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, np.degrees(z["X_ref"][:, 2]), color=INK2, lw=1.2, ls=(0, (4, 2)), label="reference")
    a2.plot(t, np.degrees(z["X"][:, 2]), color=TRAIL, lw=1.5, label="payload")
    a2.set_ylabel("degrees")
    a2.set_title(f"Payload angle (error at most {np.degrees(np.abs(z['err_psi']).max()):.2f}°)")
    a2.legend(loc="best", fontsize=8)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)
    for k in range(3):
        a3.plot(t, T[:, k, 0], color=PAIR[k], lw=1.4)
        a3.plot(t, T[:, k, 1], color=PAIR[k], lw=1.4, ls=(0, (3, 1.5)))
    a3.set_ylabel("tension")
    a3.set_ylim(0, None)
    a3.set_xlabel("time (s)")
    a3.set_title("Tendon tensions", pad=14)
    pair_legend(a3)
    for t0 in kick_times(z):
        for a in (a1, a2, a3):
            a.axvline(t0, color=PUSH, lw=0.8, ls=":")
        a1.text(t0, a1.get_ylim()[1], " push", color=PUSH, fontsize=7.5, va="top")
    fig.suptitle(f"{TITLES[name][0]}: {TITLES[name][1]}", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.text(0.01, 0.935, SUBTITLE, fontsize=8.5, color=INK2)
    fig.savefig(FIG / f"show_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/show_{name}.png")


def fig_speeds(size=(11, 5.6), dpi=150, out="show_speeds.png"):
    laps = (24, 16, 10, 6)
    rows = []
    for lap in laps:
        z = load(f"speed_{lap}")
        v = np.hypot(*np.gradient(z["X_ref"][:, :2], z["t"], axis=0).T).max()
        rows.append((v, 1e3 * z["err"].mean(), 1e3 * z["err"].max()))
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=size)
    ax.plot(rows[:, 0], rows[:, 2], "o-", color=PAIR[1], lw=2, ms=8, label="max")
    ax.plot(rows[:, 0], rows[:, 1], "o-", color=PAIR[0], lw=2, ms=8, label="mean")
    for (v, m, M), lap in zip(rows, laps):
        ax.annotate(f"{lap} s lap", (v, M), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("peak payload speed  (segment lengths / s)")
    ax.set_ylabel("payload position error  (× 10⁻³ segment lengths)")
    ax.set_ylim(0, rows[:, 2].max() * 1.3)
    ax.set_xlim(0, rows[:, 0].max() * 1.12)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Accuracy against speed: the circle of radius 0.55, one lap each", loc="left")
    fig.text(0.01, 0.01, SUBTITLE, fontsize=8.5, color=INK2)
    fig.savefig(FIG / out, dpi=dpi, bbox_inches="tight" if out == "show_speeds.png" else None)
    plt.close(fig)
    print(f"wrote figures/{out}")


# ------------------------------------------------------------------ animations
SIZE, DPI, FPS = (11, 5.6), 110, 25


def video_demo(name):
    z = load(name)
    t, T = z["t"], z["T"]
    step = max(1, int(round(1 / (FPS * float(z["dt"])))))
    frames = np.arange(0, len(t), step)
    fig = plt.figure(figsize=SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.35, 1), wspace=0.18)
    ax = fig.add_subplot(gs[0])
    base(ax)
    ax.set_ylim(-0.9, 3.0)
    ax.plot(z["X_ref"][:, 0], z["X_ref"][:, 1], color=INK2, lw=1.0, ls=(0, (4, 2)))
    trail, = ax.plot([], [], color=TRAIL, lw=2.2)
    ax.set_xlabel("x  (segment lengths)")
    ax.set_ylabel("y")
    ax.set_title("Simulated arm and payload path", fontsize=10)
    fig.suptitle(f"{TITLES[name][0]}: {TITLES[name][1]}", x=0.01, ha="left", fontweight="bold",
                 fontsize=12)
    bx = fig.add_subplot(gs[1])
    pos = np.array([0, 1, 3, 4, 6, 7], float)
    bars = bx.bar(pos, np.zeros(6), width=0.8, color=[PAIR[k] for k in range(3) for _ in range(2)])
    for b, hatch in zip(bars, ["", "///"] * 3):
        b.set_hatch(hatch)
        b.set_edgecolor(SURFACE)
    bx.set_xticks(pos, ["1 +", "1 −", "2 +", "2 −", "3 +", "3 −"])
    bx.set_ylim(0, float(T.max()) * 1.1)
    bx.set_ylabel("tension")
    bx.set_title("Tendon tensions (pair, side)")
    bx.grid(axis="x", visible=False)
    fig.text(0.01, 0.015, SUBTITLE, fontsize=8, color=INK2)
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=9)
    event = ax.text(0.02, 0.91, "", transform=ax.transAxes, va="top", fontsize=10, color=PUSH,
                    fontweight="bold")
    dyn = []
    pushes = z["w_ext"]

    def update(i):
        k = frames[i]
        for a in dyn:
            a.remove()
        dyn.clear()
        dyn.extend(draw_arm(ax, z["L"][k], poses(z, k), T[k]))
        trail.set_data(z["X"][:k + 1, 0], z["X"][:k + 1, 1])
        for b, v in zip(bars, T[k].ravel()):
            b.set_height(v)
        clock.set_text(f"t = {t[k]:5.2f} s   position error {1e3 * z['err'][k]:4.1f} × 10⁻³   "
                       f"angle {np.degrees(z['X'][k, 2]):6.1f}°")
        recent = slice(max(0, k - int(0.6 / float(z["dt"]))), k + 1)   # show a push for 0.6 s
        w = pushes[recent][np.argmax(np.abs(pushes[recent]).max(axis=1))]
        if np.abs(w).max() > 0:
            event.set_text("push!")
            X = z["X"][k]
            if np.hypot(*w[:2]) > 0:
                d = 0.35 * w[:2] / np.hypot(*w[:2])
                dyn.append(ax.annotate("", xy=X[:2], xytext=X[:2] - d,
                                       arrowprops=dict(arrowstyle="-|>", color=PUSH, lw=2.5)))
            else:
                dyn.extend(ax.plot(*X[:2], "o", ms=22, mfc="none", mec=PUSH, mew=2.5))
        else:
            event.set_text("")
        return dyn

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(FIG / f"show_{name}.mp4", writer=FFMpegWriter(
        fps=FPS, bitrate=2000, extra_args=["-pix_fmt", "yuv420p"]), dpi=DPI)
    plt.close(fig)
    print(f"wrote figures/show_{name}.mp4 ({len(frames)} frames)")


def card(path, title, lines):
    fig = plt.figure(figsize=SIZE)
    fig.text(0.07, 0.62, title, fontsize=28, fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        fig.text(0.07, 0.48 - 0.07 * i, line, fontsize=14, color=INK2)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def still_to_clip(png, mp4, seconds=3.0):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-loop", "1", "-i", str(png), "-t",
                    str(seconds), "-r", str(FPS), "-vf", "scale=1210:616,format=yuv420p",
                    "-c:v", "libx264", str(mp4)], check=True)


def showreel():
    tmp = FIG / "_reel"
    tmp.mkdir(exist_ok=True)
    clips = []
    card(tmp / "c0.png", "Bowstring tendon-driven elastica arm",
         ["Three elastica segments, three tendon pairs, a payload on the tip.",
          "Closed-loop control from a single IMU on the tip.", "Simulated; all units non-dimensional."])
    still_to_clip(tmp / "c0.png", tmp / "c0.mp4", 3.5)
    clips.append(tmp / "c0.mp4")
    for i, name in enumerate(("wrist", "waiter", "figure8", "disturb"), 1):
        z = load(name)
        card(tmp / f"c{i}.png", TITLES[name][0],
             [TITLES[name][1], f"position error: mean {1e3 * z['err'].mean():.1f}, max "
              f"{1e3 * z['err'].max():.1f} × 10⁻³ of a segment length"])
        still_to_clip(tmp / f"c{i}.png", tmp / f"c{i}.mp4", 2.5)
        clips += [tmp / f"c{i}.mp4", FIG / f"show_{name}.mp4"]
    fig_speeds(size=SIZE, dpi=DPI, out="_reel/speeds.png")
    still_to_clip(tmp / "speeds.png", tmp / "speeds.mp4", 4.0)
    clips.append(tmp / "speeds.mp4")
    (tmp / "list.txt").write_text("".join(f"file '{c}'\n" for c in clips))
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                    str(tmp / "list.txt"), "-vf", "scale=1210:616,format=yuv420p", "-r", str(FPS),
                    "-c:v", "libx264", str(FIG / "showreel.mp4")], check=True)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print("wrote figures/showreel.mp4")


if __name__ == "__main__":
    for name in TITLES:
        fig_demo(name)
    fig_speeds()
    if "--no-video" not in sys.argv:
        for name in TITLES:
            video_demo(name)
        showreel()

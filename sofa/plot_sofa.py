"""
plot_sofa.py
The closed-loop controller on SOFA against the same controller on the model's own plant.

  figures/sofa_demos.png    per demo: the SOFA arm and payload path; tracking error over time
                            (SOFA, model plant, SOFA with the earlier IMU-only sensing from
                            data/runs/imu_only/) and the pose-estimate error on SOFA
  figures/sofa_mesh.png     figure-eight on SOFA meshes of 80, 160, 320 elements per segment

usage: python sofa/plot_sofa.py      (either env)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_demo_figures import INK, INK2, MUTED, PAIR, base  # noqa: E402

RUNS, FIG = ROOT / "data/runs", ROOT / "figures"
SOFA, MODEL, EST, OLD = PAIR[0], PAIR[1], MUTED, "#c9c8c3"
DEMOS = {"wrist": "Wrist", "waiter": "Waiter", "figure8": "Figure-eight", "disturb": "Pushes",
         "speed_10": "Circle, 10 s lap"}


def payload_box(ax, X, size=0.12):
    c, s = np.cos(X[2]), np.sin(X[2])
    sq = np.array([[-0.5, 0], [0.5, 0], [0.5, 1], [-0.5, 1], [-0.5, 0]]) * size
    R = np.array([[s, c], [-c, s]])          # the block sits on the tip, "up" along the tip
    p = X[:2] + sq @ R
    ax.fill(p[:, 0], p[:, 1], color=INK2, alpha=0.85, zorder=6)


def main():
    names = [n for n in DEMOS if (RUNS / f"sofa_{n}.npz").exists()]
    fig, axes = plt.subplots(2, len(names), figsize=(4.0 * len(names), 7.6),
                             gridspec_kw={"height_ratios": (1.25, 1)})
    for j, name in enumerate(names):
        z = np.load(RUNS / f"sofa_{name}.npz")
        m = np.load(RUNS / f"show_{name}.npz")
        ax = axes[0, j]
        base(ax)
        ax.set_xlim(-0.3, 2.8)
        ax.set_ylim(-0.6, 2.6)
        ax.plot(z["X_ref"][:, 0], z["X_ref"][:, 1], color=INK2, lw=1.1, ls=(0, (4, 2)),
                label="reference")
        ax.plot(z["X"][:, 0], z["X"][:, 1], color=SOFA, lw=1.6, label="payload on SOFA")
        for k in np.linspace(0, len(z["t"]) - 1, 5).astype(int)[:-1]:
            ax.plot(z["nodes"][k, :, 0], z["nodes"][k, :, 1], color=MUTED, lw=0.9, alpha=0.5)
        ax.plot(z["nodes"][-1, :, 0], z["nodes"][-1, :, 1], color=INK, lw=2.0)
        payload_box(ax, z["X"][-1])
        ax.set_title(f"{DEMOS[name]}: {str(z['status'])}", fontsize=10)
        if j == 0:
            ax.legend(loc="lower right", fontsize=7.5)
        b = axes[1, j]
        old = RUNS / "imu_only" / f"sofa_{name}.npz"
        if old.exists():
            o = np.load(old)
            b.semilogy(o["t"], 1e3 * np.maximum(o["err"], 1e-6), color=OLD, lw=1.2,
                       label="SOFA, earlier IMU-only sensing")
        b.semilogy(m["t"], 1e3 * np.maximum(m["err"], 1e-6), color=MODEL, lw=1.3,
                   label="model plant")
        b.semilogy(z["t"], 1e3 * np.maximum(z["err"], 1e-6), color=SOFA, lw=1.5, label="SOFA")
        b.semilogy(z["t"], 1e3 * np.maximum(z["est_err"], 1e-6), color=EST, lw=1.0,
                   ls=(0, (3, 1.5)), label="SOFA: pose-estimate error")
        b.set_ylim(0.05, 1000)
        b.set_xlabel("time (s)")
        b.set_title(f"SOFA mean {1e3 * z['err'].mean():.1f}, max {1e3 * z['err'].max():.1f}\n"
                    f"model plant {1e3 * m['err'].mean():.1f}, {1e3 * m['err'].max():.1f}",
                    fontsize=9)
        if j == 0:
            b.set_ylabel("payload position error (× 10⁻³, log)")
            b.legend(loc="upper left", fontsize=7)
    fig.suptitle("Closed loop on the SOFA arm (160 beam elements per segment, SoftRobots cables): "
                 "force-driven, sensing = high-end tip IMU + tendon-length encoders", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG / "sofa_demos.png", dpi=140)
    plt.close(fig)
    print("wrote figures/sofa_demos.png")

    runs = [(ne, RUNS / f"sofa_figure8{'' if ne == 160 else f'_ne{ne}'}.npz") for ne in (80, 160, 320)]
    runs = [(ne, p) for ne, p in runs if p.exists()]
    if len(runs) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        m = np.load(RUNS / "show_figure8.npz")
        ax.plot(m["t"], 1e3 * m["err"], color=MODEL, lw=1.3, label="model plant")
        for (ne, p), col in zip(runs, ("#9ec5f0", PAIR[0], "#0d3f80")):
            z = np.load(p)
            ax.plot(z["t"], 1e3 * z["err"], color=col, lw=1.5,
                    label=f"SOFA, {ne} elements/segment (mean {1e3 * z['err'].mean():.1f})")
        ax.set_ylim(0, None)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("payload position error (× 10⁻³)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("Figure-eight on SOFA meshes of different accuracy (IMU + tendon encoders)",
                     loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "sofa_mesh.png", dpi=140)
        plt.close(fig)
        print("wrote figures/sofa_mesh.png")


if __name__ == "__main__":
    main()

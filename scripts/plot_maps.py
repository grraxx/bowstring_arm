"""
plot_maps.py
Figures of the forward and inverse maps, from data/maps/*.npz (compute_maps.py).
Writes figures/maps_*.png.

    maps_toy.png          the two maps on a one-degree-of-freedom toy: one curve
                          read two ways (2D), and the cusp surface (3D)
    maps_scurves.png      the real arm: forward-map S-curves along the payload
                          angle and along a tension, with the stability margin
    maps_forward.png      forward map over the payload position (fixed T)
    maps_cusp.png         the real arm's fold curves over (payload angle, D1), in 3D and 2D
    maps_inverse.png      inverse map over the payload force (fixed X)
    maps_inverse_3d.png   the tension differences as surfaces over the force
    maps_workspace.png    rest map: the payload pose over the tension box
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.rod import BowstringRod            # noqa: E402

DATA = ROOT / "data" / "maps"
FIG = ROOT / "figures"

# Palette (dataviz reference instance, light mode).
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
STABLE, UNSTABLE = "#2a78d6", "#e34948"
PAIR = ("#2a78d6", "#eb6834", "#1baf7a")              # tendon pairs 1-3: categorical slots 1-3
NEUTRAL = "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL, "#f2a7a6", "#e34948", "#7d1d1c"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2,
    "hatch.color": "#b9b8b2", "hatch.linewidth": 0.5,
})

ROD = BowstringRod(n=3, offsets=(0.15, 0.20, 0.25), N=40)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def stability_line(ax, x, y, margin, **kw):
    """A curve drawn solid blue where stable and dashed red where not."""
    stable = margin > 0
    for flag, color, ls in ((True, STABLE, "-"), (False, UNSTABLE, "--")):
        yy = np.where(stable == flag, y, np.nan)
        # keep the joins: extend each run by one point
        edge = np.r_[False, stable[1:] != stable[:-1]]
        yy = np.where(edge | np.r_[edge[1:], False], y, yy)
        ax.plot(x, yy, color=color, ls=ls, **kw)


def folds(x, margin):
    return np.where(np.diff(np.sign(margin)) != 0)[0]


def draw_arm(ax, theta, color=INK, tendons=True, lw=1.6, alpha=1.0, D=None, C=None):
    pos, _ = ROD.nodes(theta)
    disks = ROD.disks(theta)
    if tendons:
        for (k, s, ja, jb) in ROD.chords:
            ga, gb = ROD.hole(disks[ja], k, s), ROD.hole(disks[jb], k, s)
            ax.plot([ga[0], gb[0]], [ga[1], gb[1]], color=PAIR[k], lw=0.8, alpha=0.9 * alpha)
    ax.plot(pos[:, 0], pos[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    for j, P in enumerate(disks):
        nrm = np.array([-np.sin(P[2]), np.cos(P[2])])
        r = 0.3 if j else 0.32
        ax.plot(*np.column_stack([P[:2] - r * nrm, P[:2] + r * nrm]), color=MUTED, lw=2.2,
                alpha=alpha, solid_capstyle="round")
    tip = disks[-1]
    return tip


def draw_shape(ax, pos, color=INK, lw=1.2, alpha=1.0):
    ax.plot(pos[:, 0], pos[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    for j in range(4):
        p = pos[j * ROD.N]
        ax.plot(*p, "o", ms=2.5, color=MUTED, alpha=alpha)


# ------------------------------------------------------------------ toy

def toy():
    """U = q^4/4 - c q^2/2 + (q - X)^2/2 - D q with X = 0: the wrench is w = q,
    the 'tension' D, the 'co-contraction' c softens the middle disk."""
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    c = 2.0
    w = np.linspace(-1.45, 1.45, 600)
    D = w ** 3 + (1 - c) * w
    H = 3 * w ** 2 - c + 1
    stability_line(ax, w, D, H)
    wf = np.sqrt((c - 1) / 3)
    Df = wf ** 3 + (1 - c) * wf
    for s in (-1, 1):
        ax.plot(s * wf, s * Df, "o", color=INK, ms=6, zorder=5)
    ax.annotate("", xy=(2 * wf - 0.02, -Df), xytext=(-wf + 0.04, -Df),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, ls=(0, (4, 3))))
    ax.text(0.45 * wf, -Df + 0.05, "snap", ha="center", color=INK, fontsize=9)
    ax.annotate("", xy=(-2 * wf + 0.02, Df), xytext=(wf - 0.04, Df),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, ls=(0, (4, 3))))
    ax.text(-0.45 * wf, Df - 0.1, "snap back", ha="center", color=INK2, fontsize=8)
    ax.annotate("F1: fold of the forward map\n= maximum of the inverse map", xy=(-wf, -Df),
                xytext=(-1.0, 0.62), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(wf + 0.05, Df - 0.06, "F2", ha="left", va="top", fontsize=8, color=INK2)
    # how each map reads the curve
    wi = 0.25
    ax.annotate("", xy=(wi, wi ** 3 + (1 - c) * wi), xytext=(wi, -1.08),
                arrowprops=dict(arrowstyle="-|>", color=PAIR[1], lw=1.4))
    ax.text(wi + 0.05, -1.05, "inverse: w → D, one answer,\neven where the shape is unstable",
            color=INK2, fontsize=8, va="bottom")
    Dh = 0.2
    roots = np.sort(np.real([r for r in np.roots([1, 0, 1 - c, -Dh]) if abs(r.imag) < 1e-9]))
    ax.annotate("", xy=(roots[-1] - 0.02, Dh), xytext=(-1.45, Dh),
                arrowprops=dict(arrowstyle="-|>", color=PAIR[2], lw=1.4))
    ax.plot(roots, np.full(3, Dh), "o", mfc="white", mec=PAIR[2], ms=6, mew=1.5, zorder=6)
    ax.text(0.0, Dh + 0.035, "forward: D → w, three answers", color=INK2, fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("w  (wrench on the payload; here also the shape)")
    ax.set_ylabel("D  (tension difference)")
    ax.set_title("One curve, read two ways", loc="left")
    ax.set_xlim(-1.5, 1.75)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(handles=[Line2D([], [], color=STABLE, label="stable  (H > 0)"),
                       Line2D([], [], color=UNSTABLE, ls="--", label="unstable  (H < 0)")],
              loc="upper left", fontsize=8)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ww, cc = np.meshgrid(np.linspace(-1.35, 1.35, 140), np.linspace(0, 3, 90))
    DD = ww ** 3 + (1 - cc) * ww
    HH = 3 * ww ** 2 - cc + 1
    colors = np.where(HH[..., None] > 0, np.array([0.165, 0.47, 0.84, 0.75]),
                      np.array([0.89, 0.29, 0.28, 0.75]))
    ax.plot_surface(DD, cc, ww, facecolors=colors, rstride=2, cstride=2, linewidth=0,
                    antialiased=True, shade=True)
    cf = np.linspace(1, 3, 100)
    for s in (-1, 1):
        wf = s * np.sqrt((cf - 1) / 3)
        Df = wf ** 3 + (1 - cf) * wf
        ax.plot(Df, cf, wf, color=INK, lw=1.6)                      # fold curve
        ax.plot(Df, cf, np.full_like(cf, -1.4), color=INK, lw=1.0, ls=":")   # cusp on the floor
        ax.plot(np.full_like(cf, -2.6), cf, wf, color=PAIR[1], lw=1.2, ls=":")  # parabola on the wall
    ax.text(0.3, 3.0, -1.4, "cusp (floor): D, c with\nthree answers", fontsize=7, color=INK2)
    ax.text(-2.6, 3.0, 1.0, "wall: extrema of\nthe inverse map", fontsize=7, color=INK2)
    ax.set_xlabel("D  (tension diff.)")
    ax.set_ylabel("c  (co-contraction)")
    ax.set_zlabel("w  (wrench / shape)")
    ax.set_xlim(-2.6, 2.6)
    ax.set_zlim(-1.4, 1.4)
    ax.view_init(elev=20, azim=-62)
    ax.set_title("The cusp surface: the same equilibria over (D, c)", loc="left")
    fig.text(0.01, -0.02, "Toy model: U = q⁴/4 − c q²/2 + (q − X)²/2 − D q, X = 0, so w = q and "
             "H = 3q² − c + 1. Inverse (w, c) → D is single valued everywhere; forward (D, c) → w "
             "folds on the black curve, where H = 0.", fontsize=8, color=INK2)
    save(fig, "maps_toy.png")


# ------------------------------------------------------------------ S-curves

def scurves():
    a = np.load(DATA / "scurve_angle.npz")
    b = np.load(DATA / "scurve_tension.npz")
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 0.85, 0.75], hspace=0.45, wspace=0.22)

    # (a) along the payload angle; the bowed-up branch is the mirror (D = 0 is symmetric)
    ax = fig.add_subplot(gs[0, 0])
    p, mg = a["p"], a["margin"]
    mu = a["shape"][:, a["shape"].shape[1] // 2, 1]          # lateral deflection at mid-length
    keep = mg > -5
    st = mg > 0
    order = np.argsort(p[st])
    ps, mus = p[st][order], mu[st][order]
    stability_line(ax, p[keep], mu[keep], mg[keep])
    ax.plot(-ps, -mus, color=STABLE, alpha=0.5)
    i = folds(p, mg)[0]
    pf, muf = p[i], mu[i]
    mu_dest = -np.interp(-pf, ps, mus)
    ax.plot(pf, muf, "o", color=INK, ms=6, zorder=5)
    ax.plot(-pf, -muf, "o", color=INK, ms=6, zorder=5, alpha=0.5)
    ax.annotate("", xy=(pf, mu_dest), xytext=(pf, muf),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, ls=(0, (4, 3))))
    ax.text(pf - 0.03, 0.5 * (muf + mu_dest), "snap", ha="right", fontsize=8.5)
    ax.text(-pf + 0.03, -0.5 * (muf + mu_dest), "snap", ha="left", fontsize=8.5, color=INK2)
    ax.annotate("", xy=(-pf, -mu_dest), xytext=(-pf, -muf),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0, ls=(0, (4, 3))))
    ax.axvspan(pf, -pf, color=MUTED, alpha=0.08, lw=0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.75, 0.75)
    ax.text(0, 0.66, "bistable", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("payload angle φ_X (rad); payload position held at (2.9, 0)")
    ax.set_ylabel("y at mid-length (s = 1.5): the bow")
    ax.set_title("(a) Forward map along the payload angle, tensions fixed (D = 0)", loc="left")
    ax.legend(handles=[Line2D([], [], color=STABLE, label="stable, bowed down"),
                       Line2D([], [], color=STABLE, alpha=0.5, label="stable, bowed up (mirror)"),
                       Line2D([], [], color=UNSTABLE, ls="--", label="unstable"),
                       Line2D([], [], marker="o", color=INK, ls="", label="fold")],
              fontsize=7.5, loc="lower left")
    axm = fig.add_subplot(gs[1, 0], sharex=ax)
    axm.plot(p[keep], mg[keep], color=INK2, lw=1.3)
    axm.plot(-ps, mg[st][order], color=INK2, lw=1.3, alpha=0.4)
    axm.axhline(0, color=INK, lw=0.8)
    axm.axvline(pf, color=MUTED, lw=0.8, ls=":")
    axm.set_ylabel("margin λ_min")
    axm.set_xlabel("φ_X (rad)")

    # (b) along D1, with the branch the arm snaps to
    ax = fig.add_subplot(gs[0, 1])
    lim = float(b["D_lim"][0])
    p, fx, mg = b["p"], b["w"][:, 0], b["margin"]
    p2, fx2, mg2 = b["b2_p"], b["b2_w"][:, 0], b["b2_margin"]
    ax.axhspan(-lim, lim, color=STABLE, alpha=0.05, lw=0)
    for sgn in (-1, 1):
        ax.axhline(sgn * lim, color=MUTED, lw=0.8, ls=":")
    ax.text(6.4, lim + 0.12, "tension limit", ha="right", fontsize=7.5, color=INK2)
    stability_line(ax, fx, p, mg)
    ax.plot(fx2, p2, color=STABLE, alpha=0.5)
    i = folds(p, mg)[0]
    Df, fxf = float(b["D_fold"]), fx[i]
    fx_dest = np.interp(Df, p2, fx2)
    ax.plot(fxf, Df, "o", color=INK, ms=6, zorder=5)
    ax.annotate("", xy=(fx_dest, Df), xytext=(fxf, Df),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, ls=(0, (4, 3))))
    ax.text(0.5 * (fxf + fx_dest), Df + 0.2, f"snap at D₁ = {Df:.2f}", ha="center", fontsize=8.5)
    ax.set_xlabel("f_x  (axial force the arm applies to the payload)")
    ax.set_ylabel("D₁  (tension difference of pair 1)")
    ax.set_title("(b) Forward map along D₁; payload held at (2.9, 0, −0.40)", loc="left")
    ax.legend(handles=[Line2D([], [], color=STABLE, label="stable, branch it starts on"),
                       Line2D([], [], color=STABLE, alpha=0.5, label="stable, branch it snaps to"),
                       Line2D([], [], color=UNSTABLE, ls="--", label="unstable")],
              fontsize=7.5, loc="lower left")
    axm = fig.add_subplot(gs[1, 1], sharex=ax)
    axm.plot(fx, mg, color=INK2, lw=1.3)
    axm.plot(fx2, mg2, color=INK2, lw=1.3, alpha=0.4)
    axm.axhline(0, color=INK, lw=0.8)
    axm.axvline(fxf, color=MUTED, lw=0.8, ls=":")
    axm.set_ylabel("margin λ_min")
    axm.set_xlabel("f_x")

    # shapes
    ia = int(np.argmin(np.where(a["margin"] > 0, np.abs(a["p"]), np.inf)))
    iu = int(np.argmin(np.where(keep & (a["margin"] < 0), np.abs(a["margin"] + 2.5), np.inf)))
    shapes_a = [(a["theta"][ia], True, "bowed down, φ_X = 0"),
                (a["theta"][iu], False, f"unstable, φ_X = {a['p'][iu]:.2f}"),
                (-a["theta"][ia], True, "bowed up (mirror), φ_X = 0")]
    i0 = int(np.argmin(np.where(b["margin"] > 0, np.abs(b["p"]), np.inf)))
    ifold = folds(b["p"], b["margin"])[0]
    j = int(np.argmin(np.abs(b["b2_p"] - float(b["D_past"]))))
    shapes_b = [(b["theta"][i0], True, f"start, D₁ = {b['p'][i0]:.2f}"),
                (b["theta"][ifold], True, f"at the fold, D₁ = {b['p'][ifold]:.2f}"),
                (b["b2_theta"][j], True, f"after the snap, D₁ = {b['b2_p'][j]:.2f}")]
    for col, shapes, title in ((0, shapes_a, "(a)"), (1, shapes_b, "(b)")):
        axs = fig.add_subplot(gs[2, col])
        for n, (th, st, label) in enumerate(shapes):
            off = np.array([n * 4.2, 0.0])
            pos, _ = ROD.nodes(th)
            disks = ROD.disks(th)
            for (k, s_, ja, jb) in ROD.chords:
                ga, gb = ROD.hole(disks[ja], k, s_) + off, ROD.hole(disks[jb], k, s_) + off
                axs.plot([ga[0], gb[0]], [ga[1], gb[1]], color=PAIR[k], lw=0.7)
            axs.plot(pos[:, 0] + off[0], pos[:, 1], color=STABLE if st else UNSTABLE, lw=1.8,
                     ls="-" if st else "--")
            for P in disks:
                nrm = np.array([-np.sin(P[2]), np.cos(P[2])])
                seg = np.column_stack([P[:2] - 0.28 * nrm, P[:2] + 0.28 * nrm])
                axs.plot(seg[0] + off[0], seg[1], color=MUTED, lw=2)
            axs.text(off[0] + 1.45, -0.95, label, ha="center", fontsize=7.5, color=INK2)
        axs.set_aspect("equal")
        axs.set_ylim(-1.05, 0.75)
        axs.axis("off")
        axs.set_title(f"{title} arm shapes (tendon pairs 1–3: blue, orange, green)", loc="left", fontsize=9)
    fig.text(0.01, 0.06, "n = 3 arm, l = B = 1 per segment, C = (3, 2.5, 2), hole offsets d = (0.15, 0.2, 0.25). "
             "The payload is held; the margin is the smallest eigenvalue of the rod's second variation with "
             "the payload fixed.\nEvery fold sits where the margin crosses zero. All stable states drawn keep "
             "the rod clear of its tendons (contact margin ≥ 0.07).", fontsize=8, color=INK2)
    save(fig, "maps_scurves.png")


# ------------------------------------------------------------------ forward map

def forward():
    from matplotlib.colors import LogNorm
    f = np.load(DATA / "forward.npz")
    xs, ys, w, mg, st = f["xs"], f["ys"], f["w"], f["margin"], f["status"]
    Xr = f["X_ref"]
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    solved = st == 1
    clear = solved & (f["contact"] > 0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"wspace": 0.28})
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    def common(ax):
        ax.contourf(XX, YY, (solved & ~clear).astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["////"])
        ax.contour(XX, YY, clear.astype(float), levels=[0.5], colors=INK, linewidths=1.2)
        ax.contour(XX, YY, solved.astype(float), levels=[0.5], colors=UNSTABLE, linewidths=1.4)
        ax.add_patch(plt.Circle((0, 0), 3, fill=False, ec=MUTED, ls=":", lw=1))
        ax.plot(*Xr[:2], "o", color=PAIR[1], ms=7, zorder=5, mec="white", mew=1)
        ax.plot(0, 0, "s", color=INK, ms=5)
        ax.set_aspect("equal")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_xlabel("payload x")

    ax = axes[0]
    fx, fy = w[..., 0], w[..., 1]
    mag = np.hypot(fx, fy)
    im = ax.imshow(np.where(solved, mag, np.nan).T, origin="lower", extent=extent, cmap=SEQUENTIAL,
                   norm=LogNorm(0.1, np.nanpercentile(mag, 99)))
    fig.colorbar(im, ax=ax, shrink=0.75, label="|f|, force the arm applies (log scale)", extend="both")
    s = slice(3, None, 7)
    u = fx / np.maximum(mag, 1e-9) * np.log1p(mag)
    v = fy / np.maximum(mag, 1e-9) * np.log1p(mag)
    ax.quiver(XX[s, s], YY[s, s], u[s, s], v[s, s], color="white", edgecolor=INK, linewidth=0.3,
              width=0.004, scale=55)
    cs = ax.contour(XX, YY, np.where(solved, mg, np.nan), levels=[0.5, 2], colors=UNSTABLE,
                    linewidths=0.9, linestyles="--")
    ax.clabel(cs, fontsize=6.5, fmt="λ=%g")
    common(ax)
    ax.annotate("not reached: no stable state\ncontinues here (fold)", xy=(0.56, 0.24), xytext=(0.75, -0.45),
                fontsize=7.5, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(Xr[0] + 0.08, Xr[1] + 0.1, "rest (w = 0)", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.set_ylabel("payload y")
    ax.set_title("(a) Force f on the payload: arrows (∝ log(1+|f|)), colour |f|;\n"
                 "dashed: stability margin 0.5 and 2", loc="left")

    ax = axes[1]
    mu = np.where(solved, w[..., 2], np.nan)
    lim = np.nanpercentile(np.abs(mu), 95)
    im = ax.imshow(mu.T, origin="lower", extent=extent, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim))
    fig.colorbar(im, ax=ax, shrink=0.75, label="moment μ on the payload", extend="both")
    cs = ax.contour(XX, YY, np.where(solved, np.hypot(w[..., 0], w[..., 1]), np.nan), levels=[1, 3, 10, 30], colors=INK2, linewidths=0.7)
    ax.clabel(cs, fontsize=6.5, fmt="|f|=%g")
    common(ax)
    ax.set_title("(b) Moment μ on the payload\n(grey contours: |f|)", loc="left")

    ax = axes[2]
    for key, pos in zip(f["shape_keys"], f["shapes"]):
        i, j = map(int, key.split("_"))
        if clear[i, j]:
            draw_shape(ax, pos, color=STABLE, lw=1.1, alpha=0.75)
            ax.plot(*pos[-1], "o", ms=3, color=STABLE)
    draw_shape(ax, f["shape_ref"], color=INK, lw=2.2)
    ax.contour(XX, YY, clear.astype(float), levels=[0.5], colors=INK, linewidths=1.0)
    ax.add_patch(plt.Circle((0, 0), 3, fill=False, ec=MUTED, ls=":", lw=1))
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("(c) Held shapes inside the contact-free region\n(rest shape black)", loc="left")
    handles = [Line2D([], [], color=INK, lw=1.2, label="edge of the contact-free region"),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=INK2, hatch="////", label="a tendon crosses the rod"),
               Line2D([], [], color=UNSTABLE, lw=1.4, label="edge of the stable states reached from rest"),
               Line2D([], [], color=MUTED, ls=":", label="arm length 3")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.45, -0.04))
    D = tuple(float(x) for x in f["D"])
    C_ = tuple(float(x) for x in f["C"])
    fig.text(0.01, -0.09, f"Forward map (X, T) → w with the tensions fixed at D = {D}, C = {C_}, and the payload "
             f"angle held at {Xr[2]:.3f} rad; payload position varied. Solved outward from the rest pose, keeping "
             "only stable states.", fontsize=8, color=INK2)
    save(fig, "maps_forward.png")


def cusp():
    c = np.load(DATA / "cusp.npz")
    labels, D1s, lim = c["labels"], c["D1"], float(c["D_lim"][0])
    fig = plt.figure(figsize=(14.5, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    fold_pts = {"down": [], "up": []}
    for i, (lab, D1) in enumerate(zip(labels, D1s)):
        p, bow, mg = c[f"c{i}_p"], c[f"c{i}_bow"], c[f"c{i}_margin"]
        st = mg > 0
        alpha = 1.0 if lab == "down" else 0.55
        for flag, col, ls in ((True, STABLE, "-"), (False, UNSTABLE, "--")):
            yy = np.where(st == flag, bow, np.nan)
            edge = np.r_[False, st[1:] != st[:-1]]
            yy = np.where(edge | np.r_[edge[1:], False], bow, yy)
            ax.plot(p, np.full_like(p, D1), yy, color=col, ls=ls, lw=1.3, alpha=alpha)
        for j in folds(p, mg):
            fold_pts[lab].append((p[j], D1, bow[j]))
            ax.scatter(p[j], D1, bow[j], color=INK, s=12, depthshade=False)
    for lab, pts in fold_pts.items():
        pts = np.array(sorted(pts, key=lambda t: t[1]))
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=INK, lw=1.6)
            ax.plot(pts[:, 0], pts[:, 1], np.full(len(pts), -0.55), color=INK, lw=1.0, ls=":")
            ax2.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.8, alpha=1.0 if lab == "down" else 0.55,
                     marker="o", ms=3, label=f"fold of the {'downward' if lab == 'down' else 'upward'} bow")
    ax.set_xlabel("payload angle φ_X")
    ax.set_ylabel("D₁")
    ax.set_zlabel("bow (y at mid-length)")
    ax.set_zlim(-0.55, 0.55)
    ax.view_init(elev=24, azim=-58)
    ax.set_title("(a) Equilibria of the held arm over (φ_X, D₁): the two bows,\n"
                 "joined by unstable states; black: the fold curves", loc="left", fontsize=9.5)
    # (b) the parameter plane
    dn = np.array(sorted(fold_pts["down"], key=lambda t: t[1]))
    up = np.array(sorted(fold_pts["up"], key=lambda t: t[1]))
    if len(dn) and len(up):
        Dg = np.linspace(max(dn[:, 1].min(), up[:, 1].min()), min(dn[:, 1].max(), up[:, 1].max()), 100)
        a, b = np.interp(Dg, dn[:, 1], dn[:, 0]), np.interp(Dg, up[:, 1], up[:, 0])
        ax2.fill_betweenx(Dg, np.minimum(a, b), np.maximum(a, b), color=MUTED, alpha=0.15, lw=0,
                          label="bistable: both bows stable")
    ax2.axhspan(-lim, lim, color=STABLE, alpha=0.04, lw=0)
    ax2.plot(-0.40, 0.76, "s", color=PAIR[1], ms=7, label="snap of Fig. maps_scurves (b)")
    ax2.axvline(-0.40, color=PAIR[1], lw=0.8, ls="--")
    ax2.set_xlabel("payload angle φ_X (rad), position held at (2.9, 0)")
    ax2.set_ylabel("D₁  (tension difference of pair 1)")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-lim * 1.05, lim * 1.05)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_title("(b) Fold curves in the plane of (payload angle, tension).\nCrossing one makes the arm snap; "
                  "the forward map is two-valued between them.", loc="left", fontsize=9.5)
    fig.text(0.01, -0.01, "Each line in (a) is one sweep of the payload angle at fixed D₁ (D₂ = D₃ = 0, C = (3, 2.5, 2)), "
             "started from the downward (dark) or upward (light) bow at φ_X = 0. Stable solid blue, unstable dashed red.",
             fontsize=8, color=INK2)
    save(fig, "maps_cusp.png")


# ------------------------------------------------------------------ inverse map

def inverse():
    v = np.load(DATA / "inverse.npz")
    fs, D, mg, lim = v["fs"], v["D"], v["margin"], v["D_lim"]
    FX, FY = np.meshgrid(fs, fs, indexing="ij")
    extent = (fs[0], fs[-1], fs[0], fs[-1])
    feasible = np.all(np.abs(D) <= lim, axis=-1) & np.isfinite(D[..., 0])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.9), gridspec_kw={"wspace": 0.3})
    for k in range(3):
        ax = axes[k]
        Dk = D[..., k]
        L = np.nanpercentile(np.abs(Dk), 97)
        im = ax.imshow(Dk.T, origin="lower", extent=extent, cmap=DIVERGING, aspect="equal",
                       norm=TwoSlopeNorm(0, -L, L))
        fig.colorbar(im, ax=ax, shrink=0.78, label=f"D{k + 1}")
        ax.contour(FX, FY, np.abs(Dk) - lim[k], levels=[0], colors=PAIR[k], linewidths=1.4)
        _overlay(ax, FX, FY, mg, feasible)
        ax.set_title(f"D{k + 1}, tension difference of pair {k + 1}", loc="left")
        ax.set_xlabel("force f_x on the payload")
        if k == 0:
            ax.set_ylabel("force f_y")
    ax = axes[3]
    L = np.nanpercentile(np.abs(mg), 97)
    im = ax.imshow(mg.T, origin="lower", extent=extent, cmap=DIVERGING.reversed(), aspect="equal",
                   norm=TwoSlopeNorm(0, -L, L))
    fig.colorbar(im, ax=ax, shrink=0.78, label="stability margin λ_min")
    ax.contour(FX, FY, v["detS"], levels=[0], colors=PAIR[1], linewidths=5, alpha=0.6)
    _overlay(ax, FX, FY, mg, feasible)
    ax.set_title("Margin of the commanded shape\n(orange: det ∂D/∂w = 0)", loc="left")
    ax.set_xlabel("force f_x")
    handles = [Line2D([], [], color=INK, lw=1.6, label="margin = 0: stability lost; the map is smooth across it"),
               Line2D([], [], color=PAIR[1], lw=5, alpha=0.6, label="det ∂D/∂w = 0 (same curve)"),
               Line2D([], [], color=INK, lw=1.0, label="closed lens: feasible, every tension ≥ T_min"),
               Line2D([], [], color=MUTED, lw=1.2, ls=":", label="edge of the solved region (Newton fails)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.06))
    Xr = v["X_ref"]
    C_ = tuple(float(x) for x in v["C"])
    fig.suptitle(f"Inverse map (X, w) → D at the payload pose X = ({Xr[0]:.2f}, {Xr[1]:.2f}, {Xr[2]:.2f}) "
                 f"with moment μ = 0 and pair totals C = {C_}; coloured lines: |D_k| = C_k − 2T_min",
                 x=0.01, y=0.86, ha="left", fontsize=9.5, color=INK)
    save(fig, "maps_inverse.png")

    # 3D: the force plane mapped into tension space, with the box of tension limits
    fig = plt.figure(figsize=(13, 6.2))
    bounds = np.array([16.0, 12.0, 6.0])
    inside = np.all(np.abs(D) <= bounds, axis=-1)
    def clip(pts):
        pts = np.array(pts, float)
        pts[~np.all(np.abs(pts) <= bounds, axis=-1)] = np.nan
        return pts

    for n, (elev, azim) in enumerate(((22, -55), (55, -100))):
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        Ds = [np.where(inside, D[..., k], np.nan) for k in range(3)]
        cols = np.where((mg > 0)[..., None], np.array([0.165, 0.47, 0.84, 0.55]),
                        np.array([0.89, 0.29, 0.28, 0.75]))
        cols[feasible] = np.array([0.04, 0.21, 0.42, 1.0])
        ax.plot_surface(*Ds, facecolors=cols, rstride=1, cstride=1, linewidth=0, shade=False)
        # margin = 0 on the surface
        cs = plt.figure().add_subplot().contour(FX, FY, mg, levels=[0])
        plt.close()
        for path in cs.allsegs[0]:
            ij = [(np.argmin(np.abs(fs - a)), np.argmin(np.abs(fs - b))) for a, b in path]
            pts = clip([D[i, j] for i, j in ij])
            ax.plot(*pts.T, color=INK, lw=2)
        # tension-limit box
        L = lim
        for a in (-1, 1):
            for b in (-1, 1):
                ax.plot([-L[0], L[0]], [a * L[1]] * 2, [b * L[2]] * 2, color=MUTED, lw=1)
                ax.plot([a * L[0]] * 2, [-L[1], L[1]], [b * L[2]] * 2, color=MUTED, lw=1)
                ax.plot([a * L[0]] * 2, [b * L[1]] * 2, [-L[2], L[2]], color=MUTED, lw=1)
        # force directions: images of the f_x and f_y axes
        c0 = len(fs) // 2
        ax.plot(*clip(D[:, c0]).T, color=PAIR[1], lw=1.2, ls="--")
        ax.plot(*clip(D[c0, :]).T, color=PAIR[2], lw=1.2, ls="--")
        ax.scatter(*v["D_ref"], color=PAIR[1], s=30, depthshade=False)
        ax.set_xlim(-bounds[0], bounds[0])
        ax.set_ylim(-bounds[1], bounds[1])
        ax.set_zlim(-bounds[2], bounds[2])
        ax.set_xlabel("D₁")
        ax.set_ylabel("D₂")
        ax.set_zlabel("D₃")
        ax.view_init(elev=elev, azim=azim)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=(0.04, 0.21, 0.42), label="achievable: inside the tension box"),
               plt.Rectangle((0, 0), 1, 1, fc=(0.165, 0.47, 0.84, 0.55), label="stable, tensions out of limits"),
               plt.Rectangle((0, 0), 1, 1, fc=(0.89, 0.29, 0.28, 0.75), label="unstable shape commanded"),
               Line2D([], [], color=INK, lw=2, label="margin = 0"),
               Line2D([], [], color=MUTED, lw=1, label="box |D_k| ≤ C_k − 2T_min"),
               Line2D([], [], color=PAIR[1], ls="--", label="image of the f_x axis"),
               Line2D([], [], color=PAIR[2], ls="--", label="image of the f_y axis")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Inverse map: the plane of forces (f_x, f_y, μ = 0) carried into tension space. "
                 "Smooth through the stability loss; only a small patch fits inside the box.",
                 x=0.02, ha="left", fontsize=10)
    save(fig, "maps_inverse_3d.png")


def _overlay(ax, FX, FY, mg, feasible):
    ax.contour(FX, FY, mg, levels=[0], colors=INK, linewidths=1.6)
    ax.contour(FX, FY, feasible.astype(float), levels=[0.5], colors=INK, linewidths=1.0)
    ax.contour(FX, FY, np.isfinite(mg).astype(float), levels=[0.5], colors=MUTED, linewidths=1.2,
               linestyles=":")


# ------------------------------------------------------------------ workspace

def workspace():
    s = np.load(DATA / "workspace.npz")
    tip, contact, mg = s["tip"], s["contact"], s["margin"]
    ok = np.isfinite(mg)
    clear = ok & (contact > 0)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1)
    t = tip[ok]
    ax.scatter(tip[ok & ~clear][:, 0], tip[ok & ~clear][:, 1], s=8, facecolors="none",
               edgecolors=MUTED, linewidths=0.6, label="a tendon touches the rod")
    sc = ax.scatter(tip[clear][:, 0], tip[clear][:, 1], c=tip[clear][:, 2], s=9, cmap=DIVERGING,
                    norm=TwoSlopeNorm(0, -2.6, 2.6), linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="payload angle φ (rad)")
    for pos in s["shapes"]:
        draw_shape(ax, pos, color=INK, lw=1.4, alpha=0.8)
    ax.add_patch(plt.Circle((0, 0), 3, fill=False, ec=MUTED, ls=":", lw=1))
    ax.set_aspect("equal")
    ax.set_xlabel("payload x")
    ax.set_ylabel("payload y")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("(a) Rest map D → X (w = 0): payload positions over the tension box\n"
                 "colour: payload angle; dotted circle: arm length 3", loc="left")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(tip[clear][:, 0], tip[clear][:, 1], tip[clear][:, 2], c=tip[clear][:, 2], cmap=DIVERGING,
               norm=TwoSlopeNorm(0, -2.6, 2.6), s=5, depthshade=True)
    ax.scatter(tip[ok & ~clear][:, 0], tip[ok & ~clear][:, 1], tip[ok & ~clear][:, 2], color=MUTED, s=2,
               alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("φ")
    ax.view_init(elev=20, azim=-50)
    ax.set_title("(b) The same rest poses in (x, y, φ): a 3D patch of the pose space", loc="left")
    jumped = s["jump"] & ok
    ax = fig.axes[0]
    ax.plot(tip[jumped][:, 0], tip[jumped][:, 1], "x", color=INK, ms=6, mew=1.5, label="rest state after a snap")
    ax.legend(loc="lower left", fontsize=8)
    C_ = tuple(float(x) for x in s["C"])
    fig.text(0.01, -0.02, f"{ok.sum()} rest states over |D_k| ≤ C_k − 2T_min, C = {C_}, on a 15 × 15 × 11 grid; "
             f"at {int(jumped.sum())} grid steps the rest state snapped (x). Black: shapes at five tension settings.",
             fontsize=8, color=INK2)
    save(fig, "maps_workspace.png")


if __name__ == "__main__":
    jobs = {"toy": toy, "scurves": scurves, "cusp": cusp, "forward": forward, "inverse": inverse, "workspace": workspace}
    for name in (sys.argv[1:] or list(jobs)):
        jobs[name]()

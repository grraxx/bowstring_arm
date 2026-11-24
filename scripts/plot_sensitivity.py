"""
plot_sensitivity.py
Task maps and parameter sensitivity of the n = 3 bowstring arm, from data/sensitivity_scan.npz
(sensitivity_scan.py). Every equilibrium of the scan is one configuration holding the payload;
the arm is usable for closed-loop control where its inner disks are stable (margin_I > 0, every
segment stable). Each map cell (h = 0.1) shows the best value over the configurations and
orientations that put the payload there (what a task planner could choose), except the
parameter maps, which show the median.

Controller and payload as the console defaults: M = diag(0.1, 0.1, 0.009), wn = (5, 5, 15),
Kp = wn^2, so the closed-loop stiffness against a static push is K_cl = M Kp + K_X (PD; the
integral term then removes the steady error, within its clamp).

Writes figures/sensitivity_tasks.png, figures/sensitivity_params.png,
figures/sensitivity_configs.png and data/sensitivity_report.md (numbers by region).

usage: python scripts/plot_sensitivity.py [--scan data/sensitivity_scan.npz]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#e3eefc", "#9ec5f4", "#5a9be8", "#2a78d6", "#184f95", "#0d366b"])
SEQ.set_bad("#f4f3f0")
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
})

M = np.array([0.1, 0.1, 0.009])
WN = np.array([5.0, 5.0, 15.0])
KP = M * WN ** 2
H = 0.1
PNAMES = {"d1": "hole offset d₁", "d2": "hole offset d₂", "d3": "hole offset d₃",
          "B1": "stiffness B₁", "B2": "stiffness B₂", "B3": "stiffness B₃", "Tmin": "slack tension Tmin"}


def per_sample(z, clear=None):
    """Task numbers at every usable equilibrium. clear="all": only those where the rod stays clear
    of every tendon (the straight-chord model physically valid); clear="taut": clear of the taut
    tendons, a slack one (T = Tmin) may rest on the rod (a mild model error)."""
    use = z["ok"] & z["seg_ok"] & (z["margin_I"] > 0)
    if clear == "all":
        use &= z["contact"] >= 0
    elif clear == "taut":
        use &= z["contact_taut"] >= 0
    KX, WD, dw = z["KX"][use], z["WD"][use], z["dw"][use]
    n = len(KX)
    s = 1 / np.sqrt(M)
    lam = np.linalg.eigvalsh(KX * s[None, :, None] * s[None, None, :])[:, 0]
    out = {"X": z["P"][use, -1], "P": z["P"][use], "D": z["D"][use],
           "ol_stable": lam > 0,
           "wn_req": np.sqrt(np.maximum(0.0, -lam)),
           "T_max": z["T_max"][use],
           "enc": 1.0 / z["sig_enc"][use, -1]}
    Kcl = KX + np.diag(KP)[None]
    Ccl = np.linalg.inv(Kcl)
    out["k_cl"] = 1.0 / np.linalg.eigvalsh(0.5 * (Ccl[:, :2, :2] + np.transpose(Ccl[:, :2, :2], (0, 2, 1))))[:, -1]
    with np.errstate(all="ignore"):
        Winv = np.linalg.pinv(WD)
        out["F_T"] = 1.0 / np.linalg.svd(Winv[:, :, :2], compute_uv=False)[:, 0]
        e = np.einsum("nij,npj->npi", Ccl, dw)                   # PD static error per param
        out["e_p"] = np.hypot(e[..., 0], e[..., 1])              # (n, 7)
        out["e_model"] = np.sqrt(np.nansum(out["e_p"] ** 2, axis=1))
        out["e_model"][np.isnan(out["e_p"]).all(1)] = np.nan
        ok = out["ol_stable"]
        dX = np.full((n, 3, 3), np.nan)
        dX[ok] = -np.linalg.solve(KX[ok], WD[ok])
        out["dXdD"] = np.linalg.svd(np.nan_to_num(dX[:, :2, :]), compute_uv=False)[:, 0]
        out["dXdD"][~ok] = np.nan
    return out


def cells(X):
    return np.floor(X[:, 0] / H).astype(int), np.floor(X[:, 1] / H).astype(int)


def grid_map(S, values, how):
    """Aggregate per (x, y) cell: 'min', 'max', 'median', 'range' (of angles, deg)."""
    i, j = cells(S["X"])
    i0, j0 = i.min(), j.min()
    G = np.full((i.max() - i0 + 1, j.max() - j0 + 1), np.nan)
    order = np.lexsort((j, i))
    keys = np.stack([i[order], j[order]], 1)
    bounds = np.r_[0, np.nonzero(np.any(np.diff(keys, axis=0), axis=1))[0] + 1, len(order)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        v = values[idx]
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        if how == "range":
            ang = np.sort(np.mod(v, 360.0))
            gaps = np.diff(np.r_[ang, ang[0] + 360.0])
            r = 360.0 - gaps.max() if len(ang) > 1 else 0.0
        else:
            r = {"min": np.min, "max": np.max, "median": np.median}[how](v)
        G[i[idx[0]] - i0, j[idx[0]] - j0] = r
    extent = (i0 * H, (i.max() + 1) * H, j0 * H, (j.max() + 1) * H)
    return G, extent


def configurations(S, dpsi=15.0, sep=0.15):
    """Distinct arm shapes per (x, y) cell and orientation bin (greedy clustering of the two
    inner disk positions, separated by more than sep); the map shows the most over the bins."""
    i, j = cells(S["X"])
    k = np.floor(np.degrees(S["X"][:, 2]) / dpsi).astype(int)
    groups = {}
    for n_, key in enumerate(zip(i, j, k)):
        groups.setdefault(key, []).append(n_)
    count, example = {}, {}
    for (a, b, _), idx in groups.items():
        reps = []
        for n_ in idx:
            q = S["P"][n_, 1:3, :2].ravel()
            if all(np.abs(q - S["P"][m, 1:3, :2].ravel()).max() > sep for m in reps):
                reps.append(n_)
        if len(reps) > count.get((a, b), 0):
            count[(a, b)], example[(a, b)] = len(reps), reps
    i0, j0 = i.min(), j.min()
    G = np.full((i.max() - i0 + 1, j.max() - j0 + 1), np.nan)
    for (a, b), c in count.items():
        G[a - i0, b - j0] = c
    return G, (i0 * H, (i.max() + 1) * H, j0 * H, (j.max() + 1) * H), example


def frame(ax, extent):
    ax.set_aspect("equal")
    ax.set_xlim(-2.1, 3.2)
    ax.set_ylim(-2.8, 2.8)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3 * np.cos(t), 3 * np.sin(t), color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.plot([0, 3], [0, 0], color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.plot([-0.12, -0.12], [-0.25, 0.25], color=INK, lw=3, solid_capstyle="butt")
    ax.tick_params(length=2)


def panel(fig, ax, G, extent, title, sub, norm, cbar_label, cmap=SEQ):
    im = ax.imshow(G.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    frame(ax, extent)
    ax.set_title(title, loc="left", fontweight="bold", pad=15)
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, va="bottom", ha="left", fontsize=7.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label(cbar_label, fontsize=7.5, color=INK2)
    cb.outline.set_visible(False)


def lognorm(G, lo_pct=2, hi_pct=98):
    v = G[np.isfinite(G) & (G > 0)]
    return LogNorm(np.percentile(v, lo_pct), np.percentile(v, hi_pct))


def region_table(S, keys):
    X = S["X"]
    r, phi = np.hypot(X[:, 0], X[:, 1]), np.degrees(np.abs(np.arctan2(X[:, 1], X[:, 0])))
    regs = [("near, r < 1", r < 1), ("middle, 1 ≤ r < 2", (r >= 1) & (r < 2)), ("far, r ≥ 2", r >= 2),
            ("ahead, |bearing| < 45°", phi < 45), ("to the side, 45–135°", (phi >= 45) & (phi < 135)),
            ("behind, > 135°", phi >= 135)]
    lines = ["| region | samples | " + " | ".join(k for k, _, _, _ in keys) + " |",
             "|---|---|" + "---|" * len(keys)]
    for name, sel in regs:
        row = [name, str(int(sel.sum()))]
        for _, f, fmt, agg in keys:
            v = f(S)[sel]
            v = v[np.isfinite(v)]
            row.append(fmt(agg(v)) if len(v) else "–")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=str(ROOT / "data/sensitivity_scan.npz"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--clear", choices=("all", "taut"), default=None,
                    help="only configurations with the rod clear of every tendon (all) or of the "
                         "taut ones (taut); needs a scan with the contact fields")
    args = ap.parse_args()
    z = dict(np.load(args.scan))
    params = [str(p) for p in z["params"]]
    S = per_sample(z, args.clear)
    n_all, n_use = int(z["ok"].sum()), len(S["X"])
    print(f"{len(z['ok'])} samples, {n_all} equilibria evaluated, {n_use} usable "
          f"({S['ol_stable'].sum()} open-loop stable)")

    # ---------------------------------------------------------------- task maps
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 13.2))
    axs = axs.ravel()
    i, j = cells(S["X"])
    reach, ext = grid_map(S, np.where(S["ol_stable"], 2.0, 1.0), "max")
    cm2 = ListedColormap(["#9ec5f4", "#184f95"])
    cm2.set_bad("#f4f3f0")
    panel(fig, axs[0], reach, ext, "Where it can hold the payload",
          "light: only with feedback · dark: also without", Normalize(0.5, 2.5), "", cmap=cm2)
    axs[0].images[0].colorbar.remove()
    G, ext = grid_map(S, np.where(S["ol_stable"], np.nan, S["wn_req"]), "min")
    G2, _ = grid_map(S, np.where(S["ol_stable"], 0.0, np.nan), "min")
    G = np.where(G2 == 0, np.nan, G)
    panel(fig, axs[1], G, ext, "Feedback bandwidth needed",
          "minimum ωn to hold (blank = stable open loop)", Normalize(0, np.nanpercentile(G, 98)), "rad/s")
    G, ext = grid_map(S, np.degrees(S["X"][:, 2]), "range")
    panel(fig, axs[2], G, ext, "Reorientation", "range of payload angle ψ reachable here",
          Normalize(0, np.nanpercentile(G, 99)), "degrees")
    G, ext = grid_map(S, S["k_cl"], "max")
    panel(fig, axs[3], G, ext, "Resisting a push", "closed-loop stiffness, weakest direction (PD)",
          lognorm(G), "EI₀/L³")
    G, ext = grid_map(S, S["F_T"], "max")
    panel(fig, axs[4], G, ext, "Pushing", "force per unit tension, weakest direction", lognorm(G), "–")
    G, ext = grid_map(S, S["T_max"], "min")
    panel(fig, axs[5], G, ext, "Cost of holding", "largest tendon tension", lognorm(G), "EI₀/L²")
    G, ext = grid_map(S, S["e_model"], "min")
    panel(fig, axs[6], G, ext, "Precision with a 10 % wrong model",
          "PD position error, all 7 parameters off by 10 %", lognorm(G), "L")
    G, ext = grid_map(S, S["enc"], "min")
    panel(fig, axs[7], G, ext, "Tendon-length sensing", "pose error per unit length error",
          lognorm(G), "–")
    Gc, ext_c, examples = configurations(S)
    panel(fig, axs[8], Gc, ext_c, "Configurations", "distinct arm shapes for the same pose (±15°)",
          Normalize(0.5, max(2.5, np.nanmax(Gc) + 0.5)), "shapes")
    for ax in axs:
        ax.set_xlabel("x (L)")
        ax.set_ylabel("y (L)")
    fig.tight_layout(h_pad=2.2)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"sensitivity_tasks{args.tag}.png")
    plt.close(fig)

    # ---------------------------------------------------------------- parameter maps
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.6))
    axs = axs.ravel()
    allv = S["e_p"][np.isfinite(S["e_p"])]
    norm = LogNorm(np.percentile(allv, 2), np.percentile(allv, 98))
    for k, p in enumerate(params):
        G, ext = grid_map(S, S["e_p"][:, k], "median")
        panel(fig, axs[k], G, ext, PNAMES[p], "PD error, model value 10 % off (median)",
              norm, "L")
        axs[k].set_xlabel("x (L)")
        axs[k].set_ylabel("y (L)")
    ax = axs[7]
    med = [np.nanmedian(S["e_p"][:, k]) for k in range(len(params))]
    q75 = [np.nanpercentile(S["e_p"][:, k], 75) for k in range(len(params))]
    q25 = [np.nanpercentile(S["e_p"][:, k], 25) for k in range(len(params))]
    order = np.argsort(med)[::-1]
    y = np.arange(len(params))
    ax.barh(y, [med[k] for k in order], color="#2a78d6", height=0.55)
    ax.errorbar([med[k] for k in order], y, xerr=[[med[k] - q25[k] for k in order], [q75[k] - med[k] for k in order]],
                fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    ax.set_yticks(y, [PNAMES[params[k]] for k in order])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_title("Over the whole workspace", loc="left", fontweight="bold", pad=15)
    ax.text(0, 1.015, "median and quartiles of the position error, L", transform=ax.transAxes,
            fontsize=7.5, color=INK2, va="bottom")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(FIG / f"sensitivity_params{args.tag}.png")
    plt.close(fig)

    # ---------------------------------------------------------------- configuration examples
    many = sorted(examples.items(), key=lambda kv: -len(kv[1]))[:4]
    fig, axs = plt.subplots(1, len(many), figsize=(4 * len(many), 4))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#8f5cd6", "#b08a14", "#d6487f"]
    for ax, ((a, b), reps) in zip(np.atleast_1d(axs), many):
        for c, n_ in enumerate(reps[:6]):
            P = S["P"][n_]
            ax.plot(P[:, 0], P[:, 1], "-o", color=cols[c], ms=3, lw=1.6,
                    label=f"D = ({', '.join(f'{d:.1f}' for d in S['D'][n_])}) · ωn req {S['wn_req'][n_]:.1f}")
        frame(ax, None)
        ax.set_title(f"x ≈ {(a + .5) * H:.2f}, y ≈ {(b + .5) * H:.2f}: {len(reps)} shapes", loc="left",
                     fontweight="bold", pad=15)
        ax.text(0, 1.015, "disk positions (segment chords), same payload angle ±15°", transform=ax.transAxes,
                fontsize=7.5, color=INK2, va="bottom")
        ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / f"sensitivity_configs{args.tag}.png")
    plt.close(fig)

    # ---------------------------------------------------------------- report
    md = np.median
    keys = [("open-loop stable", lambda s: s["ol_stable"].astype(float), lambda v: f"{100 * v:.0f} %", np.mean),
            ("ωn needed where unstable (rad/s)", lambda s: np.where(s["ol_stable"], np.nan, s["wn_req"]),
             lambda v: f"{v:.1f}", md),
            ("push stiffness (EI₀/L³)", lambda s: s["k_cl"], lambda v: f"{v:.2g}", md),
            ("force per tension", lambda s: s["F_T"], lambda v: f"{v:.2g}", md),
            ("max tension (EI₀/L²)", lambda s: s["T_max"], lambda v: f"{v:.2g}", md),
            ("error, 10 % wrong model (L)", lambda s: s["e_model"], lambda v: f"{v:.2g}", md),
            ("pose error per length error", lambda s: s["enc"], lambda v: f"{v:.2g}", md)]
    rep = [f"# Sensitivity scan report{args.tag}", "", f"Scan: {args.scan}."
           + {None: "", "all": " Only configurations with the rod clear of every tendon (the model is physically valid).",
              "taut": " Only configurations with the rod clear of the taut tendons (a slack tendon may rest on it)."}[args.clear], "",
           f"{len(z['ok'])} tension samples, {n_all} equilibria, {n_use} usable (arm stable), "
           f"{int(S['ol_stable'].sum())} open-loop stable. Medians over the usable equilibria by region "
           "(r = distance of the payload from the base, bearing from the arm's straight direction); "
           "the open-loop column is the fraction, the ωn column the median over the open-loop "
           "unstable ones.", "",
           region_table(S, keys), "", "Parameter sensitivity (PD position error, L, per 10 % model error):", "",
           "| parameter | median | 75th pct | 95th pct |", "|---|---|---|---|"]
    for k in np.argsort(med)[::-1]:
        v = S["e_p"][:, k]
        rep.append(f"| {PNAMES[params[k]]} | {np.nanmedian(v):.2g} | {np.nanpercentile(v, 75):.2g} | "
                   f"{np.nanpercentile(v, 95):.2g} |")
    (ROOT / "data" / f"sensitivity_report{args.tag}.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()

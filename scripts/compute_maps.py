"""
compute_maps.py
Data for the figures of the forward and inverse maps of the n = 3 bowstring arm
(direct-rod model, bowstring/rod.py). Writes data/maps/<name>.npz.

usage: python scripts/compute_maps.py {scurve_angle,scurve_tension,forward,inverse,workspace,all}

Parameters (units l = B = 1 per segment): hole offsets d = (0.15, 0.20, 0.25),
pair totals C = (3, 2.5, 2), T_min = 0.1, so |D_k| <= C_k - 2 T_min; N = 40
elements per segment. Reference tensions D_ref = (0.8, 0.4, 0.2): the arm at
rest under them puts the payload at X_ref.
"""

import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bowstring.rod import BowstringRod            # noqa: E402
from bowstring.continuation import Branch         # noqa: E402

OUT = ROOT / "data" / "maps"
OFFSETS = (0.15, 0.20, 0.25)
C = np.array([3.0, 2.5, 2.0])
T_MIN = 0.1
D_LIM = C - 2 * T_MIN
D_REF = np.array([0.8, 0.4, 0.2])
N = 40
X_COMP = np.array([2.9, 0.0, 0.0])        # payload pulled in along the axis: the arm bows
PHI_NEAR_FOLD = -0.40                     # the angle sweep folds at -0.43 (bowed down)


def make_rod():
    return BowstringRod(n=3, offsets=OFFSETS, N=N)


def reference(rod):
    theta, margin, ok = rod.rest(rod.tensions(C, D_REF), np.zeros(rod.m))
    assert ok
    return theta, rod.tip(theta)


def compressed_state(rod, D=(0.0, 0.0, 0.0)):
    """Stable held state at X_COMP, reached by moving the payload in from rest
    (the arm buckles on the way and bows to one side, here down)."""
    T = rod.tensions(C, D)
    theta, _, _ = rod.rest(rod.tensions(C, [0.05, 0, 0]), np.zeros(rod.m))
    X0 = rod.tip(theta)
    for s in np.linspace(0, 1, 26)[1:]:
        theta, w, margin, ok = rod.held(X0 + (X_COMP - X0) * s, T, theta)
    assert ok and margin > 0
    return theta, w


def pack_branch(rod, out):
    ys = np.array([o["y"] for o in out])
    return dict(p=np.array([o["p"] for o in out]), w=np.array([o["w"] for o in out]),
                margin=np.array([o["margin"] for o in out]),
                contact=np.array([o["contact"] for o in out]),
                theta=ys[:, :rod.m],
                shape=np.array([rod.nodes(y[:rod.m])[0] for y in ys]))


def merge(back, fwd):
    """Two traces from the same start, as one curve in order."""
    return {k: np.concatenate([back[k][::-1], fwd[k][1:]]) for k in back}


# --------------------------------------------------------------- 1D: S-curves

def scurve_angle():
    """Forward map along the payload angle, position held at X_COMP, D = 0."""
    rod = make_rod()
    theta, w = compressed_state(rod)
    br = Branch(rod, X_COMP, C, np.zeros(3), "angle")
    y0 = np.r_[theta, -w, 0.0]
    parts = [pack_branch(rod, br.trace(y0, 0.05, 900, (-1.2, 1.2), direction=d)) for d in (-1, 1)]
    curve = merge(*parts)
    np.savez_compressed(OUT / "scurve_angle.npz", X=X_COMP, C=C, D=np.zeros(3), **curve)


def scurve_tension():
    """Forward map along D_1, payload held at X_COMP turned to PHI_NEAR_FOLD,
    D_2 = D_3 = 0. The branch it starts on folds; past the fold the arm snaps to a
    second branch, found by descent and traced too."""
    rod = make_rod()
    theta, w = compressed_state(rod)
    br = Branch(rod, X_COMP, C, np.zeros(3), "angle")
    out = br.trace(np.r_[theta, -w, 0.0], 0.05, 400, (PHI_NEAR_FOLD, 2.2), direction=-1)
    X = out[-1]["X"].copy()
    X[2] = PHI_NEAR_FOLD
    theta, w, margin, ok = rod.forward(X, rod.tensions(C, np.zeros(3)), out[-1]["y"][:rod.m], out[-1]["w"])
    assert ok and margin > 0
    bounds = (-2 * D_LIM[0], 2 * D_LIM[0])
    br = Branch(rod, X, C, np.zeros(3), ("D", 0))
    y0 = np.r_[theta, -w, 0.0]
    first = merge(*[pack_branch(rod, br.trace(y0, 0.05, 900, bounds, direction=d)) for d in (-1, 1)])
    # the fold, and where the arm goes past it
    i = int(np.where(np.diff(np.sign(first["margin"])) != 0)[0][0])
    D_fold = first["p"][i]
    D_past = D_fold + 0.1 * np.sign(D_fold - first["p"][max(i - 5, 0)])
    th_snap, w_snap, m_snap, ok = rod.held(X, rod.tensions(C, [D_past, 0, 0]), first["theta"][i])
    assert ok and m_snap > 0
    print(f"fold at D1 = {D_fold:.3f}; at D1 = {D_past:.3f} the arm snaps: "
          f"|dtheta| = {np.abs(th_snap - first['theta'][i]).max():.2f}, w {first['w'][i]} -> {w_snap}")
    br2 = Branch(rod, X, C, np.array([D_past, 0, 0]), ("D", 0))
    y1 = np.r_[th_snap, -w_snap, D_past]
    second = merge(*[pack_branch(rod, br2.trace(y1, 0.05, 900, bounds, direction=d)) for d in (-1, 1)])
    np.savez_compressed(OUT / "scurve_tension.npz", X=X, C=C, D=np.zeros(3), D_lim=D_LIM,
                        D_fold=D_fold, D_past=D_past, snap_from=i,
                        **first, **{f"b2_{k}": v for k, v in second.items()})


def _sweep(args):
    """One angle sweep at fixed D1 from a held state (worker)."""
    D1, theta, w = args
    rod = make_rod()
    D = np.array([D1, 0.0, 0.0])
    br = Branch(rod, X_COMP, C, D, "angle")
    y0 = np.r_[theta, -w, X_COMP[2]]
    parts = []
    for d in (-1, 1):
        out = br.trace(y0, 0.04, 500, (-1.0, 1.0), direction=d)
        rec = pack_branch(rod, out)
        cut = np.argmax(rec["margin"] < -5) if np.any(rec["margin"] < -5) else len(rec["p"])
        parts.append({k: v[:cut] for k, v in rec.items()})
    c = merge(*parts)
    return {"p": c["p"], "bow": c["shape"][:, c["shape"].shape[1] // 2, 1], "margin": c["margin"],
            "contact": c["contact"], "mu": c["w"][:, 2]}


def cusp():
    """Angle sweeps at payload position X_COMP for a range of D1, from both bows."""
    from multiprocessing import Pool
    rod = make_rod()
    theta_dn, w_dn = compressed_state(rod)
    theta_up, w_up = -theta_dn, w_dn * np.array([1, -1, -1])
    th, w, mg, ok = rod.forward(X_COMP, rod.tensions(C, np.zeros(3)), theta_up, w_up)
    assert ok and mg > 0 and np.abs(th - theta_up).max() < 1e-6
    D1s = np.linspace(-D_LIM[0], D_LIM[0], 15)
    starts = []
    for label, th0, w0 in (("down", theta_dn, w_dn), ("up", theta_up, w_up)):
        for half in (D1s[D1s >= 0], D1s[D1s < 0][::-1]):
            th, w = th0.copy(), w0.copy()
            for D1 in half:
                t2, w2, mg, ok = rod.forward(X_COMP, rod.tensions(C, [D1, 0, 0]), th, w)
                if not (ok and mg > 0 and np.abs(t2 - th).max() < 0.35):
                    break                           # this bow no longer exists at phi = 0
                th, w = t2, w2
                starts.append((label, D1, th.copy(), w.copy()))
    with Pool(min(20, len(starts))) as pool:
        curves = pool.map(_sweep, [(D1, th, w) for _, D1, th, w in starts])
    out = {"labels": np.array([s[0] for s in starts]), "D1": np.array([s[1] for s in starts]),
           "C": C, "D_lim": D_LIM, "X": X_COMP}
    for i, c in enumerate(curves):
        for k, v in c.items():
            out[f"c{i}_{k}"] = v
    np.savez_compressed(OUT / "cusp.npz", **out)
    print(f"cusp: {len(starts)} sweeps")


# --------------------------------------------------------------- 2D fronts

def front(shape, seed, seed_state, solve, max_jump=0.35):
    """Solve every node of a 2D grid outward from `seed`, each warm-started from
    a solved neighbour (nearest to the seed first). solve(node, state) returns
    (state, record) or None. A result whose theta moves more than max_jump from
    the neighbour's is a jump to another branch and is rejected."""
    states = {seed: seed_state}
    records = {}
    queue = deque()
    tried = set()

    def push(node):
        i, j = node
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (i + di, j + dj)
            if 0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and nb not in states and nb not in tried:
                tried.add(nb)
                queue.append(nb)

    rec = solve(seed, seed_state)
    assert rec is not None
    states[seed], records[seed] = rec
    push(seed)
    while queue:
        node = queue.popleft()
        i, j = node
        nbs = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                                (1, 1), (1, -1), (-1, 1), (-1, -1))]
        nbs = [nb for nb in nbs if nb in states]
        nbs.sort(key=lambda nb: abs(nb[0] - seed[0]) + abs(nb[1] - seed[1]))
        for nb in nbs[:4]:
            res = solve(node, states[nb])
            if res is None:
                continue
            state, record = res
            if np.abs(state["theta"] - states[nb]["theta"]).max() > max_jump:
                continue
            states[node], records[node] = state, record
            push(node)
            break
    return records


def forward_map():
    """Forward map over the payload position, angle held at X_ref's, T = T_ref.
    status: 1 solved on the stable branch; -1 Newton from a stable neighbour
    converged, but to an unstable state (past a fold); 0 not reached."""
    rod = make_rod()
    theta_ref, X_ref = reference(rod)
    T = rod.tensions(C, D_REF)
    nx = 111
    xs = np.linspace(-0.6, 3.2, nx)
    ys = np.linspace(-0.6, 3.2, nx)
    seed = (int(np.argmin(np.abs(xs - X_ref[0]))), int(np.argmin(np.abs(ys - X_ref[1]))))
    keys = ("w", "margin", "contact", "energy")
    grid = {k: np.full((nx, nx) + ((3,) if k == "w" else ()), np.nan) for k in keys}
    status = np.zeros((nx, nx), int)
    shapes = {}

    def solve(node, state):
        X = np.array([xs[node[0]], ys[node[1]], X_ref[2]])
        theta, w, margin, ok = rod.forward(X, T, state["theta"], state["w"])
        if ok and not margin > 0 and np.abs(theta - state["theta"]).max() < 0.35:
            status[node] = -1
        if not ok or not margin > 0:
            return None
        return ({"theta": theta, "w": w},
                {"w": w, "margin": margin, "contact": rod.contact_margin(theta),
                 "energy": rod.energy(theta, T), "shape": rod.nodes(theta)[0]})

    t0 = time.perf_counter()
    recs = front((nx, nx), seed, {"theta": theta_ref, "w": np.zeros(3)}, solve)
    for node, r in recs.items():
        status[node] = 1
        for k in keys:
            grid[k][node] = r[k]
        if node[0] % 10 == 5 and node[1] % 10 == 5:
            shapes[f"{node[0]}_{node[1]}"] = r["shape"]
    print(f"forward: {len(recs)} of {nx * nx} nodes in {time.perf_counter() - t0:.0f} s; "
          f"{(status == -1).sum()} past a fold")
    np.savez_compressed(OUT / "forward.npz", xs=xs, ys=ys, X_ref=X_ref, C=C, D=D_REF, status=status,
                        shape_ref=rod.nodes(theta_ref)[0],
                        shape_keys=np.array(list(shapes)), shapes=np.array(list(shapes.values())),
                        **grid)


def inverse_map():
    """Inverse map over the force on the payload (moment 0), payload at X_ref."""
    rod = make_rod()
    theta_ref, X_ref = reference(rod)
    fs = np.linspace(-6.0, 6.0, 121)
    seed = (60, 60)
    keys = ("D", "margin", "contact", "detS", "smin")
    grid = {k: np.full((121, 121) + ((3,) if k == "D" else ()), np.nan) for k in keys}
    shapes = {}

    def solve(node, state):
        w = np.array([fs[node[0]], fs[node[1]], 0.0])
        theta, D, margin, ok, S = rod.inverse(X_ref, w, C, state["theta"], state["D"], details=True)
        if not ok:
            return None
        return ({"theta": theta, "D": D},
                {"D": D, "margin": margin, "contact": rod.contact_margin(theta),
                 "detS": float(np.linalg.det(S)),
                 "smin": float(1.0 / np.linalg.svd(S, compute_uv=False)[0]),
                 "shape": rod.nodes(theta)[0]})

    t0 = time.perf_counter()
    recs = front((121, 121), seed, {"theta": theta_ref, "D": D_REF.copy()}, solve)
    for node, r in recs.items():
        for k in keys:
            grid[k][node] = r[k]
        if node[0] % 15 == 0 and node[1] % 15 == 0:
            shapes[f"{node[0]}_{node[1]}"] = r["shape"]
    print(f"inverse: {len(recs)} of {121 * 121} nodes in {time.perf_counter() - t0:.0f} s")
    np.savez_compressed(OUT / "inverse.npz", fs=fs, X_ref=X_ref, C=C, D_ref=D_REF, D_lim=D_LIM,
                        shape_ref=rod.nodes(theta_ref)[0],
                        shape_keys=np.array(list(shapes)), shapes=np.array(list(shapes.values())),
                        **grid)


# --------------------------------------------------------------- 3D: rest map

def workspace():
    """Rest map: the payload's pose at rest (w = 0) over the box of tension
    differences, by descent from the previous grid point (snake order)."""
    rod = make_rod()
    n1, n2, n3 = 15, 15, 11
    axes = [np.linspace(-D_LIM[k], D_LIM[k], n) for k, n in enumerate((n1, n2, n3))]
    tip = np.full((n1, n2, n3, 3), np.nan)
    margin = np.full((n1, n2, n3), np.nan)
    contact = np.full((n1, n2, n3), np.nan)
    jump = np.zeros((n1, n2, n3), bool)
    theta = np.zeros(rod.m)
    t0 = time.perf_counter()
    row_start = theta
    for k in range(n3):
        for j in range(n2):
            order = range(n1) if (j + k * n2) % 2 == 0 else range(n1 - 1, -1, -1)
            for i in order:
                D = np.array([axes[0][i], axes[1][j], axes[2][k]])
                new, mg, ok = rod.rest(rod.tensions(C, D), theta)
                if not ok:
                    new, mg, ok = rod.rest(rod.tensions(C, D), np.zeros(rod.m))
                if ok:
                    jump[i, j, k] = np.abs(new - theta).max() > 0.6
                    theta = new
                    tip[i, j, k], margin[i, j, k] = rod.tip(theta), mg
                    contact[i, j, k] = rod.contact_margin(theta)
    print(f"workspace: {np.isfinite(margin).sum()} of {margin.size} in {time.perf_counter() - t0:.0f} s, "
          f"{jump.sum()} jumps")
    shapes, Ds = [], []
    for D in ([2.6, 1.2, 0.6], [-2.0, 0.8, 1.2], [1.5, -1.5, -1.0], [-1.0, -1.8, 0.4], [0.0, 2.0, -1.2]):
        th, _, _ = rod.rest(rod.tensions(C, D), np.zeros(rod.m))
        shapes.append(rod.nodes(th)[0])
        Ds.append(D)
    np.savez_compressed(OUT / "workspace.npz", D1=axes[0], D2=axes[1], D3=axes[2], tip=tip,
                        margin=margin, contact=contact, jump=jump, C=C, D_lim=D_LIM,
                        shapes=np.array(shapes), shape_D=np.array(Ds))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {"scurve_angle": scurve_angle, "scurve_tension": scurve_tension, "cusp": cusp, "forward": forward_map,
            "inverse": inverse_map, "workspace": workspace}
    names = list(jobs) if sys.argv[1] == "all" else sys.argv[1:]
    for name in names:
        t = time.perf_counter()
        jobs[name]()
        print(f"{name}: {time.perf_counter() - t:.0f} s")

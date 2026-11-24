"""
test_chain.py
PLAN checks 1-4 for steps 1-5 (segment, table, tendons, chain):
1. derivatives match finite differences (shooting, table, tendons, chain)
2. table against shooting at random poses (V, g, K)
3. chain against the direct rod (rod.py): rest shapes and forward-map wrenches
4. inverse then forward gives back the requested wrench (n = 3)

usage: python tests/test_chain.py      (about 5 min; exits 1 if a check fails)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # small dense matrices: threads only slow
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bowstring.chain import ChainArm, relative_pose          # noqa: E402
from bowstring.rod import BowstringRod                       # noqa: E402
from bowstring.segment import integrate, pose_of, shoot      # noqa: E402
from bowstring.table import SegmentTable                     # noqa: E402
from bowstring.tendons import tendons                        # noqa: E402

rng = np.random.default_rng(1)
failed = []


def report(name, value, limit):
    ok = value <= limit
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} (limit {limit:.0e})", flush=True)
    if not ok:
        failed.append(name)


def fd_grad(f, x, eps):
    return np.array([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(len(x))]).T


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


t_start = time.perf_counter()
# ---------------------------------------------------------------- 1. derivatives
print("1. Derivatives against central differences")
poses = [pose_of(q) for q in ((0.9, 0.3, -0.2), (0.99, -0.1, 0.12), (0.9995, 0.03, -0.02),
                              (0.75, 0.8, 0.4))]
worst_g = worst_K = 0.0
for r in poses:
    st = shoot(r)
    seed = st.load

    def V(x):
        return shoot(x, seed, r0=r, stability=False).V

    def g(x):
        return shoot(x, seed, r0=r, stability=False).g
    eps = 1e-6 * (1 - st.chart[0]) ** 0.5            # finer near the stiff straight limit
    worst_g = max(worst_g, rel(st.g, fd_grad(V, r, eps)))
    worst_K = max(worst_K, rel(st.K, fd_grad(g, r, eps)))
report("shooting g = dV/dr", worst_g, 1e-5)
report("shooting K = dg/dr", worst_K, 1e-5)

table = SegmentTable()
worst_g = worst_K = 0.0
for q in ((0.9, 0.3, -0.2), (0.985, -0.2, 0.25), (0.9995, 0.03, -0.02)):
    r = pose_of(q)
    b = 1 if table.state(r, 1) is not None else -1
    st = table.state(r, b)
    eps = 1e-7
    worst_g = max(worst_g, rel(st.g, fd_grad(lambda x: table.state(x, b).V, r, eps)))
    worst_K = max(worst_K, rel(st.K, fd_grad(lambda x: table.state(x, b).g, r, eps)))
# g comes from the stored loads and V from U: they agree to interpolation accuracy only.
report("table g vs dV/dr (interpolation-level)", worst_g, 5e-2)
report("table K = dg/dr", worst_K, 5e-2)

P = np.array([[0, 0, 0], [0.9, 0.3, 0.6], [1.5, 1.0, 0.9], [2.0, 1.8, 1.1]], float)
T = rng.uniform(0.2, 3.0, (3, 2))
off = (0.15, 0.2, 0.25)
lam, lg, Ht = tendons(P, off, T)
x0 = P.ravel()
Ut = lambda x: float(np.sum(T * tendons(x.reshape(-1, 3), off)[0]))
gt = lambda x: np.einsum("ks,ksv->v", T, tendons(x.reshape(-1, 3), off)[1])
report("tendon gradient", rel(np.einsum("ks,ksv->v", T, lg), fd_grad(Ut, x0, 1e-6)), 1e-7)
report("tendon Hessian", rel(Ht, fd_grad(gt, x0, 1e-6)), 1e-6)

arm = ChainArm()
C = np.array([3.0, 2.5, 2.0])
D0 = np.array([0.8, 0.4, 0.2])
T0 = arm.tensions(C, D0)
rest = arm.rest(T0)
L0 = rest["L"].ravel()
ev = arm.evaluate(L0, T0)
report("chain dG/dL", rel(ev["dGdL"], fd_grad(lambda x: arm.evaluate(x, T0)["G"], L0, 1e-7)), 1e-5)
report("chain dP/dL", rel(ev["dPdL"], fd_grad(lambda x: arm.evaluate(x, T0)["P"][1:].ravel(),
                                              L0, 1e-7)), 1e-5)
X0 = rest["P"][-1]
fw = arm.forward(X0 + [0.04, -0.05, 0.08], T0, rest["L"])
Xf = fw["P"][-1]
w_of = lambda X: arm.forward(X, T0, fw["L"])["w"]
U_of = lambda X: arm.forward(X, T0, fw["L"])["U"]
report("chain w = -dU*/dX (envelope)", rel(fw["w"], -fd_grad(U_of, Xf, 1e-5)), 1e-5)
report("chain K_X = -dw/dX", rel(fw["K_X"], -fd_grad(w_of, Xf, 1e-6)), 1e-4)

# ---------------------------------------------------------------- 2. table vs shooting
print("2. Table against exact states at random off-node poses, both branches, up to the")
print("   energy the arm reaches in each band (from data/maps)")
bands = ((0.70, 0.96, 3.5), (0.96, 0.99, 3.5), (0.99, 0.999, 1.0), (0.999, 0.9999, 0.1))
summary = []
for lo, hi, u_max in bands:
    v, tries = [], 0
    while len(v) < 60 and tries < 5000:
        tries += 1
        d = rng.uniform(lo, hi)
        a = (16 * (1 - d) * u_max) ** 0.25
        b = int(rng.choice([1, -1]))
        ts = table.state(pose_of((d, rng.uniform(-a, a), rng.uniform(-a, a))), b)
        if ts is None or ts.V > u_max:
            continue
        # Exact state: the table's load integrated forward (no pose-matching solve), then
        # the table at that exact pose.
        pose, M1, V, J, _ = integrate(ts.load)
        tx = table.state(pose, b)
        if tx is None:
            continue
        g = np.array([ts.load[0], ts.load[1], M1])
        K = np.vstack([np.eye(3)[:2], J[3]]) @ np.linalg.inv(J[:3])
        v.append((abs(tx.V / V - 1), rel(tx.g, g), rel(tx.K, 0.5 * (K + K.T))))
    v = np.array(v)
    med, p99 = np.median(v, axis=0), np.percentile(v, 99, axis=0)
    summary.append(med)
    print(f"  d {lo}-{hi}, U <= {u_max} ({len(v)} poses): median V {med[0]:.1e} g {med[1]:.1e} "
          f"K {med[2]:.1e}; p99 V {p99[0]:.1e} g {p99[1]:.1e} K {p99[2]:.1e}", flush=True)
# Records the table's accuracy; the chain does not use the table (see chain.py).
report("table V, worst band median", float(np.max(np.array(summary)[:, 0])), 1e-2)

# ---------------------------------------------------------------- 3. chain vs rod
print("3. Chain against the direct rod (rod.py, 160 elements per segment; its stable descent)")
rod = BowstringRod(N=160)
rod20 = BowstringRod(N=20)


def rod_rest(Tt):
    """rod.py rest at 160 elements, started from the 20-element solution."""
    th20, _, _ = rod20.rest(Tt, np.zeros(rod20.m))
    s20 = np.arange(1, rod20.m + 1) / rod20.N
    s160 = np.arange(1, rod.m + 1) / rod.N
    return rod.rest(Tt, np.interp(s160, np.r_[0, s20], np.r_[0, th20]))
worst_P = worst_w = 0.0
for _ in range(4):
    D = rng.uniform(-1, 1, 3) * (C - 0.6)
    Tt = arm.tensions(C, D)
    th, _, ok_r = rod_rest(Tt)
    res = arm.rest(Tt)
    if not (ok_r and res["converged"]):
        print(f"  rest did not converge: D = {D.tolist()}, rod ok {ok_r}, chain residual "
              f"{res['residual']:.1e}")
        failed.append("rest convergence")
        continue
    worst_P = max(worst_P, float(np.abs(res["P"] - rod.disks(th)).max()))
    X = res["P"][-1] + rng.uniform(-0.05, 0.05, 3)
    fc = arm.forward(X, Tt, res["L"])
    _, wr, _, okf = rod.held(X, Tt, th)       # plain Newton can end on an unstable state
    if fc["converged"] and okf:
        err = float(np.abs(fc["w"] - wr).max())
        if err > 1e-3:
            print(f"  forward mismatch {err:.1e}: D = {D.tolist()}, X = {X.tolist()}, chain w "
                  f"{np.round(fc['w'], 5)}, rod w {np.round(wr, 5)}, chain margin "
                  f"{fc['margin']:.3g}, chain |dP| vs rod rest-start {np.abs(fc['P'] - res['P']).max():.3f}")
        worst_w = max(worst_w, err)
    else:
        print(f"  forward not converged: chain {fc['converged']}, rod {okf}")
# Limits: the 160-element rod's own discretisation error (a few 1e-4 against 320 elements).
report("rest disk poses, max |chain - rod|", worst_P, 5e-4)
report("forward wrench, max |chain - rod|", worst_w, 1e-3)

# ---------------------------------------------------------------- 4. inverse then forward
print("4. Inverse then forward (each target reached in 10 steps from rest, as the controller would)")
worst = 0.0
for _ in range(4):
    X = X0 + rng.uniform(-0.08, 0.08, 3)
    w = rng.uniform(-0.3, 0.3, 3)
    L, Dk = rest["L"], D0
    for s in np.linspace(0.1, 1.0, 10):
        inv = arm.inverse(X0 + s * (X - X0), s * w, C, L, Dk)
        if not inv["converged"]:
            break
        L, Dk = inv["L"], inv["D"]
    if not inv["converged"]:
        print(f"  inverse not converged: X - X0 = {(X - X0).tolist()}, w = {w.tolist()}, "
              f"residual {inv['residual']:.1e}, D {np.round(inv['D'], 3)}")
        failed.append("inverse convergence")
        continue
    fw2 = arm.forward(X, inv["T"], inv["L"])
    worst = max(worst, float(np.abs(fw2["w"] - w).max()))
report("n = 3: |w(forward(inverse(w))) - w|", worst, 1e-8)

print(f"{time.perf_counter() - t_start:.0f} s")
print("FAILED: " + ", ".join(failed) if failed else "All checks passed")
sys.exit(1 if failed else 0)

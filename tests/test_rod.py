"""Finite-difference and consistency checks of bowstring/rod.py."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bowstring.rod import BowstringRod

rng = np.random.default_rng(0)
rod = BowstringRod(N=20)
theta = 0.4 * rng.standard_normal(rod.m).cumsum() / rod.N
T = rng.uniform(0.2, 3.0, (3, 2))
eps = 1e-6

def grad(th):
    _, lg, _ = rod.tendons(th, T)
    return rod.Kb @ th + np.einsum("ks,ksm->m", T, lg)

U = lambda th: rod.energy(th, T)
g = grad(theta)
g_fd = np.array([(U(theta + eps * e) - U(theta - eps * e)) / (2 * eps) for e in np.eye(rod.m)])
print("grad  rel err", np.abs(g - g_fd).max() / np.abs(g).max())
_, _, Ht = rod.tendons(theta, T)
H = rod.Kb + Ht
H_fd = np.array([(grad(theta + eps * e) - grad(theta - eps * e)) / (2 * eps) for e in np.eye(rod.m)])
print("hess  rel err", np.abs(H - H_fd).max() / np.abs(H).max())

c, A, curv = rod.constraint(theta)
cfun = lambda th: rod.constraint(th)[0]
A_fd = np.array([(cfun(theta + eps * e) - cfun(theta - eps * e)) / (2 * eps) for e in np.eye(rod.m)]).T
print("A     rel err", np.abs(A - A_fd).max() / np.abs(A).max())
gw = np.array([0.7, -1.3])
Afun = lambda th: gw @ rod.constraint(th)[1][:2]
C_fd = np.array([(Afun(theta + eps * e) - Afun(theta - eps * e)) / (2 * eps) for e in np.eye(rod.m)])
print("curv  rel err", np.abs(curv(gw) - C_fd).max() / np.abs(curv(gw)).max())
print("tip vs disks", np.abs(c - rod.disks(theta)[-1]).max())

# Inverse then forward, and the wrench as -dU*/dX.
C = np.array([4.0, 3.0, 2.0])
rod = BowstringRod(N=40)
th0, _, ok = rod.rest(rod.tensions(C, [1.0, -0.5, 0.8]), np.zeros(rod.m))
X0 = rod.tip(th0)
X = X0 + np.array([0.05, -0.08, 0.1])
w = np.array([0.3, -0.2, 0.05])
th, D, marg, ok = rod.inverse(X, w, C, th0, [1.0, -0.5, 0.8])
print("inverse ok", ok, "D", D, "margin", marg)
Tt = rod.tensions(C, D)
th2, w2, marg2, ok2 = rod.forward(X, Tt, th0)
print("forward ok", ok2, "w back", w2, "err", np.abs(w2 - w).max(), "margin", marg2)
# envelope: w = -dU*/dX
def Ustar(Xv):
    t, _, _, okk = rod.forward(Xv, Tt, th)
    return rod.energy(t, Tt)
e = 1e-5
dU = np.array([(Ustar(X + e * v) - Ustar(X - e * v)) / (2 * e) for v in np.eye(3)])
print("w vs -dU*/dX", w2, -dU)

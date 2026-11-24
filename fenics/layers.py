"""
layers.py
The clamped-elastica solution space near d = 1, solved one d layer at a time on a pool
of workers (one serial Newton solve at a time per worker).

usage: python layers.py config.json [--only 0.97,0.9999] [--workers N]
       (outputs go to <config dir>/<out_dir>/layer_<d>.npz; finished layers are skipped)

This is ~/elastica/solver/main.py reduced to one layer: the same Newton solver
(solver.py), branch tracking on q = p / a(d) (branch.py), stability test
(stability.py), mirror (phi1, phi2) -> (-phi2, -phi1) and promotion pass. What differs:

- Layers are independent. Each starts cold from the exact first mode at (d, 0, 0)
  and grows a 2D front (8-neighbourhood) outwards, so layers can be solved in parallel.
- Energy cutoff. The front does not grow past U > energy_max: a node above it is
  stored, and up to `rim` rings beyond it, but nothing further. The arm never reaches
  those states, and their tensions grow like 1/(1 - d).
- The phi box and step follow the energy disk. For a nearly straight rod under
  tension, U ~ (phi1^2 + phi2^2)^2 / (16 (1 - d)), so U <= U_max lies inside
  |phi| <= phi_est = (16 (1 - d) U_max)^(1/4) (checked against the production grid at
  d = 0.97 to 0.99: 1.47 / 1.33 / 1.12 against 1.48 / 1.33 / 1.12). The box is
  box_factor * phi_est (at most pi/2) and the step scales the same way,
  step = phi_step_ref * ((1 - d) / (1 - d_ref))^(1/4), so every layer has about the
  same number of nodes across its disk.
- theta is kept only for the solved nodes of the half phi1 + phi2 >= 0; the mirror
  half is rebuilt from it.
"""

import os

# One Newton solve has ~1000 DOFs; threaded BLAS/MUMPS only adds overhead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import heapq
import json
import multiprocessing as mp
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from branch import first_mode_amplitude, first_mode_projection, is_jump  # noqa: E402
from stability import unstable_directions  # noqa: E402

# 8-neighbourhood in (j, k), face neighbours first.
OFFSETS = sorted(((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b),
                 key=lambda o: o[0] ** 2 + o[1] ** 2)
FIELDS = ("fx", "fy", "moment", "energy", "p")


# ------------------------------------------------------------------ beam

def precompute_elastica_spaces(L=1.0, N=100, grading=0.0):
    """Beam mesh and spaces (copied from main.py): s = L g(u),
    g(u) = (1 - grading) u + grading (1 - cos(pi u)) / 2, refined towards both clamps."""
    import basix.ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI
    domain = mesh.create_interval(MPI.COMM_WORLD, N, [0.0, L])
    u = domain.geometry.x[:, 0] / L
    domain.geometry.x[:, 0] = L * ((1.0 - grading) * u
                                   + grading * 0.5 * (1.0 - np.cos(np.pi * u)))
    V = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1))
    R = fem.functionspace(domain, basix.ufl.real_element(domain.basix_cell()))
    dof_x = V.tabulate_dof_coordinates()[:, 0]
    beam_order = np.argsort(dof_x)
    return V, R, dof_x[beam_order], beam_order


def end_moments(theta_sorted, beam_x, fx, fy):
    """theta'(0) and theta'(L), second order (copied from main.py)."""
    slopes = np.diff(theta_sorted) / np.diff(beam_x)
    h0, h1 = beam_x[1] - beam_x[0], beam_x[-1] - beam_x[-2]

    def curvature_rate(theta):
        return -fx * np.sin(theta) + fy * np.cos(theta)

    return (float(slopes[0] - 0.5 * h0 * curvature_rate(theta_sorted[0])),
            float(slopes[-1] + 0.5 * h1 * curvature_rate(theta_sorted[-1])))


class Beam:
    """The Newton solver on one beam mesh; states are dicts with theta in beam order."""

    def __init__(self, elements, grading, snes_rtol=1e-10, snes_atol=1e-12):
        from solver import ElasticaSolver
        V, R, self.x, self.order = precompute_elastica_spaces(1.0, elements, grading)
        self.solver = ElasticaSolver(V, R, petsc_options={
            "snes_max_it": 500, "snes_rtol": snes_rtol, "snes_atol": snes_atol})

    def state(self, solution):
        theta = solution.theta[self.order]
        m1, m2 = end_moments(theta, self.x, solution.fx, solution.fy)
        return {"theta": theta, "fx": solution.fx, "fy": solution.fy, "moment": m1,
                "moment_right": m2, "energy": solution.energy,
                "p": float(first_mode_projection(theta, self.x)),
                "iterations": solution.iterations}

    def warm(self, d, phi1, phi2, theta, fx, fy, max_it=None):
        """Solve from theta (beam order); None if Newton fails."""
        theta0 = np.empty(len(self.order))
        theta0[self.order] = theta
        try:
            return self.state(self.solver.solve(d, phi1, phi2, theta0, fx, fy, max_it=max_it))
        except RuntimeError:
            return None

    def cold(self, d, phi1, phi2, **steps):
        """Exact first mode at (d, 0, 0), continued in the clamp angles (solver.cold_solve)."""
        try:
            return self.state(self.solver.cold_solve(d, phi1, phi2, **steps))
        except RuntimeError:
            return None

    def stable(self, state):
        return not unstable_directions(state["theta"], self.x, state["fx"], state["fy"])


# ------------------------------------------------------------------ layer grid

def layer_axis(d, cfg):
    """The phi grid of layer d: symmetric about 0, through 0."""
    phi_est = (16.0 * (1.0 - d) * cfg["energy_max"]) ** 0.25
    box = min(np.pi / 2, cfg["box_factor"] * phi_est)
    step = cfg["phi_step_ref"] * ((1.0 - d) / (1.0 - cfg["d_ref"])) ** 0.25
    half = int(np.ceil(box / step - 1e-9))
    return step * np.arange(-half, half + 1), phi_est


def layer_list(cfg):
    ds = []
    for band in cfg["bands"]:
        count = int(round((band["d_stop"] - band["d_start"]) / band["d_step"])) + 1
        ds += [round(band["d_start"] + i * band["d_step"], 10) for i in range(count)]
    return sorted(set(ds))


def layer_name(d):
    return f"layer_{d:.5f}.npz"


# ------------------------------------------------------------------ one layer

_W = {}


def _init_worker(cfg):
    _W["cfg"] = cfg
    _W["beam"] = Beam(cfg["beam_elements"], cfg["beam_grading"],
                      cfg["snes_rtol"], cfg["snes_atol"])


def solve_layer(task):
    d, out_path = task
    cfg, beam = _W["cfg"], _W["beam"]
    t_start = time.perf_counter()
    phi, phi_est = layer_axis(d, cfg)
    M = len(phi)
    n_nodes = M * M
    jj, kk = np.divmod(np.arange(n_nodes), M)
    mirror = (M - 1 - kk) * M + (M - 1 - jj)
    reflect = (M - 1 - jj) * M + (M - 1 - kk)
    canonical = jj + kk >= M - 1
    centre = (M // 2) * M + M // 2
    priority = (jj - M // 2) ** 2 + (kk - M // 2) ** 2
    amp = float(first_mode_amplitude(d))
    u_max, rim = float(cfg["energy_max"]), int(cfg["rim"])
    max_it, tol = int(cfg["warm_start_max_it"]), float(cfg["jump_tol"])
    max_fail = int(cfg["max_failures_per_branch"])

    f = {k: np.full(n_nodes, np.nan) for k in FIELDS}
    solved = np.zeros(n_nodes, dtype=bool)
    beyond = np.full(n_nodes, -1, dtype=np.int8)        # rings past U > energy_max
    waited = np.zeros(n_nodes, dtype=bool)
    promoted = np.zeros(n_nodes, dtype=bool)
    order = np.full(n_nodes, -1, dtype=np.int64)
    iterations = np.zeros(n_nodes, dtype=np.int16)
    # theta of canonical solved nodes, rows allocated as needed.
    row_of = np.full(n_nodes, -1, dtype=np.int64)
    theta_rows = np.empty((max(1024, n_nodes // 4), len(beam.x)), dtype=np.float32)
    used = [0]
    counter = [0]
    stats = {"solves": 0, "diverged": 0, "jumps rejected": 0, "unstable rejected": 0,
             "deferrals": 0, "extra cold starts": 0, "cold starts failed": 0}
    cold_log = []                       # (phi1, phi2, accepted, U) of every cold start

    def theta_of(node):
        if canonical[node]:
            return theta_rows[row_of[node]]
        return -theta_rows[row_of[mirror[node]]][::-1]

    def put_theta(node, theta):
        nonlocal theta_rows
        if row_of[node] < 0:
            if used[0] == len(theta_rows):
                theta_rows = np.concatenate([theta_rows, np.empty_like(theta_rows)])
            row_of[node] = used[0]
            used[0] += 1
        theta_rows[row_of[node]] = theta

    def neighbours(node):
        j, k = divmod(int(node), M)
        for a, b in OFFSETS:
            if 0 <= j + a < M and 0 <= k + b < M:
                yield (j + a) * M + k + b

    def canon(node):
        return int(node) if canonical[node] else int(mirror[node])

    def store(node, st, theta_canonical):
        """node is canonical; also fills its mirror."""
        m = int(mirror[node])
        for idx, sign_fy, mom in ((node, 1.0, st["moment"]), (m, -1.0, st["moment_right"])):
            f["fx"][idx], f["fy"][idx] = st["fx"], sign_fy * st["fy"]
            f["moment"][idx], f["energy"][idx], f["p"][idx] = mom, st["energy"], st["p"]
            solved[idx] = True
            order[idx] = counter[0]
            iterations[idx] = st.get("iterations", 0)
        counter[0] += 1
        put_theta(node, theta_canonical)

    heap, queued = [], np.zeros(n_nodes, dtype=bool)

    def push(node):
        node = canon(node)
        if not solved[node] and not queued[node]:
            queued[node] = True
            heapq.heappush(heap, (int(priority[node]), node))

    def accept(node, st):
        store(node, st, st["theta"])
        if st["energy"] <= u_max:
            b = 0
        else:
            near = [beyond[nb] for nb in neighbours(node) if solved[nb] and nb != node]
            b = 1 + min(near) if near else 1
        beyond[node] = beyond[mirror[node]] = min(b, 127)
        if b < rim:
            for nb in neighbours(node):
                if not solved[nb]:
                    push(nb)
            for nb in neighbours(int(mirror[node])):
                if not solved[nb]:
                    push(nb)

    def cold_start(node):
        stats["solves"] += 1
        # Continuation steps scaled to the layer: cold_solve's default 0.25 rad
        # fails near d = 1, where the whole disk is a few tenths of a radian wide.
        step = phi[1] - phi[0]
        st = beam.cold(d, phi[jj[node]], phi[kk[node]], max_step=10 * step,
                       min_step=0.25 * step, step_max_it=15)
        ok = st is not None and beam.stable(st)
        cold_log.append((float(phi[jj[node]]), float(phi[kk[node]]), ok,
                         None if st is None else float(st["energy"])))
        if not ok:
            stats["cold starts failed"] += 1
            return False
        accept(node, st)
        return True

    if not cold_start(centre):
        raise RuntimeError(f"cold start at (d, 0, 0) failed for d = {d}")

    tried, failures = {}, {}
    while True:
        while heap:
            _, node = heapq.heappop(heap)
            queued[node] = False
            if solved[node]:
                continue
            node_tried = tried.setdefault(node, set())
            node_fail = failures.setdefault(node, {1.0: 0, -1.0: 0, 0.0: 0})
            accepted = None
            for nb in sorted((nb for nb in neighbours(node) if solved[nb]),
                             key=lambda nb: f["p"][nb] <= 0):
                if nb in node_tried:
                    continue
                branch = float(np.sign(f["p"][nb]))
                if node_fail[branch] >= max_fail:
                    continue
                node_tried.add(nb)
                stats["solves"] += 1
                st = beam.warm(d, phi[jj[node]], phi[kk[node]], theta_of(nb),
                               f["fx"][nb], f["fy"][nb], max_it=max_it)
                if st is None:
                    stats["diverged"] += 1
                elif is_jump(f["p"][nb] / amp, st["p"] / amp, tol):
                    stats["jumps rejected"] += 1
                elif not beam.stable(st):
                    stats["unstable rejected"] += 1
                else:
                    accepted = st
                    break
                node_fail[branch] += 1
            if accepted is None:
                stats["deferrals"] += 1
                waited[[node, mirror[node]]] = True
                continue
            tried.pop(node, None)
            failures.pop(node, None)
            accept(node, accepted)
        waiting = [nd for nd in tried if not solved[nd]]
        if not waiting or stats["extra cold starts"] >= int(cfg["max_cold_starts"]):
            break
        node = min(waiting, key=lambda nd: (priority[nd], nd))
        stats["extra cold starts"] += 1
        if not cold_start(node):
            tried.pop(node)

    # Promotion: where both branches are stable the node stores the up one.
    q = f["p"] / amp
    down = solved & (f["p"] < 0)
    nodes = np.flatnonzero(canonical & down & down[reflect])
    nodes = nodes[is_jump(q[nodes], -q[reflect[nodes]], tol)]
    src = reflect[nodes]
    copies = [(f["fx"][s], f["fy"][s], f["moment"][s], f["moment"][mirror[s]], f["energy"][s],
               f["p"][s], theta_of(s).copy()) for s in src]
    for node, (fx, fy, m1, m2, U, p, th) in zip(nodes, copies):
        # theta -> -theta: same fx and U, opposite fy, moments and p.
        store(int(node), {"fx": fx, "fy": -fy, "moment": -m1, "moment_right": -m2,
                          "energy": U, "p": -p}, -th)
        promoted[[node, mirror[node]]] = True
    promo = {"by reflection": len(nodes), "by warm start": 0}
    up = solved & (f["p"] > 0)
    seeds = [int(nd) for nd in np.flatnonzero(canonical & solved & (f["p"] < 0))
             if any(up[nb] for nb in neighbours(nd))]
    fail_up = np.zeros(n_nodes, dtype=np.int16)
    tried_up = {}
    queue, in_queue = deque(), np.zeros(n_nodes, dtype=bool)

    def push_up(node):
        node = canon(node)
        if f["p"][node] < 0 and fail_up[node] < max_fail and not in_queue[node]:
            in_queue[node] = True
            queue.append(node)

    for nd in seeds:
        push_up(nd)
    while queue:
        node = queue.popleft()
        in_queue[node] = False
        if not f["p"][node] < 0 or fail_up[node] >= max_fail:
            continue
        node_tried = tried_up.setdefault(node, set())
        for nb in neighbours(node):
            if not f["p"][nb] > 0 or nb in node_tried:
                continue
            node_tried.add(nb)
            stats["solves"] += 1
            st = beam.warm(d, phi[jj[node]], phi[kk[node]], theta_of(nb),
                           f["fx"][nb], f["fy"][nb], max_it=max_it)
            if (st is not None and st["p"] > 0 and not is_jump(f["p"][nb] / amp, st["p"] / amp, tol)
                    and beam.stable(st)):
                store(node, st, st["theta"])
                promoted[[node, mirror[node]]] = True
                promo["by warm start"] += 1
                for nb2 in list(neighbours(node)) + list(neighbours(int(mirror[node]))):
                    push_up(nb2)
                break
            fail_up[node] += 1
            if fail_up[node] >= max_fail:
                break

    # Box check: a solved node with U <= energy_max on the box edge means the box clipped the disk.
    edge = (jj == 0) | (jj == M - 1) | (kk == 0) | (kk == M - 1)
    clipped = int(np.count_nonzero(edge & solved & (f["energy"] <= u_max)))
    rows = np.flatnonzero(canonical & solved)
    elapsed = time.perf_counter() - t_start
    summary = {"d": d, "M": M, "phi_step": float(phi[1] - phi[0]), "phi_box": float(phi[-1]),
               "phi_est": phi_est, "solved": int(solved.sum()),
               "within_energy": int(np.count_nonzero(solved & (f["energy"] <= u_max))),
               "box_clipped": clipped, "unsolved_waiting": int(np.count_nonzero(waited & ~solved)),
               "promoted": promo, "seconds": elapsed, "cold_starts": cold_log, **stats}
    shape = (M, M)
    tmp = f"{out_path}.tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, d=d, phi=phi, beam_x=beam.x,
                 **{k: v.reshape(shape) for k, v in f.items()},
                 moment_right=f["moment"][mirror].reshape(shape),
                 q=(f["p"] / amp).reshape(shape), solved=solved.reshape(shape),
                 beyond=beyond.reshape(shape), waited=waited.reshape(shape),
                 promoted=promoted.reshape(shape), solve_order=order.reshape(shape),
                 iterations=iterations.reshape(shape),
                 theta_nodes=rows, theta=theta_rows[row_of[rows]],
                 summary=json.dumps(summary), settings=json.dumps(cfg))
    os.replace(tmp, out_path)
    return summary


# ------------------------------------------------------------------ driver

DEFAULTS = {"energy_max": 10.0, "rim": 2, "box_factor": 1.1, "phi_step_ref": 0.01,
            "d_ref": 0.97, "beam_elements": 1000, "beam_grading": 0.9, "snes_rtol": 1e-10,
            "snes_atol": 1e-12, "warm_start_max_it": 50, "jump_tol": 0.17,
            "max_failures_per_branch": 3, "max_cold_starts": 20, "out_dir": "layers"}


def load_config(path):
    with open(path) as fh:
        cfg = {**DEFAULTS, **json.load(fh)}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("config")
    ap.add_argument("--only", help="comma-separated d values to solve (default: all layers)")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    out_dir = config_path.parent / cfg["out_dir"]
    out_dir.mkdir(exist_ok=True)
    ds = layer_list(cfg)
    if args.only:
        wanted = [float(x) for x in args.only.split(",")]
        ds = [d for d in ds if any(abs(d - w) < 1e-9 for w in wanted)] or wanted
    tasks = [(d, str(out_dir / layer_name(d))) for d in ds
             if not (out_dir / layer_name(d)).exists()]
    workers = min(args.workers or int(cfg.get("workers", max(1, (os.cpu_count() or 2) - 2))),
                  max(1, len(tasks)))
    print(f"{len(ds)} layers, {len(ds) - len(tasks)} done; solving {len(tasks)} on {workers} "
          f"workers. Beam {cfg['beam_elements']} elements, grading {cfg['beam_grading']}; "
          f"U <= {cfg['energy_max']} plus {cfg['rim']} rings", flush=True)
    for d in ds:
        phi, est = layer_axis(d, cfg)
        print(f"  d = {d:.5f}: phi box +-{phi[-1]:.4f} (est {est:.4f}), step "
              f"{phi[1] - phi[0]:.5f}, {len(phi)}^2 nodes", flush=True)
    if not tasks:
        return
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, s in enumerate(pool.imap_unordered(solve_layer, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.1f} min] d = {s['d']:.5f}: "
                  f"{s['solved']:,} nodes ({s['within_energy']:,} with U <= {cfg['energy_max']}), "
                  f"{s['solves']:,} solves in {s['seconds'] / 60:.1f} min "
                  f"({s['seconds'] / max(s['solves'], 1) * 1e3:.0f} ms each); "
                  f"{s['jumps rejected']} jumps, {s['unstable rejected']} unstable, "
                  f"{s['diverged']} diverged rejected; {s['extra cold starts']} extra cold starts "
                  f"({s['cold starts failed']} failed); {s['unsolved_waiting']} left waiting; "
                  f"promoted {s['promoted']}; box clipped at {s['box_clipped']} nodes",
                  flush=True)
        pool.close()
        pool.join()
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""
scene.py
The bowstring arm as a SOFA scene, an independent plant for the controller.

  rod       n segments of length 1, each NE corotational beam elements (BeamFEMForceField,
            Rigid3d nodes), bending stiffness EI = 1, axial stiffness EA = 4 / r^2 EI (nearly
            inextensible); clamped at the base, kept in the x-y plane (z, rx, ry fixed).
            Light distributed mass (rod_mass in all).
  disks     the beam nodes at s = 1, 2, 3 (no mass, no thickness, as in the model).
  tendons   one SoftRobots CableConstraint per tendon, force mode: pulled from its hole on the
            base, through the holes of disks 1..k, anchored at disk k + 1; straight between
            holes. Holes at +-d_k along the disk normal (local y).
  payload   a rigid mass (m, I) on the tip node (mapped child).
  damping   Rayleigh mass damping, alpha = 0.1 (the payload damping c = alpha (m, m, I) the
            controller assumes), plus a little stiffness damping for the rod's own vibrations.
  external  a force/moment on the payload (pushes), set per step.

Cable values: SoftRobots' force mode imposes the constraint impulse, so a tension T is sent
as T dt (see set_tensions).

Integration: FreeMotionAnimationLoop, implicit Euler, direct sparse solver, dt = 1 ms.

Environment: SOFA_ROOT = the sofa-env conda prefix, SOFA_PLUGIN_PATH = SoftRobots' lib folder
(set below when not given).
"""
import math
import os
from pathlib import Path

os.environ.setdefault("SOFA_ROOT", str(Path.home() / "miniforge3/envs/sofa-env"))
os.environ.setdefault("SOFA_PLUGIN_PATH", os.pathsep.join(
    [str(Path.home() / "opt/SoftRobots/install/lib"), str(Path.home() / "opt/Cosserat/install/lib")]))

import numpy as np                           # noqa: E402
import Sofa                                  # noqa: E402
import Sofa.Core                             # noqa: E402
import Sofa.Simulation                       # noqa: E402

from bowstring.segment import integrate      # noqa: E402

PLUGINS = ["SoftRobots", "Sofa.Component.AnimationLoop", "Sofa.Component.Constraint.Lagrangian.Correction",
           "Sofa.Component.Constraint.Lagrangian.Solver", "Sofa.Component.Constraint.Projective",
           "Sofa.Component.LinearSolver.Direct", "Sofa.Component.Mapping.Linear",
           "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
           "Sofa.Component.MechanicalLoad", "Sofa.Component.ODESolver.Backward",
           "Sofa.Component.SolidMechanics.FEM.Elastic", "Sofa.Component.StateContainer",
           "Sofa.Component.Topology.Container.Dynamic"]


def quat_z(theta):
    return [0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]


def model_nodes(L, n, NE, stiffness=None):
    """Node poses (n NE + 1, 3) on the model's equilibrium centre line for segment loads L."""
    L = np.asarray(L, float).reshape(n, 3)
    B = np.ones(n) if stiffness is None else np.asarray(stiffness, float)
    out, base = [np.zeros(3)], np.zeros(3)
    for i in range(n):
        _, _, _, _, th = integrate(L[i] / B[i], record=True)
        s = np.linspace(0, 1, len(th))
        ds = s[1] - s[0]
        x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
        sq = np.linspace(0, 1, NE + 1)[1:]
        xq, yq, tq = (np.interp(sq, s, v) for v in (x, y, th))
        c, sn = math.cos(base[2]), math.sin(base[2])
        pts = np.column_stack([base[0] + c * xq - sn * yq, base[1] + sn * xq + c * yq,
                               base[2] + tq])
        out.extend(pts)
        base = pts[-1]
    return np.array(out)


class SofaArm:
    def __init__(self, nodes, T0, payload, offsets=(0.15, 0.2, 0.25), NE=20, radius=0.01,
                 rod_mass=0.015, alpha=0.1, beta=1e-3, dt=1e-3):
        n = len(offsets)
        self.n, self.NE, self.dt, self.offsets = n, NE, dt, np.asarray(offsets, float)
        N = n * NE + 1
        self.disk = [NE * j for j in range(n + 1)]
        root = Sofa.Core.Node("root")
        root.dt = dt
        root.gravity = [0.0, 0.0, 0.0]
        root.addObject("RequiredPlugin", pluginName=PLUGINS)
        root.addObject("FreeMotionAnimationLoop")
        root.addObject("BlockGaussSeidelConstraintSolver", tolerance=1e-10, maxIterations=500)
        arm = root.addChild("arm")
        arm.addObject("EulerImplicitSolver", rayleighMass=alpha, rayleighStiffness=beta)
        arm.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
        nodes = np.asarray(nodes, float)
        pos = [[x, y, 0.0] + quat_z(t) for x, y, t in nodes]
        rest = [[s, 0.0, 0.0] + quat_z(0.0) for s in np.arange(N) / NE]
        self.mo = arm.addObject("MechanicalObject", template="Rigid3d", position=pos,
                                rest_position=rest)
        arm.addObject("EdgeSetTopologyContainer", edges=[[i, i + 1] for i in range(N - 1)])
        E = 1.0 / (math.pi * radius ** 4 / 4)                 # EI = 1
        arm.addObject("BeamFEMForceField", youngModulus=E, poissonRatio=0.3, radius=radius)
        h = 1.0 / NE
        j = h * h / 12                                         # per unit mass
        arm.addObject("UniformMass", vertexMass=[rod_mass / N, 1.0, j, 0, 0, 0, j, 0, 0, 0, j])
        arm.addObject("FixedProjectiveConstraint", indices=[0])
        arm.addObject("PartialFixedProjectiveConstraint", indices=list(range(N)),
                      fixedDirections=[0, 0, 1, 1, 1, 0])
        arm.addObject("GenericConstraintCorrection")

        pl = arm.addChild("payload")
        self.pmo = pl.addObject("MechanicalObject", template="Rigid3d", position=[pos[-1]])
        Iu = payload.I / payload.m
        pl.addObject("UniformMass", vertexMass=[payload.m, 1.0, Iu, 0, 0, 0, Iu, 0, 0, 0, Iu])
        self.push = pl.addObject("ConstantForceField", indices=[0], forces=[[0.0] * 6])
        pl.addObject("SubsetMapping", indices=[N - 1])

        # holes: per disk 1..n, for each pair k and side s, at (0, s d_k, 0) in the disk frame
        holes, rigid, self.hole_index = [], [], {}
        for jd in range(1, n + 1):
            for k in range(n):
                for si, s in enumerate((1.0, -1.0)):
                    self.hole_index[(jd, k, si)] = len(holes)
                    holes.append([0.0, s * self.offsets[k], 0.0])
                    rigid.append(self.disk[jd])
        cab = arm.addChild("cables")
        cab.addObject("MechanicalObject", template="Vec3d", position=holes)
        self.cables, self.routes = {}, {}
        self.cable_mo = cab.MechanicalObject
        for k in range(n):
            for si, s in enumerate((1.0, -1.0)):
                idx = [self.hole_index[(jd, k, si)] for jd in range(1, k + 2)]
                self.routes[(k, si)] = idx
                self.cables[(k, si)] = cab.addObject(
                    "CableConstraint", name=f"cable{k}{'pm'[si]}", indices=idx,
                    pullPoint=[0.0, s * self.offsets[k], 0.0], valueType="force",
                    value=float(T0[k][si]) * dt)
        cab.addObject("RigidMapping", globalToLocalCoords=False, rigidIndexPerPoint=rigid)
        self.root = root
        Sofa.Simulation.initRoot(root)
        self.t = 0.0
        self.v_prev = self.tip()[1].copy()

    # ------------------------------------------------------------ io
    def set_tensions(self, T):
        # SOFA's constraint solver works with impulses (force x dt), and SoftRobots' force mode
        # imposes lambda = value directly: a tension T is the value T dt (checked against an
        # equivalent end moment, tests in check_static.py)
        for (k, si), c in self.cables.items():
            c.value = [float(T[k][si]) * self.dt]

    def set_push(self, w):
        self.push.forces = [[float(w[0]), float(w[1]), 0.0, 0.0, 0.0, float(w[2])]]

    def set_damping(self, alpha):
        self.root.arm.EulerImplicitSolver.rayleighMass = alpha

    def poses(self):
        """Disk poses (n + 1, 3): x, y, psi."""
        p = np.array(self.mo.position.value)[self.disk]
        return np.column_stack([p[:, 0], p[:, 1], 2 * np.arctan2(p[:, 5], p[:, 6])])

    def nodes(self):
        p = np.array(self.mo.position.value)
        return np.column_stack([p[:, 0], p[:, 1], 2 * np.arctan2(p[:, 5], p[:, 6])])

    def tendon_lengths(self):
        """True tendon lengths (n, 2) from the holes' current positions (base hole -> disk
        holes), what the motor encoders measure."""
        H = np.array(self.cable_mo.position.value)[:, :2]
        lam = np.zeros((self.n, 2))
        for (k, si), idx in self.routes.items():
            pts = np.vstack([[0.0, (1.0, -1.0)[si] * self.offsets[k]], H[idx]])
            lam[k, si] = np.hypot(*np.diff(pts, axis=0).T).sum()
        return lam

    def tip(self):
        """Payload pose (x, y, psi) and velocity (xd, yd, psid)."""
        X = self.poses()[-1]
        v = np.array(self.mo.velocity.value)[-1]
        return X, np.array([v[0], v[1], v[5]])

    def step(self):
        """One time step; returns the payload's acceleration over it (finite difference of the
        velocities, what an accelerometer on the tip would integrate to)."""
        Sofa.Simulation.animate(self.root, self.dt)
        self.t += self.dt
        v = self.tip()[1]
        a = (v - self.v_prev) / self.dt
        self.v_prev = v.copy()
        return a

"""
cosserat_scene.py
The bowstring arm in SOFA with the Cosserat plugin (piecewise-constant-strain rods): the same
interface as scene.SofaArm, an independent, geometrically exact plant.

  rod       n segments of length 1, NS sections each; the unknowns are the sections' curvatures
            (DiscreteCosseratMapping -> rigid frames). Only in-plane bending is free (torsion and
            out-of-plane bending fixed), so the rod is exactly inextensible and unshearable, like
            the elastica model; bending stiffness B_i per segment (BeamHookeLawForceField,
            variant sections). Clamped at the base. Light rod mass on the frames.
  disks     the frames at s = 1, 2, 3 (no mass or thickness, as in the model).
  tendons   SoftRobots CableConstraint per tendon, force mode, hole to hole (value = T dt, as in
            scene.py: SoftRobots imposes the constraint impulse).
  payload   rigid mass (m, I) on the tip frame; external pushes on it.
  damping   Rayleigh mass damping alpha (payload damping c = alpha (m, m, I)).
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

from sofa.scene import PLUGINS, quat_z       # noqa: E402


class SofaCosseratArm:
    def __init__(self, nodes, T0, payload, offsets=(0.15, 0.2, 0.25), stiffness=(1.0, 1.0, 1.0),
                 NS=40, radius=0.01, rod_mass=0.015, alpha=0.1, dt=1e-3):
        """nodes: initial centre line, (n NS + 1, 3) poses x, y, theta at the section ends
        (e.g. scene.model_nodes(L, n, NS)); T0 (n, 2) initial tensions."""
        n = len(offsets)
        self.n, self.NS, self.dt, self.offsets = n, NS, dt, np.asarray(offsets, float)
        N = n * NS                                         # sections
        h = 1.0 / NS
        self.disk = [NS * j for j in range(n + 1)]         # frame indices of the disks
        nodes = np.asarray(nodes, float)
        kappa = np.diff(np.unwrap(nodes[:, 2])) / h        # initial section curvatures

        root = Sofa.Core.Node("root")
        root.dt = dt
        root.gravity = [0.0, 0.0, 0.0]
        root.addObject("RequiredPlugin", pluginName=PLUGINS + ["Cosserat"])
        root.addObject("FreeMotionAnimationLoop")
        root.addObject("BlockGaussSeidelConstraintSolver", tolerance=1e-10, maxIterations=500)
        sol = root.addChild("solver")
        sol.addObject("EulerImplicitSolver", rayleighMass=alpha, rayleighStiffness=0.0)
        sol.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
        sol.addObject("GenericConstraintCorrection")

        base = sol.addChild("base")
        self.base_mo = base.addObject("MechanicalObject", template="Rigid3d", name="baseMO",
                                      position=[[0, 0, 0, 0, 0, 0, 1]])
        base.addObject("FixedProjectiveConstraint", indices=[0])

        coord = sol.addChild("coord")
        self.strain = coord.addObject(
            "MechanicalObject", template="Vec3d", name="strainMO",
            position=[[0.0, 0.0, k] for k in kappa], rest_position=[[0.0, 0.0, 0.0]] * N)
        Iz = math.pi * radius ** 4 / 4
        E = [stiffness[i // NS] / Iz for i in range(N)]
        coord.addObject("BeamHookeLawForceField", crossSectionShape="circular", length=[h] * N,
                        radius=radius, youngModulus=E[0], poissonRatio=0.3,
                        variantSections=True, youngModulusList=E, poissonRatioList=[0.3] * N)
        coord.addObject("PartialFixedProjectiveConstraint", indices=list(range(N)),
                        fixedDirections=[1, 1, 0])          # planar: only bending about z

        frames = base.addChild("frames")
        coord.addChild(frames)
        pos = [[x, y, 0.0] + quat_z(t) for x, y, t in nodes]
        self.mo = frames.addObject("MechanicalObject", template="Rigid3d", name="framesMO",
                                   position=pos)
        j = h * h / 12
        frames.addObject("UniformMass", vertexMass=[rod_mass / (N + 1), 1.0, j, 0, 0, 0, j, 0, 0,
                                                    0, j])
        s = list(np.arange(N + 1) * h)
        frames.addObject("DiscreteCosseratMapping", curv_abs_input=s, curv_abs_output=s,
                         input1=self.strain.getLinkPath(), input2=self.base_mo.getLinkPath(),
                         output=self.mo.getLinkPath(), radius=radius)

        pl = frames.addChild("payload")
        pl.addObject("MechanicalObject", template="Rigid3d", position=[pos[-1]])
        Iu = payload.I / payload.m
        pl.addObject("UniformMass", vertexMass=[payload.m, 1.0, Iu, 0, 0, 0, Iu, 0, 0, 0, Iu])
        self.push = pl.addObject("ConstantForceField", indices=[0], forces=[[0.0] * 6])
        pl.addObject("SubsetMapping", indices=[N], handleTopologyChange=False)

        holes, rigid, self.hole_index = [], [], {}
        for jd in range(1, n + 1):
            for k in range(n):
                for si, sg in enumerate((1.0, -1.0)):
                    self.hole_index[(jd, k, si)] = len(holes)
                    holes.append([0.0, sg * self.offsets[k], 0.0])
                    rigid.append(self.disk[jd])
        cab = frames.addChild("cables")
        self.cable_mo = cab.addObject("MechanicalObject", template="Vec3d", position=holes)
        self.cables, self.routes = {}, {}
        for k in range(n):
            for si, sg in enumerate((1.0, -1.0)):
                idx = [self.hole_index[(jd, k, si)] for jd in range(1, k + 2)]
                self.routes[(k, si)] = idx
                self.cables[(k, si)] = cab.addObject(
                    "CableConstraint", name=f"cable{k}{'pm'[si]}", indices=idx,
                    pullPoint=[0.0, sg * self.offsets[k], 0.0], valueType="force",
                    value=float(T0[k][si]) * dt)
        cab.addObject("RigidMapping", globalToLocalCoords=False, rigidIndexPerPoint=rigid)
        self.root = root
        Sofa.Simulation.initRoot(root)
        self.t = 0.0
        # The mapped frame velocities of DiscreteCosseratMapping are not consistent with its
        # positions (the linear part differs from d(position)/dt, even in sign right after a
        # tension change), so the tip velocity and acceleration are taken from the positions.
        self.X_prev = self.poses()[-1].copy()
        self.V = np.zeros(3)

    # ------------------------------------------------------------ io (as scene.SofaArm)
    def set_tensions(self, T):
        for (k, si), c in self.cables.items():
            c.value = [float(T[k][si]) * self.dt]

    def set_push(self, w):
        self.push.forces = [[float(w[0]), float(w[1]), 0.0, 0.0, 0.0, float(w[2])]]

    def set_damping(self, alpha):
        self.root.solver.EulerImplicitSolver.rayleighMass = alpha

    def nodes(self):
        p = np.array(self.mo.position.value)
        return np.column_stack([p[:, 0], p[:, 1], 2 * np.arctan2(p[:, 5], p[:, 6])])

    def poses(self):
        return self.nodes()[self.disk]

    def tendon_lengths(self):
        H = np.array(self.cable_mo.position.value)[:, :2]
        lam = np.zeros((self.n, 2))
        for (k, si), idx in self.routes.items():
            pts = np.vstack([[0.0, (1.0, -1.0)[si] * self.offsets[k]], H[idx]])
            lam[k, si] = np.hypot(*np.diff(pts, axis=0).T).sum()
        return lam

    def tip(self):
        return self.poses()[-1], self.V.copy()

    def step(self):
        """One time step; returns the payload's acceleration over it, from the tip positions
        (second differences, what an accelerometer on the tip measures)."""
        Sofa.Simulation.animate(self.root, self.dt)
        self.t += self.dt
        X = self.poses()[-1]
        d = X - self.X_prev
        d[2] = (d[2] + math.pi) % (2 * math.pi) - math.pi
        V = d / self.dt
        a = (V - self.V) / self.dt
        self.X_prev, self.V = X.copy(), V
        return a

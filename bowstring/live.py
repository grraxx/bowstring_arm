"""
live.py
Drive the closed loop live, in real time, from velocity commands (the console's keyboard).

A LiveSession builds the same closed loop as a console run (experiment.Session, from the panel's
parameters) and steps it on a background thread at the control rate, paced to the wall clock.
The reference is reference.Live: the command (ux, uy, ur) in [-1, 1] sets the payload's velocity,
with bounded acceleration and jerk. Each step the feedforward's `choose` (safe mode) takes the
commanded reference, or braking, or holding still, whichever the arm can do and still stop.

Real time needs the model plant: the SOFA plants are slower than real time. A command older than
0.5 s counts as released (a lost key-up or a closed tab must not keep the arm moving).
"""
import collections
import math
import threading
import time

import numpy as np

from .closed_loop import PlantError
from .experiment import DT, Session, resolve
from .reference import Live as LiveReference

CMD_TIMEOUT = 0.5          # s
FRAME_EVERY = 2            # control steps per frame sent to the page (50 Hz)


class LiveSession:
    def __init__(self, cfg):
        P = resolve(cfg)
        if P["sim.plant"] != "model":
            raise ValueError("live mode runs the model plant only: the SOFA plants are slower "
                             "than real time")
        self.P = P
        self.free = bool(P["traj.free"])
        self.X_start = np.array([P["traj.x"], P["traj.y"], math.radians(P["traj.psi"])])
        self.lock = threading.Lock()
        self.cmd, self.cmd_time = np.zeros(3), 0.0
        self.frames = collections.deque(maxlen=1500)     # 30 s at 50 Hz
        self.seq = 0
        self.state, self.message = "starting", "finding the start pose"
        self.head = None
        self.load = 0.0                                   # compute time / control period
        self.late = 0                                     # steps that missed their deadline
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ from the page
    def command(self, u):
        u = np.clip(np.asarray(u, float)[:3], -1.0, 1.0)
        with self.lock:
            self.cmd, self.cmd_time = u, time.monotonic()

    def stop(self):
        self._stop.set()

    def snapshot(self, start):
        """Frames with seq >= start, and the session's status."""
        with self.lock:
            frames = [f for f in self.frames if f["seq"] >= start]
            return {"state": self.state, "message": self.message, "head": self.head,
                    "frames": frames, "next": self.seq, "load": self.load, "late": self.late}

    # ------------------------------------------------------------ loop
    def _status(self, state, message=""):
        with self.lock:
            self.state, self.message = state, message

    def _run(self):
        P = self.P
        try:
            s = Session(P, self.X_start, self.free,
                        report=lambda **kw: kw.get("message") and self._status("starting",
                                                                               kw["message"]))
        except (PlantError, RuntimeError, ValueError, np.linalg.LinAlgError) as err:
            return self._status("failed", f"could not start: {err}")
        ref = LiveReference(s.X0, v_max=P["live.v_max"], w_max=P["live.w_max"],
                            a_max=P["live.a_max"], lo=[-3.2, -3.2, -np.inf],
                            hi=[3.2, 3.2, np.inf])
        with self.lock:
            self.head = {"offsets": list(s.true_arm.offsets), "start": s.X0.tolist(),
                         "free": self.free, "dt": DT}
        self._status("running")
        busy = collections.deque(maxlen=50)
        deadline = time.perf_counter()
        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                with self.lock:
                    cmd = self.cmd.copy() if time.monotonic() - self.cmd_time < CMD_TIMEOUT \
                        else np.zeros(3)
                t = (s.k + 1) * DT
                _, info_ff = s.ff.choose(t, ref.candidates(DT, cmd), safe=True)
                ref.accept(info_ff["choice"])
                if self.free:
                    # The controller picks the angle; the reference follows it (no pull).
                    ref.sync_angle(info_ff["P"][-1][2])
                Xr, Xdr, Xddr = ref.X.copy(), ref.V.copy(), ref.A.copy()
                row = s.advance(Xr, Xdr, Xddr, info_ff)
                if s.k % FRAME_EVERY == 0:
                    f = s.frame(0)
                    if f is not None:
                        f.update({"seq": self.seq, "t": round(row["t"], 3),
                                  "X": np.round(row["X"], 5).tolist(),
                                  "X_ref": np.round(row["X_ref"], 5).tolist(),
                                  "X_est": np.round(row["X_est"], 5).tolist(),
                                  "T": np.round(row["T"], 4).tolist(),
                                  "err": round(row["err"], 6),
                                  "ok": row["ok"], "choice": info_ff["choice"],
                                  "reason": info_ff["reason"], "cmd": cmd.round(2).tolist()})
                        with self.lock:
                            self.frames.append(f)
                            self.seq += 1
                busy.append(time.perf_counter() - t0)
                with self.lock:
                    self.load = float(np.mean(busy)) / DT
                deadline += DT
                wait = deadline - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    with self.lock:
                        self.late += 1
                    if wait < -0.1:          # far behind (e.g. the machine stalled): resync
                        deadline = time.perf_counter()
        except (PlantError, RuntimeError, ValueError, np.linalg.LinAlgError) as err:
            return self._status("failed", f"stopped at t = {s.k * DT:.2f} s: {err}")
        self._status("stopped", f"stopped at t = {s.k * DT:.2f} s")

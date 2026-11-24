"""
server.py
The bowstring console: a local web app for running and viewing closed-loop simulations.

    console/run.sh [--port 8765] [--workers 6]      (or: python console/server.py ...)
    then open http://localhost:8765 (from Windows too: WSL forwards localhost)

API (JSON):
  GET    /api/schema            parameter schema, trajectory presets, defaults
  GET    /api/runs              all runs (newest first): status, progress, summary
  POST   /api/runs              {config, label} -> {id}; queued
  GET    /api/runs/<id>         the run's result (time series, arm frames) when done, else status
  GET    /api/runs/<id>/live?from=n   the live stream from line n: {lines, next, status}
  DELETE /api/runs/<id>         cancel if running, then delete
  POST   /api/sweeps            {config, key, values, label} -> {id, runs}
  POST   /api/live/start        {config}: start the live (keyboard) session, replacing any other
  POST   /api/live/cmd          {u: [ux, uy, ur]} in [-1, 1]; resend at least every 0.5 s
  POST   /api/live/stop
  GET    /api/live?from=n       live frames from seq n, status, compute load
The live session runs on a thread in this process (bowstring/live.py), in real time.
Each run is a subprocess `python -m bowstring.experiment <dir>` in this interpreter; runs live in
data/console/runs/<id>/ and survive restarts.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
RUNS = ROOT / "data" / "console" / "runs"
sys.path.insert(0, str(ROOT))
from bowstring.experiment import IMU_GRADES, PRESETS, SCHEMA, DEFAULTS, UNITS_NOTE  # noqa: E402
from bowstring.live import LiveSession  # noqa: E402

LIVE = {"session": None}
LIVE_LOCK = threading.Lock()

RUN_ID = re.compile(r"^[0-9a-z\-]{6,40}$")


class Queue:
    def __init__(self, workers):
        self.lock = threading.Lock()
        self.pending, self.procs, self.workers = [], {}, workers
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, run_id):
        with self.lock:
            self.pending.append(run_id)

    def cancel(self, run_id):
        with self.lock:
            if run_id in self.pending:
                self.pending.remove(run_id)
            p = self.procs.pop(run_id, None)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()

    def _loop(self):
        while True:
            with self.lock:
                for rid, p in list(self.procs.items()):
                    if p.poll() is not None:
                        self.procs.pop(rid)
                        st = read_json(RUNS / rid / "status.json") or {}
                        if st.get("state") not in ("done", "failed"):
                            write_json(RUNS / rid / "status.json",
                                       {"state": "failed", "progress": 1.0,
                                        "message": f"the simulation process exited ({p.returncode})"})
                while self.pending and len(self.procs) < self.workers:
                    rid = self.pending.pop(0)
                    d = RUNS / rid
                    if not d.exists():
                        continue
                    env = dict(os.environ, OMP_NUM_THREADS="1")
                    log = open(d / "log.txt", "w")
                    self.procs[rid] = subprocess.Popen(
                        [sys.executable, "-m", "bowstring.experiment", str(d)], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, env=env)
                    write_json(d / "status.json", {"state": "starting", "progress": 0.0,
                                                   "message": "starting"})
            time.sleep(0.3)

    def position(self, run_id):
        with self.lock:
            return self.pending.index(run_id) + 1 if run_id in self.pending else 0


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def new_run(config, label, sweep=None):
    rid = time.strftime("%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = RUNS / rid
    d.mkdir(parents=True)
    clean = {k: v for k, v in config.items() if k in DEFAULTS}
    write_json(d / "config.json", clean)
    write_json(d / "meta.json", {"id": rid, "label": label or "", "created": time.time(),
                                 "sweep": sweep})
    write_json(d / "status.json", {"state": "queued", "progress": 0.0, "message": "queued"})
    QUEUE.submit(rid)
    return rid


def run_listing():
    out = []
    if not RUNS.exists():
        return out
    for d in RUNS.iterdir():
        meta = read_json(d / "meta.json")
        if not meta:
            continue
        st = read_json(d / "status.json") or {"state": "unknown", "progress": 0}
        cfg = read_json(d / "config.json") or {}
        if st.get("state") in ("queued", "starting", "running") and \
                d.name not in QUEUE.procs and d.name not in QUEUE.pending:
            st = {"state": "failed", "progress": 1.0, "message": "interrupted (server restarted)"}
        out.append({**meta, "config": cfg, "status": st, "queue": QUEUE.position(d.name)})
    out.sort(key=lambda r: r["created"], reverse=True)
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_raw(self, payload):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/schema":
            return self._json({"schema": SCHEMA, "defaults": DEFAULTS,
                               "presets": {k: {"description": d, "values": v}
                                           for k, (d, v) in PRESETS.items()},
                               "imu_grades": list(IMU_GRADES), "units_note": UNITS_NOTE})
        if self.path == "/api/runs":
            return self._json(run_listing())
        m = re.match(r"^/api/live(?:\?from=(\d+))?$", self.path)
        if m:
            ls = LIVE["session"]
            if ls is None:
                return self._json({"state": "idle", "frames": [], "next": 0})
            return self._json(ls.snapshot(int(m.group(1) or 0)))
        m = re.match(r"^/api/runs/([^/]+)/live(?:\?from=(\d+))?$", self.path)
        if m:
            rid, start = m.group(1), int(m.group(2) or 0)
            if not RUN_ID.match(rid) or not (RUNS / rid).exists():
                return self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
            lines = []
            path = RUNS / rid / "live.jsonl"
            if path.exists():
                text = path.read_text()
                complete = text[:text.rfind("\n") + 1].splitlines()   # whole lines only
                lines = [json.loads(x) for x in complete[start:]]
            body = json.dumps({"lines": lines, "next": start + len(lines),
                               "status": read_json(RUNS / rid / "status.json")})
            return self._json_raw(body.encode())
        m = re.match(r"^/api/runs/([^/]+)$", self.path)
        if m:
            rid = m.group(1)
            if not RUN_ID.match(rid) or not (RUNS / rid).exists():
                return self._json({"error": "no such run"}, HTTPStatus.NOT_FOUND)
            res = RUNS / rid / "result.json"
            if res.exists():
                meta = json.dumps(read_json(RUNS / rid / "meta.json") or {}).encode()
                payload = b'{"meta":' + meta + b',"run":' + res.read_bytes() + b'}'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            return self._json({"meta": read_json(RUNS / rid / "meta.json"),
                               "status": read_json(RUNS / rid / "status.json")})
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            body = self._body()
        except ValueError:
            return self._json({"error": "bad JSON"}, HTTPStatus.BAD_REQUEST)
        if self.path == "/api/runs":
            rid = new_run(body.get("config", {}), body.get("label", ""))
            return self._json({"id": rid})
        if self.path == "/api/live/cmd":
            ls = LIVE["session"]
            if ls is not None:
                try:
                    ls.command(body.get("u", [0, 0, 0]))
                except (TypeError, ValueError):
                    return self._json({"error": "u must be three numbers"}, HTTPStatus.BAD_REQUEST)
            return self._json({"ok": ls is not None})
        if self.path == "/api/live/start":
            cfg = {k: v for k, v in body.get("config", {}).items() if k in DEFAULTS}
            with LIVE_LOCK:
                if LIVE["session"] is not None:
                    LIVE["session"].stop()
                try:
                    LIVE["session"] = LiveSession(cfg)
                except ValueError as err:
                    LIVE["session"] = None
                    return self._json({"error": str(err)}, HTTPStatus.BAD_REQUEST)
            return self._json({"ok": True})
        if self.path == "/api/live/stop":
            with LIVE_LOCK:
                if LIVE["session"] is not None:
                    LIVE["session"].stop()
            return self._json({"ok": True})
        if self.path == "/api/sweeps":
            key, values = body.get("key"), body.get("values") or []
            if key not in DEFAULTS or not values or len(values) > 40:
                return self._json({"error": "choose a parameter and 1-40 values"},
                                  HTTPStatus.BAD_REQUEST)
            sid = "sw-" + time.strftime("%m%d-%H%M%S")
            label = body.get("label") or key
            ids = []
            for v in values:
                cfg = dict(body.get("config", {}), **{key: v})
                ids.append(new_run(cfg, f"{label} = {v:g}" if isinstance(v, (int, float))
                                   else f"{label} = {v}",
                                   sweep={"id": sid, "key": key, "value": v, "label": label}))
            return self._json({"id": sid, "runs": ids})
        return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self):
        m = re.match(r"^/api/runs/([^/]+)$", self.path)
        if not m or not RUN_ID.match(m.group(1)):
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        rid = m.group(1)
        QUEUE.cancel(rid)
        shutil.rmtree(RUNS / rid, ignore_errors=True)
        return self._json({"deleted": rid})


def main():
    global QUEUE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 4)))
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    QUEUE = Queue(args.workers)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"bowstring console on http://localhost:{args.port}  ({args.workers} parallel runs)",
          flush=True)
    srv.serve_forever()


QUEUE = None
if __name__ == "__main__":
    main()

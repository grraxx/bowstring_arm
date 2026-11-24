/* app.js: the bowstring console. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const ICON = {
  chevron: '<svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>',
  reset: '<svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/></svg>',
  trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
};
const S = {
  schema: [], defaults: {}, presets: {}, cfg: {}, runs: [], selected: null, ticked: new Set(),
  cache: new Map(), tab: "replay", follow: null, sweep: null,
};
const store = {
  get(k, d) { try { const v = localStorage.getItem("bowstring." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("bowstring." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const api = {
  async get(p) { const r = await fetch(p); if (!r.ok) throw new Error((await r.json()).error || r.statusText); return r.json(); },
  async post(p, b) {
    const r = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
    const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
  },
  async del(p) { const r = await fetch(p, { method: "DELETE" }); return r.json(); },
};
function toast(msg, bad = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast show" + (bad ? " bad" : "");
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.className = "toast" + (bad ? " bad" : "")), 2600);
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const field = (k) => S.schema.find((f) => f.key === k);
const e3 = (v) => (v == null || !isFinite(v) ? "–" : (1e3 * v).toFixed(v < 0.01 ? 2 : 1));

/* ================================================================== parameters */
function defaultFor(k) {
  const p = S.presets[S.cfg["traj.preset"]];
  if (k.startsWith("traj.") && k !== "traj.preset" && p && p.values[k] !== undefined) return p.values[k];
  return S.defaults[k];
}
function linkedValue(k) {
  return S.cfg["model.link"] && k.startsWith("model.") && k !== "model.link" ? S.cfg["robot." + k.slice(6)] : S.cfg[k];
}
function isModified(k) {
  const v = S.cfg[k], d = defaultFor(k);
  return typeof v === "number" ? Math.abs(v - d) > 1e-12 * Math.max(1, Math.abs(d)) : v !== d;
}
function fmtValue(f, v) {
  if (f.kind === "log") return Number(v).toExponential(0);
  return v;
}
function buildParams() {
  const body = $("#params-body"); body.innerHTML = "";
  const groups = [...new Set(S.schema.map((f) => f.group))];
  const openState = store.get("open", { Trajectory: true, Controller: true });
  if (S.unitsNote) {
    const key = document.createElement("details"); key.className = "units-key"; key.open = !!openState.__units;
    key.innerHTML = `<summary>${ICON.chevron}<span>Units</span><span class="units-hint">non-dimensional</span></summary><dl>${S.unitsNote.split(" · ").map((t) => {
      const i = t.indexOf(" "); return `<dt>${esc(t.slice(0, i))}</dt><dd>${esc(t.slice(i + 1))}</dd>`; }).join("")}</dl>`;
    key.addEventListener("toggle", () => { const o = store.get("open", {}); o.__units = key.open; store.set("open", o); });
    body.appendChild(key);
  }
  for (const g of groups) {
    const det = document.createElement("details"); det.className = "group"; det.open = !!openState[g];
    det.innerHTML = `<summary>${ICON.chevron}<span>${esc(g)}</span><span class="mod-count"></span></summary><div class="group-body"></div>`;
    det.addEventListener("toggle", () => { const o = store.get("open", {}); o[g] = det.open; store.set("open", o); });
    const gb = $(".group-body", det);
    if (g === "Controller's model") gb.insertAdjacentHTML("beforeend", `<p class="group-note">What the controller believes. Untick to make it differ from the true robot.</p>`);
    if (g === "Robot (true plant)") gb.insertAdjacentHTML("beforeend", `<p class="group-note">The robot that is simulated.</p>`);
    for (const f of S.schema.filter((x) => x.group === g)) gb.appendChild(buildField(f));
    body.appendChild(det);
  }
  refreshParams();
}
function buildField(f) {
  const row = document.createElement("div"); row.className = "field" + (f.kind === "select" ? " select" : ""); row.dataset.key = f.key;
  const id = "f-" + f.key.replace(/\W/g, "-");
  const unit = f.unit ? `<span class="unit">${esc(f.unit)}</span>` : "";
  const tip = f.help ? `${f.label}${f.unit ? ` [${f.unit}]` : ""}: ${f.help}` : `${f.label}${f.unit ? ` [${f.unit}]` : ""}`;
  let ctl;
  if (f.kind === "select") {
    ctl = `<select id="${id}">${f.options.map((o) => `<option value="${esc(o)}">${esc(o)}</option>`).join("")}</select>`;
  } else if (f.kind === "bool") {
    ctl = `<label class="switch"><input type="checkbox" id="${id}"></label>`;
  } else {
    const step = f.kind === "int" ? 1 : f.step == null ? "any" : f.step;
    ctl = `<input type="number" id="${id}" step="${step}" ${f.min != null ? `min="${f.min}"` : ""} ${f.max != null ? `max="${f.max}"` : ""}>`;
  }
  row.innerHTML = `<label for="${id}" title="${esc(tip)}">${esc(f.label)}</label>${f.kind === "select" ? "" : unit || `<span class="unit"></span>`}
    <div class="control">${ctl}</div>
    <button class="reset-field" title="Reset to ${esc(String(fmtValue(f, defaultFor(f.key))))}" aria-label="Reset ${esc(f.label)}">${ICON.reset}</button>`;
  const input = row.querySelector("input, select");
  input.addEventListener(f.kind === "bool" || f.kind === "select" ? "change" : "input", () => onInput(f, input));
  row.querySelector(".reset-field").addEventListener("click", () => { setValue(f.key, defaultFor(f.key)); });
  return row;
}
function onInput(f, input) {
  let v;
  if (f.kind === "bool") v = input.checked;
  else if (f.kind === "select") v = input.value;
  else {
    v = f.kind === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
    const bad = !isFinite(v) || (f.min != null && v < f.min) || (f.max != null && v > f.max);
    input.classList.toggle("invalid", bad);
    if (bad) { note(`${f.label}: enter a value between ${f.min} and ${f.max}.`, true); return; }
  }
  setValue(f.key, v, true);
}
function setValue(k, v, fromInput = false) {
  S.cfg[k] = v;
  if (k === "traj.preset") for (const f of S.schema) if (f.key.startsWith("traj.") && f.key !== k) S.cfg[f.key] = defaultFor(f.key);
  store.set("cfg", S.cfg);
  refreshParams(fromInput ? k : null);
  note("");
  updateSweepBox();
}
function refreshParams(skipKey) {
  const preset = S.cfg["traj.preset"];
  for (const row of document.querySelectorAll(".field")) {
    const f = field(row.dataset.key), input = row.querySelector("input, select");
    const v = linkedValue(f.key);
    if (f.key !== skipKey) {
      if (f.kind === "bool") input.checked = !!v;
      else if (f.kind === "log") input.value = Number(v).toExponential(0).replace("e+0", "").replace(/e\+?/, "e");
      else input.value = v;
      input.classList.remove("invalid");
    }
    const inactive = !!f.active && !Object.entries(f.active).every(([k, vs]) => vs.includes(S.cfg[k]));
    const disabled = inactive || (S.cfg["model.link"] && f.key.startsWith("model.") && f.key !== "model.link");
    input.disabled = disabled;
    row.classList.toggle("disabled", disabled);
    row.classList.toggle("hidden", !visible(f, S.cfg));
    row.classList.toggle("modified", !disabled && isModified(f.key));
    row.querySelector(".reset-field").title = `Reset to ${fmtValue(f, defaultFor(f.key))}`;
  }
  for (const det of document.querySelectorAll(".group")) {
    const n = det.querySelectorAll(".field.modified:not(.hidden)").length;
    $(".mod-count", det).textContent = n ? `${n} changed` : "";
  }
  const p = S.presets[preset];
  const tr = document.querySelector('.field[data-key="traj.preset"] label');
  if (tr && p) tr.title = p.description;
}
function labelUnit(f) { return f.unit && f.unit !== "–" ? `${f.label} [${f.unit}]` : f.label; }
function visible(f, cfg) {
  if (!f.show) return true;
  if (Array.isArray(f.show)) return f.show.includes(cfg["traj.preset"]);
  return Object.entries(f.show).every(([k, vs]) => vs.includes(cfg[k]));
}
function note(msg, bad = false) { const n = $("#foot-note"); n.textContent = msg; n.classList.toggle("bad", bad); }
function autoLabel(cfg) {
  const ch = S.schema.filter((f) => f.key !== "traj.preset" && f.key !== "sim.plant" && !(cfg["model.link"] && f.key.startsWith("model."))
    && visible(f, cfg) && isModified(f.key)).map((f) => f.key.split(".")[1]);
  return `${cfg["traj.preset"]} · ${cfg["sim.plant"]}` + (ch.length ? ` · ${ch.slice(0, 3).join(", ")}${ch.length > 3 ? "…" : ""}` : "");
}
function configForRun() {
  const c = { ...S.cfg };
  if (c["model.link"]) for (const k of Object.keys(c)) if (k.startsWith("model.") && k !== "model.link") c[k] = c["robot." + k.slice(6)];
  return c;
}

/* ================================================================== sweeps (panel) */
function sweepValues() {
  const key = $("#sweep-key").value, f = field(key);
  let a = parseFloat($("#sweep-from").value), b = parseFloat($("#sweep-to").value);
  const n = Math.max(2, Math.min(20, parseInt($("#sweep-n").value, 10) || 2));
  if (!isFinite(a) || !isFinite(b)) return [];
  const log = $("#sweep-log").checked && a > 0 && b > 0;
  const vals = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    let v = log ? Math.exp(Math.log(a) + u * (Math.log(b) - Math.log(a))) : a + u * (b - a);
    v = f.kind === "int" ? Math.round(v) : +v.toPrecision(4);
    vals.push(v);
  }
  return [...new Set(vals)];
}
function updateSweepBox() {
  const open = !$("#sweep-box").hidden;
  $("#run-btn").firstChild.textContent = open ? `Run sweep (${sweepValues().length} runs) ` : "Run ";
  if (open) $("#sweep-preview").textContent = sweepValues().map((v) => fmt(v)).join(", ");
}
function buildSweepBox() {
  const sel = $("#sweep-key");
  sel.innerHTML = S.schema.filter((f) => ["number", "int", "log"].includes(f.kind) && !f.key.startsWith("model."))
    .map((f) => `<option value="${f.key}">${esc(f.group.split(" ")[0])}: ${esc(labelUnit(f))}</option>`).join("");
  const setRange = () => {
    const f = field(sel.value), v = Number(S.cfg[f.key]);
    $("#sweep-from").value = +(f.kind === "log" ? v / 10 : Math.max(f.min ?? -Infinity, v * 0.5)).toPrecision(3);
    $("#sweep-to").value = +(f.kind === "log" ? v * 10 : Math.min(f.max ?? Infinity, v * 1.5 || 1)).toPrecision(3);
    $("#sweep-log").checked = f.kind === "log";
    updateSweepBox();
  };
  sel.addEventListener("change", setRange);
  for (const id of ["#sweep-from", "#sweep-to", "#sweep-n", "#sweep-log"]) $(id).addEventListener("input", updateSweepBox);
  $("#sweep-toggle").addEventListener("click", () => {
    const box = $("#sweep-box"); box.hidden = !box.hidden;
    $("#sweep-toggle").setAttribute("aria-expanded", String(!box.hidden));
    if (!box.hidden) setRange(); else updateSweepBox();
  });
}

/* ================================================================== submit */
async function submit() {
  if (document.querySelector(".control .invalid")) { note("Fix the highlighted value first.", true); return; }
  const btn = $("#run-btn"); btn.disabled = true;
  const cfg = configForRun(), label = $("#run-label").value.trim();
  try {
    if (!$("#sweep-box").hidden) {
      const key = $("#sweep-key").value, vals = sweepValues();
      if (vals.length < 2) throw new Error("A sweep needs two or more distinct values.");
      const r = await api.post("/api/sweeps", { config: cfg, key, values: vals, label: label || field(key).label });
      S.sweep = r.id; toast(`Sweep queued: ${vals.length} runs`);
      await poll(); setTab("sweeps");
    } else {
      const r = await api.post("/api/runs", { config: cfg, label: label || autoLabel(cfg) });
      S.follow = r.id; toast("Run queued");
      await poll(); select(r.id);
    }
    $("#run-label").value = "";
  } catch (err) { note(err.message, true); } finally { btn.disabled = false; }
}

/* ================================================================== runs list */
function runName(r) { return r.label || r.id; }
function renderRuns() {
  const list = $("#run-list"); list.innerHTML = "";
  $("#runs-empty").hidden = S.runs.length > 0;
  for (const r of S.runs) {
    const st = r.status || {}, state = st.state || "unknown";
    const li = document.createElement("li");
    li.className = "run" + (r.id === S.selected ? " selected" : "");
    const done = state === "done";
    const metric = done && st.summary ? `<span title="mean position error ×10⁻³ segment lengths">${e3(st.summary.err_mean)}</span>`
      : state === "queued" ? (r.queue ? `#${r.queue} in queue` : "queued")
      : state === "failed" ? `<span class="state failed">failed</span>`
      : `<span class="state ${state}">${Math.round(100 * (st.progress || 0))}%</span>`;
    const when = new Date(r.created * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    li.innerHTML = `<input type="checkbox" ${done ? "" : "disabled"} ${S.ticked.has(r.id) ? "checked" : ""} aria-label="Compare ${esc(runName(r))}" title="${done ? "Tick to compare" : "Comparable when finished"}">
      <div class="run-main"><div class="run-name" title="${esc(runName(r))}">${esc(runName(r))}</div>
      <div class="run-sub">${esc(r.config["sim.plant"] || "model")} · ${when}${r.sweep ? ` · <span class="sweep-tag">sweep</span>` : ""}</div></div>
      <div style="display:grid;justify-items:end;gap:2px"><span class="run-metric">${metric}</span>
      <button class="run-del" title="${done || state === "failed" ? "Delete" : "Cancel and delete"}" aria-label="Delete">${ICON.trash}</button></div>
      ${["queued", "starting", "running"].includes(state) ? `<div class="progress"><i style="width:${Math.round(100 * (st.progress || 0))}%"></i></div>` : ""}`;
    li.addEventListener("click", (e) => { if (e.target.closest("input, button")) return; select(r.id); });
    li.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) S.ticked.add(r.id); else S.ticked.delete(r.id);
      $("#compare-count").textContent = S.ticked.size || ""; if (S.tab === "compare") renderCompare();
    });
    li.querySelector(".run-del").addEventListener("click", async () => {
      await api.del("/api/runs/" + r.id); S.ticked.delete(r.id); S.cache.delete(r.id);
      if (S.selected === r.id) { S.selected = null; showReplay(null); }
      await poll();
    });
    list.appendChild(li);
  }
  const active = S.runs.filter((r) => ["running", "starting"].includes(r.status?.state)).length;
  const queued = S.runs.filter((r) => r.status?.state === "queued").length;
  $("#queue-state").textContent = active || queued ? `${active} running · ${queued} queued` : `${S.runs.length} runs · idle`;
  $("#compare-count").textContent = S.ticked.size || "";
}
let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  try {
    const prev = new Map(S.runs.map((r) => [r.id, r.status?.state]));
    S.runs = await api.get("/api/runs");
    for (const id of [...S.ticked]) if (!S.runs.find((r) => r.id === id)) S.ticked.delete(id);
    renderRuns();
    const sel = S.runs.find((r) => r.id === S.selected);
    if (sel && prev.get(sel.id) !== sel.status?.state) showReplay(sel.id);
    else if (sel && S.tab === "replay" && sel.status?.state === "queued") showPending(sel);
    if (S.tab === "sweeps") renderSweeps(false);
  } catch (err) { $("#queue-state").textContent = "server not reachable"; }
  const busy = S.runs.some((r) => ["queued", "starting", "running"].includes(r.status?.state));
  pollTimer = setTimeout(poll, busy ? 1200 : 5000);
}
function select(id) { S.selected = id; renderRuns(); setTab("replay"); showReplay(id); $("#runs").classList.remove("open"); }
async function loadRun(id) {
  if (S.cache.has(id)) return S.cache.get(id);
  const r = await api.get("/api/runs/" + id);
  if (r.run) S.cache.set(id, r);
  return r;
}

/* ================================================================== replay */
const R = { arm: null, charts: [], run: null, playing: false, simT: 0, last: 0, frameT: [] };
function statsHTML(sm, cfg) {
  const items = [
    ["Mean error", e3(sm.err_mean), "×10⁻³"], ["Max error", e3(sm.err_max), "×10⁻³"], ["RMS error", e3(sm.err_rms), "×10⁻³"],
    ["Max angle error", sm.psi_err_max_deg.toFixed(2), "deg"], ["Max estimate error", e3(sm.est_err_max), "×10⁻³"],
    ["Max tension", sm.T_max.toFixed(2), ""], ["Controller step", `${sm.ctrl_ms_mean.toFixed(1)}`, `ms (p99 ${sm.ctrl_ms_p99.toFixed(1)})`],
    ["Simulated", `${sm.simulated.toFixed(1)} / ${sm.duration.toFixed(1)}`, "s"], ["Wall time", isFinite(sm.wall_s) ? sm.wall_s.toFixed(0) : "–", "s"],
  ];
  return items.map(([k, v, u]) => `<div><dt>${k}</dt><dd>${v}<small>${u}</small></dd></div>`).join("") +
    `<div class="status-line ${sm.state === "failed" ? "bad" : ""}">${sm.state === "running" ? "<b>Running</b> · the view follows the simulation as it computes · " : ""}${esc(cfg["sim.plant"])} plant · sensing: ${esc(cfg["sense.mode"])} · errors are measured from the simulator's true payload pose to the commanded reference at the same instant. ${sm.state === "failed" ? esc(sm.status) : ""}</div>`;
}
function showPending(r) {
  $("#replay-empty").hidden = false; $("#replay").hidden = true;
  const st = r.status || {};
  $("#replay-empty").innerHTML = st.state === "failed"
    ? `<h3>${esc(runName(r))}</h3><p style="color:#f3a0a0">This run failed: ${esc(st.message || "unknown error")}</p><p style="margin-top:8px">Check that the trajectory stays in reach (try a smaller size or another centre), or that the robot and model parameters are not too far apart.</p>`
    : `<h3>${esc(runName(r))}</h3><p>${st.state === "queued" ? "Waiting in the queue." : esc(st.message || "Starting")}</p><p style="margin-top:8px;color:var(--text-3)">The live view opens here as soon as the simulation starts stepping.</p>`;
}
async function showReplay(id) {
  stop(); stopLive();
  if (!id) { $("#replay-empty").hidden = false; $("#replay").hidden = true; $("#replay-empty").innerHTML = `<h3>No run selected</h3><p>Pick a sample trajectory and its parameters on the left, then press <b>Run</b>. Runs appear in the list on the right; select one to watch it live or replay it.</p>`; return; }
  const r = S.runs.find((x) => x.id === id), st = r?.status?.state;
  if (st === "queued") return showPending(r);
  if (st === "starting" || st === "running") return startLive(id, r);
  let data;
  try { data = await loadRun(id); } catch (err) { toast(err.message, true); return; }
  if (S.selected !== id) return;
  if (!data.run) { if (r) showPending(r); return; }           // failed before the first step
  renderRun(data.run, data.meta.label || id);
  const end = R.frameT[R.frameT.length - 1];
  seek(R.pendingSeek ?? (data.run.summary.state === "failed" ? Math.max(0, end - 3) : 0)); R.pendingSeek = null;
}
function pushTimes(s) {
  const on = (w) => Math.abs(w[0]) + Math.abs(w[1]) + Math.abs(w[2]) > 0, out = [];
  s.w_ext.forEach((w, i) => { if (on(w) && !(i && on(s.w_ext[i - 1]))) out.push(s.t[i]); });
  return out;
}
function renderRun(run, name, live = false) {
  const cfg = run.config;
  R.run = run; R.live = live; R.follow = live;
  $("#replay-empty").hidden = true; $("#replay").hidden = false;
  $("#replay").classList.toggle("live", live);
  $("#replay-name").textContent = name;
  const b = $("#replay-badge");
  b.textContent = live ? "live" : run.summary.state === "failed" ? "stopped early" : `${cfg["traj.preset"]} · ${run.summary.duration.toFixed(0)} s`;
  b.className = "badge" + (live ? " live" : run.summary.state === "failed" ? " failed" : "");
  if (!R.arm) R.arm = new ArmView($("#arm-canvas"));
  const box = $("#replay-plots"); box.innerHTML = "";
  const xMax = run.duration ?? run.summary.duration;
  R.charts = [
    new LineChart(box, { title: "Payload position error", sub: "×10⁻³ segment lengths", zero: true, xMax, yFmt: (v) => fmt(v, 3) }),
    new LineChart(box, { title: "Payload angle", sub: "degrees", xMax }),
    new LineChart(box, { title: "Tendon tensions", sub: "solid + side, dashed − side", zero: true, xMax }),
    new LineChart(box, { title: "Integral wrench", sub: "M·Ki·∫e (force x, y; moment)", xMax })];
  for (const c of R.charts) c.onClick = (t) => { R.follow = false; updateFollow(); stop(); seek(t); };
  fillRun();
}
function fillRun() {
  const run = R.run, s = run.series, [c1, c2, c3, c4] = R.charts, pushes = pushTimes(s);
  for (const c of R.charts) c.o.markers = pushes;
  c1.setData(s.t, [
    { name: "tracking error |X − X*|", color: css("--accent"), values: s.err.map((v) => 1e3 * v) },
    { name: "estimate error |X − X̂|", color: css("--text-3"), dash: true, values: s.est_err.map((v) => 1e3 * v), width: 1.3 }]);
  c2.setData(s.t, [
    { name: "reference", color: css("--ref"), dash: true, values: s.X_ref.map((x) => x[2] * 180 / Math.PI), width: 1.3 },
    { name: "payload", color: css("--accent"), values: s.X.map((x) => x[2] * 180 / Math.PI) }]);
  const ten = []; for (let k = 0; k < 3; k++) for (let side = 0; side < 2; side++)
    ten.push({ name: `pair ${k + 1} ${side ? "−" : "+"}`, color: seriesColor(k), dash: side === 1, values: s.T.map((T) => T[k][side]) });
  c3.setData(s.t, ten);
  c4.setData(s.t, ["x", "y", "ψ"].map((n, i) => ({ name: n, color: seriesColor(i + 3), values: s.w_i.map((w) => w[i]) })));
  $("#replay-stats").innerHTML = statsHTML(run.summary, run.config);
  R.arm.setRun(run);
  R.frameT = run.frames.map((f) => s.t[f.k]);
  $("#scrub").max = R.frameT.length - 1;
}

/* ------------------------------------------------------------------ live view */
const L = { id: null, next: 0, run: null, name: "", timer: null };
function stopLive() { clearTimeout(L.timer); L.id = null; L.run = null; R.live = false; $("#replay").classList.remove("live"); updateFollow(); }
function startLive(id, r) {
  L.id = id; L.next = 0; L.run = null; L.name = runName(r);
  showPending(r);
  liveTick();
}
function liveSummary(run, status) {
  const s = run.series, e = s.err, n = e.length;
  let sum = 0, sq = 0, mx = 0, pm = 0, em = 0, tm = 1e-9, cm = 0;
  for (let i = 0; i < n; i++) {
    sum += e[i]; sq += e[i] * e[i]; mx = Math.max(mx, e[i]); pm = Math.max(pm, Math.abs(s.err_psi[i]));
    em = Math.max(em, s.est_err[i]); cm += s.ctrl_ms[i];
    for (const pair of s.T[i]) tm = Math.max(tm, pair[0], pair[1]);
  }
  const cs = s.ctrl_ms.slice(-400).sort((a, b) => a - b);
  return { state: "running", duration: run.duration, simulated: s.t[n - 1], err_mean: sum / n, err_max: mx,
           err_rms: Math.sqrt(sq / n), psi_err_max_deg: pm * 180 / Math.PI, est_err_max: em, T_max: tm,
           ctrl_ms_mean: cm / n, ctrl_ms_p99: cs[Math.floor(0.99 * (cs.length - 1))],
           wall_s: status?.wall_s ?? NaN };
}
async function liveTick() {
  const id = L.id; if (!id) return;
  let d = null;
  try { d = await api.get(`/api/runs/${id}/live?from=${L.next}`); } catch { /* retry */ }
  if (L.id !== id) return;
  if (d) {
    L.next = d.next;
    let grew = false;
    for (const line of d.lines) {
      if (line.head) { L.run = { ...line.head, series: {}, frames: [], summary: {} }; continue; }
      if (!L.run) continue;
      for (const [k, v] of Object.entries(line.rows)) { const a = (L.run.series[k] ||= []); for (const x of v) a.push(x); }
      for (const f of line.frames) L.run.frames.push(f);
      grew = true;
    }
    if (grew && L.run.frames.length && L.run.series.t?.length) {
      L.run.summary = liveSummary(L.run, d.status);
      if (!R.live || R.run !== L.run) renderRun(L.run, L.name, true); else fillRun();
      if (R.follow) seek(R.frameT[R.frameT.length - 1]);
      updateFollow();
    }
    const st = d.status?.state;
    if (st === "done" || st === "failed") { L.id = null; poll(); return; }
  }
  L.timer = setTimeout(liveTick, 350);
}
function updateFollow() {
  const btn = $("#follow-live");
  if (btn) btn.hidden = !(R.live && !R.follow);
}
function frameAt(t) {
  const a = R.frameT; let lo = 0, hi = a.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (a[m] <= t) lo = m; else hi = m; }
  return Math.abs(a[hi] - t) < Math.abs(a[lo] - t) ? hi : lo;
}
function seek(t) {
  if (!R.run) return;
  R.simT = Math.max(0, Math.min(t, R.frameT[R.frameT.length - 1]));
  const fi = frameAt(R.simT);
  R.arm.draw(fi);
  $("#scrub").value = fi;
  const s = R.run.series, idx = R.run.frames[fi].k, W = s.w_ext[idx];
  $("#time-readout").textContent = `${s.t[idx].toFixed(2)} s`;
  const push = Math.abs(W[0]) + Math.abs(W[1]) + Math.abs(W[2]) > 0;
  $("#arm-hud").innerHTML = `t <b>${s.t[idx].toFixed(2)} s</b><br>error <b>${e3(s.err[idx])}</b> ×10⁻³ · angle <b>${(s.X[idx][2] * 180 / Math.PI).toFixed(1)}°</b>` +
    (push ? `<br><span class="push">push</span>` : "");
  for (const c of R.charts) c.setCursor(s.t[idx]);
}
function tick(now) {
  if (!R.playing) return;
  const dt = (now - R.last) / 1000; R.last = now;
  const end = R.frameT[R.frameT.length - 1];
  let t = R.simT + dt * parseFloat($("#speed").value);
  if (t >= end) { t = end; seek(t); stop(); return; }
  seek(t); requestAnimationFrame(tick);
}
function play() {
  if (!R.run) return;
  if (R.simT >= R.frameT[R.frameT.length - 1] - 1e-9) R.simT = 0;
  R.playing = true; R.last = performance.now(); $(".transport").classList.add("playing"); $("#play-btn").setAttribute("aria-label", "Pause");
  requestAnimationFrame(tick);
}
function stop() { R.playing = false; $(".transport").classList.remove("playing"); $("#play-btn").setAttribute("aria-label", "Play"); }

/* ================================================================== live drive */
const LD = { state: "idle", running: false, next: 0, buf: [], head: null, timer: null, keys: new Set(),
             sent: "", lastSend: 0, arm: null, charts: null, chartAt: 0, bounds: null };
const KEYMAP = { ArrowUp: "up", KeyW: "up", ArrowDown: "down", KeyS: "down", ArrowLeft: "left", KeyA: "left",
                 ArrowRight: "right", KeyD: "right", KeyQ: "ccw", KeyE: "cw" };
const LIVE_WINDOW = 10;                         // seconds of history kept on screen
const LIVE_NOTE = $("#live-note").innerHTML;
function liveNote(html, bad = false) { const n = $("#live-note"); n.innerHTML = html; n.classList.toggle("bad", bad); }
function liveCmd() {
  const b = (n) => (LD.keys.has(n) ? 1 : 0);
  return [b("right") - b("left"), b("up") - b("down"), b("ccw") - b("cw")];
}
function showCaps() { for (const c of document.querySelectorAll(".keys kbd[data-cap]")) c.classList.toggle("on", LD.keys.has(c.dataset.cap)); }
/* Gamepad (standard Gamepad API; Chrome/Edge expose XInput pads such as the 8BitDo directly).
   Either stick moves the payload, LT turns it counterclockwise and RT clockwise, Start toggles
   the session. The pad wins over the keys while any of its inputs is past the deadzone. */
const GP = { index: null, name: "", startWas: false, u: [0, 0, 0] };
const PAD_DEAD = 0.08;                           // as reference.Live: below this, the pad is idle
function readPad() {
  if (GP.index == null) return null;
  const g = (navigator.getGamepads ? navigator.getGamepads() : [])[GP.index];
  if (!g) return null;
  const ax = (i) => g.axes[i] || 0, bt = (i) => g.buttons[i] || { value: 0, pressed: false };
  const L = [ax(0), ax(1)], R = [ax(2), ax(3)];
  const [sx, sy] = Math.hypot(...R) > Math.hypot(...L) ? R : L;      // the stick being used
  const r2 = (v) => Math.round(Math.max(-1, Math.min(1, v)) * 100) / 100;
  return { u: [r2(sx), r2(-sy), r2(bt(6).value - bt(7).value)], start: bt(9).pressed };
}
function currentCmd() {
  const p = readPad();
  if (p && p.u.some((v) => Math.abs(v) > PAD_DEAD)) return p.u;
  return liveCmd();
}
async function sendCmd(force = false) {
  if (!LD.running) return;
  if (LD.inflight) { LD.dirty = true; return; }   // one request at a time, so none arrive stale
  const u = currentCmd(), key = u.join(","), now = performance.now();
  if (key === LD.sent ? now - LD.lastSend < 150 : !force && now - LD.lastSend < 40) return;
  LD.sent = key; LD.lastSend = now; LD.inflight = true;
  try { await api.post("/api/live/cmd", { u }); } catch { /* the next one */ }
  LD.inflight = false;
  if (LD.dirty) { LD.dirty = false; sendCmd(true); }
}
function padStatus(p) {
  const el = $("#pad-status");
  if (GP.index == null) { el.className = "pad-status"; el.textContent = "Gamepad: press any button on it to connect"; return; }
  el.className = "pad-status on";
  const u = p ? p.u : [0, 0, 0];
  el.innerHTML = `Gamepad: ${esc(GP.name.replace(/\s*\(.*$/, "") || "connected")} · stick <b>${u[0].toFixed(2)}, ${u[1].toFixed(2)}</b> · turn <b>${u[2].toFixed(2)}</b> · Start toggles the session`;
}
function padTick() {
  const p = readPad();
  if (p && S.tab === "live") {
    if (p.start && !GP.startWas) liveToggle();
    if (LD.running) sendCmd();
  }
  GP.startWas = !!p?.start;
  if (S.tab === "live") padStatus(p);
}
function releaseKeys() { if (LD.keys.size) { LD.keys.clear(); showCaps(); sendCmd(true); } }
async function liveToggle() {
  if (LD.state === "running" || LD.state === "starting") { try { await api.post("/api/live/stop", {}); } catch { /* polled */ } return; }
  const btn = $("#live-start"); btn.disabled = true;
  try { await api.post("/api/live/start", { config: { ...configForRun(), "traj.free": $("#live-free").checked } }); }
  catch (err) { btn.disabled = false; liveNote(esc(err.message), true); return; }
  btn.disabled = false;
  Object.assign(LD, { state: "starting", running: false, next: 0, buf: [], head: null, bounds: null });
  liveNote(LIVE_NOTE);
  clearTimeout(LD.timer); livePoll();
}
async function livePoll() {
  let d = null;
  try { d = await api.get(`/api/live?from=${LD.next}`); } catch { /* retry */ }
  if (d) {
    LD.state = d.state; LD.head = d.head || LD.head; LD.next = d.next ?? LD.next;
    for (const f of d.frames || []) LD.buf.push(f);
    if (LD.buf.length) {
      const t1 = LD.buf[LD.buf.length - 1].t; let i = 0;
      while (i < LD.buf.length && LD.buf[i].t < t1 - LIVE_WINDOW) i++;
      if (i) LD.buf.splice(0, i);
    }
    LD.running = d.state === "running";
    liveRender(d);
    if (LD.running) sendCmd();
  }
  if (!d || d.state === "starting" || d.state === "running") LD.timer = setTimeout(livePoll, d ? 40 : 500);
  else { LD.timer = null; releaseKeys(); }
}
function liveRender(d) {
  const st = LD.state, active = st === "running" || st === "starting";
  const badge = $("#live-badge"), btn = $("#live-start");
  badge.textContent = st === "running" ? "live" : st === "starting" ? "starting" : st === "failed" ? "failed" : "stopped";
  badge.className = "badge" + (st === "running" ? " live" : st === "failed" ? " failed" : "");
  btn.textContent = active ? "Stop" : "Start";
  $("#live-free").disabled = active;
  btn.classList.toggle("primary", !active); btn.classList.toggle("secondary", active);
  if (st === "starting" && d.message) liveNote(esc(d.message) + "…");
  else if (st === "running" && LD.head?.free) liveNote("Orientation is free: the controller picks the payload angle, so Q, E and the triggers do nothing.");
  else if (st === "running") liveNote(LIVE_NOTE);
  else if (d?.message && (st === "failed" || st === "stopped")) liveNote(esc(d.message), st === "failed");
  const ld = $("#live-load");
  ld.textContent = st === "running" ? `compute ${Math.round(100 * d.load)} % of real time · ${d.late} late steps` : "";
  ld.classList.toggle("warn", st === "running" && d.load > 0.8);
  if (!LD.buf.length || !LD.head) return;
  const B = LD.buf, n = B.length, f = B[n - 1];
  let Tmax = 1e-9; for (const g of B) for (const p of g.T) Tmax = Math.max(Tmax, p[0], p[1]);
  const run = { offsets: LD.head.offsets, ref_path: B.map((g) => g.X_ref),
    series: { X: B.map((g) => g.X), X_ref: B.map((g) => g.X_ref), X_est: B.map((g) => g.X_est), T: B.map((g) => g.T), w_ext: B.map(() => [0, 0, 0]) },
    frames: B.map((g, i) => ({ k: i, line: g.line, disks: g.disks })), summary: { T_max: Tmax } };
  // view box: the usual workspace, grown (never shrunk) to keep the arm in sight
  let [x0, x1, y0, y1] = LD.bounds || [-0.6, 3.2, -1.2, 2.6];
  for (const p of f.line) { x0 = Math.min(x0, p[0] - 0.3); x1 = Math.max(x1, p[0] + 0.3); y0 = Math.min(y0, p[1] - 0.3); y1 = Math.max(y1, p[1] + 0.3); }
  LD.bounds = [x0, x1, y0, y1];
  if (!LD.arm) LD.arm = new ArmView($("#live-canvas"));
  LD.arm.run = run; LD.arm.overlay = null; LD.arm.bounds = LD.bounds; LD.arm.draw(n - 1);
  const why = f.choice === 0 ? "" : f.choice === 1 ? `<br><span class="warn">the arm cannot do that here: braking</span>`
    : f.choice === 2 ? `<br><span class="warn">the arm cannot do that here: holding</span>`
    : `<br><span class="bad">no feasible step (${esc(f.reason || "")}): tensions held</span>`;
  let tm = 0; for (const p of f.T) tm = Math.max(tm, p[0], p[1]);
  $("#live-hud").innerHTML = `t <b>${f.t.toFixed(1)} s</b><br>error <b>${e3(f.err)}</b> ×10⁻³ · angle <b>${(f.X[2] * 180 / Math.PI).toFixed(1)}°</b><br>`
    + `max tension <b>${tm.toFixed(2)}</b>` + why;
  const now = performance.now();
  if (now - LD.chartAt < 200) return;               // charts at 5 Hz
  LD.chartAt = now;
  if (!LD.charts) {
    const box = $("#live-plots"); box.innerHTML = "";
    LD.charts = [new LineChart(box, { title: "Payload position error", sub: "×10⁻³ segment lengths, last 10 s", zero: true, yFmt: (v) => fmt(v, 3) }),
                 new LineChart(box, { title: "Payload angle", sub: "degrees" }),
                 new LineChart(box, { title: "Tendon tensions", sub: "solid + side, dashed − side", zero: true })];
  }
  const t = B.map((g) => g.t), [c1, c2, c3] = LD.charts;
  c1.setData(t, [{ name: "tracking error |X − X*|", color: css("--accent"), values: B.map((g) => 1e3 * g.err) },
                 { name: "estimate error |X − X̂|", color: css("--text-3"), dash: true, width: 1.3, values: B.map((g) => 1e3 * Math.hypot(g.X[0] - g.X_est[0], g.X[1] - g.X_est[1])) }]);
  c2.setData(t, [{ name: "reference", color: css("--ref"), dash: true, width: 1.3, values: B.map((g) => g.X_ref[2] * 180 / Math.PI) },
                 { name: "payload", color: css("--accent"), values: B.map((g) => g.X[2] * 180 / Math.PI) }]);
  const ten = []; for (let k = 0; k < 3; k++) for (let side = 0; side < 2; side++)
    ten.push({ name: `pair ${k + 1} ${side ? "−" : "+"}`, color: seriesColor(k), dash: side === 1, values: B.map((g) => g.T[k][side]) });
  c3.setData(t, ten);
}
function bootLive() {
  $("#live-start").addEventListener("click", liveToggle);
  const free = $("#live-free");
  free.checked = store.get("liveFree", false);
  free.addEventListener("change", () => store.set("liveFree", free.checked));
  window.addEventListener("gamepadconnected", (e) => { GP.index = e.gamepad.index; GP.name = e.gamepad.id; padStatus(null); toast(`Gamepad connected: ${e.gamepad.id.replace(/\s*\(.*$/, "")}`); });
  window.addEventListener("gamepaddisconnected", (e) => { if (e.gamepad.index === GP.index) { GP.index = null; padStatus(null); sendCmd(true); } });
  setInterval(padTick, 30);
  document.addEventListener("keydown", (e) => {
    if (S.tab !== "live" || e.target.closest("input, select, textarea")) return;
    if (e.key === "Escape") { if (LD.running) { e.preventDefault(); liveToggle(); } return; }
    const k = KEYMAP[e.code];
    if (!k || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    if (!LD.keys.has(k)) { LD.keys.add(k); showCaps(); sendCmd(true); }
  });
  document.addEventListener("keyup", (e) => { const k = KEYMAP[e.code]; if (k && LD.keys.delete(k)) { showCaps(); sendCmd(true); } });
  window.addEventListener("blur", releaseKeys);
  document.addEventListener("visibilitychange", () => { if (document.hidden) releaseKeys(); });
  livePoll();                                         // pick up a session already running
}

/* ================================================================== compare */
const C = { arm: null };
async function renderCompare() {
  const ids = S.runs.filter((r) => S.ticked.has(r.id) && r.status?.state === "done").map((r) => r.id).reverse().slice(0, 6);
  const show = ids.length >= 2;
  $("#compare-empty").hidden = show; $("#compare").hidden = !show;
  if (!show) return;
  const runs = await Promise.all(ids.map(loadRun));
  if (!C.arm) C.arm = new ArmView($("#cmp-canvas"));
  C.arm.setOverlay(runs);
  const box = $("#compare-plots"); box.innerHTML = "";
  const tl = runs.reduce((a, r) => (r.run.series.t.length > a.length ? r.run.series.t : a), []);
  const pad = (arr) => tl.map((_, i) => (i < arr.length ? arr[i] : NaN));
  const c1 = new LineChart(box, { title: "Payload position error", sub: "×10⁻³ segment lengths", zero: true });
  c1.setData(tl, runs.map((r, i) => ({ name: r.meta.label || r.meta.id, color: seriesColor(i), values: pad(r.run.series.err.map((v) => 1e3 * v)) })));
  const c2 = new LineChart(box, { title: "Payload angle error", sub: "degrees" });
  c2.setData(tl, runs.map((r, i) => ({ name: r.meta.label || r.meta.id, color: seriesColor(i), values: pad(r.run.series.err_psi.map((v) => v * 180 / Math.PI)) })));
  const c3 = new LineChart(box, { title: "Estimate error", sub: "×10⁻³ segment lengths", zero: true });
  c3.setData(tl, runs.map((r, i) => ({ name: r.meta.label || r.meta.id, color: seriesColor(i), values: pad(r.run.series.est_err.map((v) => 1e3 * v)) })));
  const base = runs[0].run.config;
  const diff = (cfg) => S.schema.filter((f) => JSON.stringify(cfg[f.key]) !== JSON.stringify(base[f.key]))
    .map((f) => `${f.label}: ${fmt(base[f.key])} → <b>${fmt(cfg[f.key])}</b>`).join("; ") || (cfg === base ? "reference run" : "same parameters");
  $("#compare-table").innerHTML = `<thead><tr><th class="left">Run</th><th>Mean ×10⁻³</th><th>Max ×10⁻³</th><th>RMS ×10⁻³</th><th>Angle max °</th><th>Estimate max ×10⁻³</th><th>Max T</th><th>Ctrl ms</th><th class="left">Differs from the first</th></tr></thead><tbody>` +
    runs.map((r, i) => { const m = r.run.summary; return `<tr data-id="${r.meta.id}"><td><span class="swatch" style="background:${seriesColor(i)}"></span>${esc(r.meta.label || r.meta.id)}</td><td>${e3(m.err_mean)}</td><td>${e3(m.err_max)}</td><td>${e3(m.err_rms)}</td><td>${m.psi_err_max_deg.toFixed(2)}</td><td>${e3(m.est_err_max)}</td><td>${m.T_max.toFixed(2)}</td><td>${m.ctrl_ms_mean.toFixed(1)}</td><td class="diff">${diff(r.run.config)}</td></tr>`; }).join("") + "</tbody>";
  for (const tr of document.querySelectorAll("#compare-table tbody tr")) tr.addEventListener("click", () => select(tr.dataset.id));
}

/* ================================================================== sweeps view */
const W = { chart: null };
function sweepsList() {
  const m = new Map();
  for (const r of S.runs) if (r.sweep) { if (!m.has(r.sweep.id)) m.set(r.sweep.id, { id: r.sweep.id, key: r.sweep.key, label: r.sweep.label, runs: [] }); m.get(r.sweep.id).runs.push(r); }
  return [...m.values()].sort((a, b) => (a.id < b.id ? 1 : -1));
}
function renderSweeps(rebuildSelect = true) {
  const list = sweepsList();
  $("#sweeps-empty").hidden = list.length > 0; $("#sweeps").hidden = list.length === 0;
  if (!list.length) return;
  const sel = $("#sweep-select");
  if (!S.sweep || !list.find((s) => s.id === S.sweep)) S.sweep = list[0].id;
  const opts = list.map((s) => `<option value="${s.id}">${esc(s.label)} · ${s.runs.length} runs (${s.runs.filter((r) => r.status?.state === "done").length} done)</option>`).join("");
  if (rebuildSelect || sel.innerHTML !== opts) { sel.innerHTML = opts; sel.value = S.sweep; }
  const sw = list.find((s) => s.id === S.sweep), f = field(sw.key), metric = $("#sweep-metric").value;
  const rows = sw.runs.slice().sort((a, b) => a.sweep.value - b.sweep.value);
  const done = rows.filter((r) => r.status?.state === "done" && r.status.summary);
  const scale = metric.startsWith("err") || metric === "est_err_max" ? 1e3 : 1;
  const box = $("#sweep-plots"); box.innerHTML = "";
  const mLabel = $("#sweep-metric").selectedOptions[0].textContent + (scale === 1e3 ? " (×10⁻³)" : "");
  if (done.length) {
    W.chart = new LineChart(box, { title: `${mLabel} against ${labelUnit(f)}`, sub: `${done.length} of ${rows.length} runs finished`, tall: true, points: true,
      xFmt: (v, axis) => (axis ? fmt(v) : `${f.label} = ${fmt(v)}`), zero: true });
    W.chart.setData(done.map((r) => r.sweep.value), [{ name: mLabel, color: css("--accent"), values: done.map((r) => scale * r.status.summary[metric]) }]);
    W.chart.onClick = (x) => { const r = done.reduce((a, b) => (Math.abs(b.sweep.value - x) < Math.abs(a.sweep.value - x) ? b : a)); select(r.id); };
  } else box.innerHTML = `<div class="plot" style="padding:24px;color:var(--text-2)">Runs are queued or running; the plot fills in as they finish.</div>`;
  $("#sweep-table").innerHTML = `<thead><tr><th class="left">${esc(labelUnit(f))}</th><th class="left">State</th><th>Mean ×10⁻³</th><th>Max ×10⁻³</th><th>Angle max °</th><th>Estimate max ×10⁻³</th><th>Max T</th></tr></thead><tbody>` +
    rows.map((r) => { const m = r.status?.summary; const st = r.status?.state || "";
      return `<tr data-id="${r.id}"><td>${fmt(r.sweep.value)}</td><td class="left"><span class="state ${st}">${st === "running" ? Math.round(100 * r.status.progress) + "%" : st}</span>${st === "failed" ? ` <span style="color:var(--text-3)">${esc((r.status.message || "").slice(0, 70))}</span>` : ""}</td>` +
        (m ? `<td>${e3(m.err_mean)}</td><td>${e3(m.err_max)}</td><td>${m.psi_err_max_deg.toFixed(2)}</td><td>${e3(m.est_err_max)}</td><td>${m.T_max.toFixed(2)}</td>` : `<td>–</td><td>–</td><td>–</td><td>–</td><td>–</td>`) + "</tr>"; }).join("") + "</tbody>";
  for (const tr of document.querySelectorAll("#sweep-table tbody tr")) tr.addEventListener("click", () => select(tr.dataset.id));
}

/* ================================================================== tabs + boot */
function setTab(tab) {
  S.tab = tab;
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of document.querySelectorAll(".view")) v.hidden = v.dataset.view !== tab;
  if (tab !== "replay") stop();
  if (tab !== "live") releaseKeys();
  else if (LD.arm) LD.arm.draw();
  if (tab === "compare") renderCompare();
  if (tab === "sweeps") renderSweeps();
}
async function boot() {
  const sc = await api.get("/api/schema");
  S.schema = sc.schema; S.defaults = sc.defaults; S.presets = sc.presets; S.unitsNote = sc.units_note;
  const saved = store.get("cfg", {});
  S.cfg = { ...S.defaults };
  for (const [k, v] of Object.entries(S.presets[S.cfg["traj.preset"]]?.values || {})) S.cfg[k] = v;
  for (const [k, v] of Object.entries(saved)) if (k in S.defaults) S.cfg[k] = v;
  buildParams(); buildSweepBox(); bootLive();
  $("#run-btn").addEventListener("click", submit);
  $("#reset-all").addEventListener("click", () => { S.cfg = { ...S.defaults }; for (const [k, v] of Object.entries(S.presets[S.cfg["traj.preset"]].values)) S.cfg[k] = v; store.set("cfg", S.cfg); refreshParams(); updateSweepBox(); toast("Parameters reset to defaults"); });
  $("#load-config").addEventListener("click", () => { if (!R.run) return; for (const [k, v] of Object.entries(R.run.config)) if (k in S.defaults) S.cfg[k] = v; store.set("cfg", S.cfg); refreshParams(); updateSweepBox(); toast("Parameters loaded from this run"); });
  $("#clear-compare").addEventListener("click", () => { S.ticked.clear(); renderRuns(); if (S.tab === "compare") renderCompare(); });
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => setTab(b.dataset.tab));
  $("#play-btn").addEventListener("click", () => (R.playing ? stop() : play()));
  $("#scrub").addEventListener("input", (e) => { stop(); R.follow = false; updateFollow(); seek(R.frameT[+e.target.value]); });
  $("#follow-live").addEventListener("click", () => { stop(); R.follow = true; updateFollow(); seek(R.frameT[R.frameT.length - 1]); });
  $("#sweep-select").addEventListener("change", (e) => { S.sweep = e.target.value; renderSweeps(false); });
  $("#sweep-metric").addEventListener("change", () => renderSweeps(false));
  $("#rail-toggle").addEventListener("click", () => $("#runs").classList.toggle("open"));
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); submit(); }
    else if (e.key === " " && S.tab === "replay" && !e.target.closest("input, select, textarea, button")) { e.preventDefault(); R.playing ? stop() : play(); }
  });
  await poll();
  const h = new URLSearchParams(location.hash.slice(1));
  for (const id of (h.get("tick") || "").split(",").filter(Boolean)) S.ticked.add(id);
  const firstDone = S.runs.find((r) => ["running", "done"].includes(r.status?.state));
  const start = h.get("run") || (firstDone && firstDone.id);
  if (start) select(start); else showReplay(null);
  if (h.get("tab")) setTab(h.get("tab"));
  if (h.get("t")) R.pendingSeek = parseFloat(h.get("t"));
  renderRuns();
}
boot().catch((err) => { document.body.insertAdjacentHTML("beforeend", `<div class="toast show bad">Could not start: ${esc(err.message)}</div>`); });

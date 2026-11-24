/* charts.js: canvas line charts and the arm view for the bowstring console. */
"use strict";

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = ["--c1", "--c2", "--c3", "--c4", "--c5", "--c6"];
const seriesColor = (i) => css(SERIES[i % SERIES.length]);

const AIM_R = 0.02;                              // radius of the aim target: the laser hits inside it

function niceTicks(lo, hi, n = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (hi - lo < 1e-12) { hi = lo + 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(12));
  return out;
}
function fmt(v, digits = 3) {
  if (typeof v !== "number") return v == null ? "–" : String(v);
  if (!isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  return +v.toPrecision(digits) + "";
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

let tipEl = null;
function tooltip(html, x, y) {
  if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "tip"; document.body.appendChild(tipEl); }
  if (html == null) { tipEl.style.display = "none"; return; }
  tipEl.innerHTML = html; tipEl.style.display = "block";
  const r = tipEl.getBoundingClientRect();
  let left = x + 14, top = y + 12;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 12;
  tipEl.style.left = left + "px"; tipEl.style.top = top + "px";
}

/* ------------------------------------------------------------------ line chart */
class LineChart {
  constructor(parent, opts = {}) {
    this.o = Object.assign({ title: "", sub: "", unit: "", xLabel: "time (s)", zero: false, tall: false,
                             points: false, yFmt: (v) => fmt(v) }, opts);
    this.el = document.createElement("div");
    this.el.className = "plot" + (this.o.tall ? " tall" : "");
    this.el.innerHTML = `<div class="plot-head"><span class="plot-title"></span><span class="plot-sub"></span></div>
      <div class="legend"></div><canvas></canvas>`;
    this.el.querySelector(".plot-title").textContent = this.o.title;
    this.el.querySelector(".plot-sub").textContent = this.o.sub;
    parent.appendChild(this.el);
    this.canvas = this.el.querySelector("canvas");
    this.legend = this.el.querySelector(".legend");
    this.cursor = null; this.hoverX = null; this.data = null;
    this.onHover = null; this.onClick = null;
    this.canvas.addEventListener("mousemove", (e) => this._hover(e));
    this.canvas.addEventListener("mouseleave", () => { this.hoverX = null; tooltip(null); this.render(); this.onHover && this.onHover(null); });
    this.canvas.addEventListener("click", (e) => { const x = this._xAt(e); if (x != null && this.onClick) this.onClick(x); });
    new ResizeObserver(() => this.render()).observe(this.canvas);
  }
  setSub(text) { this.el.querySelector(".plot-sub").textContent = text; }
  setData(x, series) {
    this.data = { x, series };
    this.legend.innerHTML = series.filter((s) => !s.noLegend).map((s) =>
      `<span><i class="${s.dash ? "dashed" : ""}" style="border-top-color:${s.color}"></i>${s.name}</span>`).join("");
    this._range();
    this.render();
  }
  _range() {
    const { x, series } = this.data;
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (this.o.zero) lo = Math.min(0, lo);
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
    this.yr = [this.o.zero && lo >= 0 ? 0 : lo - pad, hi + pad];
    this.xr = [x[0], Math.max(x[x.length - 1], this.o.xMax ?? -Infinity)];
    if (this.xr[1] === this.xr[0]) { this.xr = [this.xr[0] - 1, this.xr[1] + 1]; }
    if (this.o.points) { const px = (this.xr[1] - this.xr[0]) * 0.04; this.xr = [this.xr[0] - px, this.xr[1] + px]; }
  }
  _geom() {
    const { w, h } = { w: this.canvas.clientWidth, h: this.canvas.clientHeight };
    return { L: 48, R: w - 10, T: 6, B: h - 20, w, h };
  }
  _xAt(e) {
    if (!this.data) return null;
    const r = this.canvas.getBoundingClientRect(), g = this._geom();
    const px = e.clientX - r.left;
    if (px < g.L || px > g.R) return null;
    return this.xr[0] + (px - g.L) / (g.R - g.L) * (this.xr[1] - this.xr[0]);
  }
  _nearest(xv) {
    const x = this.data.x; let lo = 0, hi = x.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (x[m] < xv) lo = m; else hi = m; }
    return Math.abs(x[lo] - xv) <= Math.abs(x[hi] - xv) ? lo : hi;
  }
  _hover(e) {
    const xv = this._xAt(e);
    if (xv == null) { tooltip(null); return; }
    this.hoverX = xv; this.render();
    const i = this._nearest(xv);
    const rows = this.data.series.filter((s) => !s.noTip).map((s) =>
      `<div class="row"><i style="border-top-color:${s.color}"></i>${s.name}: <b>${this.o.yFmt(s.values[i])}</b>${this.o.unit ? " " + this.o.unit : ""}</div>`).join("");
    const xs = this.o.xFmt ? this.o.xFmt(this.data.x[i]) : `t = ${this.data.x[i].toFixed(2)} s`;
    tooltip(`<div style="color:var(--text-3);margin-bottom:3px">${xs}</div>${rows}`, e.clientX, e.clientY);
    this.onHover && this.onHover(this.data.x[i]);
  }
  setCursor(xv) { this.cursor = xv; this.render(); }
  render() {
    if (!this.data) return;
    const { ctx, w, h } = fitCanvas(this.canvas);
    const g = this._geom();
    ctx.clearRect(0, 0, w, h);
    const X = (v) => g.L + (v - this.xr[0]) / (this.xr[1] - this.xr[0]) * (g.R - g.L);
    const Y = (v) => g.B - (v - this.yr[0]) / (this.yr[1] - this.yr[0]) * (g.B - g.T);
    ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "middle";
    // grid + y labels
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1; ctx.fillStyle = css("--text-3");
    ctx.textAlign = "right";
    for (const v of niceTicks(this.yr[0], this.yr[1], Math.max(3, Math.floor((g.B - g.T) / 26)))) {
      const y = Math.round(Y(v)) + 0.5; ctx.beginPath(); ctx.moveTo(g.L, y); ctx.lineTo(g.R, y); ctx.stroke();
      ctx.fillText(this.o.yFmt(v), g.L - 6, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const v of niceTicks(this.xr[0], this.xr[1], Math.max(2, Math.floor((g.R - g.L) / 70)))) {
      const x = Math.round(X(v)) + 0.5;
      ctx.fillText(this.o.xFmt ? this.o.xFmt(v, true) : fmt(v), x, g.B + 5);
    }
    ctx.strokeStyle = css("--line-strong"); ctx.beginPath(); ctx.moveTo(g.L, g.B + 0.5); ctx.lineTo(g.R, g.B + 0.5); ctx.stroke();
    // series
    ctx.save(); ctx.beginPath(); ctx.rect(g.L, g.T - 2, g.R - g.L, g.B - g.T + 4); ctx.clip();
    const x = this.data.x;
    for (const s of this.data.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash ? [5, 4] : []);
      ctx.globalAlpha = s.alpha || 1;
      ctx.beginPath();
      let pen = false;
      const step = Math.max(1, Math.floor(x.length / ((g.R - g.L) * 2)));
      for (let i = 0; i < x.length; i += step) {
        const v = s.values[i]; if (!isFinite(v)) { pen = false; continue; }
        const px = X(x[i]), py = Y(v);
        if (pen) ctx.lineTo(px, py); else { ctx.moveTo(px, py); pen = true; }
      }
      ctx.stroke();
      if (this.o.points || s.points) {
        ctx.fillStyle = s.color;
        for (let i = 0; i < x.length; i++) { if (!isFinite(s.values[i])) continue; ctx.beginPath(); ctx.arc(X(x[i]), Y(s.values[i]), 4, 0, 7); ctx.fill(); }
      }
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    // markers (e.g. pushes)
    for (const m of this.o.markers || []) {
      const px = Math.round(X(m)) + 0.5; ctx.strokeStyle = css("--bad"); ctx.globalAlpha = 0.6; ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(px, g.T); ctx.lineTo(px, g.B); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    ctx.restore();
    // cursor + hover
    for (const [v, col, wdt] of [[this.cursor, css("--text-2"), 1], [this.hoverX, css("--text-3"), 1]]) {
      if (v == null) continue;
      const px = Math.round(X(v)) + 0.5; ctx.strokeStyle = col; ctx.lineWidth = wdt;
      ctx.beginPath(); ctx.moveTo(px, g.T); ctx.lineTo(px, g.B); ctx.stroke();
    }
  }
}

/* ------------------------------------------------------------------ arm view */
class ArmView {
  constructor(canvas) {
    this.canvas = canvas; this.run = null; this.k = 0; this.overlay = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
  }
  _bounds(pointsList) {
    let x0 = -0.35, x1 = 3.1, y0 = -0.6, y1 = 2.5;
    for (const pts of pointsList) for (const p of pts) {
      x0 = Math.min(x0, p[0] - 0.25); x1 = Math.max(x1, p[0] + 0.25);
      y0 = Math.min(y0, p[1] - 0.25); y1 = Math.max(y1, p[1] + 0.25);
    }
    return [x0, x1, y0, y1];
  }
  setRun(run) {
    this.run = run; this.overlay = null;
    const s = run.series;
    const lines = run.frames.map((f) => f.line);
    const all = [run.ref_path || s.X_ref, s.X].concat(lines.filter((_, i) => i % 10 === 0), [lines[lines.length - 1]]);
    const aim = this._aim(run); if (aim) all.push([aim]);
    this.bounds = this._bounds(all);
    this.k = 0; this.draw();
  }
  setOverlay(runs) {
    this.run = null; this.overlay = runs;
    this.bounds = this._bounds(runs.flatMap((r) => [r.run.series.X_ref, r.run.series.X, r.run.frames[r.run.frames.length - 1].line]));
    this.draw();
  }
  _xf(w, h) {
    const [x0, x1, y0, y1] = this.bounds, pad = 14;
    const sc = Math.min((w - 2 * pad) / (x1 - x0), (h - 2 * pad) / (y1 - y0));
    const ox = pad + ((w - 2 * pad) - sc * (x1 - x0)) / 2 - sc * x0;
    const oy = h - pad - ((h - 2 * pad) - sc * (y1 - y0)) / 2 + sc * y0;
    return { X: (x) => ox + sc * x, Y: (y) => oy - sc * y, sc };
  }
  _grid(ctx, t, w, h) {
    const [x0, x1, y0, y1] = this.bounds;
    ctx.strokeStyle = "rgba(255,255,255,.045)"; ctx.lineWidth = 1;
    for (let v = Math.ceil(x0 * 2) / 2; v <= x1; v += 0.5) { const x = Math.round(t.X(v)) + .5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let v = Math.ceil(y0 * 2) / 2; v <= y1; v += 0.5) { const y = Math.round(t.Y(v)) + .5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    ctx.fillStyle = css("--text-3"); ctx.font = "10px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    for (let v = Math.ceil(x0); v <= x1; v += 1) ctx.fillText(v.toFixed(0), t.X(v) + 3, h - 4);
    ctx.textBaseline = "top";
    for (let v = Math.ceil(y0); v <= y1; v += 1) if (v !== 0) ctx.fillText(v.toFixed(0), w - 14, t.Y(v) + 2);
    // clamp at the base
    const bx = t.X(0), by0 = t.Y(-0.3), by1 = t.Y(0.3);
    ctx.fillStyle = "#2a3039"; ctx.fillRect(bx - 9, by1, 9, by0 - by1);
    ctx.strokeStyle = "#5b6470"; ctx.beginPath();
    for (let y = by1; y < by0; y += 6) { ctx.moveTo(bx - 9, y + 6); ctx.lineTo(bx, y); }
    ctx.stroke(); ctx.strokeStyle = "#7b8491"; ctx.beginPath(); ctx.moveTo(bx + .5, by1); ctx.lineTo(bx + .5, by0); ctx.stroke();
  }
  _path(ctx, t, pts, color, width, dash, upto) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath(); const n = upto == null ? pts.length : Math.min(upto + 1, pts.length);
    for (let i = 0; i < n; i++) { const x = t.X(pts[i][0]), y = t.Y(pts[i][1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.setLineDash([]);
  }
  _aim(run) {                                   // the point the payload aims at, if the run has one
    const c = run.config;
    return c && c["traj.aim"] && ["circle", "figure8", "rectangle", "line", "text"].includes(c["traj.preset"]) && !c["traj.free"]
      ? [c["traj.aim_x"], c["traj.aim_y"]] : null;
  }
  _target(ctx, t, A) {
    const x = t.X(A[0]), y = t.Y(A[1]);
    ctx.strokeStyle = "#ff6b6b"; ctx.lineWidth = 1.2; ctx.globalAlpha = 0.85;
    for (const r of [0.008, AIM_R]) { ctx.beginPath(); ctx.arc(x, y, r * t.sc, 0, 7); ctx.stroke(); }
    ctx.beginPath(); ctx.moveTo(x - 2 * AIM_R * t.sc, y); ctx.lineTo(x + 2 * AIM_R * t.sc, y); ctx.moveTo(x, y - 2 * AIM_R * t.sc); ctx.lineTo(x, y + 2 * AIM_R * t.sc); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  _laser(ctx, t, X, A) {                        // a ray from the payload's front face along its actual heading
    const c = Math.cos(X[2]), s = Math.sin(X[2]), e = [X[0] + 0.13 * c, X[1] + 0.13 * s];
    const d = (A[0] - e[0]) * c + (A[1] - e[1]) * s, miss = Math.abs(-(A[0] - e[0]) * s + (A[1] - e[1]) * c);
    const hit = d > 0 && miss <= AIM_R;           // within the target's outer ring: it stops there, else it runs on
    const L = hit ? d : 30, end = [e[0] + L * c, e[1] + L * s];
    ctx.save(); ctx.lineCap = "round"; ctx.shadowColor = "#ff2d2d"; ctx.shadowBlur = 10;
    ctx.strokeStyle = "#ff4a4a"; ctx.lineWidth = 2.4;
    ctx.beginPath(); ctx.moveTo(t.X(e[0]), t.Y(e[1])); ctx.lineTo(t.X(end[0]), t.Y(end[1])); ctx.stroke();
    ctx.shadowBlur = 0; ctx.strokeStyle = "#ffd0d0"; ctx.lineWidth = 0.9;
    ctx.beginPath(); ctx.moveTo(t.X(e[0]), t.Y(e[1])); ctx.lineTo(t.X(end[0]), t.Y(end[1])); ctx.stroke();
    if (hit) { ctx.shadowBlur = 14; ctx.fillStyle = "#ffe6e6"; ctx.beginPath(); ctx.arc(t.X(end[0]), t.Y(end[1]), 3.2, 0, 7); ctx.fill(); }
    ctx.restore();
  }
  _payload(ctx, t, X, fill) {
    const c = Math.cos(X[2]), s = Math.sin(X[2]), a = 0.13, b = 0.075;
    const corners = [[0, -b], [a, -b], [a, b], [0, b]].map(([u, v]) => [X[0] + c * u - s * v, X[1] + s * u + c * v]);
    ctx.beginPath(); corners.forEach((p, i) => (i ? ctx.lineTo(t.X(p[0]), t.Y(p[1])) : ctx.moveTo(t.X(p[0]), t.Y(p[1]))));
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = "#cfd4da"; ctx.lineWidth = 1; ctx.stroke();
  }
  draw(k) {
    if (k != null) this.k = k;
    const { ctx, w, h } = fitCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);
    if (!this.bounds) return;
    const t = this._xf(w, h);
    this._grid(ctx, t, w, h);
    if (this.overlay) return this._drawOverlay(ctx, t);
    if (!this.run) return;
    const run = this.run, s = run.series, frames = run.frames;
    const fi = Math.min(frames.length - 1, Math.max(0, this.k));
    const f = frames[fi], idx = f.k;
    this._path(ctx, t, run.ref_path || s.X_ref, css("--ref"), 1.3, [5, 4]);
    this._path(ctx, t, s.X, css("--accent"), 2, null, idx);
    // tendons
    const d = run.offsets, P = f.disks, T = s.T[idx], Tmax = Math.max(1e-9, run.summary.T_max);
    for (let k2 = 0; k2 < 3; k2++) {
      for (let side = 0; side < 2; side++) {
        const sg = side === 0 ? 1 : -1, tau = T[k2][side];
        const pts = [[0, sg * d[k2]]];
        for (let j = 1; j <= k2 + 1; j++) { const [x, y, ps] = P[j]; pts.push([x - sg * d[k2] * Math.sin(ps), y + sg * d[k2] * Math.cos(ps)]); }
        ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, tau / Tmax);
        this._path(ctx, t, pts, seriesColor(k2), 0.8 + 2.6 * Math.min(1, tau / Tmax));
      }
    }
    ctx.globalAlpha = 1;
    // backbone and disks
    this._path(ctx, t, f.line, "#e3e7ec", 3.2);
    for (let j = 1; j < P.length; j++) {
      const [x, y, ps] = P[j], r = Math.max(...d) + 0.05;
      ctx.strokeStyle = "#e3e7ec"; ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(t.X(x + r * Math.sin(ps)), t.Y(y - r * Math.cos(ps))); ctx.lineTo(t.X(x - r * Math.sin(ps)), t.Y(y + r * Math.cos(ps))); ctx.stroke();
    }
    const X = s.X[idx];
    const aim = this._aim(run);
    if (aim) this._target(ctx, t, aim);
    this._payload(ctx, t, X, "#3a414c");
    if (aim) this._laser(ctx, t, X, aim);
    // estimate
    const E = s.X_est[idx];
    ctx.strokeStyle = css("--text-2"); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(t.X(E[0]), t.Y(E[1]), 5, 0, 7); ctx.stroke();
    // reference point
    const R = s.X_ref[idx];
    ctx.fillStyle = css("--ref"); ctx.beginPath(); ctx.arc(t.X(R[0]), t.Y(R[1]), 2.6, 0, 7); ctx.fill();
    // push
    const W = s.w_ext[idx];
    if (W && (Math.abs(W[0]) + Math.abs(W[1]) + Math.abs(W[2]) > 0)) {
      ctx.strokeStyle = css("--bad"); ctx.fillStyle = css("--bad"); ctx.lineWidth = 2.5;
      const n = Math.hypot(W[0], W[1]);
      if (n > 0) {
        const ux = W[0] / n, uy = W[1] / n, L = 0.4;
        const x0 = t.X(X[0] - ux * L), y0 = t.Y(X[1] - uy * L), x1 = t.X(X[0]), y1 = t.Y(X[1]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
        const ang = Math.atan2(y1 - y0, x1 - x0); ctx.beginPath(); ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - 10 * Math.cos(ang - .45), y1 - 10 * Math.sin(ang - .45)); ctx.lineTo(x1 - 10 * Math.cos(ang + .45), y1 - 10 * Math.sin(ang + .45)); ctx.fill();
      } else { ctx.beginPath(); ctx.arc(t.X(X[0]), t.Y(X[1]), 16, 0, 7); ctx.stroke(); }
    }
  }
  _drawOverlay(ctx, t) {
    this.overlay.forEach((r, i) => {
      const s = r.run.series;
      if (i === 0) this._path(ctx, t, s.X_ref, css("--ref"), 1.3, [5, 4]);
      const f = r.run.frames[r.run.frames.length - 1];
      ctx.globalAlpha = 0.35; this._path(ctx, t, f.line, seriesColor(i), 2); ctx.globalAlpha = 1;
      this._path(ctx, t, s.X, seriesColor(i), 1.8);
    });
  }
}

window.LineChart = LineChart; window.ArmView = ArmView; window.seriesColor = seriesColor; window.fmt = fmt; window.tooltip = tooltip;

#!/usr/bin/env python3
"""Browser-based live DC motor control dashboard.

This server runs the same neural/solver DC motor loop as dc_motor_realtime_loop.py,
but exposes voltage control and live telemetry through a local web UI.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
import torch

from datasets.dc_motor import DEFAULTS, dc_motor_gradient
from util import DynamicLoad, latest_file


DEFAULT_STABLE_MODEL = (
    "stabledynamics[latent_space_dim=2,a=0.001,projfn=NN-REHU,"
    "projfn_eps=0.01,smooth_v=0,hp=60,h=100,rehu=0.001]"
)
DEFAULT_STABLE_WEIGHT = "experiments/dc-motor-stable/checkpoint_*.pth"


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DC Motor Live Control</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657181;
      --line: #d8dee8;
      --accent: #0b7fab;
      --accent-2: #bf5b04;
      --danger: #b42318;
      --ok: #087443;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.07);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 5;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 650;
      letter-spacing: 0;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      white-space: nowrap;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #98a2b3;
      display: inline-block;
    }

    .dot.running { background: var(--ok); }
    .dot.error { background: var(--danger); }

    main {
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .controls {
      align-self: start;
      position: sticky;
      top: 62px;
      padding: 14px;
    }

    .section {
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }

    .section:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .section-title {
      margin: 0 0 10px;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    label {
      display: block;
      margin: 10px 0 5px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 600;
    }

    input, select, button {
      font: inherit;
    }

    input[type="text"], input[type="number"], select {
      width: 100%;
      height: 34px;
      border: 1px solid #c8d1dc;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
      outline: none;
    }

    input[type="text"]:focus, input[type="number"]:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(11, 127, 171, 0.12);
    }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }

    button {
      height: 36px;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 0 12px;
      cursor: pointer;
      font-weight: 650;
      background: #eef2f6;
      color: var(--ink);
    }

    button.primary {
      background: var(--accent);
      color: #fff;
    }

    button.danger {
      background: #fff;
      color: var(--danger);
      border-color: #f0b4ad;
    }

    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .voltage-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 88px;
      gap: 10px;
      align-items: center;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }

    .metric {
      padding: 12px;
      min-height: 72px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
      overflow-wrap: anywhere;
    }

    .charts {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .chart-card {
      padding: 12px;
      min-height: 280px;
    }

    .chart-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }

    .chart-title {
      font-weight: 700;
      font-size: 13px;
    }

    .legend {
      display: flex;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .legend i {
      display: inline-block;
      width: 14px;
      height: 3px;
      vertical-align: middle;
      margin-right: 4px;
      background: var(--accent);
    }

    .legend i.alt { background: var(--accent-2); }

    canvas {
      display: block;
      width: 100%;
      height: 220px;
      border: 1px solid #edf0f4;
      border-radius: 6px;
      background: #fff;
    }

    .message {
      color: var(--muted);
      margin-top: 10px;
      line-height: 1.4;
      min-height: 20px;
      overflow-wrap: anywhere;
    }

    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
      .charts, .metrics {
        grid-template-columns: 1fr;
      }
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status {
        white-space: normal;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>DC Motor Live Control</h1>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Idle</span></div>
  </header>

  <main>
    <aside class="panel controls">
      <div class="section">
        <div class="section-title">Model</div>
        <label for="mode">Mode</label>
        <select id="mode">
          <option value="neural_vs_solver">Neural + solver</option>
          <option value="neural">Neural only</option>
          <option value="solver">Solver only</option>
        </select>

        <label for="modelSpec">Model spec</label>
        <input id="modelSpec" type="text" />

        <label for="weight">Checkpoint path or glob</label>
        <input id="weight" type="text" />
      </div>

      <div class="section">
        <div class="section-title">Simulation</div>
        <div class="grid-2">
          <div>
            <label for="dt">dt [s]</label>
            <input id="dt" type="number" min="0.001" max="0.2" step="0.001" value="0.01" />
          </div>
          <div>
            <label for="historyLimit">Samples</label>
            <input id="historyLimit" type="number" min="200" max="20000" step="100" value="2000" />
          </div>
          <div>
            <label for="x0Omega">x0 omega</label>
            <input id="x0Omega" type="number" step="0.1" value="0" />
          </div>
          <div>
            <label for="x0Current">x0 current</label>
            <input id="x0Current" type="number" step="0.1" value="0" />
          </div>
        </div>
        <label for="forceCpu">
          <input id="forceCpu" type="checkbox" />
          Force CPU
        </label>
        <div class="actions">
          <button id="startBtn" class="primary">Start</button>
          <button id="stopBtn" class="danger">Stop</button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Voltage</div>
        <div class="voltage-row">
          <input id="voltageSlider" type="range" min="-10" max="10" step="0.05" value="0" />
          <input id="voltageNumber" type="number" min="-100" max="100" step="0.05" value="0" />
        </div>
        <div class="actions">
          <button id="minusBtn">-0.25 V</button>
          <button id="plusBtn">+0.25 V</button>
          <button id="zeroBtn">Zero</button>
          <button id="clearBtn">Clear</button>
        </div>
        <div id="message" class="message"></div>
      </div>
    </aside>

    <section>
      <div class="metrics">
        <div class="panel metric"><span>Voltage</span><strong id="mVa">0 V</strong></div>
        <div class="panel metric"><span>Omega</span><strong id="mOmega">0</strong></div>
        <div class="panel metric"><span>Current</span><strong id="mCurrent">0</strong></div>
        <div class="panel metric"><span>Energy</span><strong id="mEnergy">0</strong></div>
      </div>

      <div class="charts">
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Omega</div><div class="legend"><span><i></i>model</span><span><i class="alt"></i>solver</span></div></div>
          <canvas id="omegaChart"></canvas>
        </div>
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Current</div><div class="legend"><span><i></i>model</span><span><i class="alt"></i>solver</span></div></div>
          <canvas id="currentChart"></canvas>
        </div>
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Voltage</div></div>
          <canvas id="voltageChart"></canvas>
        </div>
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Energy</div><div class="legend"><span><i></i>model</span><span><i class="alt"></i>solver</span></div></div>
          <canvas id="energyChart"></canvas>
        </div>
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Input Power</div></div>
          <canvas id="powerChart"></canvas>
        </div>
        <div class="panel chart-card">
          <div class="chart-head"><div class="chart-title">Phase Portrait</div><div class="legend"><span><i></i>model</span><span><i class="alt"></i>solver</span></div></div>
          <canvas id="phaseChart"></canvas>
        </div>
      </div>
    </section>
  </main>

  <script>
    const defaults = {
      modelSpec: "__MODEL_SPEC__",
      weight: "__WEIGHT__"
    };

    const state = {
      voltage: 0,
      running: false,
      timer: null,
      lastError: ""
    };

    const $ = (id) => document.getElementById(id);

    $("modelSpec").value = defaults.modelSpec;
    $("weight").value = defaults.weight;

    function setMessage(text, isError = false) {
      $("message").textContent = text || "";
      $("message").style.color = isError ? "var(--danger)" : "var(--muted)";
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const text = await response.text();
      let payload = {};
      if (text) payload = JSON.parse(text);
      if (!response.ok) {
        throw new Error(payload.error || response.statusText);
      }
      return payload;
    }

    function numberValue(id) {
      const value = Number($(id).value);
      if (!Number.isFinite(value)) throw new Error(`${id} must be numeric`);
      return value;
    }

    async function startSimulation() {
      try {
        setMessage("Starting...");
        const payload = {
          mode: $("mode").value,
          model_spec: $("modelSpec").value,
          weight: $("weight").value,
          dt: numberValue("dt"),
          x0_omega: numberValue("x0Omega"),
          x0_current: numberValue("x0Current"),
          voltage: Number($("voltageNumber").value),
          force_cpu: $("forceCpu").checked,
          history_limit: Math.trunc(numberValue("historyLimit"))
        };
        await api("/api/start", {method: "POST", body: JSON.stringify(payload)});
        setMessage("Running");
        await refreshState();
      } catch (err) {
        setMessage(err.message, true);
      }
    }

    async function stopSimulation() {
      try {
        await api("/api/stop", {method: "POST", body: "{}"});
        setMessage("Stopped");
        await refreshState();
      } catch (err) {
        setMessage(err.message, true);
      }
    }

    async function clearHistory() {
      try {
        await api("/api/clear", {method: "POST", body: "{}"});
        await refreshState();
      } catch (err) {
        setMessage(err.message, true);
      }
    }

    async function setVoltage(value) {
      const voltage = Number(value);
      if (!Number.isFinite(voltage)) return;
      state.voltage = voltage;
      $("voltageSlider").value = Math.max(-10, Math.min(10, voltage));
      $("voltageNumber").value = voltage.toFixed(2).replace(/\.00$/, "");
      try {
        await api("/api/voltage", {method: "POST", body: JSON.stringify({voltage})});
      } catch (err) {
        setMessage(err.message, true);
      }
    }

    function formatValue(value, unit = "") {
      if (value === null || value === undefined || !Number.isFinite(value)) return "-";
      const abs = Math.abs(value);
      const text = abs >= 1000 || (abs > 0 && abs < 0.001) ? value.toExponential(3) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
      return unit ? `${text} ${unit}` : text;
    }

    function updateStatus(payload) {
      state.running = payload.running;
      $("statusDot").className = `dot ${payload.running ? "running" : ""}`;
      $("statusText").textContent = payload.running
        ? `Running | ${payload.mode} | ${payload.device} | t=${formatValue(payload.t, "s")}`
        : `Idle | ${payload.mode || "not configured"}`;
      if (payload.error) {
        $("statusDot").className = "dot error";
        $("statusText").textContent = payload.error;
      }
      const latest = payload.latest || {};
      $("mVa").textContent = formatValue(latest.va || payload.voltage || 0, "V");
      $("mOmega").textContent = formatValue(latest.omega_model);
      $("mCurrent").textContent = formatValue(latest.current_model);
      $("mEnergy").textContent = formatValue(latest.energy_model);
    }

    function extent(values, padding = 0.08) {
      const finite = values.filter((v) => Number.isFinite(v));
      if (!finite.length) return [-1, 1];
      let min = Math.min(...finite);
      let max = Math.max(...finite);
      if (min === max) {
        const d = Math.max(1, Math.abs(min) * 0.2);
        min -= d;
        max += d;
      }
      const pad = (max - min) * padding;
      return [min - pad, max + pad];
    }

    function drawAxes(ctx, width, height, yMin, yMax, label) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "#e7ebf0";
      ctx.lineWidth = 1;
      for (let i = 1; i < 4; i++) {
        const y = (height * i) / 4;
        ctx.beginPath();
        ctx.moveTo(42, y);
        ctx.lineTo(width - 10, y);
        ctx.stroke();
      }
      ctx.fillStyle = "#657181";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(formatValue(yMax), 8, 16);
      ctx.fillText(formatValue(yMin), 8, height - 8);
      if (label) ctx.fillText(label, 44, 16);
    }

    function drawLineChart(canvas, rows, series, label = "") {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      const left = 42;
      const right = 10;
      const top = 10;
      const bottom = 22;
      const values = [];
      for (const s of series) {
        for (const row of rows) values.push(row[s.key]);
      }
      const [yMin, yMax] = extent(values);
      drawAxes(ctx, w, h, yMin, yMax, label);
      if (rows.length < 2) return;
      const tMin = rows[0].t;
      const tMax = rows[rows.length - 1].t;
      const xOf = (t) => left + ((t - tMin) / Math.max(1e-9, tMax - tMin)) * (w - left - right);
      const yOf = (v) => top + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (h - top - bottom);
      for (const s of series) {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (const row of rows) {
          const value = row[s.key];
          if (!Number.isFinite(value)) continue;
          const x = xOf(row.t);
          const y = yOf(value);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    }

    function drawPhaseChart(canvas, rows) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width;
      const h = rect.height;
      const left = 42;
      const right = 10;
      const top = 10;
      const bottom = 22;
      const xs = rows.flatMap((r) => [r.omega_model, r.omega_solver]);
      const ys = rows.flatMap((r) => [r.current_model, r.current_solver]);
      const [xMin, xMax] = extent(xs);
      const [yMin, yMax] = extent(ys);
      drawAxes(ctx, w, h, yMin, yMax, "omega/current");
      const xOf = (v) => left + ((v - xMin) / Math.max(1e-9, xMax - xMin)) * (w - left - right);
      const yOf = (v) => top + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * (h - top - bottom);
      ctx.strokeStyle = "#ccd3dd";
      ctx.beginPath();
      ctx.moveTo(xOf(0), top);
      ctx.lineTo(xOf(0), h - bottom);
      ctx.moveTo(left, yOf(0));
      ctx.lineTo(w - right, yOf(0));
      ctx.stroke();
      for (const series of [
        {x: "omega_model", y: "current_model", color: "#0b7fab"},
        {x: "omega_solver", y: "current_solver", color: "#bf5b04"}
      ]) {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (const row of rows) {
          const xVal = row[series.x];
          const yVal = row[series.y];
          if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue;
          const x = xOf(xVal);
          const y = yOf(yVal);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    }

    function drawCharts(payload) {
      const rows = payload.history || [];
      const model = "#0b7fab";
      const solver = "#bf5b04";
      drawLineChart($("omegaChart"), rows, [
        {key: "omega_model", color: model},
        {key: "omega_solver", color: solver}
      ]);
      drawLineChart($("currentChart"), rows, [
        {key: "current_model", color: model},
        {key: "current_solver", color: solver}
      ]);
      drawLineChart($("voltageChart"), rows, [{key: "va", color: model}], "V");
      drawLineChart($("energyChart"), rows, [
        {key: "energy_model", color: model},
        {key: "energy_solver", color: solver}
      ]);
      drawLineChart($("powerChart"), rows, [{key: "pin", color: model}]);
      drawPhaseChart($("phaseChart"), rows);
    }

    async function refreshState() {
      try {
        const payload = await api("/api/state");
        updateStatus(payload);
        drawCharts(payload);
      } catch (err) {
        $("statusDot").className = "dot error";
        $("statusText").textContent = err.message;
      }
    }

    $("startBtn").addEventListener("click", startSimulation);
    $("stopBtn").addEventListener("click", stopSimulation);
    $("clearBtn").addEventListener("click", clearHistory);
    $("zeroBtn").addEventListener("click", () => setVoltage(0));
    $("plusBtn").addEventListener("click", () => setVoltage(Number($("voltageNumber").value) + 0.25));
    $("minusBtn").addEventListener("click", () => setVoltage(Number($("voltageNumber").value) - 0.25));
    $("voltageSlider").addEventListener("input", (event) => {
      $("voltageNumber").value = Number(event.target.value).toFixed(2).replace(/\.00$/, "");
    });
    $("voltageSlider").addEventListener("change", (event) => setVoltage(event.target.value));
    $("voltageNumber").addEventListener("change", (event) => setVoltage(event.target.value));

    window.addEventListener("resize", refreshState);
    state.timer = window.setInterval(refreshState, 250);
    refreshState();
  </script>
</body>
</html>
"""


def motor_energy_np(x, params):
    omega = x[..., 0]
    current = x[..., 1]
    return 0.5 * params["J"] * omega**2 + 0.5 * params["L"] * current**2


def rk4_step(rhs_fn, x, dt):
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def load_neural_model(model_spec, weight, device):
    weight_path = latest_file(weight)
    model_module = DynamicLoad("models")(model_spec)
    model = model_module.model
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, weight_path


def nn_autonomous_rhs(model, x_np, device):
    x = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(1, 2)
    x = x.detach().requires_grad_(True)
    dx = model(x)
    return dx.detach().cpu().numpy().reshape(2).astype(np.float32)


class LiveMotorSimulator:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self.running = False
        self.error = None
        self.mode = None
        self.device = "cpu"
        self.weight_path = None
        self.model_spec = DEFAULT_STABLE_MODEL
        self.weight = DEFAULT_STABLE_WEIGHT
        self.voltage = 0.0
        self.dt = 0.01
        self.t = 0.0
        self.history = []
        self.history_limit = 2000
        self.latest = None

    def start(self, config):
        self.stop()
        with self.lock:
            self.error = None
            self.running = True
            self.mode = config["mode"]
            self.device = "cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu"
            self.model_spec = config["model_spec"]
            self.weight = config["weight"]
            self.weight_path = None
            self.voltage = float(config["voltage"])
            self.dt = float(config["dt"])
            self.t = 0.0
            self.history = []
            self.history_limit = int(config["history_limit"])
            self.latest = None

        self.thread = threading.Thread(target=self._run_loop, args=(config,), daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

    def clear(self):
        with self.lock:
            self.history = []

    def set_voltage(self, voltage):
        with self.lock:
            self.voltage = float(voltage)

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "mode": self.mode,
                "device": self.device,
                "model_spec": self.model_spec,
                "weight": self.weight,
                "weight_path": self.weight_path,
                "voltage": self.voltage,
                "dt": self.dt,
                "t": self.t,
                "latest": self.latest,
                "history": list(self.history),
            }

    def _set_error(self, exc):
        with self.lock:
            self.error = str(exc)
            self.running = False

    def _append_history(self, row):
        with self.lock:
            self.t = row["t"]
            self.latest = row
            self.history.append(row)
            if len(self.history) > self.history_limit:
                del self.history[: len(self.history) - self.history_limit]

    def _get_voltage(self):
        with self.lock:
            return self.voltage

    def _is_running(self):
        with self.lock:
            return self.running

    def _run_loop(self, config):
        try:
            params = DEFAULTS.copy()
            params["Va"] = 0.0
            params["Tl"] = 0.0
            rhs_auto = dc_motor_gradient(params)

            model = None
            if config["mode"] in ["neural", "neural_vs_solver"]:
                model, weight_path = load_neural_model(config["model_spec"], config["weight"], self.device)
                with self.lock:
                    self.weight_path = weight_path

            x_model = np.array([config["x0_omega"], config["x0_current"]], dtype=np.float32)
            x_solver = x_model.copy()
            cumulative_pos_energy = 0.0
            cumulative_net_energy = 0.0
            step = 0
            wall_start = time.perf_counter()

            while self._is_running():
                t = step * config["dt"]
                va = self._get_voltage()

                if config["mode"] == "solver":
                    x_model = x_solver.copy()

                energy_model = float(motor_energy_np(x_model, params))
                input_power = float(va * x_model[1])

                row = {
                    "t": float(t),
                    "va": float(va),
                    "omega_model": float(x_model[0]),
                    "current_model": float(x_model[1]),
                    "energy_model": energy_model,
                    "pin": input_power,
                    "ein_pos": float(cumulative_pos_energy),
                    "ein_net": float(cumulative_net_energy),
                    "omega_solver": None,
                    "current_solver": None,
                    "energy_solver": None,
                }

                if config["mode"] in ["solver", "neural_vs_solver"]:
                    row["omega_solver"] = float(x_solver[0])
                    row["current_solver"] = float(x_solver[1])
                    row["energy_solver"] = float(motor_energy_np(x_solver, params))

                self._append_history(row)

                def solver_rhs(x):
                    dx = rhs_auto(x).astype(np.float32)
                    dx[1] += va / params["L"]
                    return dx

                if config["mode"] == "solver":
                    x_solver = rk4_step(solver_rhs, x_solver, config["dt"])
                elif config["mode"] == "neural":
                    def model_rhs(x):
                        dx = nn_autonomous_rhs(model, x, self.device)
                        dx[1] += va / params["L"]
                        return dx
                    x_model = rk4_step(model_rhs, x_model, config["dt"])
                else:
                    def model_rhs(x):
                        dx = nn_autonomous_rhs(model, x, self.device)
                        dx[1] += va / params["L"]
                        return dx
                    x_model = rk4_step(model_rhs, x_model, config["dt"])
                    x_solver = rk4_step(solver_rhs, x_solver, config["dt"])

                cumulative_net_energy += input_power * config["dt"]
                cumulative_pos_energy += max(input_power, 0.0) * config["dt"]

                step += 1
                target = wall_start + step * config["dt"]
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except Exception as exc:
            self._set_error(exc)


def parse_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    body = handler.rfile.read(length).decode("utf-8")
    return json.loads(body)


def write_json(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def make_handler(simulator):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                html = INDEX_HTML.replace("__MODEL_SPEC__", DEFAULT_STABLE_MODEL).replace("__WEIGHT__", DEFAULT_STABLE_WEIGHT)
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/api/state":
                write_json(self, simulator.snapshot())
                return
            write_json(self, {"error": "not found"}, status=404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                payload = parse_json_body(self)
                if path == "/api/start":
                    config = self._config_from_payload(payload)
                    simulator.start(config)
                    write_json(self, {"ok": True})
                    return
                if path == "/api/stop":
                    simulator.stop()
                    write_json(self, {"ok": True})
                    return
                if path == "/api/clear":
                    simulator.clear()
                    write_json(self, {"ok": True})
                    return
                if path == "/api/voltage":
                    simulator.set_voltage(float(payload["voltage"]))
                    write_json(self, {"ok": True})
                    return
                write_json(self, {"error": "not found"}, status=404)
            except Exception as exc:
                write_json(self, {"error": str(exc)}, status=400)

        def _config_from_payload(self, payload):
            mode = payload.get("mode", "neural_vs_solver")
            if mode not in ["neural", "solver", "neural_vs_solver"]:
                raise ValueError("mode must be neural, solver, or neural_vs_solver")
            dt = float(payload.get("dt", 0.01))
            if dt <= 0:
                raise ValueError("dt must be positive")
            history_limit = int(payload.get("history_limit", 2000))
            if history_limit < 10:
                raise ValueError("history_limit must be at least 10")
            return {
                "mode": mode,
                "model_spec": str(payload.get("model_spec") or DEFAULT_STABLE_MODEL),
                "weight": str(payload.get("weight") or DEFAULT_STABLE_WEIGHT),
                "dt": dt,
                "x0_omega": float(payload.get("x0_omega", 0.0)),
                "x0_current": float(payload.get("x0_current", 0.0)),
                "voltage": float(payload.get("voltage", 0.0)),
                "force_cpu": bool(payload.get("force_cpu", False)),
                "history_limit": history_limit,
            }

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Start a localhost DC motor live-control dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="server bind host")
    parser.add_argument("--port", type=int, default=8765, help="server bind port")
    return parser.parse_args()


def main():
    args = parse_args()
    simulator = LiveMotorSimulator()
    handler = make_handler(simulator)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"DC motor live dashboard: {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        simulator.stop()
        server.server_close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-time DC motor neural simulation loop.

This script runs a trained autonomous DC motor neural dynamics model in a
step-by-step simulation loop and injects an external voltage signal analytically.

Important modeling note
-----------------------
The current DC motor neural model is trained on the no-input / autonomous system:

    x = [omega, current]
    x_dot = f_autonomous(x)       with Va = 0, Tl = 0

For the physical DC motor equations,

    omega_dot  = (Kt*i - b*omega - Tl) / J
    current_dot = (Va - R*i - Ke*omega) / L

voltage enters additively as +Va/L in current_dot. Therefore, this real-time
loop uses the trained neural model for the autonomous part and adds the external
voltage channel analytically:

    x_dot_model_controlled = f_nn_autonomous(x) + [0, Va(t)/L]

This is a practical first version. A fully learned controlled model f(x, u)
would require a controlled dataset and a model interface that consumes both
state and input.
"""

import argparse
import csv
import os
import select
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.dc_motor import DEFAULTS, dc_motor_gradient
from util import DynamicLoad, latest_file, setup_logging


logger = setup_logging(os.path.basename(__file__))


DEFAULT_STABLE_MODEL = (
    "stabledynamics[latent_space_dim=2,a=0.001,projfn=NN-REHU,"
    "projfn_eps=0.01,smooth_v=0,hp=60,h=100,rehu=0.001]"
)
DEFAULT_STABLE_WEIGHT = "experiments/dc-motor-stable/checkpoint_*.pth"


def input_signal(t, args):
    """Voltage input Va(t). Amplitude is interpreted as volts."""
    kind = args.input_kind.lower()
    amp = args.amplitude

    if t < args.input_start:
        return 0.0

    tau = t - args.input_start

    if kind == "none":
        return 0.0
    if kind == "constant":
        return amp
    if kind == "step":
        return amp
    if kind == "ramp":
        if args.rise_time <= 0:
            return amp
        return amp * min(max(tau / args.rise_time, 0.0), 1.0)
    if kind == "sine":
        return amp * np.sin(2.0 * np.pi * args.frequency * tau)
    if kind == "pulse":
        if args.frequency <= 0:
            return amp
        period = 1.0 / args.frequency
        phase = (tau % period) / period
        return amp if phase < args.duty else 0.0
    if kind == "live":
        return amp

    raise ValueError(f"Unknown input kind: {args.input_kind}")


def print_live_input_help(args):
    print()
    print("Live voltage control")
    print("--------------------")
    print("Type a voltage and press Enter to apply it on the next simulation step.")
    print(f"Examples: 2.5, -1, 0, +, -  (+/- use step {args.live_step} V)")
    print("Commands: zero, status, q")
    print()


def poll_live_voltage(current_va, args, t, x_model, energy_model):
    """Read all pending terminal commands without blocking the simulation loop."""
    stop_requested = False

    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            break

        line = sys.stdin.readline()
        if line == "":
            break

        command = line.strip().lower()
        if not command:
            continue

        if command in ["q", "quit", "exit", "stop"]:
            stop_requested = True
            break
        if command in ["z", "zero"]:
            current_va = 0.0
        elif command in ["+", "up"]:
            current_va += args.live_step
        elif command in ["-", "down"]:
            current_va -= args.live_step
        elif command in ["s", "status"]:
            print(
                f"t={t:.3f}s Va={current_va:.6g} "
                f"omega={x_model[0]:.6g} current={x_model[1]:.6g} "
                f"energy={energy_model:.6g}"
            )
            continue
        else:
            try:
                current_va = float(command)
            except ValueError:
                print(f"Ignored unknown live input command: {command!r}")
                continue

        print(f"t={t:.3f}s Va set to {current_va:.6g} V")

    return current_va, stop_requested


def motor_energy_np(x, params):
    omega = x[..., 0]
    current = x[..., 1]
    return 0.5 * params["J"] * omega**2 + 0.5 * params["L"] * current**2


def load_model(args, device):
    if args.solver_only:
        return None, None

    weight = latest_file(args.weight)
    model_module = DynamicLoad("models")(args.model_spec)
    model = model_module.model
    model.load_state_dict(torch.load(weight, map_location=device))
    model.to(device)
    model.eval()
    return model, weight


def nn_autonomous_rhs(model, x_np, device):
    """Evaluate autonomous neural dynamics. Works for stable dynamics requiring grad wrt x."""
    x = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(1, 2)
    x = x.detach().requires_grad_(True)
    dx = model(x)
    return dx.detach().cpu().numpy().reshape(2).astype(np.float32)


def controlled_rhs_nn(model, x_np, va, params, device):
    dx = nn_autonomous_rhs(model, x_np, device)
    dx[1] += va / params["L"]
    return dx


def controlled_rhs_solver(rhs_auto, x_np, va, params):
    dx = rhs_auto(x_np).astype(np.float32)
    dx[1] += va / params["L"]
    return dx


def rk4_step(rhs_fn, x, dt):
    k1 = rhs_fn(x)
    k2 = rhs_fn(x + 0.5 * dt * k1)
    k3 = rhs_fn(x + 0.5 * dt * k2)
    k4 = rhs_fn(x + dt * k3)
    return (x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def init_live_plot(args):
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    axes = axes.ravel()

    axes[0].set_title("omega")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("rad/s")

    axes[1].set_title("current")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("A")

    axes[2].set_title("input voltage Va(t)")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("V")

    axes[3].set_title("energy")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("J-like energy")

    axes[4].set_title("cumulative input energy")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("J")

    axes[5].set_title("phase portrait")
    axes[5].set_xlabel("omega [rad/s]")
    axes[5].set_ylabel("current [A]")

    lines = {}
    lines["omega"], = axes[0].plot([], [], label="model")
    lines["current"], = axes[1].plot([], [], label="model")
    lines["voltage"], = axes[2].plot([], [], label="Va")
    lines["energy"], = axes[3].plot([], [], label="model energy")
    lines["ein_pos"], = axes[4].plot([], [], label="positive input energy")
    lines["ein_net"], = axes[4].plot([], [], linestyle="--", label="net input energy")
    lines["phase"], = axes[5].plot([], [], label="model")
    lines["origin"] = axes[5].scatter([0.0], [0.0], marker="x", s=80)

    if args.compare_solver:
        lines["omega_solver"], = axes[0].plot([], [], linestyle="--", label="solver")
        lines["current_solver"], = axes[1].plot([], [], linestyle="--", label="solver")
        lines["energy_solver"], = axes[3].plot([], [], linestyle="--", label="solver energy")
        lines["phase_solver"], = axes[5].plot([], [], linestyle="--", label="solver")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    return fig, axes, lines


def update_plot(fig, axes, lines, hist, args):
    t = np.asarray(hist["t"])
    x = np.asarray(hist["x_model"])
    va = np.asarray(hist["va"])
    energy = np.asarray(hist["energy_model"])
    ein_pos = np.asarray(hist["ein_pos"])
    ein_net = np.asarray(hist["ein_net"])

    lines["omega"].set_data(t, x[:, 0])
    lines["current"].set_data(t, x[:, 1])
    lines["voltage"].set_data(t, va)
    lines["energy"].set_data(t, energy)
    lines["ein_pos"].set_data(t, ein_pos)
    lines["ein_net"].set_data(t, ein_net)
    lines["phase"].set_data(x[:, 0], x[:, 1])

    if args.compare_solver:
        xs = np.asarray(hist["x_solver"])
        es = np.asarray(hist["energy_solver"])
        lines["omega_solver"].set_data(t, xs[:, 0])
        lines["current_solver"].set_data(t, xs[:, 1])
        lines["energy_solver"].set_data(t, es)
        lines["phase_solver"].set_data(xs[:, 0], xs[:, 1])

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    axes[5].set_aspect("equal", adjustable="datalim")
    fig.canvas.draw_idle()
    plt.pause(0.001)


def write_csv(path, hist, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "t", "omega_model", "current_model", "Va", "energy_model",
            "input_power", "cumulative_positive_input_energy", "cumulative_net_input_energy",
        ]
        if args.compare_solver:
            header += ["omega_solver", "current_solver", "energy_solver"]
        writer.writerow(header)

        n = len(hist["t"])
        for i in range(n):
            row = [
                hist["t"][i],
                hist["x_model"][i][0],
                hist["x_model"][i][1],
                hist["va"][i],
                hist["energy_model"][i],
                hist["pin"][i],
                hist["ein_pos"][i],
                hist["ein_net"][i],
            ]
            if args.compare_solver:
                row += [
                    hist["x_solver"][i][0],
                    hist["x_solver"][i][1],
                    hist["energy_solver"][i],
                ]
            writer.writerow(row)


def run(args):
    params = DEFAULTS.copy()
    for key in ["J", "b", "Kt", "Ke", "R", "L", "Va", "Tl"]:
        val = getattr(args, key)
        if val is not None:
            params[key] = float(val)

    # The neural model represents the autonomous part. Keep the internal dataset
    # voltage/load at zero and inject the runtime input externally.
    params["Va"] = 0.0
    params["Tl"] = 0.0
    rhs_auto = dc_motor_gradient(params)

    if args.input_kind == "live" and not args.realtime:
        args.realtime = True
        print("Live input enabled wall-clock pacing (--realtime).")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, weight = load_model(args, device)
    if model is None and not args.solver_only:
        raise RuntimeError("Model failed to load.")

    print("Real-time DC motor neural loop")
    print("--------------------------------")
    print(f"device:          {device}")
    print(f"model:           {'solver only' if model is None else args.model_spec}")
    print(f"weight:          {weight}")
    print(f"input kind:      {args.input_kind}")
    print(f"amplitude [V]:   {args.amplitude}")
    print(f"dt [s]:          {args.dt}")
    print(f"duration [s]:    {args.duration}")
    print(f"energy budget:   {args.energy_budget}")
    if args.input_kind == "live":
        print_live_input_help(args)

    x_model = np.array([args.x0_omega, args.x0_current], dtype=np.float32)
    x_solver = x_model.copy()
    live_va = float(args.amplitude)

    hist = {
        "t": [], "x_model": [], "va": [], "energy_model": [],
        "pin": [], "ein_pos": [], "ein_net": [],
    }
    if args.compare_solver:
        hist["x_solver"] = []
        hist["energy_solver"] = []

    fig = axes = lines = None
    if not args.no_plot:
        fig, axes, lines = init_live_plot(args)

    cumulative_pos_energy = 0.0
    cumulative_net_energy = 0.0
    steps = int(np.ceil(args.duration / args.dt)) + 1
    t0_wall = time.perf_counter()

    for step in range(steps):
        t = step * args.dt

        energy_model = float(motor_energy_np(x_model, params))
        if args.input_kind == "live":
            live_va, stop_requested = poll_live_voltage(live_va, args, t, x_model, energy_model)
            va = live_va
        else:
            stop_requested = False
            va = float(input_signal(t, args))

        current_for_power = float(x_model[1])
        input_power = va * current_for_power

        hist["t"].append(t)
        hist["x_model"].append(x_model.copy())
        hist["va"].append(va)
        hist["energy_model"].append(energy_model)
        hist["pin"].append(input_power)
        hist["ein_pos"].append(cumulative_pos_energy)
        hist["ein_net"].append(cumulative_net_energy)

        if args.compare_solver:
            hist["x_solver"].append(x_solver.copy())
            hist["energy_solver"].append(float(motor_energy_np(x_solver, params)))

        if not args.no_plot and step % args.plot_every == 0:
            update_plot(fig, axes, lines, hist, args)

        if args.energy_budget is not None and cumulative_pos_energy >= args.energy_budget:
            print(f"Stopping: positive input energy budget reached at t={t:.4f}s")
            break

        if stop_requested:
            print(f"Stopping: live input requested stop at t={t:.4f}s")
            break

        if step == steps - 1:
            break

        def rhs_model_local(x):
            if args.solver_only:
                return controlled_rhs_solver(rhs_auto, x, va, params)
            return controlled_rhs_nn(model, x, va, params, device)

        x_model = rk4_step(rhs_model_local, x_model, args.dt)

        if args.compare_solver:
            x_solver = rk4_step(lambda x: controlled_rhs_solver(rhs_auto, x, va, params), x_solver, args.dt)

        # Electrical power integrated over time. Positive energy is what the
        # source injects into the motor. Net energy can be negative during regen.
        cumulative_net_energy += input_power * args.dt
        cumulative_pos_energy += max(input_power, 0.0) * args.dt

        if args.realtime:
            target_wall = t0_wall + (step + 1) * args.dt
            sleep_s = target_wall - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / args.csv_name
    write_csv(csv_path, hist, args)

    if not args.no_plot:
        update_plot(fig, axes, lines, hist, args)
        png_path = output_dir / args.png_name
        fig.savefig(png_path, dpi=150)
        if args.hold:
            plt.ioff()
            plt.show()
        else:
            plt.close(fig)
    else:
        png_path = None

    print("Done.")
    print(f"csv: {csv_path}")
    if png_path is not None:
        print(f"plot: {png_path}")
    print(f"final state model: {hist['x_model'][-1]}")
    print(f"final model energy: {hist['energy_model'][-1]}")
    print(f"positive input energy: {hist['ein_pos'][-1]}")
    print(f"net input energy: {hist['ein_net'][-1]}")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time DC motor neural dynamics loop with voltage input signals.")

    parser.add_argument("--model-spec", default=DEFAULT_STABLE_MODEL, help="DynamicLoad model specification")
    parser.add_argument("--weight", default=DEFAULT_STABLE_WEIGHT, help="checkpoint path or glob")
    parser.add_argument("--solver-only", action="store_true", help="skip neural model and run only physical solver with the same input")
    parser.add_argument("--compare-solver", action="store_true", help="also run the analytic solver in parallel for comparison")
    parser.add_argument("--cpu", action="store_true", help="force CPU")

    parser.add_argument("--input-kind", default="step", choices=["none", "constant", "step", "ramp", "sine", "pulse", "live"], help="input voltage profile")
    parser.add_argument("--amplitude", type=float, default=2.0, help="input voltage amplitude [V]")
    parser.add_argument("--live-step", type=float, default=0.25, help="voltage increment for +/- commands in live input mode [V]")
    parser.add_argument("--input-start", type=float, default=0.5, help="time when input starts [s]")
    parser.add_argument("--rise-time", type=float, default=2.0, help="ramp rise time [s]")
    parser.add_argument("--frequency", type=float, default=1.0, help="sine/pulse frequency [Hz]")
    parser.add_argument("--duty", type=float, default=0.5, help="pulse duty ratio in [0, 1]")
    parser.add_argument("--energy-budget", type=float, default=None, help="stop when positive injected electrical energy reaches this value [J]")

    parser.add_argument("--x0-omega", type=float, default=0.0, help="initial omega [rad/s]")
    parser.add_argument("--x0-current", type=float, default=0.0, help="initial current [A]")
    parser.add_argument("--dt", type=float, default=0.01, help="simulation step [s]")
    parser.add_argument("--duration", type=float, default=20.0, help="simulation duration [s]")
    parser.add_argument("--realtime", action="store_true", help="sleep between steps to approximately match wall-clock time")
    parser.add_argument("--plot-every", type=int, default=5, help="plot update period in simulation steps")
    parser.add_argument("--no-plot", action="store_true", help="do not open live matplotlib plot")
    parser.add_argument("--hold", action="store_true", help="keep plot window open at the end")

    parser.add_argument("--output-dir", default="experiments/dc-motor-realtime", help="output directory")
    parser.add_argument("--csv-name", default="realtime_loop.csv", help="CSV output filename")
    parser.add_argument("--png-name", default="realtime_loop.png", help="final plot filename")

    # Optional physical parameter overrides for experimentation. Va/Tl are kept
    # zero internally for the autonomous model and runtime input injection.
    parser.add_argument("--J", type=float, default=None)
    parser.add_argument("--b", type=float, default=None)
    parser.add_argument("--Kt", type=float, default=None)
    parser.add_argument("--Ke", type=float, default=None)
    parser.add_argument("--R", type=float, default=None)
    parser.add_argument("--L", type=float, default=None)
    parser.add_argument("--Va", type=float, default=None)
    parser.add_argument("--Tl", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

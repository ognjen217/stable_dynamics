#!/usr/bin/env python3
"""Rollout evaluation for the autonomous DC motor benchmark.

This script mirrors the role of pendulum_error.py, but is intentionally specific
and simpler because the DC motor state is Euclidean and two-dimensional:
  x = [omega, current]

It compares an RK4 rollout of the true DC motor ODE against an RK4 rollout of a
trained neural dynamics model. It writes:
  - rollout_metrics.csv
  - rollout_error.png
  - rollout_phase.png
  - rollout_states.png
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from util import DynamicLoad, latest_file, setup_logging, to_variable


logger = setup_logging(os.path.basename(__file__))


def rk4_numpy(rhs, x, h):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * h * k1)
    k3 = rhs(x + 0.5 * h * k2)
    k4 = rhs(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk4_torch(model, x, h):
    x = x.detach().requires_grad_(True)
    k1 = model(x)
    k2 = model((x + 0.5 * h * k1).detach().requires_grad_(True))
    k3 = model((x + 0.5 * h * k2).detach().requires_grad_(True))
    k4 = model((x + h * k3).detach().requires_grad_(True))
    return (x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).detach()


def motor_energy(x, params):
    """Simple physical energy proxy: 0.5*J*omega^2 + 0.5*L*i^2."""
    return 0.5 * params["J"] * x[..., 0] ** 2 + 0.5 * params["L"] * x[..., 1] ** 2


def choose_initial_states(args, rng):
    omega = rng.uniform(args.omega_min, args.omega_max, size=(args.number, 1)).astype(np.float32)
    current = rng.uniform(args.current_min, args.current_max, size=(args.number, 1)).astype(np.float32)
    return np.concatenate([omega, current], axis=1).astype(np.float32)


def plot_error(outdir, steps, mse, mean_norm_error, true_energy, pred_energy):
    t = np.arange(steps)

    plt.figure()
    plt.plot(t, mse, label="state MSE")
    plt.plot(t, mean_norm_error, label="mean ||error||")
    plt.xlabel("step")
    plt.ylabel("error")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "rollout_error.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(t, true_energy, label="true mean energy")
    plt.plot(t, pred_energy, label="model mean energy")
    plt.xlabel("step")
    plt.ylabel("0.5*J*omega^2 + 0.5*L*i^2")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "rollout_energy.png", dpi=150)
    plt.close()


def plot_phase(outdir, true_states, pred_states, max_paths):
    count = min(max_paths, true_states.shape[1])

    plt.figure()
    for idx in range(count):
        plt.plot(true_states[:, idx, 0], true_states[:, idx, 1], linewidth=1.0)
    plt.xlabel("omega [rad/s]")
    plt.ylabel("current [A]")
    plt.title("True DC motor trajectories")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_phase_true.png", dpi=150)
    plt.close()

    plt.figure()
    for idx in range(count):
        plt.plot(pred_states[:, idx, 0], pred_states[:, idx, 1], linewidth=1.0)
    plt.xlabel("omega [rad/s]")
    plt.ylabel("current [A]")
    plt.title("Neural model trajectories")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_phase_model.png", dpi=150)
    plt.close()

    plt.figure()
    for idx in range(count):
        plt.plot(true_states[:, idx, 0], true_states[:, idx, 1], linewidth=1.0, linestyle="-")
        plt.plot(pred_states[:, idx, 0], pred_states[:, idx, 1], linewidth=1.0, linestyle="--")
    plt.xlabel("omega [rad/s]")
    plt.ylabel("current [A]")
    plt.title("True solid vs model dashed")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_phase.png", dpi=150)
    plt.close()


def plot_state_traces(outdir, true_states, pred_states, max_paths):
    count = min(max_paths, true_states.shape[1])
    t = np.arange(true_states.shape[0])

    plt.figure()
    for idx in range(count):
        plt.plot(t, true_states[:, idx, 0], linewidth=1.0, linestyle="-")
        plt.plot(t, pred_states[:, idx, 0], linewidth=1.0, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("omega [rad/s]")
    plt.title("omega: true solid vs model dashed")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_omega.png", dpi=150)
    plt.close()

    plt.figure()
    for idx in range(count):
        plt.plot(t, true_states[:, idx, 1], linewidth=1.0, linestyle="-")
        plt.plot(t, pred_states[:, idx, 1], linewidth=1.0, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("current [A]")
    plt.title("current: true solid vs model dashed")
    plt.tight_layout()
    plt.savefig(outdir / "rollout_current.png", dpi=150)
    plt.close()


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = args.model.model
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        model.cuda()

    if not hasattr(args.data, "_dc_motor_rhs"):
        raise RuntimeError("Dataset does not expose _dc_motor_rhs. Use dc_motor[...] as the data argument.")

    rhs = args.data._dc_motor_rhs
    params = args.data._dc_motor_params
    logger.info(f"Loaded DC motor params: {params}")

    rng = np.random.default_rng(args.seed)
    x0 = choose_initial_states(args, rng)

    true_states = np.zeros((args.steps, args.number, 2), dtype=np.float32)
    pred_states = np.zeros_like(true_states)
    true_states[0] = x0
    pred_states[0] = x0

    x_true = x0.copy()
    x_pred = to_variable(torch.tensor(x0), cuda=use_cuda)

    for step in range(1, args.steps):
        x_true = rk4_numpy(rhs, x_true, args.timestep).astype(np.float32)
        x_pred = rk4_torch(model, x_pred, args.timestep)

        true_states[step] = x_true
        pred_states[step] = x_pred.cpu().numpy().astype(np.float32)

        if args.print_every and step % args.print_every == 0:
            err = true_states[step] - pred_states[step]
            logger.info(f"step={step} mse={np.mean(err ** 2):.6g}")

    errors = true_states - pred_states
    mse = np.mean(errors ** 2, axis=(1, 2))
    mean_norm_error = np.mean(np.linalg.norm(errors, axis=2), axis=1)
    true_mean_energy = np.mean(motor_energy(true_states, params), axis=1)
    pred_mean_energy = np.mean(motor_energy(pred_states, params), axis=1)

    csv_path = outdir / "rollout_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "state_mse", "mean_norm_error", "true_mean_energy", "model_mean_energy"])
        for step in range(args.steps):
            writer.writerow([step, mse[step], mean_norm_error[step], true_mean_energy[step], pred_mean_energy[step]])

    np.savez(outdir / "rollout_states.npz", true_states=true_states, pred_states=pred_states)

    plot_error(outdir, args.steps, mse, mean_norm_error, true_mean_energy, pred_mean_energy)
    plot_phase(outdir, true_states, pred_states, args.plot_paths)
    plot_state_traces(outdir, true_states, pred_states, args.plot_paths)

    print("DC motor rollout evaluation")
    print("--------------------------")
    print(f"weight:             {args.weight}")
    print(f"steps:              {args.steps}")
    print(f"timestep:           {args.timestep}")
    print(f"number trajectories:{args.number}")
    print(f"final_state_mse:    {mse[-1]}")
    print(f"mean_state_mse:     {np.mean(mse)}")
    print(f"final_norm_error:   {mean_norm_error[-1]}")
    print(f"true_final_energy:  {true_mean_energy[-1]}")
    print(f"model_final_energy: {pred_mean_energy[-1]}")
    print(f"outputs:            {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DC motor neural dynamics rollout.")
    parser.add_argument("data", type=DynamicLoad("datasets"), help="dataset, expected: dc_motor[...] or compatible")
    parser.add_argument("model", type=DynamicLoad("models"), help="model definition string")
    parser.add_argument("weight", type=latest_file, help="checkpoint glob or exact checkpoint path")
    parser.add_argument("--steps", type=int, default=300, help="number of rollout steps")
    parser.add_argument("--number", type=int, default=128, help="number of initial states")
    parser.add_argument("--timestep", type=float, default=0.01, help="RK4 step size")
    parser.add_argument("--seed", type=int, default=2024, help="seed for rollout initial states")
    parser.add_argument("--omega-min", type=float, default=-3.0, help="rollout initial omega min")
    parser.add_argument("--omega-max", type=float, default=3.0, help="rollout initial omega max")
    parser.add_argument("--current-min", type=float, default=-3.0, help="rollout initial current min")
    parser.add_argument("--current-max", type=float, default=3.0, help="rollout initial current max")
    parser.add_argument("--outdir", type=str, default="experiments/dc-motor-rollout", help="output directory")
    parser.add_argument("--plot-paths", type=int, default=16, help="number of trajectories to draw")
    parser.add_argument("--print-every", type=int, default=50, help="progress logging period; 0 disables")
    parser.add_argument("--cpu", action="store_true", help="force CPU evaluation")
    main(parser.parse_args())

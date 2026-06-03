#!/usr/bin/env python3
"""Autonomous DC motor dynamics dataset.

The state is x = [omega, current], where:
  omega  - shaft angular velocity [rad/s]
  current - armature current [A]

The default system is intentionally autonomous / no-input:
  Va = 0, Tl = 0

Equations:
  d omega / dt = (Kt * i - b * omega - Tl) / J
  d i     / dt = (Va - R * i - Ke * omega) / L

With positive R and b, and Va=Tl=0, the origin is a dissipative equilibrium.
This makes the system a good non-pendulum benchmark for the stable dynamics model.
"""

from pathlib import Path

import numpy as np
import torch


CACHE = Path("dc-motor-cache/")


DEFAULTS = {
    "J": 0.01,          # rotor inertia [kg m^2]
    "b": 0.1,           # viscous friction [N m s/rad]
    "Kt": 0.01,         # torque constant [N m/A]
    "Ke": 0.01,         # back-emf constant [V s/rad]
    "R": 1.0,           # armature resistance [Ohm]
    "L": 0.5,           # armature inductance [H]
    "Va": 0.0,          # fixed armature voltage [V], zero by default = no input
    "Tl": 0.0,          # fixed load torque [N m], zero by default = no input disturbance
    "omega_min": -5.0,
    "omega_max": 5.0,
    "current_min": -5.0,
    "current_max": 5.0,
}


def _get_float(props, key):
    return float(props[key]) if key in props else float(DEFAULTS[key])


def dc_motor_gradient(params):
    """Return a vectorized RHS function f(X) for the DC motor ODE."""
    J = float(params["J"])
    b = float(params["b"])
    Kt = float(params["Kt"])
    Ke = float(params["Ke"])
    R = float(params["R"])
    L = float(params["L"])
    Va = float(params["Va"])
    Tl = float(params["Tl"])

    if J <= 0 or L <= 0:
        raise ValueError("J and L must be positive.")
    if R < 0 or b < 0:
        raise ValueError("R and b should be non-negative for a dissipative motor model.")

    def gradient(x):
        x = np.asarray(x, dtype=np.float32)
        squeeze = False
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
            squeeze = True

        omega = x[:, 0]
        current = x[:, 1]

        y = np.zeros_like(x, dtype=np.float32)
        y[:, 0] = (Kt * current - b * omega - Tl) / J
        y[:, 1] = (Va - R * current - Ke * omega) / L

        if squeeze:
            return y[0]
        return y

    return gradient


def _cache_name(n, test, params):
    mode = "test" if test else "train"
    pieces = [
        f"n{n}", mode,
        f"J{params['J']}", f"b{params['b']}", f"Kt{params['Kt']}",
        f"Ke{params['Ke']}", f"R{params['R']}", f"L{params['L']}",
        f"Va{params['Va']}", f"Tl{params['Tl']}",
        f"om{params['omega_min']}_{params['omega_max']}",
        f"i{params['current_min']}_{params['current_max']}",
    ]
    safe = "_".join(str(p).replace("-", "m").replace(".", "p") for p in pieces)
    return CACHE / f"dc-{safe}.npz"


def build(props):
    """Build a TensorDataset of states X and derivatives Y=f(X).

    Supported options:
      n=<int>                  number of random samples, default 50000
      test                     use deterministic test seed instead of train seed
      nocache                  disable .npz caching
      seed=<int>               override deterministic seed
      J,b,Kt,Ke,R,L,Va,Tl      motor parameters
      omega_min,omega_max      omega sampling range
      current_min,current_max  current sampling range
    """
    n = int(props["n"]) if "n" in props else 50000
    test = "test" in props
    cache = "nocache" not in props

    params = {key: _get_float(props, key) for key in DEFAULTS}

    seed = int(props["seed"]) if "seed" in props else (12345 if test else 54321)
    rng = np.random.default_rng(seed)

    cache_path = _cache_name(n, test, params)
    if cache and cache_path.exists():
        loaded = np.load(cache_path)
        X = loaded["X"].astype(np.float32)
        Y = loaded["Y"].astype(np.float32)
    else:
        omega = rng.uniform(params["omega_min"], params["omega_max"], size=(n, 1)).astype(np.float32)
        current = rng.uniform(params["current_min"], params["current_max"], size=(n, 1)).astype(np.float32)
        X = np.concatenate([omega, current], axis=1).astype(np.float32)

        rhs = dc_motor_gradient(params)
        Y = rhs(X).astype(np.float32)

        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, X=X, Y=Y)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    dataset._dc_motor_rhs = dc_motor_gradient(params)
    dataset._dc_motor_params = params
    dataset._state_dim = 2
    dataset._equilibrium = np.array([0.0, 0.0], dtype=np.float32)
    return dataset

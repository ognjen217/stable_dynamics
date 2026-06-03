# DC Motor Stable Dynamics Benchmark

This repository now includes a non-pendulum autonomous dynamics benchmark based on a simple DC motor model.

## System definition

State:

```text
x = [omega, current]
```

where:

- `omega` is shaft angular velocity `[rad/s]`
- `current` is armature current `[A]`

Default no-input equations:

```text
d omega / dt = (Kt * current - b * omega - Tl) / J
d current / dt = (Va - R * current - Ke * omega) / L
```

By default:

```text
Va = 0
Tl = 0
```

so the benchmark is autonomous and dissipative. With positive friction/resistance, the origin is a stable equilibrium.

## Files added

```text
datasets/dc_motor.py        # synthetic DC motor dataset: X -> dX/dt
train_dc_motor_simple       # simple MLP baseline training wrapper
train_dc_motor_stable       # stable dynamics training wrapper
dc_motor_rollout.py         # RK4 rollout evaluation + plots
```

## Train simple baseline

```sh
bash train_dc_motor_simple
```

This trains:

```text
simple[a=2,b=64]
```

Output:

```text
experiments/dc-motor-simple/
runs/experiments/dc-motor-simple/
```

## Train stable dynamics model

```sh
bash train_dc_motor_stable
```

This trains:

```text
stabledynamics[
  latent_space_dim=2,
  a=0.001,
  projfn=NN-REHU,
  projfn_eps=0.01,
  smooth_v=0,
  hp=60,
  h=100,
  rehu=0.001
]
```

Output:

```text
experiments/dc-motor-stable/
runs/experiments/dc-motor-stable/
```

## TensorBoard

```sh
tensorboard --logdir runs/
```

## Run rollout evaluation manually

Simple model:

```sh
python dc_motor_rollout.py \
  "dc_motor[n=10000,test]" \
  "simple[a=2,b=64]" \
  "experiments/dc-motor-simple/checkpoint_*.pth" \
  --outdir experiments/dc-motor-simple/eval
```

Stable model:

```sh
python dc_motor_rollout.py \
  "dc_motor[n=10000,test]" \
  "stabledynamics[latent_space_dim=2,a=0.001,projfn=NN-REHU,projfn_eps=0.01,smooth_v=0,hp=60,h=100,rehu=0.001]" \
  "experiments/dc-motor-stable/checkpoint_*.pth" \
  --outdir experiments/dc-motor-stable/eval
```

## Evaluation outputs

The rollout evaluator writes:

```text
rollout_metrics.csv       # per-step MSE, norm error, true/model energy
rollout_states.npz        # true and predicted trajectories
rollout_error.png         # rollout error over time
rollout_energy.png        # true vs model energy over time
rollout_phase_true.png    # true phase portrait
rollout_phase_model.png   # model phase portrait
rollout_phase.png         # true solid vs model dashed
rollout_omega.png         # omega traces
rollout_current.png       # current traces
```

## Dataset options

Example:

```text
dc_motor[n=50000,J=0.01,b=0.1,Kt=0.01,Ke=0.01,R=1.0,L=0.5,Va=0,Tl=0]
```

Supported options:

```text
n                         number of samples
test                      deterministic test seed
nocache                   disable npz cache
seed                      override RNG seed
J,b,Kt,Ke,R,L,Va,Tl       motor parameters
omega_min,omega_max       omega sampling range
current_min,current_max   current sampling range
```

Cache files are written to:

```text
dc-motor-cache/
```

## Notes

- This first version intentionally keeps `Va=0` and `Tl=0`, so the motor has no control input.
- A controlled version `x_dot = f(x, u)` would require changing the model interface so `fhat` can consume `[x, u]` while the Lyapunov function remains defined over `x`.
- The stable model is most meaningful here because the DC motor without input has a dissipative equilibrium at zero.

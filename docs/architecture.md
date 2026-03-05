## Mouse Accel ML/RL Suite – Architecture

### 1. Core idea

We model mouse acceleration as a parametric curve mapping mouse speed (counts/s) → sensitivity and use:
- RL to **auto-tune** curve parameters in a simulator.
- Supervised models to **infer** or **recommend** curves.
- A sequence model to **detect anomalies / assist** in aim traces.

All components share the same acceleration parameterization.

---

### 2. Code structure

- `env/aim_sim/`
  - `env_core.py`
    - `AccelParams` (imported from `models/curve_param_config.py`) controls the accel curve.
    - `SimpleAimTask`: 1D yaw-only target aiming task.
    - `AimEnv` (Gym-like): RL environment exposing `reset()` / `step()` for training an RL agent to improve `AccelParams`.

- `models/`
  - `curve_param_config.py`
    - `AccelParams`: dataclass for curve params (`k1, a, k2, b, v0, sens_min, sens_max`).
    - `sensitivity_from_speed`: maps mouse speed → sensitivity using a 2-branch power curve.
    - `apply_accel`: converts raw mouse deltas to view deltas under a given curve.
  - `inverse_curve_net.py`
    - `InverseCurveNet`: sequence-to-vector regressor (LSTM-based) predicting curve params from time series of mouse + view data.
  - `ideal_curve_net.py`
    - `IdealCurveNet`: MLP mapping player/session features to curve parameters (ideal curve recommendation).
  - `aim_anomaly_net.py`
    - `AimAnomalyNet`: sequence classifier (LSTM-based) for normal vs assisted aim traces.

- `experiments/`
  - `rl_auto_tune/train_rl_auto_tune.py`
    - Trains PPO on `AimEnv` to learn better accel parameters.
  - `inverse_model/gen_synthetic_data.py`
    - Uses `SimpleAimTask` + random `AccelParams` to generate synthetic sequences and ground-truth curve parameters.
  - `inverse_model/train_inverse_model.py`
    - Trains `InverseCurveNet` on the synthetic inverse dataset.
  - `ideal_curve/train_ideal_curve.py`
    - Trains `IdealCurveNet` on synthetic player/session profiles for curve recommendation.
  - `aim_anomaly/train_aim_anomaly.py`
    - Generates normal/assisted aim sequences with the simulator and trains `AimAnomalyNet`.
  - `run_experiment.py`
    - Small CLI to dispatch to the above scripts via a name or YAML config.

- `raw_input/`
  - `collector_design.md`: high-level design for logging real mouse + view data from a PC FPS / aim trainer.
  - `logger.py`: simple prototype for logging raw mouse deltas + timestamps (no game integration).

- `data/`
  - (Git-ignored) houses generated `.npz` datasets and logs.

- `notebooks/`
  - Jupyter notebooks for visualization (curves, RL performance, inverse model error, anomaly ROC, etc.).

---

### 3. Data flow overview

```mermaid
flowchart LR
  rawMouse["Raw mouse deltas"]
  simEnv["Aim simulator (SimpleAimTask)"]
  accelParams["AccelParams"]
  rlAgent["RL agent (PPO)"]
  inverseNet["InverseCurveNet"]
  idealNet["IdealCurveNet"]
  anomalyNet["AimAnomalyNet"]

  accelParams --> simEnv
  rawMouse --> simEnv

  subgraph P1 [P1: RL auto-tuning]
    simEnv --> rlAgent
    rlAgent --> accelParams
  end

  subgraph P2 [P2: Inverse modeling]
    simEnv -->|"mouse+view sequences"| inverseNet
    inverseNet -->|"predicted curve params"| accelParams
  end

  subgraph P3 [P3: Ideal curve]
    playerStats["Player/session stats"] --> idealNet
    idealNet -->|"recommended curve params"| accelParams
  end

  subgraph P4 [P4: Anomaly detection]
    simEnv -->|"mouse+view traces"| anomalyNet
  end
"""
CLI dispatcher for running experiments by name or config file.

Usage:
    python experiments/run_experiments.py --name rl_auto_tune
    python experiments/run_experiments.py --config experiments/configs/inverse_model.yaml
"""
import argparse
import importlib

import yaml


EXPERIMENT_ENTRYPOINTS = {
    "rl_auto_tune": "experiments.rl_auto_tune.train_rl_auto_tune:main",
    "inverse_model": "experiments.inverse_model.train_inverse_model:main",
    "gen_synthetic_data": "experiments.inverse_model.gen_synthetic_data:main",
    "ideal_curve": "experiments.ideal_curve.train_ideal_curve:main",
    "aim_anomaly": "experiments.aim_anomaly.train_aim_anomaly:main",
}


def run_from_name(name: str, config_path: str = None):
    target = EXPERIMENT_ENTRYPOINTS[name]
    module_name, func_name = target.split(":")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    # Pass config_path if the function accepts it
    import inspect
    sig = inspect.signature(func)
    if "config_path" in sig.parameters:
        func(config_path=config_path)
    else:
        func()


def main():
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--config", type=str, required=False, help="YAML config with experiment.name")
    parser.add_argument("--name", type=str, required=False, help="Experiment name if not using config")
    args = parser.parse_args()

    config_path = args.config
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = cfg["experiment"]["name"]
    else:
        if not args.name:
            print("Available experiments:")
            for k in EXPERIMENT_ENTRYPOINTS:
                print(f"  - {k}")
            raise SystemExit("\nProvide --name or --config")
        name = args.name

    if name not in EXPERIMENT_ENTRYPOINTS:
        raise SystemExit(f"Unknown experiment name '{name}'. Available: {list(EXPERIMENT_ENTRYPOINTS.keys())}")

    print(f"Running experiment: {name}")
    run_from_name(name, config_path=config_path)


if __name__ == "__main__":
    main()
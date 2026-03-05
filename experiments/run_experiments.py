import argparse
import importlib
import yaml


EXPERIMENT_ENTRYPOINTS = {
    "rl_auto_tune": "experiments.rl_auto_tune.train_rl_auto_tune:main",
    "inverse_model": "experiments.inverse_model.train_inverse_model:main",
    "ideal_curve": "experiments.ideal_curve.train_ideal_curve:main",
    "aim_anomaly": "experiments.aim_anomaly.train_aim_anomaly:main",
}


def run_from_name(name: str):
    target = EXPERIMENT_ENTRYPOINTS[name]
    module_name, func_name = target.split(":")
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    func()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="YAML config with experiment.name")
    parser.add_argument("--name", type=str, required=False, help="Experiment name if not using config")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = cfg["experiment"]["name"]
    else:
        if not args.name:
            raise SystemExit("Provide --name or --config")
        name = args.name

    if name not in EXPERIMENT_ENTRYPOINTS:
        raise SystemExit(f"Unknown experiment name {name}")
    run_from_name(name)


if __name__ == "__main__":
    main()
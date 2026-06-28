from __future__ import annotations

import argparse
from pathlib import Path

from .config import deep_get, load_config
from .evaluate import run_evaluate
from .failure_analysis import run_failure_analysis
from .output_summary import write_output_summary
from .tampering_type_analysis import run_tampering_type_analysis
from .train import run_train
from .validate_experiment import validate_experiment


def run_all(config_path: str | Path) -> None:
    config = load_config(config_path)
    checkpoint = run_train(config_path)
    run_evaluate(config_path, str(checkpoint))
    run_failure_analysis(config_path, str(checkpoint))
    if bool(deep_get(config, "tampering_type_analysis.enabled", True)):
        run_tampering_type_analysis(config_path, str(checkpoint))
    write_output_summary(config_path)
    ok, errors = validate_experiment(config_path)
    if not ok:
        raise RuntimeError("Experiment validation failed:\n" + "\n".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_all(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

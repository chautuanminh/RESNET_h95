from __future__ import annotations

import argparse
from pathlib import Path

from .config import deep_get, load_config, resolve_output_paths
from .evaluate import run_evaluate
from .failure_analysis import run_failure_analysis
from .output_summary import write_output_summary
from .tampering_type_analysis import run_tampering_type_analysis
from .validate_experiment import validate_experiment


def run_post_train_all(config_path: str | Path, checkpoint: str | None = None) -> None:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    resolved_checkpoint = checkpoint or str(paths["run"] / "checkpoints" / "best_model.pth")
    if not Path(resolved_checkpoint).exists():
        raise FileNotFoundError(f"Best checkpoint not found: {resolved_checkpoint}")

    run_evaluate(config_path, resolved_checkpoint)
    run_failure_analysis(config_path, resolved_checkpoint)
    if bool(deep_get(config, "tampering_type_analysis.enabled", True)):
        run_tampering_type_analysis(config_path, resolved_checkpoint)
    write_output_summary(config_path)
    ok, errors = validate_experiment(config_path)
    if not ok:
        raise RuntimeError("Experiment validation failed:\n" + "\n".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    run_post_train_all(args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

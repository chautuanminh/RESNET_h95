from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, resolve_output_paths
from .report import write_res_summary
from .utils import read_csv_rows


def write_output_summary(config_path: str | Path) -> Path:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    summary_path = paths["run"] / "official_threshold_0.5_metrics.csv"
    rows = read_csv_rows(summary_path) if summary_path.exists() else []
    write_res_summary(paths["root"], rows)
    return paths["root"] / "RES_SUMMARY.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    write_output_summary(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

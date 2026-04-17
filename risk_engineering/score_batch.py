"""Command-line batch scoring entrypoint for the engineered risk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .modeling import load_model_package, score_with_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a batch with a trained risk model package.")
    parser.add_argument("--input", required=True, help="Scoring CSV path.")
    parser.add_argument("--model-package", default="risk_engineering_outputs/model_package.pkl")
    parser.add_argument("--output-dir", default="risk_engineering_scoring_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    package = load_model_package(args.model_package)
    scoring_df = pd.read_csv(args.input)
    approval_output, monitoring_report = score_with_package(scoring_df, package)

    approval_output.to_csv(output_dir / "approval_decision_output.csv", index=False)
    if monitoring_report is not None:
        monitoring_report.to_csv(output_dir / "monitoring_report.csv", index=False)

    print(f"Scored rows: {len(approval_output)}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()


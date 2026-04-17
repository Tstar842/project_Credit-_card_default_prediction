"""Command-line training entrypoint for the engineered risk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .modeling import fit_credit_risk_pipeline, save_model_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit default risk pipeline.")
    parser.add_argument("--input", default="sample.csv", help="Training CSV path.")
    parser.add_argument("--output-dir", default="risk_engineering_outputs", help="Output directory.")
    parser.add_argument("--model-package", default="model_package.pkl", help="Model package file name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    package, outputs = fit_credit_risk_pipeline(df)

    for name, table in outputs.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    save_model_package(package, output_dir / args.model_package)
    print(f"Best model: {package.model_name}")
    print(f"Business threshold: {package.threshold:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()


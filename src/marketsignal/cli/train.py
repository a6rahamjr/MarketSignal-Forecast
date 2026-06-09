from __future__ import annotations

import argparse
from dataclasses import replace

from marketsignal.config import load_config
from marketsignal.data.loaders import load_market_data
from marketsignal.training.pipeline import train_model
from marketsignal.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate MarketSignal Forecast.")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--source", choices=["synthetic", "csv", "yfinance"])
    parser.add_argument("--symbol")
    parser.add_argument("--csv-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--no-tune", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging()
    config = load_config(args.config)

    data_config = replace(
        config.data,
        source=args.source or config.data.source,
        symbol=args.symbol or config.data.symbol,
        csv_path=args.csv_path or config.data.csv_path,
    )
    training_config = replace(
        config.training,
        tune=False if args.no_tune else config.training.tune,
    )
    project_config = replace(
        config.project,
        artifact_dir=args.artifact_dir or config.project.artifact_dir,
    )
    config = replace(
        config,
        data=data_config,
        training=training_config,
        project=project_config,
    )

    market_data = load_market_data(config)
    result = train_model(market_data, config)
    model_metrics = result.metrics["model"]
    print(f"Run: {result.run_id}")
    print(f"Model: {result.model_path}")
    print(
        "Holdout metrics: "
        f"accuracy={model_metrics['accuracy']:.3f}, "
        f"f1={model_metrics['f1']:.3f}, "
        f"roc_auc={model_metrics['roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()

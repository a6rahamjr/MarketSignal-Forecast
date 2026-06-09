from __future__ import annotations

import argparse
import json

from marketsignal.data.loaders import load_csv
from marketsignal.inference.predictor import Predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict the next-session direction.")
    parser.add_argument("--model", default="artifacts/latest/model")
    parser.add_argument("--input", required=True, help="CSV with date/open/high/low/close/volume.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = Predictor.from_artifact(args.model)
    prediction = predictor.predict_latest(load_csv(args.input))
    print(json.dumps(prediction.__dict__, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from marketsignal.data.synthetic import generate_synthetic_ohlcv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic OHLCV data.")
    parser.add_argument("--rows", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2018-01-01")
    parser.add_argument("--output", default="data/generated/synthetic_ohlcv.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_ohlcv(
        rows=args.rows,
        start_date=args.start_date,
        seed=args.seed,
    )
    frame.reset_index().to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows to {destination}")


if __name__ == "__main__":
    main()

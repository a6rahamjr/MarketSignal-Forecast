from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from marketsignal.config import AppConfig
from marketsignal.data.schema import normalize_ohlcv
from marketsignal.data.synthetic import generate_synthetic_ohlcv

MAX_CSV_BYTES = 100 * 1024 * 1024
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.^=-]{1,20}$")
ALLOWED_PERIODS = {"1y", "2y", "5y", "10y", "max"}
ALLOWED_INTERVALS = {"1d", "5d", "1wk"}


def load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")
    if not csv_path.is_file() or csv_path.is_symlink():
        raise ValueError("CSV input must be a regular file.")
    if csv_path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV input exceeds the 100 MB limit.")
    return normalize_ohlcv(pd.read_csv(csv_path))


def load_yfinance(symbol: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    symbol = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError("Invalid market symbol.")
    if period not in ALLOWED_PERIODS:
        raise ValueError(f"Unsupported period: {period}")
    if interval not in ALLOWED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Install the market extra to use Yahoo Finance: pip install -e \".[market]\""
        ) from exc

    frame = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for symbol {symbol!r}.")
    return normalize_ohlcv(frame)


def load_market_data(config: AppConfig) -> pd.DataFrame:
    source = config.data.source
    if source == "synthetic":
        return generate_synthetic_ohlcv(
            rows=config.data.rows,
            start_date=config.data.start_date,
            seed=config.project.random_seed,
        )
    if source == "csv":
        if not config.data.csv_path:
            raise ValueError("data.csv_path is required when data.source='csv'.")
        return load_csv(config.data.csv_path)
    if source == "yfinance":
        return load_yfinance(
            symbol=config.data.symbol,
            period=config.data.period,
            interval=config.data.interval,
        )
    raise ValueError(f"Unsupported data source: {source}")

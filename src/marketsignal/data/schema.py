from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
MAX_PRICE = 1e12
MAX_VOLUME = 1e15


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Market data is empty.")

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], utc=True, errors="coerce")
        if normalized["date"].isna().any():
            raise ValueError("Market data contains invalid timestamps.")
        normalized = normalized.set_index("date")
    else:
        normalized.index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
        if normalized.index.isna().any():
            raise ValueError("Market data contains invalid timestamps.")

    missing = sorted(set(REQUIRED_COLUMNS) - set(normalized.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    normalized = normalized.loc[:, list(REQUIRED_COLUMNS)]
    normalized = normalized.apply(pd.to_numeric, errors="coerce")
    if normalized.isna().any().any():
        raise ValueError("Market data contains missing or non-numeric OHLCV values.")
    if normalized.index.duplicated().any():
        raise ValueError("OHLCV timestamps must be unique.")
    normalized = normalized.sort_index()
    validate_ohlcv(normalized)
    return normalized


def validate_ohlcv(frame: pd.DataFrame, minimum_rows: int = 2) -> None:
    if len(frame) < minimum_rows:
        raise ValueError(f"At least {minimum_rows} OHLCV rows are required.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be sorted in ascending order.")
    if not frame.index.is_unique:
        raise ValueError("OHLCV timestamps must be unique.")
    if not np.isfinite(frame.loc[:, list(REQUIRED_COLUMNS)].to_numpy()).all():
        raise ValueError("OHLCV values must be finite.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (frame[["open", "high", "low", "close"]] > MAX_PRICE).any().any():
        raise ValueError("OHLC prices exceed the supported range.")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    if (frame["volume"] > MAX_VOLUME).any():
        raise ValueError("Volume exceeds the supported range.")

    row_high = frame[["open", "close", "low"]].max(axis=1)
    row_low = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < row_high).any() or (frame["low"] > row_low).any():
        raise ValueError("OHLC price relationships are inconsistent.")


def records_to_frame(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    return normalize_ohlcv(pd.DataFrame.from_records(records))

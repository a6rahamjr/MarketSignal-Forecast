import pandas as pd

import pytest

from marketsignal.data.schema import REQUIRED_COLUMNS, normalize_ohlcv
from marketsignal.data.synthetic import generate_synthetic_ohlcv


def test_synthetic_generation_is_reproducible():
    first = generate_synthetic_ohlcv(rows=180, seed=9)
    second = generate_synthetic_ohlcv(rows=180, seed=9)
    pd.testing.assert_frame_equal(first, second)


def test_generated_dataset_obeys_ohlcv_contract():
    frame = generate_synthetic_ohlcv(rows=180, seed=3)
    assert tuple(frame.columns) == REQUIRED_COLUMNS
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)).all()
    assert (frame["low"] <= frame[["open", "close", "high"]].min(axis=1)).all()
    pd.testing.assert_frame_equal(frame, normalize_ohlcv(frame))


def test_duplicate_timestamps_are_rejected():
    frame = generate_synthetic_ohlcv(rows=180, seed=3)
    duplicate = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(ValueError, match="unique"):
        normalize_ohlcv(duplicate)


def test_pathological_prices_are_rejected():
    frame = generate_synthetic_ohlcv(rows=180, seed=3)
    frame.iloc[-1, frame.columns.get_loc("close")] = 1e100
    frame.iloc[-1, frame.columns.get_loc("high")] = 1e100
    with pytest.raises(ValueError, match="supported range"):
        normalize_ohlcv(frame)

from __future__ import annotations

import json

import numpy as np
import pytest

from marketsignal.config import ModelConfig
from marketsignal.features.technical import FEATURE_COLUMNS, build_supervised_dataset
from marketsignal.models.artifact import load_artifact, save_artifact
from marketsignal.models.factory import build_forest


def test_safe_artifact_matches_sklearn(tmp_path, synthetic_market):
    dataset = build_supervised_dataset(synthetic_market)
    features = dataset[FEATURE_COLUMNS]
    target = dataset["target"]
    pipeline = build_forest(
        ModelConfig(n_estimators=60, min_samples_leaf=6, max_features=0.75, max_depth=12),
        seed=11,
    )
    pipeline.fit(features, target)

    path = tmp_path / "model"
    save_artifact(
        pipeline,
        path,
        feature_columns=FEATURE_COLUMNS,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        decision_threshold=0.5,
        abstain_margin=0.03,
        forecast_horizon=1,
        metadata={"run_id": "test"},
    )
    loaded = load_artifact(path)
    sample = features.tail(12)
    expected = pipeline.predict_proba(sample)[:, 1]
    actual = loaded.predict_probability(sample).to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_artifact_checksum_rejects_tampering(tmp_path, trained_result):
    source = trained_result.model_path
    target = tmp_path / "tampered"
    target.mkdir()
    (target / "forest.npz").write_bytes((source / "forest.npz").read_bytes() + b"x")
    (target / "manifest.json").write_text(
        (source / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        load_artifact(target)


def test_manifest_contains_no_pickle_payload(trained_result):
    manifest = json.loads(
        (trained_result.model_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == "marketsignal.extra-trees"
    assert sorted(path.name for path in trained_result.model_path.iterdir()) == [
        "forest.npz",
        "manifest.json",
    ]

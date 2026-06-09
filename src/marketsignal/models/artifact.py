from __future__ import annotations

import hashlib
import hmac
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

FORMAT_NAME = "marketsignal.extra-trees"
FORMAT_VERSION = 1
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_TREES = 2_000
MAX_NODES = 5_000_000
ARRAY_NAMES = {
    "offsets",
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "probability_up",
}


@dataclass(frozen=True)
class ForestArtifact:
    feature_columns: list[str]
    imputer_values: np.ndarray
    offsets: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    probability_up: np.ndarray
    calibration_slope: float
    calibration_intercept: float
    decision_threshold: float
    abstain_margin: float
    forecast_horizon: int
    metadata: dict[str, Any]

    def predict_probability(self, features: pd.DataFrame) -> pd.Series:
        matrix = features.loc[:, self.feature_columns].to_numpy(dtype=np.float64, copy=True)
        missing = ~np.isfinite(matrix)
        if missing.any():
            matrix[missing] = np.take(self.imputer_values, np.where(missing)[1])

        raw = np.zeros(len(matrix), dtype=np.float64)
        for tree_index in range(len(self.offsets) - 1):
            start = int(self.offsets[tree_index])
            stop = int(self.offsets[tree_index + 1])
            nodes = np.zeros(len(matrix), dtype=np.int32)

            while True:
                absolute = start + nodes
                split_features = self.feature[absolute]
                active = split_features >= 0
                if not active.any():
                    break

                rows = np.flatnonzero(active)
                node_features = split_features[rows]
                go_left = matrix[rows, node_features] <= self.threshold[absolute[rows]]
                left = self.children_left[absolute[rows]]
                right = self.children_right[absolute[rows]]
                nodes[rows] = np.where(go_left, left, right)

                if (nodes < 0).any() or (nodes >= stop - start).any():
                    raise ValueError("Model tree contains invalid child indices.")

            raw += self.probability_up[start + nodes]

        raw /= len(self.offsets) - 1
        calibrated = _sigmoid(
            self.calibration_intercept
            + self.calibration_slope * _logit(np.clip(raw, 1e-6, 1 - 1e-6))
        )
        return pd.Series(calibrated, index=features.index, name="probability_up")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1 - values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_forest(pipeline: Pipeline) -> tuple[np.ndarray, ...]:
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, ExtraTreesClassifier):
        raise TypeError("Only ExtraTreesClassifier pipelines can be exported.")
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("Classifier must use binary classes [0, 1].")

    offsets = [0]
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    threshold_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    for estimator in classifier.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        totals = values.sum(axis=1)
        probability = np.divide(
            values[:, 1],
            totals,
            out=np.zeros_like(totals, dtype=np.float64),
            where=totals > 0,
        )
        left_parts.append(tree.children_left.astype(np.int32))
        right_parts.append(tree.children_right.astype(np.int32))
        feature_parts.append(tree.feature.astype(np.int32))
        threshold_parts.append(tree.threshold.astype(np.float32))
        probability_parts.append(probability.astype(np.float32))
        offsets.append(offsets[-1] + tree.node_count)

    return (
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(left_parts),
        np.concatenate(right_parts),
        np.concatenate(feature_parts),
        np.concatenate(threshold_parts),
        np.concatenate(probability_parts),
    )


def save_artifact(
    pipeline: Pipeline,
    destination: str | Path,
    *,
    feature_columns: list[str],
    calibration_slope: float,
    calibration_intercept: float,
    decision_threshold: float,
    abstain_margin: float,
    forecast_horizon: int,
    metadata: dict[str, Any],
) -> Path:
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Model directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    imputer = pipeline.named_steps["imputer"]
    arrays = _extract_forest(pipeline)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        archive_path = temporary / "forest.npz"
        np.savez_compressed(
            archive_path,
            offsets=arrays[0],
            children_left=arrays[1],
            children_right=arrays[2],
            feature=arrays[3],
            threshold=arrays[4],
            probability_up=arrays[5],
        )
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "archive_sha256": _sha256(archive_path),
            "feature_columns": feature_columns,
            "imputer_values": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "calibration": {
                "slope": calibration_slope,
                "intercept": calibration_intercept,
            },
            "decision_threshold": decision_threshold,
            "abstain_margin": abstain_margin,
            "forecast_horizon": forecast_horizon,
            "metadata": metadata,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def _regular_file(path: Path, size_limit: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Artifact entry must be a regular file: {path.name}")
    if path.stat().st_size > size_limit:
        raise ValueError(f"Artifact entry exceeds its size limit: {path.name}")


def _read_manifest(path: Path) -> dict[str, Any]:
    _regular_file(path, MAX_MANIFEST_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Model manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError("Model manifest must contain a JSON object.")
    return value


def _validate_archive(path: Path) -> None:
    _regular_file(path, MAX_ARCHIVE_BYTES)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            expected_names = {f"{name}.npy" for name in ARRAY_NAMES}
            if {entry.filename for entry in entries} != expected_names:
                raise ValueError("Model archive contains unexpected files.")
            if any(entry.flag_bits & 0x1 for entry in entries):
                raise ValueError("Encrypted model archives are not supported.")
            total_size = sum(entry.file_size for entry in entries)
            if total_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Model archive expands beyond the safe size limit.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Model archive is not a valid NPZ file.") from exc


def load_artifact(path: str | Path) -> ForestArtifact:
    model_dir = Path(path)
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError("Model path must be a regular directory.")

    manifest = _read_manifest(model_dir / "manifest.json")
    archive_path = model_dir / "forest.npz"
    _validate_archive(archive_path)

    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")
    if not hmac.compare_digest(
        str(manifest.get("archive_sha256", "")),
        _sha256(archive_path),
    ):
        raise ValueError("Model archive checksum does not match its manifest.")

    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_NAMES:
            raise ValueError("Model archive contains unexpected arrays.")
        arrays = {name: np.asarray(archive[name]) for name in ARRAY_NAMES}

    feature_columns = manifest.get("feature_columns")
    imputer_values = np.asarray(manifest.get("imputer_values"), dtype=np.float64)
    if not isinstance(feature_columns, list) or not all(
        isinstance(name, str) for name in feature_columns
    ):
        raise ValueError("Model feature contract is invalid.")
    if imputer_values.shape != (len(feature_columns),) or not np.isfinite(imputer_values).all():
        raise ValueError("Model imputer values are invalid.")

    offsets = arrays["offsets"].astype(np.int64, copy=False)
    node_count = len(arrays["feature"])
    if not 1 < len(offsets) <= MAX_TREES + 1:
        raise ValueError("Model tree count is invalid.")
    if node_count > MAX_NODES or offsets[0] != 0 or offsets[-1] != node_count:
        raise ValueError("Model node offsets are invalid.")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("Model tree offsets must be increasing.")
    if any(len(arrays[name]) != node_count for name in ARRAY_NAMES - {"offsets"}):
        raise ValueError("Model tree arrays have inconsistent lengths.")
    if not np.isfinite(arrays["threshold"]).all() or not np.isfinite(
        arrays["probability_up"]
    ).all():
        raise ValueError("Model tree arrays contain non-finite values.")
    if np.any((arrays["probability_up"] < 0) | (arrays["probability_up"] > 1)):
        raise ValueError("Model leaf probabilities are outside [0, 1].")

    feature_count = len(feature_columns)
    for tree_index in range(len(offsets) - 1):
        start = int(offsets[tree_index])
        stop = int(offsets[tree_index + 1])
        size = stop - start
        tree_features = arrays["feature"][start:stop]
        left = arrays["children_left"][start:stop]
        right = arrays["children_right"][start:stop]
        leaf = tree_features < 0
        if np.any((tree_features[~leaf] < 0) | (tree_features[~leaf] >= feature_count)):
            raise ValueError("Model tree references an unknown feature.")
        if np.any(left[leaf] != -1) or np.any(right[leaf] != -1):
            raise ValueError("Model leaf nodes contain child references.")
        if np.any((left[~leaf] < 0) | (left[~leaf] >= size)):
            raise ValueError("Model tree contains an invalid left child.")
        if np.any((right[~leaf] < 0) | (right[~leaf] >= size)):
            raise ValueError("Model tree contains an invalid right child.")
        node_index = np.arange(size)
        if np.any(left[~leaf] <= node_index[~leaf]) or np.any(
            right[~leaf] <= node_index[~leaf]
        ):
            raise ValueError("Model tree contains a cycle or backward reference.")

    calibration = manifest.get("calibration", {})
    slope = float(calibration.get("slope"))
    intercept = float(calibration.get("intercept"))
    threshold = float(manifest.get("decision_threshold"))
    margin = float(manifest.get("abstain_margin"))
    if not all(math.isfinite(value) for value in (slope, intercept, threshold, margin)):
        raise ValueError("Model decision parameters are invalid.")
    if not 0 < threshold < 1 or not 0 <= margin < 0.25:
        raise ValueError("Model decision parameters are outside safe bounds.")

    forecast_horizon = int(manifest.get("forecast_horizon"))
    metadata = manifest.get("metadata")
    if forecast_horizon < 1:
        raise ValueError("Model forecast horizon is invalid.")
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")

    return ForestArtifact(
        feature_columns=feature_columns,
        imputer_values=imputer_values,
        offsets=offsets,
        children_left=arrays["children_left"].astype(np.int32, copy=False),
        children_right=arrays["children_right"].astype(np.int32, copy=False),
        feature=arrays["feature"].astype(np.int32, copy=False),
        threshold=arrays["threshold"].astype(np.float32, copy=False),
        probability_up=arrays["probability_up"].astype(np.float32, copy=False),
        calibration_slope=slope,
        calibration_intercept=intercept,
        decision_threshold=threshold,
        abstain_margin=margin,
        forecast_horizon=forecast_horizon,
        metadata=metadata,
    )

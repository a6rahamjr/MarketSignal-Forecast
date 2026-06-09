from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from marketsignal.config import ModelConfig


def build_baseline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_forest(config: ModelConfig, seed: int) -> Pipeline:
    depth = config.max_depth or None
    classifier = ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        max_depth=depth,
        class_weight="balanced",
        criterion="log_loss",
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def search_space() -> dict[str, list[object]]:
    return {
        "classifier__min_samples_leaf": [4, 6, 8, 12, 16, 24],
        "classifier__max_features": [0.4, 0.6, 0.75, 0.9, 1.0],
        "classifier__max_depth": [8, 12, 16, 24, None],
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

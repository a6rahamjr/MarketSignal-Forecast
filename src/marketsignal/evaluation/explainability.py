from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_feature_importance(
    estimator: object,
    features: pd.DataFrame,
    target: pd.Series,
    random_seed: int,
) -> pd.DataFrame:
    scoring = "roc_auc" if target.nunique() == 2 else "accuracy"
    result = permutation_importance(
        estimator,
        features,
        target,
        n_repeats=7,
        random_state=random_seed,
        scoring=scoring,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

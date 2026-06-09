import json

from marketsignal.models.artifact import load_artifact


def test_training_pipeline_writes_complete_artifacts(trained_result):
    assert trained_result.model_path.exists()
    assert (trained_result.run_dir / "metrics.json").exists()
    assert (trained_result.run_dir / "feature_importance.csv").exists()
    assert (trained_result.run_dir / "predictions.csv").exists()

    metrics = json.loads((trained_result.run_dir / "metrics.json").read_text())
    assert 0 <= metrics["model"]["accuracy"] <= 1
    assert 0 <= metrics["model"]["roc_auc"] <= 1
    assert metrics["data"]["train_rows"] > metrics["data"]["test_rows"]

    model = load_artifact(trained_result.model_path)
    assert model.metadata["run_id"] == trained_result.run_id
    assert model.feature_columns
    assert 0 < trained_result.metrics["decision_threshold"] < 1

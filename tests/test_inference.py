from marketsignal.inference.predictor import Predictor


def test_inference_is_deterministic_and_well_formed(trained_result, synthetic_market):
    predictor = Predictor.from_artifact(trained_result.model_path)
    first = predictor.predict_latest(synthetic_market)
    second = predictor.predict_latest(synthetic_market)

    assert first == second
    assert 0 <= first.probability_up <= 1
    assert first.direction in {"up", "down", "uncertain"}
    assert 0 <= first.confidence <= 1
    assert first.forecast_horizon == 1

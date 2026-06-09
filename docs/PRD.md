# Product Requirements

## Product

**Name:** MarketSignal Forecast
**Purpose:** Estimate the probability of next-session price direction from daily OHLCV history.

## Problem

The original project generated and executed Python through a local LLM. That workflow could produce
useful charts, but it had no stable prediction target, repeatable feature set, temporal evaluation,
or safe model-serving boundary.

MarketSignal provides a measurable forecasting workflow. A run should be reproducible, its
holdout results should be inspectable, and the exact trained behavior should be usable from both
the command line and HTTP without executing generated code.

## Users

- ML engineers building time-series baselines
- Analysts evaluating OHLCV signals
- Backend engineers integrating probabilistic forecasts
- Students learning leakage-aware model evaluation

## Inputs

Daily `date`, `open`, `high`, `low`, `close`, and `volume` observations. Timestamps must be unique;
prices must be positive and internally consistent; volume must be finite and non-negative.

Supported sources:

- Seeded synthetic data for offline testing
- Versioned CSV files
- Yahoo Finance for research runs

## Outputs

Inference returns:

- Probability of an upward next-session close
- `up`, `down`, or `uncertain`
- Confidence, learned threshold, timestamp, and horizon
- Model version and run ID through the API

Training returns:

- Safe model artifact
- Holdout metrics
- Holdout predictions
- Permutation feature importance
- Cost-aware directional strategy diagnostics

## MVP Requirements

- Strict OHLCV validation
- Past-only feature generation
- Chronological holdout
- Expanding-window model search
- Logistic baseline
- Nonlinear tree ensemble
- Out-of-fold probability calibration
- Learned threshold and uncertainty band
- CLI and FastAPI inference
- Reproducible run artifacts
- Automated tests and CI

## Success Measures

- Primary accuracy, F1, and ROC-AUC are reported against the baseline.
- The holdout remains untouched until final evaluation.
- Identical input and artifact produce identical output.
- Tampered or structurally invalid artifacts are rejected.
- The default benchmark completes in under one minute on the reference machine.
- API requests with invalid market data fail before inference.

## Constraints

- Market relationships change over time.
- Synthetic results do not demonstrate real-world alpha.
- Daily OHLCV does not contain news, order-book, macroeconomic, or fundamental information.
- Directional backtests are diagnostic and include only a simple transaction-cost assumption.
- An internet-facing deployment still requires a reverse proxy, TLS, rate limiting, and secret
  management.

## Future Work

- Walk-forward retraining and drift reports
- Multi-asset panel models
- Corporate-action and exchange-calendar handling
- Probability calibration monitoring
- Portfolio-level costs and position sizing
- Fundamental, macroeconomic, and news features
- Model registry and scheduled promotion rules

## Stack

Python 3.11, pandas, NumPy, scikit-learn, FastAPI, Pydantic, pytest, TOML, Docker, and GitHub Actions.

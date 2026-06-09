# Original vs MarketSignal Forecast

## Architecture

| Original | MarketSignal |
|---|---|
| Four CrewAI agents in two main files | Separate data, feature, model, training, evaluation, inference, and API modules |
| Required Ollama and a local DeepSeek model | Runs with standard Python ML dependencies |
| Generated and executed Python at request time | Serves a fixed, validated numeric model artifact |
| Machine-specific shell paths | Portable package commands, TOML configuration, Docker, and CI |

## Machine Learning

The original repository did not train a forecasting model. It used an LLM to decide what analysis
code to write, so there was no fixed target or repeatable benchmark.

MarketSignal defines one task: predict whether the next close is higher. It adds:

- 36 fixed trailing features
- Logistic baseline
- Extra Trees classifier
- Expanding-window hyperparameter search
- Out-of-fold calibration and threshold selection
- Final chronological holdout
- Classification, probability, confidence, and strategy metrics

## Data

The old flow depended on whatever Yahoo Finance code the LLM generated. MarketSignal validates one
documented OHLCV schema and supports seeded synthetic data, CSV snapshots, or an explicit Yahoo
Finance loader.

## Security

The original system allowed generated-code execution and provided a tool that ran a saved Python
file. MarketSignal has no code-execution endpoint.

The first rewrite used joblib, which still inherited pickle's unsafe-deserialization behavior. The
current implementation removes it. Models are JSON metadata plus numeric NPZ arrays loaded with
`allow_pickle=False`, checksum verification, size limits, and tree-structure validation.

The API also adds request limits, strict schemas, trusted hosts, optional API-key authentication,
sanitized load errors, and an unprivileged Docker user.

## Measured Performance

The original has no comparable predictive metric. On the fixed seeded reference run:

| Metric | Logistic baseline | MarketSignal |
|---|---:|---:|
| Accuracy | 0.506 | 0.589 |
| F1 | 0.573 | 0.691 |
| ROC-AUC | 0.534 | 0.562 |
| Brier score | 0.251 | 0.245 |

The primary model improved accuracy by 8.3 percentage points and F1 by 0.118. With the uncertainty
band enabled, 67.7% of holdout observations received directional predictions and those were 61.6%
accurate.

These are synthetic measurements intended to verify the implementation. They are not a claim of
live trading performance.

## Efficiency

- Feature generation is vectorized and normalizes data once per training path.
- Parameter search uses smaller forests, then applies the selected parameters to the full model.
- The reference tuned run fell from roughly 92 seconds to 49 seconds.
- The exported model is about 220 KB and inference does not import scikit-learn estimators.

## Maintainability

- One responsibility per module
- Shared training/inference feature contract
- Human-readable configuration and manifests
- Immutable run outputs
- Regression tests for model parity and tamper rejection
- No hard-coded machine paths or LLM prompts

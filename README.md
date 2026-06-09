# MarketSignal Forecast

MarketSignal Forecast trains a next-session direction classifier from daily OHLCV data. It includes
the parts that are usually missing from market-model demos: chronological validation, probability
calibration, an uncertainty band, reproducible artifacts, and a small HTTP API.

It is a research tool, not trading advice.

## What It Does

The model estimates the probability that the next closing price will be above the latest close.
Predictions are reported as `up`, `down`, or `uncertain`. The uncertain result is deliberate: the
service does not force a directional answer when the probability is too close to the threshold
learned during temporal validation.

The default model is an Extra Trees classifier using 36 price, candle, trend, volatility, and
volume features. Probabilities are calibrated from expanding-window out-of-fold predictions. The
latest 20% of observations are kept untouched until final evaluation.

## Measured Reference Run

The checked-in configuration was run against the seeded synthetic dataset on June 9, 2026:

| Metric | Logistic baseline | MarketSignal |
|---|---:|---:|
| Accuracy | 0.506 | 0.589 |
| F1 | 0.573 | 0.691 |
| ROC-AUC | 0.534 | 0.562 |
| Brier score | 0.251 | 0.245 |

At the configured uncertainty margin, the model issued a directional prediction for 67.7% of the
holdout and those predictions were 61.6% accurate. These are synthetic benchmark results. They
verify the pipeline; they do not establish live-market profitability.

## Layout

```text
MarketSignal-Forecast/
├── app/                       FastAPI service
├── configs/                   TOML configuration
├── data/generated/            Reproducible local datasets
├── docs/                      PRD, architecture, and comparison notes
├── src/marketsignal/
│   ├── cli/                   Command-line entry points
│   ├── data/                  Loaders, validation, synthetic generator
│   ├── evaluation/            Metrics and feature importance
│   ├── features/              OHLCV feature engineering
│   ├── inference/             Prediction service
│   ├── models/                Estimator factory and safe artifact format
│   ├── training/              Search, calibration, evaluation, publishing
│   └── utils/
└── tests/
```

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Train

The default run is offline and reproducible:

```bash
marketsignal-train --config configs/default.toml
```

Skip parameter search for a quick check:

```bash
marketsignal-train --config configs/default.toml --no-tune
```

Use a CSV:

```bash
marketsignal-train --source csv --csv-path data/my_symbol.csv
```

Use Yahoo Finance:

```bash
python -m pip install -e ".[market]"
marketsignal-train --source yfinance --symbol SPY
```

Each run writes:

```text
artifacts/runs/<UTC_RUN_ID>/
├── model/
│   ├── forest.npz
│   └── manifest.json
├── metrics.json
├── feature_importance.csv
└── predictions.csv
```

`artifacts/latest/` is replaced only after a complete run is ready.

## Predict

Generate the reference CSV:

```bash
marketsignal-generate --output data/generated/synthetic_ohlcv.csv
```

Run inference:

```bash
marketsignal-predict \
  --model artifacts/latest/model \
  --input data/generated/synthetic_ohlcv.csv
```

Python:

```python
from marketsignal.data.loaders import load_csv
from marketsignal.inference import Predictor

predictor = Predictor.from_artifact("artifacts/latest/model")
result = predictor.predict_latest(load_csv("data/generated/synthetic_ohlcv.csv"))
print(result)
```

## API

Set an API key before exposing the service outside a trusted machine:

```bash
set MARKETSIGNAL_API_KEY=replace-me
set MARKETSIGNAL_ALLOWED_HOSTS=localhost,127.0.0.1
uvicorn app.api:app --host 127.0.0.1 --port 8000 --no-server-header
```

Use `$env:MARKETSIGNAL_API_KEY="replace-me"` in PowerShell.

Endpoints:

- `GET /health`
- `GET /model`
- `POST /predict`

`/model` and `/predict` require `X-API-Key` when `MARKETSIGNAL_API_KEY` is set. Production mode
disables the interactive API documentation:

```bash
set MARKETSIGNAL_ENV=production
```

Requests accept 80 to 2,500 candles and reject unknown fields, duplicate timestamps, non-finite
numbers, invalid candle ranges, and payloads above the configured HTTP limit.

## Artifact Security

The serving path does not use pickle or joblib. Models are exported as:

- A JSON manifest containing the feature contract and decision parameters.
- Numeric NumPy arrays loaded with `allow_pickle=False`.

Loading checks the SHA-256 digest, file type, compressed and expanded size, tree count, node count,
array names, array lengths, feature references, child references, and probability bounds. This
prevents the arbitrary code execution risk associated with loading untrusted pickle artifacts.

## Tests

```bash
python -m pytest
python -m ruff check src app tests
```

The tests cover data validation, deterministic generation, training outputs, safe artifact
round-tripping, tamper rejection, inference, API behavior, and optional API-key enforcement.

## Docker

The image runs as an unprivileged user. Mount a trained model at runtime:

```bash
docker build -t marketsignal-forecast .
docker run --rm -p 8000:8000 \
  -e MARKETSIGNAL_ALLOWED_HOSTS=localhost,127.0.0.1 \
  -e MARKETSIGNAL_API_KEY=replace-me \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  marketsignal-forecast
```

## Documentation

- [Product requirements](docs/PRD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Original project comparison](docs/COMPARISON.md)

## License

MIT

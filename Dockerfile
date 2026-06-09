FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system marketsignal && adduser --system --ingroup marketsignal marketsignal

COPY --chown=marketsignal:marketsignal src ./src
COPY --chown=marketsignal:marketsignal app ./app
COPY --chown=marketsignal:marketsignal configs ./configs
COPY --chown=marketsignal:marketsignal artifacts ./artifacts

USER marketsignal

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]

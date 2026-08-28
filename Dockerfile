FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/champion.joblib \
    COOLING_LOAD_CONFIG=/app/configs/base.yaml

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY configs configs
COPY src src
COPY models models

RUN chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"

CMD ["uvicorn", "cooling_load.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

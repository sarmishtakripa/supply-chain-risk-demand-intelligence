FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLATFORM_DATA_DIR=/app/data \
    PLATFORM_ARTIFACT_DIR=/app/artifacts \
    PLATFORM_SOURCE=demo

WORKDIR /app

COPY . /app
RUN python -m pip install -r requirements.lock && \
    python -m pip install --no-deps -e .

EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

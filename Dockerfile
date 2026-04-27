FROM python:3.10-slim

WORKDIR /app

# Installa dipendenze di sistema necessarie per Pillow e torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Timeout esteso e retry: torch+tensorflow sono pacchetti grossi (~700MB e ~500MB)
# e i download da pypi possono andare in timeout su connessioni lente.
RUN pip install --no-cache-dir --default-timeout 600 --retries 5 -r requirements.txt

COPY src/ ./src/
COPY experiments/ ./experiments/

ENV PORT=8000
EXPOSE 8000

# Shell form per espandere $PORT in caso venga sovrascritto
CMD exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT}

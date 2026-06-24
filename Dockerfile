FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY cacheir ./cacheir
COPY examples ./examples
COPY cpp ./cpp

RUN python -m pip install --upgrade pip \
  && python -m pip install -e ".[server]"

EXPOSE 8000
CMD ["cacheir", "profile"]

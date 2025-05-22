FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential wget curl unzip \
    libnss3 libnspr4 libatk-bridge2.0-0 libcups2 libdbus-1-3 libxkbcommon0 libgtk-3-0 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --upgrade pip poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without dev

# Install playwright browsers
RUN playwright install chromium --with-deps

COPY src ./src
COPY .env .env

RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

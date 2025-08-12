# Run command:
# docker build -t craa . && docker run --rm -it -v %CD%/data:/app/data -v %CD%/src:/app/src -p 8000:8000 craa

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    git build-essential wget curl unzip \
    libnss3 libnspr4 libatk-bridge2.0-0 libcups2 libdbus-1-3 libxkbcommon0 libgtk-3-0 libgbm1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --without dev
RUN playwright install chromium --with-deps

RUN pip install tiktoken

COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

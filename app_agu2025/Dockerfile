#docker build -f Dockerfile.crawler -t crawl_env . && docker run --rm -it -v %CD%:/app crawl_env
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    alembic \
    beautifulsoup4 \
    python-dotenv \
    lxml \
    openai \
    playwright \
    pyyaml \
    requests \
    sqlalchemy \
    tqdm \
    faiss-cpu \
    fastapi \
    uvicorn[standard] \
    jinja2 \
    tiktoken


# Expose FastAPI port
EXPOSE 8000

# Launch FastAPI app
# Adjust "main:app" if your entrypoint module is named differently
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

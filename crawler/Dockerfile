#docker build -t crawl_env . && docker run --rm -it -v %CD%:/app crawl_env
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pyyaml \
    beautifulsoup4 \
    lxml \
    requests \
    playwright \
    sqlalchemy && python -m playwright install --with-deps chromium

CMD ["bash"]

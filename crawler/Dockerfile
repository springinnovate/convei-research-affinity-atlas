#docker build -t crawl_env . && docker run --rm -it -v %CD%:/app crawl_env
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pyyaml

CMD ["bash"]

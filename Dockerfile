# Backend-only container. Desktop tools intentionally remain unavailable in containers.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY aria ./aria
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["aria"]

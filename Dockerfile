FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# The service needs only the API extras; model training stays outside this image.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]" \
    && groupadd --system gridwise \
    && useradd --system --uid 10001 --gid gridwise --no-create-home gridwise

USER gridwise
EXPOSE 8000
CMD ["uvicorn", "gridwise.api:app", "--host", "0.0.0.0", "--port", "8000"]

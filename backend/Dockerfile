FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY src /app/src
COPY backend /app/backend
ENV PYTHONPATH=/app/src:/app
ENV PORT=8080
ENV JAX_PLATFORMS=cpu
ENV CUDA_VISIBLE_DEVICES=""

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# SupportIQ API image (light: no PyTorch, embeddings come from the Gemini API).
#   docker build -t supportiq-api .
#   docker run -p 8000:8000 -e GEMINI_API_KEY=... -e DATABASE_URL=... supportiq-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY data/knowledge_base ./data/knowledge_base
COPY ml/artifacts/classifier.joblib ./ml/artifacts/classifier.joblib

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

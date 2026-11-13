# SupportIQ API image. Built for Hugging Face Spaces (Docker) but runs anywhere.
#   docker build -t supportiq-api .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... -e DATABASE_URL=... supportiq-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# CPU-only PyTorch keeps the image a few GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Download the embedding model at build time so cold starts do not hit the network.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app ./app
COPY data/knowledge_base ./data/knowledge_base
COPY ml/artifacts/classifier.joblib ./ml/artifacts/classifier.joblib

# Spaces runs as a non-root user; make writable dirs for SQLite fallback and Chroma.
RUN useradd -m appuser && mkdir -p /app/data/chroma && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

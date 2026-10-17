# Image for hosting the web UI + API (e.g. a Hugging Face Docker Space). Serving only:
# ingestion (yt-dlp, Whisper) runs locally. Secrets come from the host's settings:
# DATABASE_URL (use a read-only role) and GEMINI_API_KEY.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir --user -r requirements-serve.txt

COPY --chown=user api/ api/
COPY --chown=user ingest/ ingest/
COPY --chown=user retrieval/ retrieval/

# Download both models at build time so the first visitor doesn't wait for them.
RUN python -c "from ingest.embed import get_model; get_model('cpu'); \
from retrieval.rerank import get_reranker; get_reranker()"

# Public-demo limits (see api/limits.py); past the daily cap /ask answers in search-only mode.
ENV ASK_LIMIT_PER_HOUR=10 \
    LLM_DAILY_LIMIT=150

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

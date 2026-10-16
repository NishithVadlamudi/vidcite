"""Embed chunk text with BAAI/bge-small-en-v1.5 (384 dims, runs locally).

bge models expect an instruction prefix on QUERIES but not on passages,
so embed_passages and embed_query differ. Vectors are normalized, which
makes cosine similarity equal to a dot product.
"""
from __future__ import annotations

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIM = 384

# One short query is ~8x faster on CPU than on the Mac GPU (measured on an M2:
# 14 ms vs 111 ms warm, 0.04 s vs 1.7 s first call): GPU transfer and warm-up
# cost more than the compute saves. Batches of passages still use the GPU.
QUERY_DEVICE = "cpu"

_models: dict[str, object] = {}


def get_model(device: str | None = None):
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"  # Mac GPU when available
    if device not in _models:
        from sentence_transformers import SentenceTransformer

        try:  # use the cached copy without asking Hugging Face for updates (~1.2 s)
            _models[device] = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
        except OSError:  # not downloaded yet
            _models[device] = SentenceTransformer(MODEL_NAME, device=device)
    return _models[device]


def embed_passages(texts: list[str], model=None, batch_size: int = 32) -> list[list[float]]:
    model = model or get_model()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return [list(map(float, v)) for v in vectors]


def embed_query(question: str, model=None) -> list[float]:
    model = model or get_model(QUERY_DEVICE)
    vec = model.encode([QUERY_PREFIX + question], normalize_embeddings=True, show_progress_bar=False)[0]
    return list(map(float, vec))

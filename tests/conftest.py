import pytest


@pytest.fixture(autouse=True)
def no_real_reranker(monkeypatch):
    """The app reranks by default, which loads a cross-encoder model. Unit tests use fake
    databases and embedders, so keep real models out; reranking tests pass rerank=True
    with a fake scorer."""
    import retrieval.search

    monkeypatch.setattr(retrieval.search, "RERANK", False)

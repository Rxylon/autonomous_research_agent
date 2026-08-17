"""Tests for the embedding tier selection and its fallbacks."""

from __future__ import annotations

import pytest

from app.services import embeddings as module
from app.services.embeddings import EMBEDDING_DIMENSIONS, _hash_embedding, embed_query, embed_texts


def _clear_caches():
    module.get_embedding_model.cache_clear()
    module._openai_embeddings_enabled.cache_clear()
    module.embedding_backend.cache_clear()


@pytest.fixture(autouse=True)
def isolated_caches():
    _clear_caches()
    yield
    _clear_caches()


class TestHashFallback:
    def test_vectors_are_the_expected_width(self):
        assert len(_hash_embedding("some text")) == EMBEDDING_DIMENSIONS

    def test_vectors_are_normalised(self):
        magnitude = sum(component**2 for component in _hash_embedding("a b c")) ** 0.5
        assert magnitude == pytest.approx(1.0)

    def test_identical_text_gives_identical_vectors(self):
        assert _hash_embedding("repeatable") == _hash_embedding("repeatable")

    def test_different_text_gives_different_vectors(self):
        assert _hash_embedding("alpha beta") != _hash_embedding("gamma delta")

    def test_empty_text_does_not_divide_by_zero(self):
        assert len(_hash_embedding("")) == EMBEDDING_DIMENSIONS


class TestBackendSelection:
    def test_hash_preference_skips_the_local_model(self, monkeypatch):
        """`EMBEDDING_BACKEND=hash` must not import or load torch — that is the whole
        point of the setting on a memory-constrained host."""
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "hash")
        _clear_caches()
        assert module.get_embedding_model() is None
        assert module.embedding_backend() == "hash"

    def test_openai_preference_without_a_key_falls_back(self, monkeypatch):
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "openai")
        monkeypatch.setattr(module.settings, "openai_api_key", None)
        _clear_caches()
        assert module.embedding_backend() != "openai"

    def test_openai_preference_with_a_key_is_selected(self, monkeypatch):
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "openai")
        monkeypatch.setattr(module.settings, "openai_api_key", "sk-test")
        _clear_caches()
        assert module.embedding_backend() == "openai"

    def test_failed_openai_call_falls_through_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "openai")
        monkeypatch.setattr(module.settings, "openai_api_key", "sk-invalid")
        monkeypatch.setattr(module, "_openai_embeddings", lambda texts: None)
        _clear_caches()

        vectors = embed_texts(["fallback should still produce a vector"])
        assert len(vectors) == 1
        assert len(vectors[0]) in {EMBEDDING_DIMENSIONS, 384, 768}


class TestEmbedTexts:
    def test_empty_input_returns_empty_list(self):
        assert embed_texts([]) == []

    def test_one_vector_per_text(self, monkeypatch):
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "hash")
        _clear_caches()
        assert len(embed_texts(["a", "b", "c"])) == 3

    def test_embed_query_returns_a_single_vector(self, monkeypatch):
        monkeypatch.setattr(module.settings, "embedding_backend_preference", "hash")
        _clear_caches()
        assert len(embed_query("a query")) == EMBEDDING_DIMENSIONS

    def test_local_model_failure_falls_back_to_hashing(self, monkeypatch):
        class BrokenModel:
            def encode(self, texts, **kwargs):
                raise RuntimeError("out of memory")

        monkeypatch.setattr(module, "get_embedding_model", lambda: BrokenModel())
        monkeypatch.setattr(module, "_openai_embeddings_enabled", lambda: False)
        vectors = embed_texts(["text that must still embed"])
        assert len(vectors[0]) == EMBEDDING_DIMENSIONS

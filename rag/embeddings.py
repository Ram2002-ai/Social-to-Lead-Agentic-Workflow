"""Embedding model factory for the RAG pipeline."""

from __future__ import annotations

from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import settings


def get_embedding_model(*, prefer_openai: bool = True) -> OpenAIEmbeddings | FakeEmbeddings:
    """Return an embedding model suitable for the current environment."""
    if prefer_openai and settings.use_openai and settings.openai_api_key:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            max_retries=0,
            request_timeout=8,
        )

    return FakeEmbeddings(size=1536)
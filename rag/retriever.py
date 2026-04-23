"""Local JSON knowledge-base retriever backed by FAISS with keyword fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from config.settings import settings
from rag.embeddings import get_embedding_model


class KnowledgeBaseRetriever:
    """Build and query a FAISS vector index from the local knowledge base."""

    def __init__(self, knowledge_base_path: Path | None = None) -> None:
        self.knowledge_base_path = knowledge_base_path or settings.knowledge_base_path
        self.documents = self._load_documents()
        self._use_real_embeddings = bool(settings.use_openai and settings.openai_api_key)
        
        try:
            self._vector_store = FAISS.from_documents(self.documents, get_embedding_model())
        except Exception:
            self._vector_store = FAISS.from_documents(
                self.documents,
                get_embedding_model(prefer_openai=False),
            )
            self._use_real_embeddings = False

    def retrieve(self, query: str, *, k: int = 3) -> list[Document]:
        """Return the most relevant knowledge-base documents."""
        if not self._use_real_embeddings:
            return self._keyword_retrieve(query, k)
        return self._vector_store.similarity_search(query, k=k)

    def context_for(self, query: str) -> str:
        """Return retrieved documents formatted as plain context."""
        documents = self.retrieve(query)
        return "\n".join(doc.page_content for doc in documents)

    def _keyword_retrieve(self, query: str, k: int) -> list[Document]:
        """Fallback keyword-based retrieval when embeddings aren't available."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for word in query_words if word in content_lower)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def _load_documents(self) -> list[Document]:
        """Load JSON and flatten it into semantic documents."""
        with self.knowledge_base_path.open("r", encoding="utf-8") as file:
            raw_data: dict[str, Any] = json.load(file)

        documents: list[Document] = []
        for section, value in raw_data.items():
            documents.extend(self._flatten_section(section, value))
        return documents

    def _flatten_section(self, path: str, value: Any) -> list[Document]:
        """Convert nested JSON values into retrievable documents."""
        if isinstance(value, dict):
            documents: list[Document] = []
            for key, child_value in value.items():
                documents.extend(self._flatten_section(f"{path}.{key}", child_value))
            return documents

        content = f"{path}: {value}"
        return [Document(page_content=content, metadata={"source": path})]
"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the AutoStream agent."""

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
    use_openai: bool = os.getenv("AUTOSTREAM_USE_OPENAI", "true").lower() == "true"
    use_gemini: bool = os.getenv("AUTOSTREAM_USE_GEMINI", "false").lower() == "true"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    knowledge_base_path: Path = BASE_DIR / "data" / "knowledge_base.json"
    faiss_index_path: Path = BASE_DIR / "data" / "faiss_index"


settings = Settings()
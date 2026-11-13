from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

CATEGORIES = ["billing", "delivery", "technical", "returns", "account"]
PRIORITIES = ["low", "medium", "high"]
STATUSES = ["open", "pending", "resolved"]
SENTIMENTS = ["positive", "neutral", "dissatisfied", "frustrated", "angry"]


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"  # free tier: Flash is capped at ~20 req/day, Lite is far higher
    gemini_fallback_model: str = "gemini-3.5-flash"  # used on 503 overload / 429 rate limit
    database_url: str = f"sqlite:///{BASE_DIR / 'supportiq.db'}"
    enable_rag: bool = True
    # Shared secret for the internal /tickets/* API (agent dashboard). Empty = no auth (local dev).
    # The customer-facing /public/* endpoints never require it.
    api_key: str = ""

    classifier_path: Path = BASE_DIR / "ml" / "artifacts" / "classifier.joblib"
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"

    # Embeddings: "gemini" (API, multilingual, no heavy dependencies; the deployment default)
    # or "local" (Sentence Transformers; needs `pip install sentence-transformers`, ~1.5 GB RAM).
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    local_embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    @field_validator("database_url")
    @classmethod
    def _normalise_database_url(cls, url: str) -> str:
        """Hosted Postgres providers hand out `postgres://` or `postgresql://` URLs.
        SQLAlchemy needs the driver spelled out, and we ship psycopg (v3)."""
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        return url

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


settings = Settings()

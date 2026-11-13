from pathlib import Path

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

    classifier_path: Path = BASE_DIR / "ml" / "artifacts" / "classifier.joblib"
    knowledge_base_dir: Path = BASE_DIR / "data" / "knowledge_base"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"  # English + Hindi; 384-dim like the English-only MiniLM

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")


settings = Settings()

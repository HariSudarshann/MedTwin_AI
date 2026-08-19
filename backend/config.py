"""
MediTwin AI — Central Configuration
Reads from .env file; all paths and settings are defined here.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────
    llm_provider: str = Field("groq", env="LLM_PROVIDER")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    anthropic_model: str = Field("claude-3-haiku-20240307", env="ANTHROPIC_MODEL")
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", env="OLLAMA_MODEL")

    # ── Kaggle ────────────────────────────────────────────────
    kaggle_username: str = Field("", env="KAGGLE_USERNAME")
    kaggle_key: str = Field("", env="KAGGLE_KEY")

    # ── Paths ─────────────────────────────────────────────────
    data_dir: Path = Field(BASE_DIR / "data", env="DATA_DIR")
    models_dir: Path = Field(BASE_DIR / "models", env="MODELS_DIR")
    outputs_dir: Path = Field(BASE_DIR / "outputs", env="OUTPUTS_DIR")
    rag_corpus_dir: Path = Field(BASE_DIR / "rag_corpus", env="RAG_CORPUS_DIR")
    chroma_persist_dir: Path = Field(BASE_DIR / "outputs" / "chromadb", env="CHROMA_PERSIST_DIR")

    # ── Imaging Model ─────────────────────────────────────────
    imaging_model_path: Path = Field(BASE_DIR / "models" / "resnet18_xray.pt")
    imaging_input_size: int = 224
    imaging_batch_size: int = 16
    imaging_epochs: int = 10
    imaging_lr: float = 1e-4
    imaging_classes: list = ["NORMAL", "PNEUMONIA"]

    # ── Tabular Risk Model ────────────────────────────────────
    tabular_model_path: Path = Field(BASE_DIR / "models" / "heart_risk_rf.joblib")
    tabular_dataset: str = "heart_disease"

    # ── RAG ───────────────────────────────────────────────────
    rag_collection_name: str = "medical_guidelines"
    rag_top_k: int = 3
    rag_embedding_model: str = "all-MiniLM-L6-v2"

    # ── Backend ───────────────────────────────────────────────
    backend_host: str = Field("0.0.0.0", env="BACKEND_HOST")
    backend_port: int = Field(8000, env="BACKEND_PORT")

    class Config:
        env_file = str(BASE_DIR / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Ensure output dirs exist
for d in [settings.outputs_dir / "reports",
          settings.outputs_dir / "fhir",
          settings.outputs_dir / "xai_plots",
          settings.chroma_persist_dir,
          settings.models_dir]:
    d.mkdir(parents=True, exist_ok=True)

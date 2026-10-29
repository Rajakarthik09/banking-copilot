"""Settings, read from environment variables (and .env when present)."""

from datetime import date
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_url: str = "postgresql://copilot:copilot@localhost:5432/copilot"

    # Any OpenAI-compatible endpoint: Ollama (default), OpenAI, Azure OpenAI.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "copilot-qwen"  # qwen2.5:3b with num_ctx 8192, see ollama/Modelfile
    llm_timeout_s: float = 300.0
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # nomic-embed-text was trained with task prefixes; other models don't want them.
    embed_task_prefixes: bool = True
    # LLM_MOCK=1 swaps in MockLLM + MockEmbedder (tests / CI, no model needed).
    llm_mock: bool = False

    # All date logic is relative to this fixed date, never CURRENT_DATE, so the
    # synthetic data and the eval answers do not drift over time.
    as_of_date: date = date(2026, 10, 31)

    # Retrieval gate (calibrated by scripts/calibrate_threshold.py; RRF scores are rank-based
    # and can't be thresholded). Retrieval is "strong" if the best cosine similarity is
    # >= rag_min_sim, or >= rag_min_sim_with_keyword AND the best full-text rank >= rag_min_fts.
    # Borderline questions that pass the gate are left to the prompt's refusal instruction.
    rag_min_sim: float = 0.60
    rag_min_sim_with_keyword: float = 0.53
    rag_min_fts: float = 0.025
    rag_top_k: int = 5

    max_tool_calls: int = 5
    tool_row_limit: int = 20

    # Demo API keys -> rm_id. In production this would come from SSO / a JWT.
    api_keys: str = "rm1-demo-key:RM-1,rm2-demo-key:RM-2,rm3-demo-key:RM-3,rm4-demo-key:RM-4,rm5-demo-key:RM-5"

    def api_key_map(self) -> dict[str, str]:
        pairs = (p.split(":", 1) for p in self.api_keys.split(",") if ":" in p)
        return {k.strip(): v.strip() for k, v in pairs}


@lru_cache
def get_settings() -> Settings:
    return Settings()

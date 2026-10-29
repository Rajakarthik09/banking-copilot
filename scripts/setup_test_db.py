"""Create and seed the isolated test database (mock embeddings), never touching the dev DB.

Used by tests/conftest.py and `eval/run_eval.py --mock`. The test DB name is the dev DB
name + "_test" unless TEST_DB_URL is set (CI points it at its own service container).

Run:  python scripts/setup_test_db.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Thresholds for the mock (hashed bag-of-words) embedder. Its similarity scale differs from
# nomic-embed-text (common words dominate), so only the keyword path may pass the gate.
# Calibrated with scripts/calibrate_threshold.py under use_test_env(): 16/16 in, 9/10 out.
MOCK_ENV = {"LLM_MOCK": "1", "RAG_MIN_SIM": "1.01", "RAG_MIN_SIM_WITH_KEYWORD": "0.15", "RAG_MIN_FTS": "0.025"}


def test_db_url() -> str:
    if url := os.environ.get("TEST_DB_URL"):
        return url
    from app.config import Settings

    prefix, name = Settings().db_url.rsplit("/", 1)
    name = name.split("?")[0]
    return f"{prefix}/{name if name.endswith('_test') else name + '_test'}"


def use_test_env() -> str:
    """Point this process at the test DB with mock models. Idempotent; call before app code reads settings."""
    url = test_db_url()
    os.environ.update({**MOCK_ENV, "DB_URL": url, "TEST_DB_URL": url})
    from app.config import get_settings

    get_settings.cache_clear()
    return url


def setup_test_db() -> str:
    import psycopg

    url = use_test_env()
    admin_url, name = url.rsplit("/", 1)
    name = name.split("?")[0]
    with psycopg.connect(f"{admin_url}/postgres", autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(f'CREATE DATABASE "{name}"')

    sys.path.insert(0, str(ROOT / "data"))
    import generate_synthetic

    from app.rag.ingest import ingest

    generate_synthetic.main()
    ingest()
    return url


if __name__ == "__main__":
    print("test DB ready:", setup_test_db())

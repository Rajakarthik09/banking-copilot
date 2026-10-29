"""Postgres connection pool (psycopg 3). Rows come back as dicts."""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().db_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn


def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    with get_conn() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    with get_conn() as conn:
        return conn.execute(sql, params).fetchone()

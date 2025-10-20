from __future__ import annotations
import datetime as dt
import os
from typing import Optional

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from settings import load_config


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    if val is None:
        return default
    return val.strip().strip('"').strip("'")


def _build_url_from_laravel_env() -> Optional[str]:
    """Build SQLAlchemy URL from Laravel .env (DB_* keys). Supports MySQL."""
    conn = (_env("DB_CONNECTION") or "").lower()
    host = _env("DB_HOST") or "localhost"
    port = _env("DB_PORT") or "3306"
    db = _env("DB_DATABASE")
    user = _env("DB_USERNAME")
    pwd = _env("DB_PASSWORD")

    if not conn or not db or not user:
        return None

    if conn in ("mysql", "mariadb"):
        # Use PyMySQL driver
        return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"

    # Extend here for other drivers if needed (e.g., PostgreSQL)
    return None


def get_engine() -> Engine:
    """Create SQLAlchemy engine from RECO_DB_URL or Laravel DB_* envs."""
    cfg = load_config()
    url = (cfg.db.url or "").strip()
    if not url:
        url = _build_url_from_laravel_env() or ""

    if not url:
        raise RuntimeError(
            "Database URL not configured. Set RECO_DB_URL in .env (e.g., mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4) "
            "or ensure Laravel DB_* variables are set."
        )

    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    return engine


def load_products() -> pl.DataFrame:
    """Load basic product metadata suitable for content-based features."""
    engine = get_engine()
    sql = text(
        """
        SELECT
            p.id,
            p.title,
            p.slug,
            p.sku,
            p.sale_price,
            p.stock_quantity,
            p.status,
            p.published_at,
            p.seller_id,
            p.category_id
        FROM products p
        WHERE p.deleted_at IS NULL
        """
    )
    with engine.connect() as conn:
        rows = list(conn.execute(sql).mappings())
    return pl.DataFrame(rows) if rows else pl.DataFrame([])


def load_order_items(start_date: dt.date, end_date: dt.date) -> pl.DataFrame:
    """Load order items joined with orders, filtered by status completed and date range."""
    engine = get_engine()
    sql = text(
        """
        SELECT
            oi.id,
            oi.order_id,
            oi.product_id,
            oi.variety_id AS variant_id,
            oi.quantity,
            oi.price,
            oi.sale_price,
            oi.total_price,
            oi.created_at AS item_created_at,
            o.user_id AS order_user_id,
            o.total_price AS order_total_price,
            o.created_at AS order_created_at
        FROM order_items oi
        INNER JOIN orders o ON o.id = oi.order_id
        WHERE o.status = :completed
          AND o.created_at >= :start_dt
          AND o.created_at <= :end_dt
        """
    )
    params = {
        "completed": "completed",
        "start_dt": f"{start_date} 00:00:00",
        "end_dt": f"{end_date} 23:59:59",
    }
    with get_engine().connect() as conn:
        rows = list(conn.execute(sql, params).mappings())
    return pl.DataFrame(rows) if rows else pl.DataFrame([])
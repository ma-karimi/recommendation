"""
dataframe_loader.py - DataFrame-based data loading
این ماژول داده‌ها را به صورت Polars DataFrames برمی‌گرداند
مناسب برای: پردازش داده، Machine Learning، و تحلیل

نکته: برای Object-based loading از object_loader.py استفاده کنید
"""
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


# Global engine cache
_engine_cache: Optional[Engine] = None

def get_engine(force_new: bool = False) -> Engine:
    """
    Create SQLAlchemy engine from RECO_DB_URL or Laravel DB_* envs.
    
    Args:
        force_new: If True, create a new engine even if one exists (useful for resetting broken connections)
    """
    global _engine_cache
    
    if _engine_cache is not None and not force_new:
        return _engine_cache
    
    cfg = load_config()
    url = (cfg.db.url or "").strip()
    if not url:
        url = _build_url_from_laravel_env() or ""

    if not url:
        raise RuntimeError(
            "Database URL not configured. Set RECO_DB_URL in .env (e.g., mysql+pymysql://user:pass@host:3306/db?charset=utf8mb4) "
            "or ensure Laravel DB_* variables are set."
        )

    # Enhanced connection pool settings for MySQL/PyMySQL
    engine = create_engine(
        url,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,    # Recycle connections after 1 hour
        pool_size=5,          # Number of connections to maintain
        max_overflow=10,      # Additional connections allowed
        echo=False,
        connect_args={
            "connect_timeout": 10,
            "read_timeout": 30,
            "write_timeout": 30,
        }
    )
    
    _engine_cache = engine
    return engine

def reset_engine() -> None:
    """Reset the engine cache (useful when connections are broken)"""
    global _engine_cache
    if _engine_cache is not None:
        try:
            _engine_cache.dispose()
        except:
            pass
    _engine_cache = None


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
          AND p.status = 1
          AND p.stock_quantity > 0
        """
    )
    with engine.connect() as conn:
        rows = [dict(row) for row in conn.execute(sql).mappings()]
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
        rows = [dict(row) for row in conn.execute(sql, params).mappings()]
    return pl.DataFrame(rows) if rows else pl.DataFrame([])
from __future__ import annotations
import datetime as dt
from typing import List, Optional
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from models import User, Product, Order, OrderItem, UserBehavior, ProductInteraction
from settings import load_config


def get_engine() -> Engine:
    """ایجاد اتصال به پایگاه داده"""
    cfg = load_config()
    url = (cfg.db.url or "").strip()
    if not url:
        raise RuntimeError("Database URL not configured. Set RECO_DB_URL in .env")
    
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    return engine


def load_users() -> List[User]:
    """بارگذاری کاربران"""
    engine = get_engine()
    sql = text("""
        SELECT id, email, name, created_at
        FROM users
        WHERE deleted_at IS NULL
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings()
        return [User(
            id=row['id'],
            email=row.get('email'),
            name=row.get('name'),
            created_at=row.get('created_at')
        ) for row in rows]


def load_products() -> List[Product]:
    """بارگذاری محصولات"""
    engine = get_engine()
    sql = text("""
        SELECT id, title, slug, sku, sale_price, stock_quantity, status, 
               published_at, seller_id, category_id
        FROM products
        WHERE deleted_at IS NULL AND status = 'published'
    """)
    
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings()
        return [Product(
            id=row['id'],
            title=row['title'],
            slug=row['slug'],
            sku=row['sku'],
            sale_price=float(row['sale_price'] or 0),
            stock_quantity=int(row['stock_quantity'] or 0),
            status=row['status'],
            published_at=row.get('published_at'),
            seller_id=row.get('seller_id'),
            category_id=row.get('category_id')
        ) for row in rows]


def load_orders(start_date: dt.date, end_date: dt.date) -> List[Order]:
    """بارگذاری سفارشات در بازه زمانی"""
    engine = get_engine()
    sql = text("""
        SELECT id, user_id, total_price, status, created_at, sub_order_id
        FROM orders
        WHERE created_at >= :start_date AND created_at <= :end_date
        AND status = 'completed'
    """)
    
    params = {
        "start_date": f"{start_date} 00:00:00",
        "end_date": f"{end_date} 23:59:59"
    }
    
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings()
        return [Order(
            id=row['id'],
            user_id=row['user_id'],
            total_price=float(row['total_price']),
            status=row['status'],
            created_at=row['created_at'],
            sub_order_id=row.get('sub_order_id')
        ) for row in rows]


def load_order_items(start_date: dt.date, end_date: dt.date) -> List[OrderItem]:
    """بارگذاری آیتم‌های سفارش در بازه زمانی"""
    engine = get_engine()
    sql = text("""
        SELECT oi.id, oi.order_id, oi.product_id, oi.variety_id as variant_id,
               oi.quantity, oi.price, oi.sale_price, oi.total_price, oi.created_at
        FROM order_items oi
        INNER JOIN orders o ON o.id = oi.order_id
        WHERE o.created_at >= :start_date AND o.created_at <= :end_date
        AND o.status = 'completed'
    """)
    
    params = {
        "start_date": f"{start_date} 00:00:00",
        "end_date": f"{end_date} 23:59:59"
    }
    
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings()
        return [OrderItem(
            id=row['id'],
            order_id=row['order_id'],
            product_id=row['product_id'],
            variant_id=row.get('variant_id'),
            quantity=int(row['quantity']),
            price=float(row['price']),
            sale_price=float(row['sale_price']),
            total_price=float(row['total_price']),
            created_at=row['created_at']
        ) for row in rows]


def load_user_purchase_history(user_id: int, days_back: int = 365) -> List[ProductInteraction]:
    """بارگذاری تاریخچه خرید کاربر"""
    engine = get_engine()
    start_date = dt.date.today() - dt.timedelta(days=days_back)
    
    sql = text("""
        SELECT oi.product_id, oi.total_price, oi.created_at
        FROM order_items oi
        INNER JOIN orders o ON o.id = oi.order_id
        WHERE o.user_id = :user_id
        AND o.created_at >= :start_date
        AND o.status = 'completed'
    """)
    
    params = {
        "user_id": user_id,
        "start_date": f"{start_date} 00:00:00"
    }
    
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings()
        return [ProductInteraction(
            user_id=user_id,
            product_id=row['product_id'],
            interaction_type='purchase',
            timestamp=row['created_at'],
            value=float(row['total_price'])
        ) for row in rows]


def load_user_interactions_from_matomo(user_id: int, matomo_data_path: str) -> List[ProductInteraction]:
    """بارگذاری تعاملات کاربر از داده‌های Matomo"""
    # این تابع نیاز به پیاده‌سازی دارد تا داده‌های Matomo را با کاربران مرتبط کند
    # فعلاً یک پیاده‌سازی ساده
    interactions = []
    
    try:
        # خواندن فایل‌های Parquet Matomo
        events_df = pl.read_parquet(f"{matomo_data_path}/matomo_events_*.parquet")
        pageviews_df = pl.read_parquet(f"{matomo_data_path}/matomo_pageviews_*.parquet")
        
        # استخراج تعاملات مرتبط با محصولات
        # این بخش نیاز به منطق پیچیده‌تری دارد تا URL ها را به product_id تبدیل کند
        
    except Exception as e:
        print(f"خطا در خواندن داده‌های Matomo: {e}")
    
    return interactions

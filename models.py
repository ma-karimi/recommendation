from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import datetime as dt


@dataclass
class User:
    """مدل کاربر"""
    id: int
    email: Optional[str] = None
    name: Optional[str] = None
    created_at: Optional[dt.datetime] = None


@dataclass
class Product:
    """مدل محصول"""
    id: int
    title: str
    slug: str
    sku: str
    sale_price: float
    stock_quantity: int
    status: str
    published_at: Optional[dt.datetime] = None
    seller_id: Optional[int] = None
    category_id: Optional[int] = None


@dataclass
class Order:
    """مدل سفارش"""
    id: int
    user_id: int
    total_price: float
    status: str
    created_at: dt.datetime
    sub_order_id: Optional[int] = None  # معادل store


@dataclass
class OrderItem:
    """مدل آیتم سفارش"""
    id: int
    order_id: int
    product_id: int
    quantity: int
    price: float
    sale_price: float
    total_price: float
    created_at: dt.datetime
    variant_id: Optional[int] = None


@dataclass
class UserBehavior:
    """رفتار کاربر از Matomo"""
    user_id: int
    page_views: int
    events: int
    goals: int
    session_duration: float
    bounce_rate: float


@dataclass
class ProductInteraction:
    """تعامل کاربر با محصول"""
    user_id: int
    product_id: int
    interaction_type: str  # 'view', 'purchase', 'cart_add', 'wishlist'
    timestamp: dt.datetime
    value: float = 0.0


@dataclass
class Recommendation:
    """توصیه محصول"""
    user_id: int
    product_id: int
    score: float
    reason: str  # توضیح دلیل توصیه
    confidence: float  # میزان اطمینان (0-1)

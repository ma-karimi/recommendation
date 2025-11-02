"""
API برای سیستم توصیه محصولات

این ماژول endpointهای REST API را برای دریافت توصیه‌های محصولات ارائه می‌دهد.
"""
from __future__ import annotations
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from hybrid_recommender import HybridRecommender
from models import Product
from object_loader import load_products

# تنظیم logger
logger = logging.getLogger(__name__)

# مدل‌های Pydantic برای API
class RecommendationResponse(BaseModel):
    product_id: int
    score: float
    reason: str
    confidence: float
    product_title: Optional[str] = None
    product_price: Optional[float] = None
    product_stock: Optional[int] = None
    collaborative_details: Optional[Dict[str, Any]] = None  # جزئیات collaborative


class UserInsightsResponse(BaseModel):
    total_interactions: int
    preferred_categories: List[tuple]
    average_purchase_value: float
    similar_users: List[tuple]


class PopularProductsResponse(BaseModel):
    product_id: int
    purchase_count: int
    product_title: Optional[str] = None
    product_price: Optional[float] = None


class SimilarProductsResponse(BaseModel):
    product_id: int
    similarity_score: float
    product_title: Optional[str] = None
    product_price: Optional[float] = None


# State management برای مدل و cache
class AppState:
    """مدیریت state اپلیکیشن"""
    def __init__(self):
        self.recommender: Optional[HybridRecommender] = None
        self.products_cache: Dict[int, Product] = {}
        self.storage = None
    
    def init_redis_storage(self):
        """دریافت یا initialize کردن Redis storage"""
        if self.storage is None:
            try:
                from recommendation_storage import get_storage
                self.storage = get_storage()
                logger.info("Redis storage initialized successfully")
            except ImportError:
                logger.warning("recommendation_storage module not available")
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
        return self.storage


# ایجاد instance سراسری state
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت چرخه حیات اپلیکیشن"""
    logger.info("Starting API initialization...")
    
    # بارگذاری اولیه (فقط کش محصولات - توصیه‌ها از Redis خوانده می‌شوند)
    try:
        products = load_products()
        app_state.products_cache = {p.id: p for p in products}
        logger.info(f"Loaded {len(app_state.products_cache)} products into cache")
    except Exception as e:
        logger.error(f"Error loading products: {e}")
        app_state.products_cache = {}
    
    # تست اتصال به Redis
    storage = app_state.init_redis_storage()
    if storage and storage.test_connection():
        logger.info("Redis connection established")
    else:
        logger.warning("Redis not available - will use fallback mode")
    
    yield
    
    # پاکسازی
    logger.info("Shutting down API...")
    app_state.products_cache.clear()


app = FastAPI(
    title="سیستم توصیه محصولات",
    description="API برای دریافت توصیه‌های محصولات کاربرمحور",
    version="1.0.0",
    lifespan=lifespan
)


def get_recommender() -> Optional[HybridRecommender]:
    """دریافت نمونه سیستم توصیه (برای fallback)"""
    if app_state.recommender is None:
        try:
            logger.info("Initializing recommender for fallback mode...")
            app_state.recommender = HybridRecommender()
            app_state.recommender.train()
            logger.info("Recommender initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing recommender: {e}")
    return app_state.recommender


def get_product_info(product_id: int) -> Dict[str, Any]:
    """دریافت اطلاعات محصول"""
    product = app_state.products_cache.get(product_id)
    if not product:
        return {}
    
    return {
        "product_title": product.title,
        "product_price": product.sale_price,
        "product_stock": product.stock_quantity
    }


def parse_collaborative_details(collaborative_details: Any) -> Optional[Dict[str, Any]]:
    """Parse کردن collaborative_details از JSON string یا dict"""
    if not collaborative_details:
        return None
    
    # اگر قبلاً dict است، مستقیماً برگردان
    if isinstance(collaborative_details, dict):
        return collaborative_details
    
    # اگر string است، parse کن
    if isinstance(collaborative_details, str):
        try:
            return json.loads(collaborative_details)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing collaborative_details: {e}")
            return None
    
    return None


@app.get("/")
async def root():
    """صفحه اصلی"""
    return {
        "message": "سیستم توصیه محصولات",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    """بررسی سلامت سیستم"""
    return {
        "status": "healthy",
        "recommender_ready": app_state.recommender is not None,
        "products_loaded": len(app_state.products_cache)
    }


def _convert_recommendation_to_response(
    rec_data: Dict[str, Any],
    product_info: Dict[str, Any]
) -> RecommendationResponse:
    """تبدیل داده توصیه به Response model"""
    collaborative_details = parse_collaborative_details(
        rec_data.get('collaborative_details')
    )
    
    return RecommendationResponse(
        product_id=rec_data['product_id'],
        score=rec_data['score'],
        reason=rec_data['reason'],
        confidence=rec_data['confidence'],
        product_title=product_info.get("product_title"),
        product_price=product_info.get("product_price"),
        product_stock=product_info.get("product_stock"),
        collaborative_details=collaborative_details
    )


@app.get("/recommendations/{user_id}", response_model=List[RecommendationResponse])
async def get_user_recommendations(
    user_id: int,
    limit: int = 10,
    use_redis: bool = True
):
    """دریافت توصیه‌های کاربر از Redis یا تولید مستقیم"""
    try:
        # تلاش برای خواندن از Redis
        if use_redis:
            storage = app_state.init_redis_storage()
            if storage and storage.exists(user_id):
                recs_dict = storage.get_recommendations(user_id)
                
                if recs_dict:
                    response = [
                        _convert_recommendation_to_response(
                            rec,
                            get_product_info(rec['product_id'])
                        )
                        for rec in recs_dict[:limit]
                    ]
                    logger.debug(f"Retrieved {len(response)} recommendations from Redis for user {user_id}")
                    return response
        
        # Fallback: اگر Redis در دسترس نباشد یا توصیه وجود نداشته باشد
        logger.info(f"Using fallback mode for user {user_id}")
        recommender_instance = get_recommender()
        if recommender_instance:
            recommendations = recommender_instance.get_recommendations(user_id, limit)
            
            response = [
                _convert_recommendation_to_response(
                    {
                        'product_id': rec.product_id,
                        'score': rec.score,
                        'reason': rec.reason,
                        'confidence': rec.confidence,
                        'collaborative_details': rec.collaborative_details
                    },
                    get_product_info(rec.product_id)
                )
                for rec in recommendations
            ]
            
            logger.debug(f"Generated {len(response)} recommendations via fallback for user {user_id}")
            return response
        else:
            raise HTTPException(
                status_code=503, 
                detail="توصیه‌ای برای این کاربر یافت نشد و سیستم در حال بارگذاری است"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا در دریافت توصیه‌ها: {str(e)}")


@app.get("/insights/{user_id}", response_model=UserInsightsResponse)
async def get_user_insights(
    user_id: int,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت بینش‌های کاربر"""
    try:
        if recommender_instance is None:
            raise HTTPException(
                status_code=503,
                detail="سیستم توصیه در حال بارگذاری است"
            )
        insights = recommender_instance.get_user_insights(user_id)
        return UserInsightsResponse(**insights)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting insights for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا در دریافت بینش‌ها: {str(e)}")


@app.get("/popular", response_model=List[PopularProductsResponse])
async def get_popular_products(
    limit: int = 10,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت محصولات محبوب"""
    try:
        if recommender_instance is None:
            raise HTTPException(
                status_code=503,
                detail="سیستم توصیه در حال بارگذاری است"
            )
        popular_products = recommender_instance.get_popular_products(limit)
        
        response = [
            PopularProductsResponse(
                product_id=product_id,
                purchase_count=count,
                product_title=get_product_info(product_id).get("product_title"),
                product_price=get_product_info(product_id).get("product_price")
            )
            for product_id, count in popular_products
        ]
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting popular products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا در دریافت محصولات محبوب: {str(e)}")


@app.get("/similar/{product_id}", response_model=List[SimilarProductsResponse])
async def get_similar_products(
    product_id: int,
    limit: int = 5,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت محصولات مشابه"""
    try:
        if recommender_instance is None:
            raise HTTPException(
                status_code=503,
                detail="سیستم توصیه در حال بارگذاری است"
            )
        similar_products = recommender_instance.get_similar_products(product_id, limit)
        
        response = [
            SimilarProductsResponse(
                product_id=similar_product_id,
                similarity_score=similarity,
                product_title=get_product_info(similar_product_id).get("product_title"),
                product_price=get_product_info(similar_product_id).get("product_price")
            )
            for similar_product_id, similarity in similar_products
        ]
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar products for {product_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا در دریافت محصولات مشابه: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """بازآموزی مدل"""
    try:
        logger.info("Starting model retraining...")
        app_state.recommender = HybridRecommender()
        app_state.recommender.train()
        
        # به‌روزرسانی کش محصولات
        products = load_products()
        app_state.products_cache = {p.id: p for p in products}
        
        logger.info(f"Model retrained successfully. {len(app_state.products_cache)} products cached")
        return {
            "message": "مدل با موفقیت بازآموزی شد",
            "products_count": len(app_state.products_cache)
        }
    except Exception as e:
        logger.error(f"Error retraining model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"خطا در بازآموزی مدل: {str(e)}")


@app.get("/stats")
async def get_system_stats():
    """دریافت آمار سیستم"""
    redis_stats = {}
    storage = app_state.init_redis_storage()
    redis_connected = False
    
    if storage:
        try:
            redis_connected = storage.test_connection()
            if redis_connected:
                redis_stats = storage.get_stats()
        except Exception as e:
            logger.warning(f"Error getting Redis stats: {e}")
    
    return {
        "total_products": len(app_state.products_cache),
        "recommender_ready": app_state.recommender is not None,
        "redis_connected": redis_connected,
        "redis_stats": redis_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



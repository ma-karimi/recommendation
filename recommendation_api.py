from __future__ import annotations
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime as dt
from contextlib import asynccontextmanager

from hybrid_recommender import HybridRecommender
from models import Recommendation, User, Product
from object_loader import load_users, load_products


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


# متغیر سراسری برای مدل
recommender: Optional[HybridRecommender] = None
products_cache: Dict[int, Product] = {}
storage = None  # Redis storage

def init_redis_storage():
    """دریافت یا initialize کردن Redis storage"""
    global storage
    if storage is None:
        try:
            from recommendation_storage import get_storage
            storage = get_storage()
        except ImportError:
            print("⚠️  recommendation_storage در دسترس نیست")
        except Exception as e:
            print(f"⚠️  خطا در اتصال به Redis: {e}")
    return storage


@asynccontextmanager
async def lifespan(app: FastAPI):
    """مدیریت چرخه حیات اپلیکیشن"""
    global recommender, products_cache
    
    # بارگذاری اولیه (فقط کش محصولات - توصیه‌ها از Redis خوانده می‌شوند)
    print("شروع بارگذاری API...")
    
    # کش محصولات برای پاسخ سریع
    products = load_products()
    products_cache = {p.id: p for p in products}
    print(f"✅ {len(products_cache)} محصول کش شد")
    
    # تست اتصال به Redis
    storage = init_redis_storage()
    if storage and storage.test_connection():
        print("✅ اتصال به Redis برقرار است")
    else:
        print("⚠️  Redis در دسترس نیست - استفاده از fallback")
    
    yield
    
    # پاکسازی
    print("بستن API...")


app = FastAPI(
    title="سیستم توصیه محصولات",
    description="API برای دریافت توصیه‌های محصولات کاربرمحور",
    version="1.0.0",
    lifespan=lifespan
)


def get_recommender() -> Optional[HybridRecommender]:
    """دریافت نمونه سیستم توصیه (برای fallback)"""
    global recommender
    if recommender is None:
        try:
            recommender = HybridRecommender()
            recommender.train()
        except:
            pass
    return recommender


def get_product_info(product_id: int) -> Dict[str, Any]:
    """دریافت اطلاعات محصول"""
    product = products_cache.get(product_id)
    if not product:
        return {}
    
    return {
        "product_title": product.title,
        "product_price": product.sale_price,
        "product_stock": product.stock_quantity
    }


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
        "recommender_ready": recommender is not None,
        "products_loaded": len(products_cache)
    }


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
            storage = init_redis_storage()
            if storage and storage.exists(user_id):
                recs_dict = storage.get_recommendations(user_id)
                
                response = []
                for rec in recs_dict[:limit]:
                    product_info = get_product_info(rec['product_id'])
                    
                    # Parse collaborative_details if exists
                    collaborative_details = None
                    if rec.get('collaborative_details'):
                        import json
                        try:
                            collaborative_details = json.loads(rec['collaborative_details'])
                        except:
                            pass
                    
                    response.append(RecommendationResponse(
                        product_id=rec['product_id'],
                        score=rec['score'],
                        reason=rec['reason'],
                        confidence=rec['confidence'],
                        product_title=product_info.get("product_title"),
                        product_price=product_info.get("product_price"),
                        product_stock=product_info.get("product_stock"),
                        collaborative_details=collaborative_details
                    ))
                
                if response:
                    return response
        
        # Fallback: اگر Redis در دسترس نباشد یا توصیه وجود نداشته باشد
        recommender_instance = get_recommender()
        if recommender_instance:
            recommendations = recommender_instance.get_recommendations(user_id, limit)
            
            response = []
            for rec in recommendations:
                product_info = get_product_info(rec.product_id)
                
                # Parse collaborative_details if exists
                collaborative_details = None
                if rec.collaborative_details:
                    import json
                    try:
                        collaborative_details = json.loads(rec.collaborative_details)
                    except:
                        pass
                
                response.append(RecommendationResponse(
                    product_id=rec.product_id,
                    score=rec.score,
                    reason=rec.reason,
                    confidence=rec.confidence,
                    product_title=product_info.get("product_title"),
                    product_price=product_info.get("product_price"),
                    product_stock=product_info.get("product_stock"),
                    collaborative_details=collaborative_details
                ))
            
            return response
        else:
            raise HTTPException(
                status_code=503, 
                detail="توصیه‌ای برای این کاربر یافت نشد و سیستم در حال بارگذاری است"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت توصیه‌ها: {str(e)}")


@app.get("/insights/{user_id}", response_model=UserInsightsResponse)
async def get_user_insights(
    user_id: int,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت بینش‌های کاربر"""
    try:
        insights = recommender_instance.get_user_insights(user_id)
        return UserInsightsResponse(**insights)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت بینش‌ها: {str(e)}")


@app.get("/popular", response_model=List[PopularProductsResponse])
async def get_popular_products(
    limit: int = 10,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت محصولات محبوب"""
    try:
        popular_products = recommender_instance.get_popular_products(limit)
        
        response = []
        for product_id, count in popular_products:
            product_info = get_product_info(product_id)
            response.append(PopularProductsResponse(
                product_id=product_id,
                purchase_count=count,
                product_title=product_info.get("product_title"),
                product_price=product_info.get("product_price")
            ))
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت محصولات محبوب: {str(e)}")


@app.get("/similar/{product_id}", response_model=List[SimilarProductsResponse])
async def get_similar_products(
    product_id: int,
    limit: int = 5,
    recommender_instance: HybridRecommender = Depends(get_recommender)
):
    """دریافت محصولات مشابه"""
    try:
        similar_products = recommender_instance.get_similar_products(product_id, limit)
        
        response = []
        for similar_product_id, similarity in similar_products:
            product_info = get_product_info(similar_product_id)
            response.append(SimilarProductsResponse(
                product_id=similar_product_id,
                similarity_score=similarity,
                product_title=product_info.get("product_title"),
                product_price=product_info.get("product_price")
            ))
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در دریافت محصولات مشابه: {str(e)}")


@app.post("/retrain")
async def retrain_model():
    """بازآموزی مدل"""
    global recommender, products_cache
    
    try:
        print("شروع بازآموزی مدل...")
        recommender = HybridRecommender()
        recommender.train()
        
        # به‌روزرسانی کش محصولات
        products = load_products()
        products_cache = {p.id: p for p in products}
        
        return {
            "message": "مدل با موفقیت بازآموزی شد",
            "products_count": len(products_cache)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در بازآموزی مدل: {str(e)}")


@app.get("/stats")
async def get_system_stats():
    """دریافت آمار سیستم"""
    # دریافت آمار Redis
    redis_stats = {}
    storage = init_redis_storage()
    if storage and storage.test_connection():
        redis_stats = storage.get_stats()
    
    return {
        "total_products": len(products_cache),
        "recommender_ready": recommender is not None,
        "redis_connected": storage is not None and storage.test_connection() if storage else False,
        "redis_stats": redis_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



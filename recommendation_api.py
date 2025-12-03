"""
API برای سیستم توصیه محصولات

این ماژول endpointهای REST API را برای دریافت توصیه‌های محصولات ارائه می‌دهد.
"""
from __future__ import annotations
import json
import logging
import uuid
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
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


class JobStatus(str, Enum):
    """وضعیت job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateRecommendationsRequest(BaseModel):
    """درخواست تولید توصیه"""
    user_ids: List[int]
    top_k: int = 20


class JobResponse(BaseModel):
    """پاسخ job"""
    job_id: str
    status: JobStatus
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_users: int = 0
    processed_users: int = 0
    failed_users: int = 0
    error: Optional[str] = None


# State management برای مدل و cache
class AppState:
    """مدیریت state اپلیکیشن"""
    def __init__(self):
        self.recommender: Optional[HybridRecommender] = None
        self.products_cache: Dict[int, Product] = {}
        self.storage = None
        # Job tracking
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()
    
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


def _update_job_status(job_id: str, **kwargs):
    """به‌روزرسانی وضعیت job"""
    with app_state.jobs_lock:
        if job_id in app_state.jobs:
            app_state.jobs[job_id].update(kwargs)
            app_state.jobs[job_id]['updated_at'] = datetime.now().isoformat()


def _generate_recommendations_background(job_id: str, user_ids: List[int], top_k: int):
    """تولید توصیه در پس‌زمینه"""
    try:
        _update_job_status(job_id, status=JobStatus.RUNNING, started_at=datetime.now().isoformat())
        logger.info(f"Job {job_id}: Starting recommendation generation for {len(user_ids)} users")
        
        # Import here to avoid circular imports
        import sys
        import os
        # اضافه کردن مسیر فعلی به sys.path برای import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from generate_recommendations import generate_recommendations_for_users, main_for_specific_users
        
        # اجرای تولید توصیه
        # استفاده از main_for_specific_users که همه چیز را مدیریت می‌کند
        main_for_specific_users(user_ids, top_k=top_k)
        
        _update_job_status(
            job_id,
            status=JobStatus.COMPLETED,
            completed_at=datetime.now().isoformat(),
            processed_users=len(user_ids),
            message=f"توصیه‌ها با موفقیت برای {len(user_ids)} کاربر تولید شد"
        )
        logger.info(f"Job {job_id}: Completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Job {job_id}: Failed with error: {error_msg}", exc_info=True)
        _update_job_status(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            error=error_msg,
            message=f"خطا در تولید توصیه: {error_msg}"
        )


@app.post("/generate-recommendations", response_model=JobResponse)
async def generate_recommendations(
    request: GenerateRecommendationsRequest,
    background_tasks: BackgroundTasks
):
    """
    تولید توصیه برای کاربران مشخص در پس‌زمینه
    
    این endpoint فوراً برمی‌گردد و فرآیند تولید توصیه در پس‌زمینه اجرا می‌شود.
    برای بررسی وضعیت از endpoint /job-status/{job_id} استفاده کنید.
    """
    if not request.user_ids:
        raise HTTPException(status_code=400, detail="لیست user_id ها نمی‌تواند خالی باشد")
    
    if request.top_k <= 0 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k باید بین 1 تا 100 باشد")
    
    # ایجاد job ID
    job_id = str(uuid.uuid4())
    
    # ثبت job
    with app_state.jobs_lock:
        app_state.jobs[job_id] = {
            'job_id': job_id,
            'status': JobStatus.PENDING,
            'message': 'در انتظار شروع...',
            'created_at': datetime.now().isoformat(),
            'total_users': len(request.user_ids),
            'processed_users': 0,
            'failed_users': 0,
            'user_ids': request.user_ids,
            'top_k': request.top_k
        }
    
    # اضافه کردن task به پس‌زمینه
    background_tasks.add_task(
        _generate_recommendations_background,
        job_id,
        request.user_ids,
        request.top_k
    )
    
    logger.info(f"Job {job_id} created for {len(request.user_ids)} users")
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="درخواست ثبت شد و در حال پردازش است",
        created_at=app_state.jobs[job_id]['created_at'],
        total_users=len(request.user_ids)
    )


@app.get("/job-status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """بررسی وضعیت job"""
    with app_state.jobs_lock:
        job = app_state.jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} یافت نشد")
    
    return JobResponse(
        job_id=job['job_id'],
        status=JobStatus(job['status']),
        message=job.get('message', ''),
        created_at=job['created_at'],
        started_at=job.get('started_at'),
        completed_at=job.get('completed_at'),
        total_users=job.get('total_users', 0),
        processed_users=job.get('processed_users', 0),
        failed_users=job.get('failed_users', 0),
        error=job.get('error')
    )


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(limit: int = 10):
    """لیست آخرین job ها"""
    with app_state.jobs_lock:
        # مرتب‌سازی بر اساس created_at (جدیدترین اول)
        sorted_jobs = sorted(
            app_state.jobs.values(),
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )[:limit]
    
    return [
        JobResponse(
            job_id=job['job_id'],
            status=JobStatus(job['status']),
            message=job.get('message', ''),
            created_at=job['created_at'],
            started_at=job.get('started_at'),
            completed_at=job.get('completed_at'),
            total_users=job.get('total_users', 0),
            processed_users=job.get('processed_users', 0),
            failed_users=job.get('failed_users', 0),
            error=job.get('error')
        )
        for job in sorted_jobs
    ]


@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """حذف job (فقط job های completed یا failed)"""
    with app_state.jobs_lock:
        job = app_state.jobs.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} یافت نشد")
        
        status = JobStatus(job['status'])
        if status in [JobStatus.RUNNING, JobStatus.PENDING]:
            raise HTTPException(
                status_code=400,
                detail=f"نمی‌توان job در حال اجرا (status: {status}) را حذف کرد"
            )
        
        del app_state.jobs[job_id]
        logger.info(f"Job {job_id} deleted")
    
    return {"message": f"Job {job_id} حذف شد"}


class UsersWithoutRecommendationsResponse(BaseModel):
    """پاسخ لیست کاربران بدون توصیه"""
    total_users: int
    users_with_recommendations: int
    users_without_recommendations: int
    user_ids_without_recommendations: List[int]
    percentage_with_recommendations: float
    percentage_without_recommendations: float


@app.get("/users-without-recommendations", response_model=UsersWithoutRecommendationsResponse)
async def get_users_without_recommendations(
    limit: Optional[int] = None,
    check_all: bool = False
):
    """
    دریافت لیست کاربرانی که توصیه برایشان ایجاد نشده است
    
    Args:
        limit: محدود کردن تعداد کاربران برای بررسی (None = همه)
        check_all: اگر True باشد، همه کاربران را بررسی می‌کند (ممکن است کند باشد)
    
    Returns:
        لیست user_id های کاربران بدون توصیه
    """
    try:
        # بارگذاری کاربران از دیتابیس
        from generate_recommendations import load_users_from_db
        import polars as pl
        
        logger.info("Loading users from database...")
        users_df = load_users_from_db()
        
        if users_df.is_empty():
            raise HTTPException(status_code=404, detail="هیچ کاربری یافت نشد")
        
        # محدود کردن اگر limit مشخص شده
        if limit and limit < len(users_df):
            users_df = users_df.head(limit)
            logger.info(f"Limited to {limit} users for checking")
        
        # دریافت storage
        storage = app_state.init_redis_storage()
        if not storage or not storage.test_connection():
            raise HTTPException(
                status_code=503,
                detail="Redis در دسترس نیست. برای بررسی نیاز به Redis است."
            )
        
        # بررسی وجود توصیه برای هر کاربر
        logger.info(f"Checking recommendations for {len(users_df)} users...")
        user_ids = users_df['id'].to_list()
        
        users_with_recommendations = []
        users_without_recommendations = []
        
        # بررسی به صورت batch برای سرعت بیشتر
        batch_size = 100
        for i in range(0, len(user_ids), batch_size):
            batch = user_ids[i:i + batch_size]
            
            for user_id in batch:
                if storage.exists(user_id):
                    users_with_recommendations.append(user_id)
                else:
                    users_without_recommendations.append(user_id)
            
            # Log progress
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(user_ids):
                logger.info(f"Checked {min(i + batch_size, len(user_ids))}/{len(user_ids)} users...")
        
        total_users = len(user_ids)
        users_with_count = len(users_with_recommendations)
        users_without_count = len(users_without_recommendations)
        
        percentage_with = (users_with_count / total_users * 100) if total_users > 0 else 0
        percentage_without = (users_without_count / total_users * 100) if total_users > 0 else 0
        
        logger.info(
            f"Summary: {users_with_count} with recommendations, "
            f"{users_without_count} without ({percentage_without:.1f}%)"
        )
        
        return UsersWithoutRecommendationsResponse(
            total_users=total_users,
            users_with_recommendations=users_with_count,
            users_without_recommendations=users_without_count,
            user_ids_without_recommendations=users_without_recommendations,
            percentage_with_recommendations=round(percentage_with, 2),
            percentage_without_recommendations=round(percentage_without, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users without recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"خطا در دریافت لیست کاربران: {str(e)}"
        )


@app.get("/users-without-recommendations/count")
async def get_users_without_recommendations_count():
    """
    دریافت تعداد کاربران بدون توصیه (سریع‌تر از endpoint کامل)
    
    این endpoint فقط تعداد را برمی‌گرداند و لیست user_id ها را نمی‌دهد.
    """
    try:
        from generate_recommendations import load_users_from_db
        
        users_df = load_users_from_db()
        if users_df.is_empty():
            return {
                "total_users": 0,
                "users_without_recommendations": 0
            }
        
        storage = app_state.init_redis_storage()
        if not storage or not storage.test_connection():
            raise HTTPException(
                status_code=503,
                detail="Redis در دسترس نیست"
            )
        
        user_ids = users_df['id'].to_list()
        users_without_count = 0
        
        # بررسی سریع (نمونه‌گیری)
        sample_size = min(1000, len(user_ids))
        sample_ids = user_ids[:sample_size]
        
        for user_id in sample_ids:
            if not storage.exists(user_id):
                users_without_count += 1
        
        # تخمین برای کل
        estimated_percentage = (users_without_count / sample_size * 100) if sample_size > 0 else 0
        estimated_total_without = int(len(user_ids) * estimated_percentage / 100)
        
        return {
            "total_users": len(user_ids),
            "sample_size": sample_size,
            "users_without_in_sample": users_without_count,
            "estimated_percentage_without": round(estimated_percentage, 2),
            "estimated_users_without_recommendations": estimated_total_without,
            "note": "این یک تخمین است. برای لیست دقیق از /users-without-recommendations استفاده کنید"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error counting users without recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"خطا در شمارش کاربران: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



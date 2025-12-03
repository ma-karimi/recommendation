#!/usr/bin/env python3
"""
اسکریپت تولید توصیه‌ها برای همه کاربران

این اسکریپت:
1. داده‌های کاربران، محصولات و سفارشات را از دیتابیس می‌خواند
2. داده‌های Matomo را از فایل‌های parquet می‌خواند
3. سیستم توصیه را آموزش می‌دهد
4. برای همه کاربران توصیه تولید می‌کند
5. نتایج را در فایل parquet و Redis ذخیره می‌کند
"""
from __future__ import annotations
import datetime as dt
import glob
import logging
import os
import gc
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import polars as pl
from sqlalchemy import text

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from dataframe_loader import get_engine, load_order_items
from hybrid_recommender import HybridRecommender
from models import Product, ProductInteraction, User
from object_loader import load_products, load_users
from settings import load_config

# تنظیم logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_users_from_db() -> pl.DataFrame:
    """بارگذاری کاربران از دیتابیس به صورت DataFrame با retry logic"""
    from dataframe_loader import get_engine, reset_engine
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            engine = get_engine(force_new=(attempt > 0))
            sql = text("""
                SELECT id, email, 
                       CONCAT(COALESCE(first_name, ''), ' ', COALESCE(last_name, '')) as name,
                       created_at
                FROM users
                ORDER BY id
            """)
            
            with engine.connect() as conn:
                rows = [dict(row) for row in conn.execute(sql).mappings()]
            
            if not rows:
                return pl.DataFrame()
            
            return pl.DataFrame(rows)
        except Exception as e:
            if "Packet sequence" in str(e) or "InternalError" in str(e):
                if attempt < max_retries - 1:
                    logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}), resetting connection...")
                    reset_engine()
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise
            else:
                raise


def load_products_from_db() -> pl.DataFrame:
    """بارگذاری محصولات از دیتابیس به صورت DataFrame با retry logic"""
    from dataframe_loader import get_engine, reset_engine
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            engine = get_engine(force_new=(attempt > 0))
            sql = text("""
                SELECT id, title, slug, sku, sale_price, stock_quantity, 
                       status, published_at, seller_id, category_id
                FROM products
                WHERE deleted_at IS NULL AND status = 1
                ORDER BY id
            """)
            
            with engine.connect() as conn:
                rows = [dict(row) for row in conn.execute(sql).mappings()]
            
            if not rows:
                return pl.DataFrame()
            
            return pl.DataFrame(rows)
        except Exception as e:
            if "Packet sequence" in str(e) or "InternalError" in str(e):
                if attempt < max_retries - 1:
                    logger.warning(f"Database connection error (attempt {attempt + 1}/{max_retries}), resetting connection...")
                    reset_engine()
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise
            else:
                raise


def create_user_product_interactions(order_items_df: pl.DataFrame) -> List[ProductInteraction]:
    """ایجاد لیست تعاملات کاربر-محصول از داده‌های سفارشات"""
    interactions = []
    
    if order_items_df.is_empty():
        return interactions
    
    # تبدیل به دیکشنری برای پردازش سریع‌تر
    for row in order_items_df.iter_rows(named=True):
        interaction = ProductInteraction(
            user_id=row['order_user_id'],
            product_id=row['product_id'],
            interaction_type='purchase',
            timestamp=row['order_created_at'],
            value=float(row['total_price'])
        )
        interactions.append(interaction)
    
    logger.info(f"Extracted {len(interactions)} purchase interactions")
    return interactions


def load_matomo_product_popularity() -> Dict[int, float]:
    """بارگذاری محبوبیت محصولات از داده‌های Matomo"""
    cfg = load_config()
    pageviews_files = sorted(glob.glob(
        os.path.join(cfg.output_dir, "matomo_pageviews_*.parquet")
    ))
    
    if not pageviews_files:
        logger.warning("Matomo pageviews file not found")
        return {}
    
    # استفاده از آخرین فایل
    latest_file = pageviews_files[-1]
    df = pl.read_parquet(latest_file)
    
    product_popularity = {}
    
    # فیلتر کردن صفحات محصول و استخراج محبوبیت
    if 'label' in df.columns and 'nb_visits' in df.columns:
        product_rows = df.filter(pl.col('label') == 'product')
        
        for row in product_rows.iter_rows(named=True):
            # استخراج product_id از URL یا label
            # این قسمت نیاز به تنظیم بر اساس ساختار URL‌های شما دارد
            # فعلاً فقط تعداد بازدید کلی محصولات را نگه می‌داریم
            popularity_score = float(row['nb_visits'])
            product_popularity[0] = popularity_score  # کلیدی برای محصولات عمومی
    
    logger.info("Matomo popularity data loaded")
    return product_popularity


def generate_recommendations_for_users(
    recommender: HybridRecommender,
    user_ids: List[int],
    top_k: int = 20
) -> pl.DataFrame:
    """
    تولید توصیه برای لیست مشخصی از کاربران
    
    Args:
        recommender: مدل توصیه‌گر (باید قبلاً train شده باشد)
        user_ids: لیست ID های کاربران
        top_k: تعداد توصیه برای هر کاربر
        
    Returns:
        DataFrame شامل توصیه‌ها
    """
    if not user_ids:
        logger.warning("No user IDs provided")
        return pl.DataFrame()
    
    recommendations_data = []
    
    logger.info(f"Starting recommendation generation for {len(user_ids)} specific users...")
    
    # شمارنده برای نمایش پیشرفت
    total_users = len(user_ids)
    users_with_recommendations = 0
    users_without_recommendations = 0
    
    for idx, user_id in enumerate(user_ids, 1):
        # نمایش پیشرفت
        if idx % 10 == 0 or idx == total_users:
            logger.info(f"Processing user {idx}/{total_users} (User ID: {user_id})...")
        
        try:
            # دریافت توصیه‌ها
            recommendations = recommender.get_recommendations(user_id, top_k)
            
            if recommendations:
                users_with_recommendations += 1
                
                # اضافه کردن توصیه‌ها به لیست
                for rank, rec in enumerate(recommendations, 1):
                    recommendations_data.append({
                        'user_id': user_id,
                        'product_id': rec.product_id,
                        'score': rec.score,
                        'rank': rank,
                        'confidence': rec.confidence,
                        'reason': rec.reason,
                        'collaborative_details': rec.collaborative_details,
                        'generated_at': dt.datetime.now()
                    })
            else:
                users_without_recommendations += 1
                logger.debug(f"No recommendations for user {user_id}")
                
        except Exception as e:
            users_without_recommendations += 1
            logger.warning(f"Error for user {user_id}: {e}")
    
    logger.info(
        f"Summary: {users_with_recommendations} users with recommendations, "
        f"{users_without_recommendations} without. "
        f"Total recommendations: {len(recommendations_data)}"
    )
    
    if not recommendations_data:
        logger.error("No recommendations generated!")
        return pl.DataFrame()
    
    # تبدیل به DataFrame
    recommendations_df = pl.DataFrame(recommendations_data)
    
    return recommendations_df


def generate_recommendations_for_all_users(
    recommender: HybridRecommender,
    users_df: pl.DataFrame,
    top_k: int = 20,
    sample_size: int = None
) -> pl.DataFrame:
    """
    تولید توصیه برای کاربران
    
    Args:
        recommender: مدل توصیه‌گر
        users_df: DataFrame کاربران
        top_k: تعداد توصیه برای هر کاربر
        sample_size: تعداد کاربران برای تست (None = همه کاربران)
    """
    
    # محدود کردن به sample اگر مشخص شده
    if sample_size and sample_size < len(users_df):
        users_df = users_df.head(sample_size)
        logger.warning(f"Test mode: Only processing first {sample_size} users")
    
    recommendations_data = []
    
    logger.info(f"Starting recommendation generation for {len(users_df)} users...")
    
    # شمارنده برای نمایش پیشرفت
    total_users = len(users_df)
    users_with_recommendations = 0
    users_without_recommendations = 0
    
    for idx, row in enumerate(users_df.iter_rows(named=True), 1):
        user_id = row['id']
        
        # نمایش پیشرفت
        if idx % 100 == 0 or idx == total_users:
            logger.info(f"Processing user {idx}/{total_users} (User ID: {user_id})...")
        
        try:
            # دریافت توصیه‌ها
            recommendations = recommender.get_recommendations(user_id, top_k)
            
            if recommendations:
                users_with_recommendations += 1
                
                # اضافه کردن توصیه‌ها به لیست
                for rank, rec in enumerate(recommendations, 1):
                    recommendations_data.append({
                        'user_id': user_id,
                        'product_id': rec.product_id,
                        'score': rec.score,
                        'rank': rank,
                        'confidence': rec.confidence,
                        'reason': rec.reason,
                        'collaborative_details': rec.collaborative_details,
                        'generated_at': dt.datetime.now()
                    })
            else:
                users_without_recommendations += 1
                
        except Exception as e:
            users_without_recommendations += 1
            if idx <= 10:  # فقط 10 خطای اول را نمایش می‌دهیم
                logger.warning(f"Error for user {user_id}: {e}")
    
    logger.info(
        f"Summary: {users_with_recommendations} users with recommendations, "
        f"{users_without_recommendations} without. "
        f"Total recommendations: {len(recommendations_data)}"
    )
    
    if not recommendations_data:
        logger.error("No recommendations generated!")
        return pl.DataFrame()
    
    # تبدیل به DataFrame
    recommendations_df = pl.DataFrame(recommendations_data)
    
    return recommendations_df


def save_recommendations(recommendations_df: pl.DataFrame, output_dir: str) -> str:
    """ذخیره توصیه‌ها در فایل parquet"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"user_recommendations_{timestamp}.parquet")
    
    recommendations_df.write_parquet(output_file)
    logger.info(f"Recommendations saved to: {output_file}")
    
    # ذخیره نسخه CSV برای بررسی راحت‌تر
    csv_file = output_file.replace('.parquet', '.csv')
    recommendations_df.write_csv(csv_file)
    logger.info(f"CSV version saved to: {csv_file}")
    
    return output_file


def print_sample_recommendations(recommendations_df: pl.DataFrame, products_df: pl.DataFrame, n_users: int = 5):
    """نمایش نمونه‌ای از توصیه‌ها"""
    if recommendations_df.is_empty():
        return
    
    # تبدیل محصولات به دیکشنری برای جستجوی سریع
    products_dict = {}
    for row in products_df.iter_rows(named=True):
        products_dict[row['id']] = row['title']
    
    print(f"\n{'='*80}")
    print(f"نمونه توصیه‌ها برای {n_users} کاربر اول:")
    print(f"{'='*80}\n")
    
    # گروه‌بندی بر اساس user_id
    unique_users = recommendations_df['user_id'].unique().sort()[:n_users]
    
    for user_id in unique_users:
        user_recs = recommendations_df.filter(pl.col('user_id') == user_id).sort('rank')
        
        print(f"کاربر {user_id} - {len(user_recs)} توصیه:")
        print("-" * 80)
        
        # نمایش 5 توصیه اول
        for row in user_recs.head(5).iter_rows(named=True):
            product_title = products_dict.get(row['product_id'], f"محصول {row['product_id']}")
            print(f"  {row['rank']}. {product_title}")
            print(f"     امتیاز: {row['score']:.4f} | اطمینان: {row['confidence']:.2f}")
            print(f"     دلیل: {row['reason'][:100]}...")
            
            # نمایش جزئیات collaborative اگر وجود داشته باشد
            if 'collaborative_details' in row and row['collaborative_details']:
                print(f"     جزئیات: {row['collaborative_details'][:150]}...")
        
        print()


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0

def log_memory_usage(stage: str):
    """Log memory usage at a specific stage"""
    if PSUTIL_AVAILABLE:
        mem_mb = get_memory_usage_mb()
        print(f"   💾 استفاده از حافظه ({stage}): {mem_mb:.1f} MB")
        return mem_mb
    return 0.0

def main(sample_size: int = None):
    """
    تابع اصلی
    
    Args:
        sample_size: تعداد کاربران برای تست (None = همه کاربران)
    """
    print("="*80)
    print("سیستم تولید توصیه محصولات")
    if sample_size:
        print(f"🧪 حالت تست - {sample_size} کاربر")
    else:
        print("🚀 حالت کامل - همه کاربران")
    print("="*80)
    print()
    
    cfg = load_config()
    
    # 1. بارگذاری کاربران
    print("📥 بارگذاری کاربران از دیتابیس...")
    initial_memory = log_memory_usage("شروع")
    users_df = load_users_from_db()
    if users_df.is_empty():
        print("❌ هیچ کاربری یافت نشد!")
        return
    print(f"✅ {len(users_df)} کاربر بارگذاری شد")
    log_memory_usage("بعد از بارگذاری کاربران")
    
    # 2. بارگذاری محصولات
    print("\n📥 بارگذاری محصولات از دیتابیس...")
    products_df = load_products_from_db()
    if products_df.is_empty():
        print("❌ هیچ محصولی یافت نشد!")
        return
    print(f"✅ {len(products_df)} محصول بارگذاری شد")
    log_memory_usage("بعد از بارگذاری محصولات")
    
    # 3. بارگذاری سفارشات (آخر 180 روز)
    print("\n📥 بارگذاری سفارشات از دیتابیس...")
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=180)  # 6 ماه گذشته
    
    order_items_df = load_order_items(start_date, end_date)
    if order_items_df.is_empty():
        print("⚠️  هیچ سفارشی در 180 روز گذشته یافت نشد!")
        print("   تلاش برای بارگذاری تمام سفارشات...")
        # سعی می‌کنیم تمام سفارشات را بارگذاری کنیم
        start_date = dt.date(2020, 1, 1)
        order_items_df = load_order_items(start_date, end_date)
        
        if order_items_df.is_empty():
            print("❌ هیچ سفارشی یافت نشد! سیستم توصیه نیاز به داده دارد.")
            return
    
    print(f"✅ {len(order_items_df)} آیتم سفارش بارگذاری شد")
    print(f"   بازه زمانی: {start_date} تا {end_date}")
    
    # 4. ایجاد تعاملات کاربر-محصول
    print("\n🔄 ایجاد ماتریس تعاملات کاربر-محصول...")
    interactions = create_user_product_interactions(order_items_df)
    
    if not interactions:
        print("❌ هیچ تعاملی ایجاد نشد!")
        return
    
    # 5. بارگذاری محبوبیت محصولات از Matomo (اختیاری)
    print("\n📥 بارگذاری داده‌های محبوبیت از Matomo...")
    matomo_popularity = load_matomo_product_popularity()
    
    # 6. تبدیل داده‌ها به فرمت مدل‌ها (به صورت lazy برای صرفه‌جویی در حافظه)
    print("\n🔄 تبدیل داده‌ها به فرمت مدل‌ها...")
    
    # فقط محصولات مورد نیاز را بارگذاری می‌کنیم (نه همه)
    # برای content-based فقط محصولاتی که در تعاملات هستند
    products_in_interactions = set()
    for interaction in interactions:
        products_in_interactions.add(interaction.product_id)
    
    print(f"   تعداد محصولات منحصر به فرد در تعاملات: {len(products_in_interactions)}")
    
    # فقط محصولات مرتبط را بارگذاری می‌کنیم
    products_list = []
    products_dict = {}  # برای دسترسی سریع
    for row in products_df.iter_rows(named=True):
        product_id = row['id']
        if product_id in products_in_interactions:
            from models import Product
            product = Product(
                id=product_id,
                title=row['title'],
                slug=row['slug'],
                sku=row['sku'],
                sale_price=float(row['sale_price'] or 0),
                stock_quantity=int(row['stock_quantity'] or 0),
                status='published' if row['status'] == 1 else 'draft',
                published_at=row.get('published_at'),
                seller_id=row.get('seller_id'),
                category_id=row.get('category_id')
            )
            products_list.append(product)
            products_dict[product_id] = product
    
    # فیلتر کردن products_df برای فقط محصولاتی که در تعاملات هستند (برای استفاده بعدی)
    # این باعث صرفه‌جویی در حافظه می‌شود
    products_df = products_df.filter(pl.col('id').is_in(list(products_in_interactions)))
    
    import gc
    gc.collect()
    
    print(f"   ✅ {len(products_list)} محصول مرتبط بارگذاری شد")
    
    # گروه‌بندی تعاملات بر اساس user_id (قبل از استفاده در users_df)
    user_interactions = defaultdict(list)
    for interaction in interactions:
        user_interactions[interaction.user_id].append(interaction)
    
    # کاربران را به صورت lazy نگه می‌داریم (فقط ID ها)
    users_dict = {}
    for row in users_df.iter_rows(named=True):
        from models import User
        user = User(
            id=row['id'],
            email=row.get('email'),
            name=row.get('name'),
            created_at=row.get('created_at')
        )
        users_dict[user.id] = user
    
    print(f"   ✅ {len(users_dict)} کاربر بارگذاری شد")
    
    # نگه داشتن users_df برای استفاده بعدی در generate_recommendations_for_all_users
    # اما فقط برای کاربرانی که در تعاملات هستند (برای صرفه‌جویی در حافظه)
    users_with_interactions = set(user_interactions.keys())
    if len(users_with_interactions) < len(users_df):
        # فیلتر کردن users_df برای فقط کاربرانی که تعامل دارند
        users_df = users_df.filter(pl.col('id').is_in(list(users_with_interactions)))
        print(f"   ✅ فیلتر شد به {len(users_df)} کاربر با تعامل")
    
    # اگر sample_size مشخص شده، فقط همان تعداد را نگه می‌داریم
    if sample_size and sample_size < len(users_df):
        users_df = users_df.head(sample_size)
        print(f"   ✅ محدود شد به {len(users_df)} کاربر (sample_size)")
    
    # 7. آموزش سیستم توصیه
    print("\n🧠 آموزش سیستم توصیه...")
    print("   این ممکن است چند دقیقه طول بکشد...")
    
    recommender = HybridRecommender()
    
    # تنظیم داده‌ها - فقط محصولات و کاربران مرتبط
    recommender.users = list(users_dict.values())
    recommender.products = products_list
    
    # user_interactions قبلاً ساخته شده
    recommender.user_interactions = dict(user_interactions)
    
    print(f"   تعداد کاربران با تعامل: {len(user_interactions)}")
    
    # آموزش مدل‌ها
    try:
        from collaborative_filtering import train_collaborative_model
        from content_based_filtering import train_content_based_model
        
        print("   🔹 آموزش مدل Collaborative Filtering...")
        recommender.collaborative_model = train_collaborative_model(interactions)
        
        # پاک کردن ماتریس از حافظه بعد از ذخیره
        if recommender.collaborative_model and recommender.collaborative_model.use_storage:
            if recommender.collaborative_model.user_item_matrix is not None:
                del recommender.collaborative_model.user_item_matrix
            if recommender.collaborative_model.user_similarities is not None:
                del recommender.collaborative_model.user_similarities
            gc.collect()
            print("   ✅ ماتریس‌ها در DuckDB ذخیره شد و از حافظه پاک شد")
        
        print("   🔹 آموزش مدل Content-Based Filtering...")
        recommender.content_model = train_content_based_model(products_list, user_interactions)
        
        # پاک کردن داده‌های موقت از حافظه
        if recommender.content_model:
            # Product features در storage ذخیره شده، از حافظه پاک می‌کنیم
            if hasattr(recommender.content_model, 'product_features'):
                del recommender.content_model.product_features
            gc.collect()
        
        print("✅ سیستم توصیه با موفقیت آموزش داده شد!")
        log_memory_usage("بعد از آموزش")
        
        # پاک‌سازی نهایی
        gc.collect()
        log_memory_usage("بعد از پاک‌سازی")
        
    except Exception as e:
        print(f"❌ خطا در آموزش مدل: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. تولید توصیه برای کاربران
    if sample_size:
        print(f"\n🎯 تولید توصیه برای {sample_size} کاربر (نمونه تست)...")
    else:
        print("\n🎯 تولید توصیه برای همه کاربران...")
    
    recommendations_df = generate_recommendations_for_all_users(
        recommender,
        users_df,
        top_k=20,  # 20 توصیه برای هر کاربر
        sample_size=sample_size
    )
    
    if recommendations_df.is_empty():
        print("❌ هیچ توصیه‌ای تولید نشد!")
        return
    
    # 9. ذخیره توصیه‌ها
    print("\n💾 ذخیره توصیه‌ها...")
    
    # ذخیره در فایل (backup)
    output_file = save_recommendations(recommendations_df, cfg.output_dir)
    
    # ذخیره در DuckDB (persistent storage)
    try:
        if recommender.storage:
            recommender.storage.save_recommendations_batch(recommendations_df, overwrite=True)
            duckdb_stats = recommender.storage.get_recommendations_stats()
            print(f"\n📊 آمار DuckDB:")
            print(f"   تعداد کل توصیه‌ها: {duckdb_stats['total_recommendations']:,}")
            print(f"   تعداد کاربران با توصیه: {duckdb_stats['users_with_recommendations']:,}")
            if duckdb_stats.get('last_generated_at'):
                print(f"   آخرین تولید: {duckdb_stats['last_generated_at']}")
        else:
            print("⚠️  ModelStorage در دسترس نیست - توصیه‌ها در DuckDB ذخیره نشد")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره DuckDB: {e}")
        import traceback
        traceback.print_exc()
    
    # ذخیره در Redis (cache)
    try:
        from recommendation_storage import get_storage
        storage = get_storage()
        
        if storage.test_connection():
            stats = storage.store_batch_from_dataframe(recommendations_df, batch_size=1000)
            storage_stats = storage.get_stats()
            print(f"\n📊 آمار Redis (Cache):")
            print(f"   تعداد توصیه‌ها در حافظه: {storage_stats['total_recommendations']}")
            print(f"   استفاده از حافظه: {storage_stats['memory_usage_mb']} MB")
        else:
            print("⚠️  Redis در دسترس نیست - فقط DuckDB و فایل ذخیره شد")
    except ImportError:
        print("⚠️  ماژول recommendation_storage پیدا نشد - فقط DuckDB و فایل ذخیره شد")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره Redis: {e}")
        print("   ✅ DuckDB و فایل‌ها به درستی ذخیره شدند")
    
    # 10. نمایش نمونه توصیه‌ها
    # اطمینان از وجود products_df (برای جلوگیری از خطا)
    try:
        _ = len(products_df)
    except (NameError, UnboundLocalError):
        # اگر products_df وجود ندارد، دوباره بارگذاری می‌کنیم (فقط برای نمایش)
        print("⚠️  بارگذاری مجدد products_df برای نمایش...")
        products_df = load_products_from_db()
        # فیلتر کردن فقط محصولات مرتبط
        products_in_interactions = set()
        for interaction in interactions:
            products_in_interactions.add(interaction.product_id)
        if products_in_interactions:
            products_df = products_df.filter(pl.col('id').is_in(list(products_in_interactions)))
    
    print_sample_recommendations(recommendations_df, products_df, n_users=5)
    
    # 11. آمار نهایی
    print(f"\n{'='*80}")
    print("✅ فرآیند تولید توصیه با موفقیت کامل شد!")
    print(f"{'='*80}")
    print(f"📊 آمار نهایی:")
    print(f"   تعداد کاربران: {len(users_df)}")
    try:
        print(f"   تعداد محصولات: {len(products_df)}")
    except (NameError, UnboundLocalError):
        print(f"   تعداد محصولات: N/A")
    try:
        print(f"   تعداد سفارشات: {len(order_items_df)}")
    except (NameError, UnboundLocalError):
        print(f"   تعداد سفارشات: N/A")
    print(f"   تعداد کل توصیه‌ها: {len(recommendations_df)}")
    print(f"   فایل خروجی: {output_file}")
    print(f"{'='*80}\n")


def main_for_specific_users(user_ids: List[int], top_k: int = 20):
    """
    تولید توصیه برای کاربران مشخص (بدون train کردن مجدد مدل)
    
    Args:
        user_ids: لیست ID های کاربران
        top_k: تعداد توصیه برای هر کاربر
    """
    print("="*80)
    print("سیستم تولید توصیه محصولات")
    print(f"🎯 حالت کاربران مشخص - {len(user_ids)} کاربر")
    print("="*80)
    print()
    
    cfg = load_config()
    
    # 1. بارگذاری محصولات (برای فیلترهای تجاری)
    print("📥 بارگذاری محصولات از دیتابیس...")
    products_df = load_products_from_db()
    if products_df.is_empty():
        print("❌ هیچ محصولی یافت نشد!")
        return
    print(f"✅ {len(products_df)} محصول بارگذاری شد")
    
    # 2. تبدیل محصولات به لیست
    products_list = []
    for row in products_df.iter_rows(named=True):
        from models import Product
        product = Product(
            id=row['id'],
            title=row['title'],
            slug=row['slug'],
            sku=row['sku'],
            sale_price=float(row['sale_price'] or 0),
            stock_quantity=int(row['stock_quantity'] or 0),
            status='published' if row['status'] == 1 else 'draft',
            published_at=row.get('published_at'),
            seller_id=row.get('seller_id'),
            category_id=row.get('category_id')
        )
        products_list.append(product)
    
    # 3. Initialize recommender (از storage استفاده می‌کند)
    print("\n🔄 بارگذاری مدل از storage...")
    print("   (مدل باید قبلاً train شده باشد)")
    
    recommender = HybridRecommender(use_storage=True)
    
    # بارگذاری داده‌های پایه (بدون train کردن)
    recommender.users = load_users()
    recommender.products = products_list
    
    # بارگذاری تعاملات کاربران (فقط برای کاربران مشخص)
    print(f"📥 بارگذاری تعاملات برای {len(user_ids)} کاربر...")
    # فقط تعاملات کاربران مشخص را بارگذاری می‌کنیم
    recommender.user_interactions = {}
    from object_loader import load_user_purchase_history
    for user_id in user_ids:
        purchase_history = load_user_purchase_history(user_id, days_back=365)
        recommender.user_interactions[user_id] = purchase_history
    
    # بارگذاری مدل‌های train شده از storage
    try:
        from collaborative_filtering import CollaborativeFiltering
        from content_based_filtering import ContentBasedFiltering
        
        # بارگذاری collaborative model از storage
        print("   🔹 بارگذاری مدل Collaborative Filtering از storage...")
        recommender.collaborative_model = CollaborativeFiltering(use_storage=True, storage=recommender.storage)
        
        # بارگذاری mappings از storage
        if recommender.storage:
            conn = recommender.storage._get_connection(read_only=True)
            
            # بارگذاری user mappings
            user_mappings = conn.execute("SELECT user_id, user_index FROM user_index_mapping").fetchall()
            recommender.collaborative_model.user_to_index = {row[0]: row[1] for row in user_mappings}
            recommender.collaborative_model.index_to_user = {row[1]: row[0] for row in user_mappings}
            
            # بارگذاری product mappings
            product_mappings = conn.execute("SELECT product_id, product_index FROM product_index_mapping").fetchall()
            recommender.collaborative_model.product_to_index = {row[0]: row[1] for row in product_mappings}
            recommender.collaborative_model.index_to_product = {row[1]: row[0] for row in product_mappings}
            
            logger.info(f"Loaded {len(recommender.collaborative_model.user_to_index)} user mappings and {len(recommender.collaborative_model.product_to_index)} product mappings")
        
        # بارگذاری content-based model از storage
        print("   🔹 بارگذاری مدل Content-Based Filtering از storage...")
        recommender.content_model = ContentBasedFiltering(use_storage=True, storage=recommender.storage)
        
        # بارگذاری ANN index
        if not recommender.content_model._load_ann_index():
            print("⚠️  ANN index یافت نشد. ممکن است مدل هنوز train نشده باشد.")
            print("   لطفاً ابتدا مدل را train کنید:")
            print("   python generate_recommendations.py --sample 100")
            return
        
        # بارگذاری user profiles از storage (اگر وجود داشته باشد)
        if recommender.storage:
            try:
                conn = recommender.storage._get_connection(read_only=True)
                result = conn.execute("SELECT user_id, profile_data FROM user_profiles").fetchall()
                user_profiles = {}
                for row in result:
                    import pickle
                    user_profiles[row[0]] = pickle.loads(row[1])
                
                if user_profiles:
                    recommender.content_model.user_profiles = user_profiles
                    logger.info(f"Loaded {len(user_profiles)} user profiles from storage")
            except Exception as e:
                logger.warning(f"Could not load user profiles: {e}")
        
        print("✅ مدل‌ها از storage بارگذاری شدند!")
        
    except Exception as e:
        print(f"⚠️  خطا در بارگذاری مدل از storage: {e}")
        import traceback
        traceback.print_exc()
        print("\n   ممکن است مدل هنوز train نشده باشد. لطفاً ابتدا مدل را train کنید:")
        print("   python generate_recommendations.py --sample 100")
        return
    
    # 4. تولید توصیه برای کاربران مشخص
    print(f"\n🎯 تولید توصیه برای {len(user_ids)} کاربر...")
    
    recommendations_df = generate_recommendations_for_users(
        recommender,
        user_ids,
        top_k=top_k
    )
    
    if recommendations_df.is_empty():
        print("❌ هیچ توصیه‌ای تولید نشد!")
        return
    
    # 5. ذخیره توصیه‌ها
    print("\n💾 ذخیره توصیه‌ها...")
    
    # ذخیره در فایل
    output_file = save_recommendations(recommendations_df, cfg.output_dir)
    
    # ذخیره در DuckDB (persistent storage)
    try:
        if recommender.storage:
            recommender.storage.save_recommendations_batch(recommendations_df, overwrite=True)
            duckdb_stats = recommender.storage.get_recommendations_stats()
            print(f"\n📊 آمار DuckDB:")
            print(f"   تعداد کل توصیه‌ها: {duckdb_stats['total_recommendations']:,}")
            print(f"   تعداد کاربران با توصیه: {duckdb_stats['users_with_recommendations']:,}")
            if duckdb_stats.get('last_generated_at'):
                print(f"   آخرین تولید: {duckdb_stats['last_generated_at']}")
        else:
            print("⚠️  ModelStorage در دسترس نیست - توصیه‌ها در DuckDB ذخیره نشد")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره DuckDB: {e}")
        import traceback
        traceback.print_exc()
    
    # ذخیره در Redis (cache)
    try:
        from recommendation_storage import get_storage
        storage = get_storage()
        
        if storage.test_connection():
            stats = storage.store_batch_from_dataframe(recommendations_df, batch_size=1000)
            storage_stats = storage.get_stats()
            print(f"\n📊 آمار Redis (Cache):")
            print(f"   تعداد توصیه‌ها در حافظه: {storage_stats['total_recommendations']}")
            print(f"   استفاده از حافظه: {storage_stats['memory_usage_mb']} MB")
        else:
            print("⚠️  Redis در دسترس نیست - فقط DuckDB و فایل ذخیره شد")
    except ImportError:
        print("⚠️  ماژول recommendation_storage پیدا نشد - فقط DuckDB و فایل ذخیره شد")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره Redis: {e}")
        print("   ✅ DuckDB و فایل‌ها به درستی ذخیره شدند")
    
    # 6. نمایش نمونه توصیه‌ها
    print_sample_recommendations(recommendations_df, products_df, n_users=min(5, len(user_ids)))
    
    # 7. آمار نهایی
    print(f"\n{'='*80}")
    print("✅ فرآیند تولید توصیه با موفقیت کامل شد!")
    print(f"{'='*80}")
    print(f"📊 آمار نهایی:")
    print(f"   تعداد کاربران: {len(user_ids)}")
    print(f"   تعداد محصولات: {len(products_df)}")
    print(f"   تعداد کل توصیه‌ها: {len(recommendations_df)}")
    print(f"   فایل خروجی: {output_file}")
    print(f"{'='*80}\n")


def find_users_without_recommendations(limit: Optional[int] = None, output_file: Optional[str] = None) -> List[int]:
    """
    پیدا کردن کاربرانی که توصیه برایشان ایجاد نشده است
    
    Args:
        limit: محدود کردن تعداد کاربران برای بررسی (None = همه)
        output_file: مسیر فایل برای ذخیره لیست (اختیاری)
        
    Returns:
        لیست user_id های کاربران بدون توصیه
    """
    print("="*80)
    print("جستجوی کاربران بدون توصیه")
    print("="*80)
    print()
    
    # 1. بارگذاری کاربران
    print("📥 بارگذاری کاربران از دیتابیس...")
    users_df = load_users_from_db()
    if users_df.is_empty():
        print("❌ هیچ کاربری یافت نشد!")
        return []
    
    print(f"✅ {len(users_df)} کاربر بارگذاری شد")
    
    # محدود کردن اگر limit مشخص شده
    if limit and limit < len(users_df):
        users_df = users_df.head(limit)
        print(f"⚠️  محدود شد به {limit} کاربر برای بررسی")
    
    # 2. بررسی وجود توصیه در Redis
    print("\n🔍 بررسی وجود توصیه‌ها در Redis...")
    try:
        from recommendation_storage import get_storage
        storage = get_storage()
        
        if not storage.test_connection():
            print("❌ Redis در دسترس نیست!")
            return []
        
        print("✅ اتصال به Redis برقرار شد")
        
    except ImportError:
        print("❌ ماژول recommendation_storage پیدا نشد!")
        return []
    except Exception as e:
        print(f"❌ خطا در اتصال به Redis: {e}")
        return []
    
    # 3. بررسی هر کاربر
    user_ids = users_df['id'].to_list()
    users_with_recommendations = []
    users_without_recommendations = []
    
    total_users = len(user_ids)
    print(f"\n📊 بررسی {total_users} کاربر...")
    
    # بررسی به صورت batch
    batch_size = 100
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        
        for user_id in batch:
            if storage.exists(user_id):
                users_with_recommendations.append(user_id)
            else:
                users_without_recommendations.append(user_id)
        
        # نمایش پیشرفت
        checked = min(i + batch_size, total_users)
        if checked % 1000 == 0 or checked == total_users:
            print(f"   بررسی شده: {checked}/{total_users} ({checked/total_users*100:.1f}%)")
    
    # 4. نمایش نتایج
    print(f"\n{'='*80}")
    print("📊 نتایج:")
    print(f"{'='*80}")
    print(f"   کل کاربران بررسی شده: {total_users:,}")
    print(f"   کاربران با توصیه: {len(users_with_recommendations):,} ({len(users_with_recommendations)/total_users*100:.1f}%)")
    print(f"   کاربران بدون توصیه: {len(users_without_recommendations):,} ({len(users_without_recommendations)/total_users*100:.1f}%)")
    print(f"{'='*80}\n")
    
    # 5. ذخیره در فایل اگر مشخص شده
    if output_file and users_without_recommendations:
        import os
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # ذخیره به صورت CSV
        import polars as pl
        df = pl.DataFrame({'user_id': users_without_recommendations})
        df.write_csv(output_file)
        print(f"💾 لیست کاربران بدون توصیه در فایل ذخیره شد: {output_file}")
        
        # همچنین یک فایل txt ساده
        txt_file = output_file.replace('.csv', '.txt')
        with open(txt_file, 'w') as f:
            for user_id in users_without_recommendations:
                f.write(f"{user_id}\n")
        print(f"💾 نسخه TXT نیز ذخیره شد: {txt_file}")
    
    return users_without_recommendations


if __name__ == "__main__":
    import sys
    import argparse
    
    # تنظیم CLI arguments
    parser = argparse.ArgumentParser(
        description="تولید توصیه محصولات برای کاربران",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌های استفاده:
  # تست با 1000 کاربر
  python generate_recommendations.py --sample 1000
  
  # تست با 100 کاربر
  python generate_recommendations.py --sample 100
  
  # تولید برای همه کاربران
  python generate_recommendations.py
  python generate_recommendations.py --all
  
  # تولید برای کاربران مشخص (از command line)
  python generate_recommendations.py --users 123 456 789
  
  # تولید برای کاربران مشخص (از فایل)
  python generate_recommendations.py --users-file user_ids.txt
  
  # تولید برای یک کاربر جدید
  python generate_recommendations.py --user 12345
        """
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        metavar='N',
        help='تعداد کاربران برای تست (مثال: 1000). اگر مشخص نشود، همه کاربران پردازش می‌شوند.'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='پردازش همه کاربران (پیش‌فرض)'
    )
    
    parser.add_argument(
        '--users',
        type=int,
        nargs='+',
        metavar='USER_ID',
        help='لیست ID های کاربران برای تولید توصیه (مثال: --users 123 456 789)'
    )
    
    parser.add_argument(
        '--user',
        type=int,
        metavar='USER_ID',
        help='تولید توصیه برای یک کاربر (مثال: --user 12345)'
    )
    
    parser.add_argument(
        '--users-file',
        type=str,
        metavar='FILE',
        help='فایل حاوی لیست user_id ها (هر خط یک user_id)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        metavar='K',
        help='تعداد توصیه برای هر کاربر (پیش‌فرض: 20)'
    )
    
    parser.add_argument(
        '--find-without-recommendations',
        action='store_true',
        help='پیدا کردن کاربران بدون توصیه'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        metavar='FILE',
        help='مسیر فایل برای ذخیره لیست کاربران بدون توصیه (مثال: users_without_recs.csv)'
    )
    
    args = parser.parse_args()
    
    # اگر find-without-recommendations استفاده شده
    if args.find_without_recommendations:
        try:
            users_without = find_users_without_recommendations(
                limit=args.sample,
                output_file=args.output_file
            )
            if users_without:
                print(f"\n✅ {len(users_without)} کاربر بدون توصیه پیدا شد")
                print(f"\nبرای تولید توصیه برای این کاربران:")
                print(f"python generate_recommendations.py --users {' '.join(map(str, users_without[:10]))}")
                if len(users_without) > 10:
                    print(f"   (فقط 10 کاربر اول نمایش داده شد)")
            else:
                print("\n✅ همه کاربران توصیه دارند!")
            sys.exit(0)
        except KeyboardInterrupt:
            print("\n\n⚠️  فرآیند توسط کاربر متوقف شد")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ خطای غیرمنتظره: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # پردازش arguments
    if args.user:
        # یک کاربر
        user_ids = [args.user]
        try:
            main_for_specific_users(user_ids, top_k=args.top_k)
        except KeyboardInterrupt:
            print("\n\n⚠️  فرآیند توسط کاربر متوقف شد")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ خطای غیرمنتظره: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.users:
        # لیست کاربران از command line
        user_ids = args.users
        try:
            main_for_specific_users(user_ids, top_k=args.top_k)
        except KeyboardInterrupt:
            print("\n\n⚠️  فرآیند توسط کاربر متوقف شد")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ خطای غیرمنتظره: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    elif args.users_file:
        # لیست کاربران از فایل
        try:
            with open(args.users_file, 'r') as f:
                user_ids = [int(line.strip()) for line in f if line.strip() and not line.strip().startswith('#')]
            
            if not user_ids:
                print(f"❌ هیچ user_id معتبری در فایل {args.users_file} یافت نشد!")
                sys.exit(1)
            
            print(f"📄 بارگذاری {len(user_ids)} user_id از فایل {args.users_file}")
            try:
                main_for_specific_users(user_ids, top_k=args.top_k)
            except KeyboardInterrupt:
                print("\n\n⚠️  فرآیند توسط کاربر متوقف شد")
                sys.exit(1)
            except Exception as e:
                print(f"\n\n❌ خطای غیرمنتظره: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        except FileNotFoundError:
            print(f"❌ فایل {args.users_file} یافت نشد!")
            sys.exit(1)
        except ValueError as e:
            print(f"❌ خطا در خواندن user_id از فایل: {e}")
            sys.exit(1)
    else:
        # حالت عادی (همه کاربران یا sample)
        sample_size = None if args.all else args.sample
        
        try:
            main(sample_size=sample_size)
        except KeyboardInterrupt:
            print("\n\n⚠️  فرآیند توسط کاربر متوقف شد")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ خطای غیرمنتظره: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


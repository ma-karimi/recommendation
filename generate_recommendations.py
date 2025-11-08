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
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import polars as pl
from sqlalchemy import text

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
    """بارگذاری کاربران از دیتابیس به صورت DataFrame"""
    engine = get_engine()
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


def load_products_from_db() -> pl.DataFrame:
    """بارگذاری محصولات از دیتابیس به صورت DataFrame"""
    engine = get_engine()
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
    users_df = load_users_from_db()
    if users_df.is_empty():
        print("❌ هیچ کاربری یافت نشد!")
        return
    print(f"✅ {len(users_df)} کاربر بارگذاری شد")
    
    # 2. بارگذاری محصولات
    print("\n📥 بارگذاری محصولات از دیتابیس...")
    products_df = load_products_from_db()
    if products_df.is_empty():
        print("❌ هیچ محصولی یافت نشد!")
        return
    print(f"✅ {len(products_df)} محصول بارگذاری شد")
    
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
    
    # 6. تبدیل داده‌ها به فرمت مدل‌ها
    print("\n🔄 تبدیل داده‌ها به فرمت مدل‌ها...")
    users_list = []
    for row in users_df.iter_rows(named=True):
        from models import User
        users_list.append(User(
            id=row['id'],
            email=row.get('email'),
            name=row.get('name'),
            created_at=row.get('created_at')
        ))
    
    products_list = []
    for row in products_df.iter_rows(named=True):
        from models import Product
        products_list.append(Product(
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
        ))
    
    # 7. آموزش سیستم توصیه
    print("\n🧠 آموزش سیستم توصیه...")
    print("   این ممکن است چند دقیقه طول بکشد...")
    
    recommender = HybridRecommender()
    
    # تنظیم داده‌ها به صورت دستی
    recommender.users = users_list
    recommender.products = products_list
    
    # گروه‌بندی تعاملات بر اساس user_id
    user_interactions = defaultdict(list)
    for interaction in interactions:
        user_interactions[interaction.user_id].append(interaction)
    
    recommender.user_interactions = dict(user_interactions)
    
    print(f"   تعداد کاربران با تعامل: {len(user_interactions)}")
    
    # آموزش مدل‌ها
    try:
        from collaborative_filtering import train_collaborative_model
        from content_based_filtering import train_content_based_model
        
        print("   🔹 آموزش مدل Collaborative Filtering...")
        recommender.collaborative_model = train_collaborative_model(interactions)
        
        print("   🔹 آموزش مدل Content-Based Filtering...")
        # استفاده از Sparse Matrix برای صرفه‌جویی در حافظه
        recommender.content_model = train_content_based_model(
            products_list, 
            user_interactions,
            use_sparse=True,  # استفاده از Sparse Matrix
            max_similar_products=50  # حداکثر 50 محصول مشابه برای هر محصول
        )
        
        print("✅ سیستم توصیه با موفقیت آموزش داده شد!")
        
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
    
    # ذخیره در Redis
    try:
        from recommendation_storage import get_storage
        storage = get_storage()
        
        if storage.test_connection():
            stats = storage.store_batch_from_dataframe(recommendations_df, batch_size=1000)
            storage_stats = storage.get_stats()
            print(f"\n📊 آمار Redis:")
            print(f"   تعداد توصیه‌ها در حافظه: {storage_stats['total_recommendations']}")
            print(f"   استفاده از حافظه: {storage_stats['memory_usage_mb']} MB")
        else:
            print("⚠️  Redis در دسترس نیست - فقط فایل ذخیره شد")
    except ImportError:
        print("⚠️  ماژول recommendation_storage پیدا نشد - فقط فایل ذخیره شد")
    except Exception as e:
        print(f"⚠️  خطا در ذخیره Redis: {e}")
        print("   ✅ فایل‌ها به درستی ذخیره شدند")
    
    # 10. نمایش نمونه توصیه‌ها
    print_sample_recommendations(recommendations_df, products_df, n_users=5)
    
    # 11. آمار نهایی
    print(f"\n{'='*80}")
    print("✅ فرآیند تولید توصیه با موفقیت کامل شد!")
    print(f"{'='*80}")
    print(f"📊 آمار نهایی:")
    print(f"   تعداد کاربران: {len(users_df)}")
    print(f"   تعداد محصولات: {len(products_df)}")
    print(f"   تعداد سفارشات: {len(order_items_df)}")
    print(f"   تعداد کل توصیه‌ها: {len(recommendations_df)}")
    print(f"   فایل خروجی: {output_file}")
    print(f"{'='*80}\n")


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
    
    args = parser.parse_args()
    
    # اگر --all استفاده شده، sample_size رو None می‌کنیم
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


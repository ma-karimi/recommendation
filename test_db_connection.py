#!/usr/bin/env python3
"""
اسکریپت تست اتصال دیتابیس و بررسی داده‌های موجود
"""

from __future__ import annotations
import sys

def test_database_connection():
    """تست اتصال به دیتابیس"""
    print("="*60)
    print("تست اتصال دیتابیس")
    print("="*60)
    
    try:
        from dataframe_loader import get_engine
        from sqlalchemy import text
        
        print("\n1️⃣  تست اتصال...")
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            print("   ✅ اتصال برقرار است")
        
        # تست جداول
        print("\n2️⃣  بررسی جداول...")
        
        with engine.connect() as conn:
            # بررسی جدول users
            result = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
            print(f"   ✅ جدول users: {result} کاربر")
            
            # بررسی جدول products
            result = conn.execute(text("SELECT COUNT(*) FROM products WHERE deleted_at IS NULL AND status = 1")).scalar()
            print(f"   ✅ جدول products (فعال): {result} محصول")
            
            # بررسی جدول orders
            result = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()
            print(f"   ✅ جدول orders: {result} سفارش")
            
            # بررسی جدول order_items
            result = conn.execute(text("SELECT COUNT(*) FROM order_items")).scalar()
            print(f"   ✅ جدول order_items: {result} آیتم")
            
            # بررسی سفارشات تکمیل شده
            result = conn.execute(text("SELECT COUNT(*) FROM orders WHERE status = 'completed'")).scalar()
            print(f"   ✅ سفارشات تکمیل شده: {result}")
        
        print("\n3️⃣  بررسی داده‌های نمونه...")
        
        with engine.connect() as conn:
            # نمایش اولین کاربر
            result = conn.execute(text(
                "SELECT id, email, CONCAT(COALESCE(first_name, ''), ' ', COALESCE(last_name, '')) as name FROM users LIMIT 1"
            )).mappings().first()
            if result:
                print(f"   📌 نمونه کاربر: ID={result['id']}, Email={result['email']}, Name={result['name']}")
            
            # نمایش اولین محصول
            result = conn.execute(text(
                "SELECT id, title, sale_price, stock_quantity FROM products WHERE deleted_at IS NULL AND status = 1 LIMIT 1"
            )).mappings().first()
            if result:
                print(f"   📌 نمونه محصول: ID={result['id']}, Title={result['title'][:50]}")
        
        print("\n" + "="*60)
        print("✅ همه چیز آماده است!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ خطا: {e}")
        print("\n💡 راهنمایی:")
        print("   1. بررسی کنید فایل .env وجود دارد")
        print("   2. متغیرهای DB_* یا RECO_DB_URL را تنظیم کنید")
        print("   3. اطمینان حاصل کنید دیتابیس در دسترس است")
        return False


def test_matomo_files():
    """بررسی فایل‌های Matomo"""
    print("\n" + "="*60)
    print("بررسی فایل‌های Matomo")
    print("="*60)
    
    try:
        from settings import load_config
        import os
        import glob
        
        cfg = load_config()
        output_dir = cfg.output_dir
        
        print(f"\n📂 مسیر: {output_dir}")
        
        # بررسی فایل‌های موجود
        pageviews_files = glob.glob(os.path.join(output_dir, "matomo_pageviews_*.parquet"))
        events_files = glob.glob(os.path.join(output_dir, "matomo_events_*.parquet"))
        goals_files = glob.glob(os.path.join(output_dir, "matomo_goals_*.parquet"))
        
        if pageviews_files:
            print(f"   ✅ فایل‌های pageviews: {len(pageviews_files)}")
            print(f"      آخرین: {os.path.basename(pageviews_files[-1])}")
        else:
            print("   ⚠️  هیچ فایل pageviews یافت نشد")
        
        if events_files:
            print(f"   ✅ فایل‌های events: {len(events_files)}")
        else:
            print("   ⚠️  هیچ فایل events یافت نشد")
        
        if goals_files:
            print(f"   ✅ فایل‌های goals: {len(goals_files)}")
        else:
            print("   ⚠️  هیچ فایل goals یافت نشد")
        
        if not (pageviews_files or events_files or goals_files):
            print("\n💡 برای دریافت داده‌های Matomo:")
            print("   python pipeline.py --start 2024-01-01 --end 2024-12-31")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  خطا در بررسی Matomo: {e}")
        return False


def main():
    """تابع اصلی"""
    print("\n" + "🔍 تست سیستم توصیه محصولات" + "\n")
    
    # تست اتصال دیتابیس
    db_ok = test_database_connection()
    
    # تست فایل‌های Matomo
    matomo_ok = test_matomo_files()
    
    print("\n" + "="*60)
    if db_ok:
        print("✅ سیستم آماده برای تولید توصیه است!")
        print("\nبرای شروع:")
        print("   python generate_recommendations.py")
    else:
        print("❌ لطفاً مشکلات را برطرف کنید و دوباره تلاش کنید")
        sys.exit(1)
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  متوقف شد")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ خطای غیرمنتظره: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


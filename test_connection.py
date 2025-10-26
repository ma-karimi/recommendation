#!/usr/bin/env python3
"""
اسکریپت تست اتصال به پایگاه داده
"""
import os
from sqlalchemy import create_engine, text
from settings import load_config


def test_database_connection():
    """تست اتصال به پایگاه داده"""
    print("🔍 بررسی تنظیمات پایگاه داده...")
    
    try:
        cfg = load_config()
        print(f"✅ تنظیمات بارگذاری شد")
        
        db_url = cfg.db.url
        if not db_url:
            print("❌ RECO_DB_URL تنظیم نشده است")
            print("لطفاً در فایل .env مقدار زیر را اضافه کنید:")
            print("RECO_DB_URL=mysql+pymysql://username:password@localhost:3306/database_name?charset=utf8mb4")
            return False
        
        print(f"📡 تلاش برای اتصال به: {db_url.split('@')[1] if '@' in db_url else 'نامشخص'}")
        
        # ایجاد اتصال
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_timeout=30,
            echo=False
        )
        
        # تست اتصال
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            
        if test_value == 1:
            print("✅ اتصال به پایگاه داده موفق!")
            
            # تست جداول مورد نیاز
            print("\n🔍 بررسی جداول مورد نیاز...")
            required_tables = ['users', 'products', 'orders', 'order_items']
            
            with engine.connect() as conn:
                for table in required_tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table} LIMIT 1"))
                        count = result.fetchone()[0]
                        print(f"  ✅ جدول {table}: {count} رکورد")
                    except Exception as e:
                        print(f"  ❌ جدول {table}: خطا - {e}")
            
            return True
        else:
            print("❌ تست اتصال ناموفق")
            return False
            
    except Exception as e:
        print(f"❌ خطا در اتصال: {e}")
        print("\nراه‌حل‌های ممکن:")
        print("1. بررسی نام کاربری و رمز عبور")
        print("2. بررسی آدرس سرور و پورت")
        print("3. بررسی نام پایگاه داده")
        print("4. اطمینان از اجرای MySQL")
        return False


def create_sample_env():
    """ایجاد فایل .env نمونه"""
    env_content = """# Matomo Analytics
MATOMO_BASE_URL=https://your-matomo.example.com
MATOMO_SITE_ID=1
MATOMO_TOKEN_AUTH=your_token_here
MATOMO_VERIFY_SSL=true

# Database - لطفاً مقادیر زیر را با اطلاعات پایگاه داده خود جایگزین کنید
RECO_DB_URL=mysql+pymysql://username:password@localhost:3306/database_name?charset=utf8mb4

# Storage Path
STORAGE_PATH=storage/app/recommendation
"""
    
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"📝 فایل {env_path} ایجاد شد")
        print("لطفاً مقادیر پایگاه داده را در این فایل تنظیم کنید")
    else:
        print(f"📝 فایل {env_path} از قبل وجود دارد")


if __name__ == "__main__":
    print("🚀 تست اتصال به پایگاه داده\n")
    
    # بررسی وجود فایل .env
    if not os.path.exists(".env"):
        print("❌ فایل .env یافت نشد")
        create_sample_env()
        print("\nلطفاً فایل .env را ویرایش کرده و دوباره اجرا کنید")
        exit(1)
    
    # تست اتصال
    success = test_database_connection()
    
    if success:
        print("\n🎉 اتصال موفق! می‌توانید سیستم توصیه را اجرا کنید")
    else:
        print("\n❌ اتصال ناموفق. لطفاً تنظیمات را بررسی کنید")



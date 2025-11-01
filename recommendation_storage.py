"""
recommendation_storage.py - ذخیره و بازیابی توصیه‌ها از Redis

این ماژول برای ذخیره و بازیابی توصیه‌های محصولات از Redis استفاده می‌کند.

ساختار کلیدهای Redis:
- recommendation:{user_id} -> JSON array با 20 توصیه برتر
- recommendation_meta:{user_id} -> metadata (تاریخ تولید، تعداد، etc.)

مزایای Redis:
1. سرعت بالا (O(1) read/write)
2. TTL خودکار (expiration)
3. حافظه بهینه
4. پشتیبانی از JSON
5. بهترین انتخاب برای caching
"""
from __future__ import annotations
import json
import datetime as dt
from typing import List, Optional, Dict, Any
import polars as pl

try:
    import redis
    from redis.connection import ConnectionPool
except ImportError:
    redis = None
    ConnectionPool = None

from models import Recommendation


class RecommendationStorage:
    """کلاس مدیریت ذخیره و بازیابی توصیه‌ها از Redis"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
        ttl_seconds: int = 604800  # 7 روز (پیش‌فرض)
    ):
        """
        Args:
            host: آدرس سرور Redis
            port: پورت Redis
            db: شماره دیتابیس Redis (0-15)
            password: رمز عبور Redis (اگر نیاز باشد)
            decode_responses: تبدیل پاسخ‌ها به string
            ttl_seconds: مدت زمان انقضا به ثانیه
        """
        if redis is None:
            raise ImportError(
                "Redis is not installed. Install it with: pip install redis"
            )
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl_seconds = ttl_seconds
        
        # ایجاد connection pool برای عملکرد بهتر
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses,
            max_connections=50
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
    
    def _get_key(self, user_id: int) -> str:
        """ساخت کلید Redis برای کاربر"""
        return f"recommendation:{user_id}"
    
    def _get_meta_key(self, user_id: int) -> str:
        """ساخت کلید metadata برای کاربر"""
        return f"recommendation_meta:{user_id}"
    
    def store_recommendations(
        self,
        user_id: int,
        recommendations: List[Recommendation],
        overwrite: bool = True
    ) -> bool:
        """
        ذخیره توصیه‌ها برای یک کاربر
        
        Args:
            user_id: شناسه کاربر
            recommendations: لیست توصیه‌ها
            overwrite: آیا توصیه‌های قبلی را بازنویسی کند؟
        
        Returns:
            True اگر موفق بود، False اگر خطا داشت
        """
        try:
            key = self._get_key(user_id)
            meta_key = self._get_meta_key(user_id)
            
            # بررسی وجود توصیه‌های قبلی
            if not overwrite and self.client.exists(key):
                return False
            
            # تبدیل Recommendation objects به dict
            recs_dict = []
            for rec in recommendations:
                rec_dict = {
                    'product_id': rec.product_id,
                    'score': float(rec.score),
                    'reason': rec.reason,
                    'confidence': float(rec.confidence),
                    'collaborative_details': rec.collaborative_details
                }
                recs_dict.append(rec_dict)
            
            # ذخیره توصیه‌ها
            self.client.setex(
                key,
                self.ttl_seconds,
                json.dumps(recs_dict, ensure_ascii=False)
            )
            
            # ذخیره metadata
            metadata = {
                'user_id': user_id,
                'count': len(recommendations),
                'generated_at': dt.datetime.now().isoformat(),
                'ttl_seconds': self.ttl_seconds
            }
            self.client.setex(
                meta_key,
                self.ttl_seconds,
                json.dumps(metadata, ensure_ascii=False)
            )
            
            return True
            
        except Exception as e:
            print(f"❌ خطا در ذخیره توصیه‌های کاربر {user_id}: {e}")
            return False
    
    def get_recommendations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        دریافت توصیه‌های یک کاربر
        
        Args:
            user_id: شناسه کاربر
        
        Returns:
            لیست توصیه‌ها (dict) یا [] اگر وجود نداشت
        """
        try:
            key = self._get_key(user_id)
            data = self.client.get(key)
            
            if data is None:
                return []
            
            recommendations = json.loads(data)
            return recommendations
            
        except Exception as e:
            print(f"❌ خطا در دریافت توصیه‌های کاربر {user_id}: {e}")
            return []
    
    def get_metadata(self, user_id: int) -> Optional[Dict[str, Any]]:
        """دریافت metadata توصیه‌های یک کاربر"""
        try:
            meta_key = self._get_meta_key(user_id)
            data = self.client.get(meta_key)
            
            if data is None:
                return None
            
            return json.loads(data)
            
        except Exception as e:
            print(f"❌ خطا در دریافت metadata کاربر {user_id}: {e}")
            return None
    
    def exists(self, user_id: int) -> bool:
        """بررسی وجود توصیه‌ها برای کاربر"""
        try:
            key = self._get_key(user_id)
            return self.client.exists(key) > 0
        except:
            return False
    
    def delete(self, user_id: int) -> bool:
        """حذف توصیه‌های کاربر"""
        try:
            key = self._get_key(user_id)
            meta_key = self._get_meta_key(user_id)
            
            self.client.delete(key)
            self.client.delete(meta_key)
            
            return True
        except Exception as e:
            print(f"❌ خطا در حذف توصیه‌های کاربر {user_id}: {e}")
            return False
    
    def store_batch_from_dataframe(
        self,
        recommendations_df: pl.DataFrame,
        batch_size: int = 1000
    ) -> Dict[str, int]:
        """
        ذخیره توصیه‌ها از DataFrame به صورت batch
        
        Args:
            recommendations_df: DataFrame با ستون‌های user_id, product_id, score, etc.
            batch_size: تعداد کاربران در هر batch
        
        Returns:
            آمار ذخیره‌سازی: {success_count, failed_count}
        """
        stats = {'success_count': 0, 'failed_count': 0, 'total_users': 0}
        
        if recommendations_df.is_empty():
            return stats
        
        # گروه‌بندی بر اساس user_id
        users = recommendations_df['user_id'].unique().sort()
        stats['total_users'] = len(users)
        
        print(f"\n💾 شروع ذخیره {len(users)} کاربر در Redis...")
        
        # پردازش batch به batch
        for i in range(0, len(users), batch_size):
            batch_users = users[i:i+batch_size]
            
            for user_id in batch_users:
                user_recs_df = recommendations_df.filter(
                    pl.col('user_id') == user_id
                ).sort('rank')
                
                # تبدیل به لیست Recommendation
                recommendations = []
                for row in user_recs_df.iter_rows(named=True):
                    rec = Recommendation(
                        user_id=int(row['user_id']),
                        product_id=int(row['product_id']),
                        score=float(row['score']),
                        reason=row.get('reason', ''),
                        confidence=float(row.get('confidence', 0.0)),
                        collaborative_details=row.get('collaborative_details')
                    )
                    recommendations.append(rec)
                
                # ذخیره
                if self.store_recommendations(user_id, recommendations):
                    stats['success_count'] += 1
                else:
                    stats['failed_count'] += 1
                
                # نمایش پیشرفت
                if len(recommendations) > 0:
                    print(f"  ✅ کاربر {user_id}: {len(recommendations)} توصیه ذخیره شد", end='\r')
            
            # نمایش progress
            processed = min(i + batch_size, len(users))
            print(f"\n  📊 پیشرفت: {processed}/{len(users)} کاربر ({processed*100//len(users)}%)")
        
        print(f"\n{'='*60}")
        print(f"✅ ذخیره‌سازی تکمیل شد!")
        print(f"  ✅ موفق: {stats['success_count']} کاربر")
        print(f"  ❌ ناموفق: {stats['failed_count']} کاربر")
        print(f"{'='*60}\n")
        
        return stats
    
    def get_stats(self) -> Dict[str, int]:
        """دریافت آمار کلی از Redis"""
        try:
            keys = self.client.keys("recommendation:*")
            return {
                'total_recommendations': len(keys),
                'memory_usage_mb': round(
                    self.client.info('memory')['used_memory'] / 1024 / 1024, 2
                )
            }
        except:
            return {'total_recommendations': 0, 'memory_usage_mb': 0}
    
    def test_connection(self) -> bool:
        """تست اتصال به Redis"""
        try:
            self.client.ping()
            print("✅ اتصال به Redis برقرار است")
            return True
        except Exception as e:
            print(f"❌ خطا در اتصال به Redis: {e}")
            return False
    
    def close(self):
        """بستن اتصال"""
        try:
            self.client.close()
        except:
            pass


def get_storage() -> RecommendationStorage:
    """تابع helper برای ساخت instance از RecommendationStorage"""
    import os
    
    def _env(key: str, default: str = "") -> str:
        val = os.getenv(key, default)
        if val is None:
            return default
        return val.strip().strip('"').strip("'")
    
    host = _env("REDIS_HOST", "localhost")
    port = int(_env("REDIS_PORT", "6379"))
    db = int(_env("REDIS_DB", "0"))
    password = _env("REDIS_PASSWORD", "")
    ttl = int(_env("REDIS_TTL_SECONDS", "604800"))  # 7 روز
    
    return RecommendationStorage(
        host=host,
        port=port,
        db=db,
        password=password if password else None,
        ttl_seconds=ttl
    )


# برای استفاده در Laravel PHP
"""
Laravel Integration Example:

use Illuminate\\Support\\Facades\\Redis;

// دریافت توصیه‌های کاربر
$userId = 123;
$key = "recommendation:{$userId}";
$recommendations = json_decode(Redis::get($key), true);

if (!$recommendations) {
    // دریافت از منبع جایگزین (CSV, DB, etc.)
}

// نمایش توصیه‌ها
foreach ($recommendations as $rec) {
    echo "Product: {$rec['product_id']}\\n";
    echo "Score: {$rec['score']}\\n";
    echo "Reason: {$rec['reason']}\\n";
    echo "Confidence: {$rec['confidence']}\\n";
    
    if (!empty($rec['collaborative_details'])) {
        $details = json_decode($rec['collaborative_details'], true);
        echo "Similar Users: " . $details['total_similar_users'] . "\\n";
    }
}
"""


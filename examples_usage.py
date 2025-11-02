#!/usr/bin/env python3
"""
نمونه کدهای استفاده از توصیه‌های ذخیره شده در Redis

این فایل مثال‌های متعددی از نحوه استفاده از RecommendationStorage را نشان می‌دهد.
"""
from __future__ import annotations
import json
from typing import List, Dict, Any

from recommendation_storage import get_storage


def example_1_basic_usage():
    """مثال 1: استفاده پایه - دریافت توصیه‌های یک کاربر"""
    print("\n" + "="*70)
    print("مثال 1: دریافت توصیه‌های یک کاربر")
    print("="*70)
    
    storage = get_storage()
    
    # دریافت توصیه‌ها برای کاربر 1
    user_id = 1
    recommendations = storage.get_recommendations(user_id)
    
    if recommendations:
        print(f"\n✅ {len(recommendations)} توصیه برای کاربر {user_id} یافت شد:")
        
        for i, rec in enumerate(recommendations[:5], 1):  # نمایش 5 تا اول
            print(f"\n{i}. Product ID: {rec['product_id']}")
            print(f"   Score: {rec['score']}")
            print(f"   Confidence: {rec['confidence']}")
            print(f"   Reason: {rec['reason'][:80]}...")
            
            # نمایش جزئیات Collaborative اگر موجود باشد
            if rec.get('collaborative_details'):
                details = json.loads(rec['collaborative_details'])
                print(f"   👥 Similar Users: {details['total_similar_users']}")
                if details.get('similar_users'):
                    for sim_user in details['similar_users'][:3]:
                        print(f"      - User {sim_user['user_id']}: {sim_user['similarity_percent']:.1f}% similar")
    else:
        print(f"\n⚠️  هیچ توصیه‌ای برای کاربر {user_id} یافت نشد")
        print("   احتمالاً هنوز توصیه‌ها تولید نشده‌اند")


def example_2_check_user_exists():
    """مثال 2: بررسی وجود توصیه‌ها بدون دریافت کامل"""
    print("\n" + "="*70)
    print("مثال 2: بررسی وجود توصیه‌ها (سریع)")
    print("="*70)
    
    storage = get_storage()
    
    # تست چند کاربر
    test_users = [1, 9194798,9194809,9194445]
    
    for user_id in test_users:
        exists = storage.exists(user_id)
        status = "✅ موجود" if exists else "❌ موجود نیست"
        print(f"کاربر {user_id}: {status}")


def example_3_get_metadata():
    """مثال 3: دریافت metadata توصیه‌ها"""
    print("\n" + "="*70)
    print("مثال 3: دریافت metadata")
    print("="*70)
    
    storage = get_storage()
    
    # دریافت metadata برای کاربر 1
    user_id = 1
    metadata = storage.get_metadata(user_id)
    
    if metadata:
        print(f"\n📋 Metadata برای کاربر {user_id}:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    else:
        print(f"\n⚠️  هیچ metadata‌ای برای کاربر {user_id} یافت نشد")


def example_4_batch_query():
    """مثال 4: دریافت توصیه‌های چند کاربر به صورت batch"""
    print("\n" + "="*70)
    print("مثال 4: دریافت توصیه‌های batch (چند کاربر)")
    print("="*70)
    
    storage = get_storage()
    
    # لیست کاربران (با ID های واقعی)
    import redis
    client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    keys = client.keys('recommendation:*')
    user_ids = [int(key.split(':')[1]) for key in keys[:3]]
    
    if not user_ids:
        print("\n⚠️  هیچ کاربری با توصیه یافت نشد")
        return
    
    print(f"\nدریافت توصیه‌ها برای {len(user_ids)} کاربر...")
    
    results = {}
    for user_id in user_ids:
        recommendations = storage.get_recommendations(user_id)
        if recommendations:
            results[user_id] = recommendations[:5]  # فقط 5 تا اول
            print(f"  ✅ کاربر {user_id}: {len(recommendations)} توصیه")
        else:
            print(f"  ⚠️  کاربر {user_id}: بدون توصیه")
    
    # نمایش خلاصه
    print(f"\n📊 خلاصه:")
    print(f"   کاربران با توصیه: {len(results)}")
    total_recs = sum(len(recs) for recs in results.values())
    print(f"   تعداد کل توصیه‌ها: {total_recs}")


def example_5_filter_by_score():
    """مثال 5: فیلتر کردن توصیه‌ها بر اساس امتیاز"""
    print("\n" + "="*70)
    print("مثال 5: فیلتر توصیه‌ها (امتیاز بالا)")
    print("="*70)
    
    storage = get_storage()
    
    user_id = 1
    recommendations = storage.get_recommendations(user_id)
    
    if recommendations:
        # فیلتر فقط توصیه‌های با امتیاز بالا
        min_score = 4.0
        high_score_recs = [
            rec for rec in recommendations 
            if rec['score'] >= min_score
        ]
        
        print(f"\n✅ {len(high_score_recs)} توصیه با امتیاز >= {min_score} یافت شد:")
        
        for i, rec in enumerate(high_score_recs[:5], 1):
            print(f"\n{i}. Product {rec['product_id']}: Score {rec['score']:.2f}")
            print(f"   {rec['reason'][:70]}...")


def example_6_get_top_products():
    """مثال 6: دریافت محصولات برتر از همه کاربران"""
    print("\n" + "="*70)
    print("مثال 6: محاسبه محبوب‌ترین محصولات (sample)")
    print("="*70)
    
    storage = get_storage()
    
    # نمونه: بررسی 100 کاربر اول
    product_counts = {}
    
    for user_id in range(1, 101):  # 100 کاربر
        if storage.exists(user_id):
            recommendations = storage.get_recommendations(user_id)
            for rec in recommendations:
                product_id = rec['product_id']
                product_counts[product_id] = product_counts.get(product_id, 0) + 1
    
    # مرتب‌سازی
    top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 محبوب‌ترین محصولات (از 100 کاربر نمونه):")
    for i, (product_id, count) in enumerate(top_products[:10], 1):
        print(f"{i}. Product {product_id}: {count} توصیه")


def example_7_get_stats():
    """مثال 7: دریافت آمار کلی Redis"""
    print("\n" + "="*70)
    print("مثال 7: آمار کلی Redis")
    print("="*70)
    
    storage = get_storage()
    
    if storage.test_connection():
        stats = storage.get_stats()
        print(f"\n📊 آمار سیستم:")
        print(f"   تعداد توصیه\u200cها: {stats['total_recommendations']}")
        print(f"   مصرف حافظه: {stats['memory_usage_mb']} MB")
        
        # محاسبه آمار تخمینی
        if stats['total_recommendations'] > 0:
            avg_per_user = 20  # فرض
            estimated_users = stats['total_recommendations'] / avg_per_user
            print(f"   کاربران تخمینی: ~{estimated_users:.0f}")


def example_8_usage_in_api():
    """مثال 8: شبیه‌سازی استفاده در API"""
    print("\n" + "="*70)
    print("مثال 8: استفاده در API (نمونه)")
    print("="*70)
    
    storage = get_storage()
    
    # شبیه‌سازی درخواست API
    def api_get_recommendations(user_id: int, limit: int = 10):
        """تابع شبیه‌سازی API"""
        recommendations = storage.get_recommendations(user_id)
        
        if not recommendations:
            return {
                "user_id": user_id,
                "recommendations": [],
                "message": "No recommendations available"
            }
        
        # محدود کردن به limit
        results = []
        for rec in recommendations[:limit]:
            result = {
                "product_id": rec['product_id'],
                "score": float(rec['score']),
                "confidence": float(rec['confidence']),
                "reason": rec['reason']
            }
            
            # اضافه کردن collaborative_details اگر موجود باشد
            if rec.get('collaborative_details'):
                result['collaborative_details'] = json.loads(rec['collaborative_details'])
            
            results.append(result)
        
        return {
            "user_id": user_id,
            "count": len(results),
            "recommendations": results
        }
    
    # تست
    response = api_get_recommendations(user_id=1, limit=5)
    print(f"\n📤 پاسخ API برای کاربر {response['user_id']}:")
    
    if 'count' in response:
        print(f"   تعداد: {response['count']} توصیه")
        
        for i, rec in enumerate(response['recommendations'], 1):
            print(f"\n{i}. Product {rec['product_id']}")
            print(f"   Score: {rec['score']}")
            print(f"   Reason: {rec['reason'][:70]}...")
    else:
        print(f"   پیام: {response.get('message', 'بدون توصیه')}")


def main():
    """اجرای همه مثال‌ها"""
    print("\n" + "🎯"*35)
    print("نمونه کدهای استفاده از RecommendationStorage")
    print("🎯"*35)
    
    # اگر Redis در دسترس نباشد، نمایش پیام
    try:
        storage = get_storage()
        if not storage.test_connection():
            print("\n❌ Redis در دسترس نیست!")
            print("برای نصب Redis:")
            print("  macOS:   brew install redis && brew services start redis")
            print("  Linux:   sudo apt install redis-server && sudo systemctl start redis")
            return
    except Exception as e:
        print(f"\n❌ خطا در اتصال به Redis: {e}")
        return
    
    # اجرای مثال‌ها
    example_1_basic_usage()
    example_2_check_user_exists()
    example_3_get_metadata()
    example_4_batch_query()
    example_5_filter_by_score()
    example_6_get_top_products()
    example_7_get_stats()
    example_8_usage_in_api()
    
    print("\n" + "="*70)
    print("✅ همه مثال‌ها اجرا شد!")
    print("="*70)
    print("\n💡 برای استفاده در Laravel، فایل README.md را مطالعه کنید.")


if __name__ == "__main__":
    main()


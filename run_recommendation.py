#!/usr/bin/env python3
"""
اسکریپت اجرای سیستم توصیه محصولات
"""
from __future__ import annotations
import argparse
import sys
from typing import Optional

from hybrid_recommender import HybridRecommender
from recommendation_api import app
import uvicorn


def run_training():
    """اجرای آموزش مدل"""
    print("شروع آموزش سیستم توصیه...")
    
    try:
        recommender = HybridRecommender()
        recommender.train()
        print("✅ آموزش مدل با موفقیت انجام شد")
        
        # تست سیستم
        print("\nتست سیستم...")
        users = recommender.users
        if users:
            test_user = users[0]
            recommendations = recommender.get_recommendations(test_user.id, 5)
            print(f"توصیه‌های نمونه برای کاربر {test_user.id}:")
            for rec in recommendations:
                print(f"  - محصول {rec.product_id}: {rec.score:.2f} ({rec.reason})")
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در آموزش مدل: {e}")
        return False


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """اجرای API سرور"""
    print(f"شروع API سرور روی {host}:{port}")
    print("📖 مستندات API: http://localhost:8000/docs")
    print("🔍 بررسی سلامت: http://localhost:8000/health")
    
    uvicorn.run(app, host=host, port=port)


def get_recommendations_for_user(user_id: int, limit: int = 10):
    """دریافت توصیه‌های کاربر"""
    try:
        recommender = HybridRecommender()
        recommender.train()
        
        recommendations = recommender.get_recommendations(user_id, limit)
        
        print(f"توصیه‌های کاربر {user_id}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. محصول {rec.product_id}")
            print(f"   امتیاز: {rec.score:.2f}")
            print(f"   دلیل: {rec.reason}")
            print(f"   اطمینان: {rec.confidence:.2f}")
            print()
        
        return recommendations
        
    except Exception as e:
        print(f"❌ خطا در دریافت توصیه‌ها: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="سیستم توصیه محصولات")
    subparsers = parser.add_subparsers(dest="command", help="دستورات موجود")
    
    # دستور آموزش
    train_parser = subparsers.add_parser("train", help="آموزش مدل")
    train_parser.set_defaults(func=run_training)
    
    # دستور API
    api_parser = subparsers.add_parser("api", help="اجرای API سرور")
    api_parser.add_argument("--host", default="0.0.0.0", help="آدرس سرور")
    api_parser.add_argument("--port", type=int, default=8000, help="پورت سرور")
    api_parser.set_defaults(func=lambda args: run_api(args.host, args.port))
    
    # دستور توصیه
    recommend_parser = subparsers.add_parser("recommend", help="دریافت توصیه‌ها")
    recommend_parser.add_argument("user_id", type=int, help="شناسه کاربر")
    recommend_parser.add_argument("--limit", type=int, default=10, help="تعداد توصیه‌ها")
    recommend_parser.set_defaults(func=lambda args: get_recommendations_for_user(args.user_id, args.limit))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "api":
        run_api(args.host, args.port)
    elif args.command == "train":
        success = run_training()
        sys.exit(0 if success else 1)
    elif args.command == "recommend":
        get_recommendations_for_user(args.user_id, args.limit)


if __name__ == "__main__":
    main()



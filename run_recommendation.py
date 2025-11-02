#!/usr/bin/env python3
"""
اسکریپت اجرای سیستم توصیه محصولات

این اسکریپت امکان اجرای دستورات مختلف سیستم توصیه را فراهم می‌کند:
- train: آموزش مدل
- api: اجرای API سرور
- recommend: دریافت توصیه‌ها برای یک کاربر
"""
from __future__ import annotations
import argparse
import logging
import sys
from typing import Optional

import uvicorn

from hybrid_recommender import HybridRecommender
from recommendation_api import app

# تنظیم logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training():
    """اجرای آموزش مدل"""
    logger.info("Starting recommendation system training...")
    
    try:
        recommender = HybridRecommender()
        recommender.train()
        logger.info("Model training completed successfully")
        
        # تست سیستم
        logger.info("Testing system...")
        users = recommender.users
        if users:
            test_user = users[0]
            recommendations = recommender.get_recommendations(test_user.id, 5)
            logger.info(f"Sample recommendations for user {test_user.id}:")
            for rec in recommendations:
                logger.info(f"  - Product {rec.product_id}: {rec.score:.2f} ({rec.reason})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return False


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """اجرای API سرور"""
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"📖 API Documentation: http://localhost:{port}/docs")
    logger.info(f"🔍 Health Check: http://localhost:{port}/health")
    
    try:
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.error(f"Error running API server: {e}", exc_info=True)
        sys.exit(1)


def get_recommendations_for_user(user_id: int, limit: int = 10):
    """دریافت توصیه‌های کاربر"""
    try:
        logger.info(f"Getting recommendations for user {user_id}...")
        recommender = HybridRecommender()
        recommender.train()
        
        recommendations = recommender.get_recommendations(user_id, limit)
        
        logger.info(f"Found {len(recommendations)} recommendations for user {user_id}:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"{i}. Product {rec.product_id} - "
                f"Score: {rec.score:.2f}, "
                f"Confidence: {rec.confidence:.2f} - "
                f"Reason: {rec.reason}"
            )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}", exc_info=True)
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





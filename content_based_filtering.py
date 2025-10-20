from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from models import User, Product, ProductInteraction, Recommendation
from data_loader import load_user_purchase_history


class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا"""
    
    def __init__(self):
        self.product_features = None
        self.product_similarities = None
        self.user_profiles = None
    
    def extract_product_features(self, products: List[Product]) -> Dict[int, Dict]:
        """استخراج ویژگی‌های محصولات"""
        features = {}
        
        for product in products:
            # ویژگی‌های متنی
            text_features = f"{product.title} {product.slug} {product.sku}".lower()
            
            # ویژگی‌های عددی
            price_range = self._get_price_range(product.sale_price)
            stock_level = self._get_stock_level(product.stock_quantity)
            
            features[product.id] = {
                'text': text_features,
                'price_range': price_range,
                'stock_level': stock_level,
                'category_id': product.category_id,
                'seller_id': product.seller_id,
                'price': product.sale_price,
                'stock': product.stock_quantity
            }
        
        return features
    
    def _get_price_range(self, price: float) -> str:
        """دسته‌بندی قیمت"""
        if price < 100000:
            return "low"
        elif price < 500000:
            return "medium"
        elif price < 1000000:
            return "high"
        else:
            return "premium"
    
    def _get_stock_level(self, stock: int) -> str:
        """سطح موجودی"""
        if stock == 0:
            return "out_of_stock"
        elif stock < 5:
            return "low_stock"
        elif stock < 20:
            return "medium_stock"
        else:
            return "high_stock"
    
    def build_product_similarity_matrix(self, product_features: Dict[int, Dict]) -> None:
        """ساخت ماتریس شباهت محصولات"""
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        # استخراج متن برای TF-IDF
        texts = [product_features[pid]['text'] for pid in product_ids]
        
        # محاسبه TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # برای فارسی
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # محاسبه شباهت کسینوسی
        self.product_similarities = cosine_similarity(tfidf_matrix)
        
        # ذخیره نگاشت
        self.product_to_index = {pid: i for i, pid in enumerate(product_ids)}
        self.index_to_product = {i: pid for pid, i in self.product_to_index.items()}
        
        # اضافه کردن شباهت بر اساس ویژگی‌های عددی
        self._add_numerical_similarities(product_features)
    
    def _add_numerical_similarities(self, product_features: Dict[int, Dict]) -> None:
        """اضافه کردن شباهت بر اساس ویژگی‌های عددی"""
        n_products = len(self.product_to_index)
        
        for i in range(n_products):
            for j in range(i + 1, n_products):
                product1_id = self.index_to_product[i]
                product2_id = self.index_to_product[j]
                
                features1 = product_features[product1_id]
                features2 = product_features[product2_id]
                
                # شباهت قیمت
                price_sim = self._price_similarity(features1['price'], features2['price'])
                
                # شباهت دسته‌بندی
                category_sim = 1.0 if features1['category_id'] == features2['category_id'] else 0.0
                
                # شباهت فروشنده
                seller_sim = 1.0 if features1['seller_id'] == features2['seller_id'] else 0.0
                
                # ترکیب شباهت‌ها
                numerical_sim = (price_sim * 0.4 + category_sim * 0.4 + seller_sim * 0.2)
                
                # ترکیب با شباهت متنی
                self.product_similarities[i, j] = (
                    self.product_similarities[i, j] * 0.7 + numerical_sim * 0.3
                )
                self.product_similarities[j, i] = self.product_similarities[i, j]
    
    def _price_similarity(self, price1: float, price2: float) -> float:
        """محاسبه شباهت قیمت"""
        if price1 == 0 or price2 == 0:
            return 0.0
        
        ratio = min(price1, price2) / max(price1, price2)
        return ratio
    
    def build_user_profiles(self, user_interactions: Dict[int, List[ProductInteraction]], 
                          product_features: Dict[int, Dict]) -> None:
        """ساخت پروفایل کاربران"""
        self.user_profiles = {}
        
        for user_id, interactions in user_interactions.items():
            if not interactions:
                continue
            
            # محاسبه وزن‌های محصولات
            product_weights = defaultdict(float)
            for interaction in interactions:
                weight = self._get_interaction_weight(interaction)
                product_weights[interaction.product_id] += weight
            
            # نرمال‌سازی وزن‌ها
            total_weight = sum(product_weights.values())
            if total_weight > 0:
                for product_id in product_weights:
                    product_weights[product_id] /= total_weight
            
            # ساخت پروفایل کاربر
            user_profile = {
                'product_weights': dict(product_weights),
                'preferred_categories': self._get_preferred_categories(
                    product_weights, product_features
                ),
                'preferred_price_range': self._get_preferred_price_range(
                    product_weights, product_features
                ),
                'preferred_sellers': self._get_preferred_sellers(
                    product_weights, product_features
                )
            }
            
            self.user_profiles[user_id] = user_profile
    
    def _get_interaction_weight(self, interaction: ProductInteraction) -> float:
        """محاسبه وزن تعامل"""
        weights = {
            'purchase': 5.0,
            'wishlist': 3.0,
            'cart_add': 2.0,
            'view': 1.0
        }
        return weights.get(interaction.interaction_type, 1.0)
    
    def _get_preferred_categories(self, product_weights: Dict[int, float], 
                                product_features: Dict[int, Dict]) -> Dict[int, float]:
        """دسته‌بندی‌های مورد علاقه کاربر"""
        category_weights = defaultdict(float)
        
        for product_id, weight in product_weights.items():
            if product_id in product_features:
                category_id = product_features[product_id]['category_id']
                if category_id:
                    category_weights[category_id] += weight
        
        return dict(category_weights)
    
    def _get_preferred_price_range(self, product_weights: Dict[int, float], 
                                 product_features: Dict[int, Dict]) -> Dict[str, float]:
        """محدوده قیمت مورد علاقه کاربر"""
        price_weights = defaultdict(float)
        
        for product_id, weight in product_weights.items():
            if product_id in product_features:
                price_range = product_features[product_id]['price_range']
                price_weights[price_range] += weight
        
        return dict(price_weights)
    
    def _get_preferred_sellers(self, product_weights: Dict[int, float], 
                             product_features: Dict[int, Dict]) -> Dict[int, float]:
        """فروشندگان مورد علاقه کاربر"""
        seller_weights = defaultdict(float)
        
        for product_id, weight in product_weights.items():
            if product_id in product_features:
                seller_id = product_features[product_id]['seller_id']
                if seller_id:
                    seller_weights[seller_id] += weight
        
        return dict(seller_weights)
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Recommendation]:
        """دریافت توصیه‌های کاربر"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        product_weights = user_profile['product_weights']
        
        # پیدا کردن محصولات مشابه به محصولات مورد علاقه
        recommendations = []
        seen_products = set(product_weights.keys())
        
        for liked_product_id, weight in product_weights.items():
            if liked_product_id not in self.product_to_index:
                continue
            
            product_idx = self.product_to_index[liked_product_id]
            similarities = self.product_similarities[product_idx]
            
            # پیدا کردن محصولات مشابه
            for similar_idx, similarity in enumerate(similarities):
                similar_product_id = self.index_to_product[similar_idx]
                
                if similar_product_id not in seen_products:
                    score = similarity * weight
                    recommendations.append((similar_product_id, score))
        
        # مرتب‌سازی و حذف تکراری‌ها
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # حذف تکراری‌ها و انتخاب بهترین‌ها
        unique_recommendations = {}
        for product_id, score in recommendations:
            if product_id not in unique_recommendations:
                unique_recommendations[product_id] = score
        
        final_recommendations = []
        for product_id, score in list(unique_recommendations.items())[:top_k]:
            final_recommendations.append(Recommendation(
                user_id=user_id,
                product_id=product_id,
                score=score,
                reason="توصیه بر اساس محصولات مشابه",
                confidence=min(score, 1.0)
            ))
        
        return final_recommendations


def train_content_based_model(products: List[Product], 
                            user_interactions: Dict[int, List[ProductInteraction]]) -> ContentBasedFiltering:
    """آموزش مدل content-based filtering"""
    model = ContentBasedFiltering()
    product_features = model.extract_product_features(products)
    model.build_product_similarity_matrix(product_features)
    model.build_user_profiles(user_interactions, product_features)
    return model

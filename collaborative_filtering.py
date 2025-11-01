from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math
import json

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history


class CollaborativeFiltering:
    """سیستم توصیه مبتنی بر همکاری کاربران"""
    
    def __init__(self, min_common_items: int = 2):
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_similarities = None
    
    def build_user_item_matrix(self, interactions: List[ProductInteraction]) -> None:
        """ساخت ماتریس کاربر-محصول"""
        user_items = defaultdict(dict)
        
        for interaction in interactions:
            user_id = interaction.user_id
            product_id = interaction.product_id
            
            # محاسبه امتیاز بر اساس نوع تعامل
            if interaction.interaction_type == 'purchase':
                score = 5.0 + interaction.value / 100  # امتیاز خرید
            elif interaction.interaction_type == 'view':
                score = 1.0
            elif interaction.interaction_type == 'cart_add':
                score = 3.0
            elif interaction.interaction_type == 'wishlist':
                score = 4.0
            else:
                score = 1.0
            
            # تجمیع امتیازات
            if product_id in user_items[user_id]:
                user_items[user_id][product_id] += score
            else:
                user_items[user_id][product_id] = score
        
        # تبدیل به ماتریس
        all_users = list(user_items.keys())
        all_products = set()
        for user_products in user_items.values():
            all_products.update(user_products.keys())
        all_products = list(all_products)
        
        self.user_item_matrix = np.zeros((len(all_users), len(all_products)))
        self.user_to_index = {user_id: i for i, user_id in enumerate(all_users)}
        self.product_to_index = {product_id: i for i, product_id in enumerate(all_products)}
        self.index_to_user = {i: user_id for user_id, i in self.user_to_index.items()}
        self.index_to_product = {i: product_id for product_id, i in self.product_to_index.items()}
        
        for user_id, products in user_items.items():
            user_idx = self.user_to_index[user_id]
            for product_id, score in products.items():
                product_idx = self.product_to_index[product_id]
                self.user_item_matrix[user_idx, product_idx] = score
    
    def calculate_user_similarities(self) -> None:
        """محاسبه شباهت بین کاربران"""
        n_users = self.user_item_matrix.shape[0]
        self.user_similarities = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            for j in range(i + 1, n_users):
                similarity = self._cosine_similarity(
                    self.user_item_matrix[i],
                    self.user_item_matrix[j]
                )
                self.user_similarities[i, j] = similarity
                self.user_similarities[j, i] = similarity
    
    def _cosine_similarity(self, user1: np.ndarray, user2: np.ndarray) -> float:
        """محاسبه شباهت کسینوسی"""
        dot_product = np.dot(user1, user2)
        norm1 = np.linalg.norm(user1)
        norm2 = np.linalg.norm(user2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_user_recommendations(self, user_id: int, top_k: int = 10) -> List[Recommendation]:
        """دریافت توصیه‌های کاربر"""
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        user_ratings = self.user_item_matrix[user_idx]
        
        # پیدا کردن محصولات که کاربر هنوز ندیده
        unseen_products = []
        for product_idx, rating in enumerate(user_ratings):
            if rating == 0:  # محصول ندیده
                product_id = self.index_to_product[product_idx]
                unseen_products.append((product_idx, product_id))
        
        # محاسبه امتیاز پیش‌بینی برای هر محصول
        predictions = []
        for product_idx, product_id in unseen_products:
            score, similar_users_details = self._predict_rating(user_idx, product_idx)
            if score > 0:
                predictions.append((product_id, score, similar_users_details))
        
        # مرتب‌سازی و انتخاب بهترین‌ها
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product_id, score, similar_users_details in predictions[:top_k]:
            # ساخت جزئیات JSON
            if similar_users_details:
                # محدود کردن به 5 کاربر مشابه برتر
                top_similar = similar_users_details[:5]
                details_json = json.dumps({
                    'similar_users': [
                        {'user_id': user_id, 'similarity': round(sim, 4), 
                         'similarity_percent': round(sim * 100, 2)}
                        for user_id, sim in top_similar
                    ],
                    'total_similar_users': len(similar_users_details)
                }, ensure_ascii=False)
                
                # ساخت reason با جزئیات
                similar_users_str = ', '.join([f"user_{uid}" for uid, _ in top_similar])
                reason = f"Collaborative: {len(similar_users_details)} کاربران مشابه ({similar_users_str}) این محصول را خریده‌اند"
            else:
                details_json = None
                reason = "Collaborative: توصیه بر اساس رفتار کاربران مشابه"
            
            recommendations.append(Recommendation(
                user_id=user_id,
                product_id=product_id,
                score=score,
                reason=reason,
                confidence=min(score / 5.0, 1.0),
                collaborative_details=details_json
            ))
        
        return recommendations
    
    def _predict_rating(self, user_idx: int, product_idx: int) -> Tuple[float, List[Tuple[int, float]]]:
        """پیش‌بینی امتیاز محصول برای کاربر و برگرداندن جزئیات کاربران مشابه"""
        similarities = self.user_similarities[user_idx]
        ratings = self.user_item_matrix[:, product_idx]
        
        # پیدا کردن کاربران مشابه که این محصول را خریده‌اند
        similar_users = []
        similar_users_details = []
        
        for other_user_idx, similarity in enumerate(similarities):
            if other_user_idx != user_idx and ratings[other_user_idx] > 0:
                similar_users.append((similarity, ratings[other_user_idx]))
                # ذخیره جزئیات (user_id, similarity)
                other_user_id = self.index_to_user[other_user_idx]
                similar_users_details.append((other_user_id, similarity))
        
        if not similar_users:
            return 0.0, []
        
        # مرتب‌سازی بر اساس شباهت (برای نمایش دقیق‌تر)
        similar_users_details.sort(key=lambda x: x[1], reverse=True)
        
        # محاسبه میانگین وزنی
        total_weight = 0.0
        weighted_sum = 0.0
        
        for similarity, rating in similar_users:
            if similarity > 0:
                total_weight += similarity
                weighted_sum += similarity * rating
        
        if total_weight == 0:
            return 0.0, []
        
        return weighted_sum / total_weight, similar_users_details
    
    def get_similar_users(self, user_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """دریافت کاربران مشابه"""
        if user_id not in self.user_to_index:
            return []
        
        user_idx = self.user_to_index[user_id]
        similarities = self.user_similarities[user_idx]
        
        # مرتب‌سازی بر اساس شباهت
        similar_users = []
        for other_user_idx, similarity in enumerate(similarities):
            if other_user_idx != user_idx and similarity > 0:
                other_user_id = self.index_to_user[other_user_idx]
                similar_users.append((other_user_id, similarity))
        
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users[:top_k]


def train_collaborative_model(interactions: List[ProductInteraction]) -> CollaborativeFiltering:
    """آموزش مدل collaborative filtering"""
    model = CollaborativeFiltering()
    model.build_user_item_matrix(interactions)
    model.calculate_user_similarities()
    return model

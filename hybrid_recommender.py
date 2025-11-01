from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict

from models import User, Product, ProductInteraction, Recommendation
from collaborative_filtering import CollaborativeFiltering, train_collaborative_model
from content_based_filtering import ContentBasedFiltering, train_content_based_model
from object_loader import load_users, load_products, load_user_purchase_history


class HybridRecommender:
    """سیستم توصیه ترکیبی (Hybrid Recommendation System)"""
    
    def __init__(self, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.collaborative_model = None
        self.content_model = None
        self.users = []
        self.products = []
        self.user_interactions = {}
    
    def train(self, start_date=None, end_date=None) -> None:
        """آموزش مدل‌های توصیه"""
        print("شروع بارگذاری داده‌ها...")
        
        # بارگذاری داده‌های پایه
        self.users = load_users()
        self.products = load_products()
        
        print(f"بارگذاری {len(self.users)} کاربر و {len(self.products)} محصول")
        
        # بارگذاری تعاملات کاربران
        self._load_user_interactions()
        
        # آموزش مدل collaborative filtering
        print("آموزش مدل collaborative filtering...")
        all_interactions = []
        for interactions in self.user_interactions.values():
            all_interactions.extend(interactions)
        
        if all_interactions:
            self.collaborative_model = train_collaborative_model(all_interactions)
            print("مدل collaborative filtering آموزش داده شد")
        else:
            print("هشدار: هیچ تعامل کاربری یافت نشد")
        
        # آموزش مدل content-based filtering
        print("آموزش مدل content-based filtering...")
        if self.products and self.user_interactions:
            self.content_model = train_content_based_model(self.products, self.user_interactions)
            print("مدل content-based filtering آموزش داده شد")
        else:
            print("هشدار: محصولات یا تعاملات کاربری کافی نیست")
    
    def _load_user_interactions(self) -> None:
        """بارگذاری تعاملات کاربران"""
        self.user_interactions = {}
        
        for user in self.users:
            # بارگذاری تاریخچه خرید
            purchase_history = load_user_purchase_history(user.id, days_back=365)
            self.user_interactions[user.id] = purchase_history
    
    def get_recommendations(self, user_id: int, top_k: int = 10) -> List[Recommendation]:
        """دریافت توصیه‌های ترکیبی برای کاربر"""
        recommendations = []
        
        # دریافت توصیه‌های collaborative filtering
        if self.collaborative_model:
            collab_recs = self.collaborative_model.get_user_recommendations(user_id, top_k * 2)
            # نکته: reason و collaborative_details قبلاً در مدل collaborative تنظیم شده است
            for rec in collab_recs:
                rec.score *= self.collaborative_weight
            recommendations.extend(collab_recs)
        
        # دریافت توصیه‌های content-based filtering
        if self.content_model:
            content_recs = self.content_model.get_user_recommendations(user_id, top_k * 2)
            for rec in content_recs:
                rec.score *= self.content_weight
                rec.reason = f"Content-based: {rec.reason}"
            recommendations.extend(content_recs)
        
        # ترکیب و رتبه‌بندی نهایی
        final_recommendations = self._combine_recommendations(recommendations, top_k)
        
        # اضافه کردن فیلترهای تجاری
        final_recommendations = self._apply_business_filters(final_recommendations, user_id)
        
        return final_recommendations[:top_k]
    
    def _combine_recommendations(self, recommendations: List[Recommendation], top_k: int) -> List[Recommendation]:
        """ترکیب و رتبه‌بندی توصیه‌ها"""
        # گروه‌بندی بر اساس product_id
        product_scores = defaultdict(list)
        
        for rec in recommendations:
            product_scores[rec.product_id].append(rec)
        
        # محاسبه امتیاز نهایی برای هر محصول
        final_recommendations = []
        for product_id, recs in product_scores.items():
            if len(recs) == 1:
                # فقط یک نوع توصیه
                final_rec = recs[0]
            else:
                # ترکیب چندین نوع توصیه
                combined_score = sum(rec.score for rec in recs)
                combined_confidence = max(rec.confidence for rec in recs)
                combined_reason = " + ".join(set(rec.reason for rec in recs))
                
                # حفظ collaborative_details اگر یکی از rec ها آن را دارد
                collaborative_details = None
                for rec in recs:
                    if rec.collaborative_details:
                        collaborative_details = rec.collaborative_details
                        break
                
                final_rec = Recommendation(
                    user_id=recs[0].user_id,
                    product_id=product_id,
                    score=combined_score,
                    reason=combined_reason,
                    confidence=combined_confidence,
                    collaborative_details=collaborative_details
                )
            
            final_recommendations.append(final_rec)
        
        # مرتب‌سازی بر اساس امتیاز
        final_recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return final_recommendations
    
    def _apply_business_filters(self, recommendations: List[Recommendation], user_id: int) -> List[Recommendation]:
        """اعمال فیلترهای تجاری"""
        filtered_recommendations = []
        
        # ایجاد دیکشنری محصولات برای دسترسی سریع
        products_dict = {p.id: p for p in self.products}
        
        for rec in recommendations:
            product = products_dict.get(rec.product_id)
            if not product:
                continue
            
            # فیلتر محصولات موجود
            if product.stock_quantity <= 0:
                continue
            
            # فیلتر محصولات منتشر شده
            if product.status != 'published':
                continue
            
            # فیلتر محصولات که کاربر قبلاً خریده
            user_purchases = {interaction.product_id for interaction in self.user_interactions.get(user_id, [])}
            if rec.product_id in user_purchases:
                continue
            
            # اضافه کردن اطلاعات اضافی
            rec.reason += f" (قیمت: {product.sale_price:,} تومان، موجودی: {product.stock_quantity})"
            
            filtered_recommendations.append(rec)
        
        return filtered_recommendations
    
    def get_similar_products(self, product_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """دریافت محصولات مشابه"""
        if not self.content_model or product_id not in self.content_model.product_to_index:
            return []
        
        product_idx = self.content_model.product_to_index[product_id]
        similarities = self.content_model.product_similarities[product_idx]
        
        similar_products = []
        for similar_idx, similarity in enumerate(similarities):
            if similar_idx != product_idx and similarity > 0:
                similar_product_id = self.content_model.index_to_product[similar_idx]
                similar_products.append((similar_product_id, similarity))
        
        similar_products.sort(key=lambda x: x[1], reverse=True)
        return similar_products[:top_k]
    
    def get_user_insights(self, user_id: int) -> Dict:
        """دریافت بینش‌های کاربر"""
        insights = {
            'total_interactions': len(self.user_interactions.get(user_id, [])),
            'preferred_categories': [],
            'average_purchase_value': 0.0,
            'similar_users': []
        }
        
        # محاسبه آمار تعاملات
        user_interactions = self.user_interactions.get(user_id, [])
        if user_interactions:
            purchase_values = [interaction.value for interaction in user_interactions if interaction.value > 0]
            if purchase_values:
                insights['average_purchase_value'] = sum(purchase_values) / len(purchase_values)
        
        # پیدا کردن دسته‌بندی‌های مورد علاقه
        if self.content_model and user_id in self.content_model.user_profiles:
            user_profile = self.content_model.user_profiles[user_id]
            preferred_categories = user_profile.get('preferred_categories', {})
            insights['preferred_categories'] = sorted(
                preferred_categories.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        
        # پیدا کردن کاربران مشابه
        if self.collaborative_model and user_id in self.collaborative_model.user_to_index:
            similar_users = self.collaborative_model.get_similar_users(user_id, top_k=3)
            insights['similar_users'] = similar_users
        
        return insights
    
    def get_popular_products(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """دریافت محصولات محبوب"""
        product_counts = defaultdict(int)
        
        for interactions in self.user_interactions.values():
            for interaction in interactions:
                if interaction.interaction_type == 'purchase':
                    product_counts[interaction.product_id] += 1
        
        popular_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        return popular_products[:top_k]

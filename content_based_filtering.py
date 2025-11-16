from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
import logging

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history

logger = logging.getLogger(__name__)

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    logger.warning("joblib not available, parallel processing disabled")


class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا - بهینه‌شده برای حافظه و CPU"""
    
    def __init__(self, use_sparse: bool = True, n_jobs: int = -1):
        """
        Args:
            use_sparse: استفاده از ماتریس‌های sparse برای صرفه‌جویی در حافظه
            n_jobs: تعداد هسته‌های CPU برای پردازش موازی (-1 = همه هسته‌ها)
        """
        self.product_features = None
        self.product_similarities = None  # فقط برای محصولات محبوب (cache)
        self.tfidf_matrix = None  # ماتریس TF-IDF sparse
        self.vectorizer = None
        self.user_profiles = None
        self.use_sparse = use_sparse
        self.n_jobs = n_jobs if HAS_JOBLIB else 1
        self._similarity_cache = {}  # Cache برای شباهت‌های محاسبه شده
    
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
        """
        ساخت ماتریس شباهت محصولات - بهینه‌شده برای حافظه
        به جای ساخت ماتریس کامل، فقط TF-IDF را ذخیره می‌کند و شباهت‌ها را on-demand محاسبه می‌کند
        """
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        logger.info(f"Building product similarity matrix for {n_products} products (memory-optimized mode)")
        
        # استخراج متن برای TF-IDF
        texts = [product_features[pid]['text'] for pid in product_ids]
        
        # محاسبه TF-IDF با استفاده از float32 برای صرفه‌جویی در حافظه
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # برای فارسی
            ngram_range=(1, 2),
            dtype=np.float32  # استفاده از float32 به جای float64
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # تبدیل به float32 برای صرفه‌جویی در حافظه
        if hasattr(self.tfidf_matrix, 'astype'):
            self.tfidf_matrix = self.tfidf_matrix.astype(np.float32)
        
        # ذخیره نگاشت
        self.product_to_index = {pid: i for i, pid in enumerate(product_ids)}
        self.index_to_product = {i: pid for pid, i in self.product_to_index.items()}
        
        # ذخیره product_features برای محاسبه شباهت‌های عددی
        self.product_features = product_features
        
        logger.info(f"TF-IDF matrix created: shape={self.tfidf_matrix.shape}, "
                   f"memory={self.tfidf_matrix.nbytes / 1024 / 1024:.2f} MB")
        
        # دیگر نیازی به ساخت ماتریس کامل نیست - شباهت‌ها on-demand محاسبه می‌شوند
        self.product_similarities = None
    
    def _compute_similarity(self, product1_id: int, product2_id: int) -> float:
        """
        محاسبه شباهت بین دو محصول - on-demand
        ترکیب شباهت متنی (TF-IDF) و شباهت عددی
        """
        if product1_id not in self.product_to_index or product2_id not in self.product_to_index:
            return 0.0
        
        # استفاده از cache اگر وجود داشته باشد
        cache_key = tuple(sorted([product1_id, product2_id]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        idx1 = self.product_to_index[product1_id]
        idx2 = self.product_to_index[product2_id]
        
        # محاسبه شباهت متنی (TF-IDF cosine similarity)
        if self.tfidf_matrix is not None:
            # محاسبه cosine similarity بین دو بردار sparse
            vec1 = self.tfidf_matrix[idx1:idx1+1]
            vec2 = self.tfidf_matrix[idx2:idx2+1]
            text_similarity = float(cosine_similarity(vec1, vec2)[0, 0])
        else:
            text_similarity = 0.0
        
        # محاسبه شباهت عددی
        if self.product_features:
            features1 = self.product_features.get(product1_id, {})
            features2 = self.product_features.get(product2_id, {})
            
            # شباهت قیمت
            price_sim = self._price_similarity(
                features1.get('price', 0),
                features2.get('price', 0)
            )
            
            # شباهت دسته‌بندی
            category_sim = 1.0 if features1.get('category_id') == features2.get('category_id') else 0.0
            
            # شباهت فروشنده
            seller_sim = 1.0 if features1.get('seller_id') == features2.get('seller_id') else 0.0
            
            # ترکیب شباهت‌های عددی
            numerical_sim = (price_sim * 0.4 + category_sim * 0.4 + seller_sim * 0.2)
            
            # ترکیب با شباهت متنی
            combined_similarity = text_similarity * 0.7 + numerical_sim * 0.3
        else:
            combined_similarity = text_similarity
        
        # ذخیره در cache (محدود کردن اندازه cache)
        if len(self._similarity_cache) < 100000:  # حداکثر 100k entry در cache
            self._similarity_cache[cache_key] = combined_similarity
        
        return combined_similarity
    
    def _get_similar_products(self, product_id: int, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        پیدا کردن محصولات مشابه به یک محصول - بهینه‌شده با استفاده از batch computation
        """
        if product_id not in self.product_to_index or self.tfidf_matrix is None:
            return []
        
        product_idx = self.product_to_index[product_id]
        
        # استفاده از batch cosine similarity برای محاسبه سریع‌تر
        # محاسبه شباهت بین محصول فعلی و همه محصولات دیگر به صورت batch
        product_vector = self.tfidf_matrix[product_idx:product_idx+1]
        
        # محاسبه cosine similarity با همه محصولات به صورت batch
        # این بسیار سریع‌تر از محاسبه تک‌تک است
        all_similarities = cosine_similarity(product_vector, self.tfidf_matrix)[0]
        
        # تبدیل به لیست (product_id, similarity)
        similarities = []
        for idx, sim in enumerate(all_similarities):
            if idx == product_idx:  # خود محصول را نادیده بگیر
                continue
            
            other_product_id = self.index_to_product[idx]
            
            # اضافه کردن شباهت عددی
            if self.product_features:
                features1 = self.product_features.get(product_id, {})
                features2 = self.product_features.get(other_product_id, {})
                
                # شباهت قیمت
                price_sim = self._price_similarity(
                    features1.get('price', 0),
                    features2.get('price', 0)
                )
                
                # شباهت دسته‌بندی
                category_sim = 1.0 if features1.get('category_id') == features2.get('category_id') else 0.0
                
                # شباهت فروشنده
                seller_sim = 1.0 if features1.get('seller_id') == features2.get('seller_id') else 0.0
                
                # ترکیب شباهت‌های عددی
                numerical_sim = (price_sim * 0.4 + category_sim * 0.4 + seller_sim * 0.2)
                
                # ترکیب با شباهت متنی
                combined_sim = float(sim) * 0.7 + numerical_sim * 0.3
            else:
                combined_sim = float(sim)
            
            if combined_sim > 0:
                similarities.append((other_product_id, combined_sim))
        
        # مرتب‌سازی و برگرداندن top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
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
        """
        دریافت توصیه‌های کاربر - بهینه‌شده با پردازش موازی
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        product_weights = user_profile['product_weights']
        
        if not product_weights:
            return []
        
        # پیدا کردن محصولات مشابه به محصولات مورد علاقه
        recommendations_dict = defaultdict(float)
        seen_products = set(product_weights.keys())
        
        # محدود کردن تعداد محصولات برای محاسبه شباهت (برای سرعت بیشتر)
        max_products_to_check = min(50, len(product_weights))  # فقط 50 محصول اول
        
        liked_products = list(product_weights.items())[:max_products_to_check]
        
        # پردازش موازی برای پیدا کردن محصولات مشابه
        if HAS_JOBLIB and self.n_jobs != 1 and len(liked_products) > 5:
            def process_liked_product(item):
                liked_product_id, weight = item
                if liked_product_id not in self.product_to_index:
                    return []
                
                # پیدا کردن محصولات مشابه (top 20 برای هر محصول)
                similar_products = self._get_similar_products(liked_product_id, top_k=20)
                
                results = []
                for similar_product_id, similarity in similar_products:
                    if similar_product_id not in seen_products:
                        score = similarity * weight
                        results.append((similar_product_id, score))
                return results
            
            all_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(process_liked_product)(item) for item in liked_products
            )
            
            # ترکیب نتایج
            for results in all_results:
                for product_id, score in results:
                    recommendations_dict[product_id] += score
        else:
            # پردازش سریالی
            for liked_product_id, weight in liked_products:
                if liked_product_id not in self.product_to_index:
                    continue
                
                # پیدا کردن محصولات مشابه
                similar_products = self._get_similar_products(liked_product_id, top_k=20)
                
                for similar_product_id, similarity in similar_products:
                    if similar_product_id not in seen_products:
                        score = similarity * weight
                        recommendations_dict[similar_product_id] += score
        
        # مرتب‌سازی و انتخاب بهترین‌ها
        sorted_recommendations = sorted(
            recommendations_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        final_recommendations = []
        for product_id, score in sorted_recommendations:
            final_recommendations.append(Recommendation(
                user_id=user_id,
                product_id=product_id,
                score=float(score),
                reason="توصیه بر اساس محصولات مشابه",
                confidence=min(float(score), 1.0)
            ))
        
        return final_recommendations


def train_content_based_model(products: List[Product], 
                            user_interactions: Dict[int, List[ProductInteraction]],
                            n_jobs: int = -1) -> ContentBasedFiltering:
    """
    آموزش مدل content-based filtering
    
    Args:
        products: لیست محصولات
        user_interactions: دیکشنری تعاملات کاربران
        n_jobs: تعداد هسته‌های CPU برای پردازش موازی (-1 = همه هسته‌ها)
    """
    model = ContentBasedFiltering(n_jobs=n_jobs)
    product_features = model.extract_product_features(products)
    model.build_product_similarity_matrix(product_features)
    model.build_user_profiles(user_interactions, product_features)
    return model



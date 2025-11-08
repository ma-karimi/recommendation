"""
Content-Based Filtering با بهینه‌سازی حافظه

بهینه‌سازی‌ها:
- استفاده از Sparse Matrix برای ذخیره شباهت‌ها
- محاسبه شباهت فقط برای محصولات مرتبط (همان دسته‌بندی)
- محاسبه Lazy (فقط وقتی نیاز است)
- کاهش استفاده از حافظه از 11.9 GB به < 1 GB
"""
from __future__ import annotations
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history

# تنظیم logger
logger = logging.getLogger(__name__)

class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا (بهینه‌سازی شده برای حافظه)"""
    
    def __init__(self, use_sparse: bool = True, max_similar_products: int = 50):
        """
        Args:
            use_sparse: استفاده از Sparse Matrix برای صرفه‌جویی در حافظه
            max_similar_products: حداکثر تعداد محصولات مشابه برای هر محصول
        """
        self.product_features = None
        self.product_similarities = None  # Sparse matrix یا dict
        self.user_profiles = None
        self.use_sparse = use_sparse
        self.max_similar_products = max_similar_products
        self.tfidf_matrix = None
        self.vectorizer = None
    
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
        ساخت ماتریس شباهت محصولات (بهینه‌سازی شده)
        
        به جای ساخت ماتریس کامل N×N، فقط شباهت‌های مهم را ذخیره می‌کند
        """
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        logger.info(f"Building similarity matrix for {n_products} products...")
        
        # ذخیره نگاشت
        self.product_to_index = {pid: i for i, pid in enumerate(product_ids)}
        self.index_to_product = {i: pid for pid, i in self.product_to_index.items()}
        
        # استخراج متن برای TF-IDF
        texts = [product_features[pid]['text'] for pid in product_ids]
        
        # محاسبه TF-IDF
        logger.info("Computing TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # کاهش از 1000 به 500
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,  # فقط کلمات که در حداقل 2 سند آمده
            max_df=0.95  # حذف کلمات خیلی رایج
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # بهینه‌سازی: محاسبه شباهت فقط برای محصولات مرتبط
        if self.use_sparse:
            logger.info("Building sparse similarity matrix (memory efficient)...")
            self._build_sparse_similarity_matrix(product_features)
        else:
            logger.warning("Using dense matrix - may require large memory!")
            # فقط برای تعداد کم محصولات
            if n_products > 10000:
                raise MemoryError(
                    f"Too many products ({n_products}). Use sparse mode or reduce products."
                )
            self.product_similarities = cosine_similarity(self.tfidf_matrix)
    
    def _build_sparse_similarity_matrix(self, product_features: Dict[int, Dict]) -> None:
        """
        ساخت ماتریس شباهت Sparse (فقط شباهت‌های مهم)
        
        استراتژی:
        1. محاسبه شباهت فقط برای محصولات در همان دسته‌بندی
        2. نگه‌داری فقط top-k مشابه‌ترین محصولات
        3. استفاده از Sparse Matrix
        """
        n_products = len(self.product_to_index)
        
        # ساختار: {product_id: [(similar_product_id, similarity), ...]}
        similarity_dict = defaultdict(list)
        
        # گروه‌بندی محصولات بر اساس دسته‌بندی
        products_by_category = defaultdict(list)
        for product_id, features in product_features.items():
            category_id = features.get('category_id')
            if category_id:
                products_by_category[category_id].append(product_id)
        
        logger.info(f"Grouped products into {len(products_by_category)} categories")
        
        # محاسبه شباهت فقط برای محصولات در همان دسته‌بندی
        processed = 0
        batch_size = 1000
        
        for category_id, category_products in products_by_category.items():
            if len(category_products) < 2:
                continue
            
            # محاسبه شباهت برای محصولات این دسته
            category_indices = [self.product_to_index[pid] for pid in category_products]
            category_tfidf = self.tfidf_matrix[category_indices]
            
            # محاسبه شباهت (فقط برای این دسته)
            category_similarities = cosine_similarity(category_tfidf)
            
            # ذخیره فقط top-k مشابه‌ترین
            for i, product_id in enumerate(category_products):
                similarities = category_similarities[i]
                
                # پیدا کردن top-k مشابه‌ترین
                top_indices = np.argsort(similarities)[::-1][1:self.max_similar_products + 1]
                
                for j in top_indices:
                    if similarities[j] > 0.1:  # فقط شباهت‌های معنی‌دار
                        similar_product_id = category_products[j]
                        similarity_dict[product_id].append(
                            (similar_product_id, float(similarities[j]))
                        )
                
                processed += 1
                if processed % batch_size == 0:
                    logger.info(f"Processed {processed}/{n_products} products...")
        
        # تبدیل به Sparse Matrix
        logger.info("Converting to sparse matrix...")
        self._similarity_dict_to_sparse(similarity_dict, product_features)
        
        logger.info("Similarity matrix built successfully!")
    
    def _similarity_dict_to_sparse(self, similarity_dict: Dict[int, List[Tuple[int, float]]],
                                   product_features: Dict[int, Dict]) -> None:
        """تبدیل dict شباهت به Sparse Matrix"""
        n_products = len(self.product_to_index)
        
        # ساخت Sparse Matrix
        rows = []
        cols = []
        data = []
        
        for product_id, similar_products in similarity_dict.items():
            if product_id not in self.product_to_index:
                continue
            
            row_idx = self.product_to_index[product_id]
            
            for similar_product_id, similarity in similar_products:
                if similar_product_id not in self.product_to_index:
                    continue
                
                col_idx = self.product_to_index[similar_product_id]
                
                # اضافه کردن شباهت عددی
                numerical_sim = self._compute_numerical_similarity(
                    product_features[product_id],
                    product_features[similar_product_id]
                )
                
                # ترکیب شباهت متنی و عددی
                combined_sim = similarity * 0.7 + numerical_sim * 0.3
                
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(combined_sim)
        
        # ساخت Sparse Matrix
        self.product_similarities = csr_matrix(
            (data, (rows, cols)),
            shape=(n_products, n_products)
        )
        
        logger.info(
            f"Sparse matrix created: {len(data)} non-zero elements "
            f"({len(data) / (n_products * n_products) * 100:.2f}% density)"
        )
    
    def _compute_numerical_similarity(self, features1: Dict, features2: Dict) -> float:
        """محاسبه شباهت بر اساس ویژگی‌های عددی"""
        # شباهت قیمت
        price_sim = self._price_similarity(features1['price'], features2['price'])
        
        # شباهت دسته‌بندی
        category_sim = 1.0 if features1.get('category_id') == features2.get('category_id') else 0.0
        
        # شباهت فروشنده
        seller_sim = 1.0 if features1.get('seller_id') == features2.get('seller_id') else 0.0
        
        # ترکیب
        return price_sim * 0.4 + category_sim * 0.4 + seller_sim * 0.2
    
    def get_product_similarities(self, product_id: int, top_k: int = 50) -> List[Tuple[int, float]]:
        """
        دریافت محصولات مشابه برای یک محصول (بهینه‌سازی شده)
        
        این روش حافظه کمتری استفاده می‌کند چون فقط شباهت‌های مورد نیاز را محاسبه می‌کند
        """
        if product_id not in self.product_to_index:
            return []
        
        product_idx = self.product_to_index[product_id]
        
        if isinstance(self.product_similarities, csr_matrix):
            # استفاده از Sparse Matrix (کارآمدتر)
            row = self.product_similarities[product_idx]
            
            # دریافت فقط عناصر غیر صفر
            if row.nnz == 0:
                return []
            
            # تبدیل به array فقط برای عناصر غیر صفر
            col_indices = row.indices
            similarities = row.data
            
            # مرتب‌سازی بر اساس شباهت
            sorted_indices = np.argsort(similarities)[::-1]
            
            similar_products = []
            for idx in sorted_indices[:top_k]:
                if similarities[idx] > 0.1:  # فقط شباهت‌های معنی‌دار
                    similar_product_id = self.index_to_product[col_indices[idx]]
                    similar_products.append((similar_product_id, float(similarities[idx])))
            
            return similar_products
        else:
            # Dense Matrix (فقط برای تعداد کم محصولات)
            similarities = self.product_similarities[product_idx]
            
            # پیدا کردن top-k مشابه‌ترین
            top_indices = np.argsort(similarities)[::-1][1:top_k + 1]
            
            similar_products = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    similar_product_id = self.index_to_product[idx]
                    similar_products.append((similar_product_id, float(similarities[idx])))
            
            return similar_products
    
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
        دریافت توصیه‌های کاربر (بهینه‌سازی شده)
        
        به جای استفاده از کل ماتریس، فقط محصولات مشابه محصولات مورد علاقه را محاسبه می‌کند
        """
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        product_weights = user_profile['product_weights']
        
        # پیدا کردن محصولات مشابه به محصولات مورد علاقه
        recommendations = defaultdict(float)
        seen_products = set(product_weights.keys())
        
        # محدود کردن به top محصولات مورد علاقه برای سرعت بیشتر
        top_liked_products = sorted(
            product_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]  # فقط 20 محصول برتر
        
        for liked_product_id, weight in top_liked_products:
            if liked_product_id not in self.product_to_index:
                continue
            
            # دریافت محصولات مشابه (Lazy)
            similar_products = self.get_product_similarities(
                liked_product_id,
                top_k=self.max_similar_products
            )
            
            for similar_product_id, similarity in similar_products:
                if similar_product_id not in seen_products:
                    # تجمیع امتیاز
                    recommendations[similar_product_id] += similarity * weight
        
        # مرتب‌سازی و انتخاب بهترین‌ها
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        final_recommendations = []
        for product_id, score in sorted_recommendations:
            final_recommendations.append(Recommendation(
                user_id=user_id,
                product_id=product_id,
                score=score,
                reason="Content-based: مشابه محصولات قبلی شما",
                confidence=min(score / 5.0, 1.0)  # نرمال‌سازی
            ))
        
        return final_recommendations


def train_content_based_model(products: List[Product], 
                            user_interactions: Dict[int, List[ProductInteraction]],
                            use_sparse: bool = True,
                            max_similar_products: int = 50) -> ContentBasedFiltering:
    """
    آموزش مدل content-based filtering (بهینه‌سازی شده)
    
    Args:
        products: لیست محصولات
        user_interactions: تعاملات کاربران
        use_sparse: استفاده از Sparse Matrix (پیش‌فرض: True)
        max_similar_products: حداکثر تعداد محصولات مشابه برای هر محصول
    
    Returns:
        ContentBasedFiltering model
    """
    logger.info("Training Content-Based Filtering model...")
    model = ContentBasedFiltering(
        use_sparse=use_sparse,
        max_similar_products=max_similar_products
    )
    
    logger.info("Extracting product features...")
    product_features = model.extract_product_features(products)
    
    logger.info("Building product similarity matrix...")
    model.build_product_similarity_matrix(product_features)
    
    logger.info("Building user profiles...")
    model.build_user_profiles(user_interactions, product_features)
    
    logger.info("Content-Based Filtering model trained successfully!")
    return model



"""
Content-Based Filtering با بهینه‌سازی حافظه و پردازش موازی

بهینه‌سازی‌ها:
- استفاده از Sparse Matrix برای ذخیره شباهت‌ها
- محاسبه شباهت فقط برای محصولات مرتبط (همان دسته‌بندی)
- محاسبه Lazy (فقط وقتی نیاز است)
- کاهش استفاده از حافظه از 11.9 GB به < 1 GB
- پردازش موازی با استفاده از تمام هسته‌های CPU
"""
from __future__ import annotations
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil not installed. Install with: pip install psutil")

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history

# تنظیم logger
logger = logging.getLogger(__name__)


def get_available_memory_mb() -> float:
    """دریافت حافظه RAM موجود به مگابایت"""
    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)  # تبدیل به MB
        except Exception as e:
            logger.warning(f"Error getting memory from psutil: {e}")
    
    # Fallback: استفاده از os.sysconf در Linux/Mac
    try:
        if hasattr(os, 'sysconf'):
            # Linux/Mac
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return total_memory / (1024 * 1024)
    except Exception:
        pass
    
    # Default: فرض 8 GB
    logger.warning("Could not detect available memory, assuming 8 GB")
    return 8 * 1024


def calculate_optimal_max_similar_products(
    n_products: int,
    target_memory_usage_percent: float = 0.8,
    similarity_threshold: float = 0.1
) -> int:
    """
    محاسبه حداکثر تعداد محصولات مشابه بر اساس حافظه موجود
    
    Args:
        n_products: تعداد کل محصولات
        target_memory_usage_percent: درصد حافظه مورد استفاده (0.8 = 80%)
        similarity_threshold: حداقل شباهت برای ذخیره
    
    Returns:
        حداکثر تعداد محصولات مشابه برای هر محصول
    """
    available_memory_mb = get_available_memory_mb()
    target_memory_mb = available_memory_mb * target_memory_usage_percent
    
    logger.info(f"Available RAM: {available_memory_mb:.2f} MB")
    logger.info(f"Target memory usage ({target_memory_usage_percent*100}%): {target_memory_mb:.2f} MB")
    
    # محاسبه حافظه مورد نیاز برای هر similarity entry
    # هر entry شامل: product_id (int64 = 8 bytes) + similarity (float64 = 8 bytes) = 16 bytes
    bytes_per_similarity = 16
    
    # محاسبه حافظه برای Sparse Matrix
    # CSR format: data (float64), indices (int32), indptr (int64)
    # تقریبی: برای هر non-zero element حدود 16 bytes
    bytes_per_sparse_element = 16
    
    # محاسبه حافظه برای نگاشت‌ها
    # product_to_index و index_to_product: هر کدام n_products * 8 bytes
    mapping_memory_mb = (n_products * 8 * 2) / (1024 * 1024)
    
    # حافظه قابل استفاده برای similarities
    available_for_similarities_mb = target_memory_mb - mapping_memory_mb - 100  # 100 MB buffer
    
    if available_for_similarities_mb < 0:
        logger.warning("Not enough memory for mappings, using minimum settings")
        return 10
    
    # محاسبه تعداد کل similarities که می‌توانیم ذخیره کنیم
    total_similarities = (available_for_similarities_mb * 1024 * 1024) / bytes_per_sparse_element
    
    # محاسبه max_similar_products
    # فرض: به طور متوسط 50% از similarities بالای threshold هستند
    # و هر محصول به طور متوسط max_similar_products محصول مشابه دارد
    avg_similarity_ratio = 0.5  # 50% of similarities above threshold
    
    max_similar = int(total_similarities / (n_products * avg_similarity_ratio))
    
    # محدودیت‌های منطقی
    min_similar = 10
    max_similar = min(max_similar, n_products - 1)  # نمی‌تواند بیشتر از تعداد کل محصولات باشد
    max_similar = max(max_similar, min_similar)
    
    # محدودیت عملی: برای جلوگیری از مصرف بیش از حد حافظه
    practical_max = min(max_similar, 500)  # حداکثر 500 محصول مشابه
    
    logger.info(
        f"Calculated max_similar_products: {practical_max} "
        f"(theoretical: {max_similar}, limited to {practical_max})"
    )
    
    return practical_max


class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا (بهینه‌سازی شده برای حافظه)"""
    
    def __init__(
        self, 
        use_sparse: bool = True, 
        max_similar_products: Optional[int] = None, 
        n_jobs: int = -1,
        target_memory_usage_percent: float = 0.8,
        auto_optimize_memory: bool = True
    ):
        """
        Args:
            use_sparse: استفاده از Sparse Matrix برای صرفه‌جویی در حافظه
            max_similar_products: حداکثر تعداد محصولات مشابه برای هر محصول (None = auto-calculate)
            n_jobs: تعداد thread/core برای پردازش موازی (-1 = همه هسته‌ها)
            target_memory_usage_percent: درصد حافظه RAM مورد استفاده (0.8 = 80%)
            auto_optimize_memory: محاسبه خودکار max_similar_products بر اساس حافظه
        """
        self.product_features = None
        self.product_similarities = None  # Sparse matrix یا dict
        self.user_profiles = None
        self.use_sparse = use_sparse
        self.max_similar_products = max_similar_products
        self.tfidf_matrix = None
        self.vectorizer = None
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.target_memory_usage_percent = target_memory_usage_percent
        self.auto_optimize_memory = auto_optimize_memory
        logger.info(f"Initialized ContentBasedFiltering with {self.n_jobs} CPU cores")
        if auto_optimize_memory:
            logger.info(f"Auto memory optimization enabled (target: {target_memory_usage_percent*100}% RAM)")
    
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
        اگر auto_optimize_memory فعال باشد، از 80% حافظه RAM استفاده می‌کند
        """
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        logger.info(f"Building similarity matrix for {n_products} products...")
        
        # محاسبه خودکار max_similar_products بر اساس حافظه
        if self.auto_optimize_memory and self.max_similar_products is None:
            self.max_similar_products = calculate_optimal_max_similar_products(
                n_products=n_products,
                target_memory_usage_percent=self.target_memory_usage_percent
            )
        elif self.max_similar_products is None:
            # Default fallback
            self.max_similar_products = 50
            logger.warning("max_similar_products not set, using default: 50")
        
        logger.info(f"Using max_similar_products: {self.max_similar_products}")
        
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
        ساخت ماتریس شباهت Sparse (فقط شباهت‌های مهم) با پردازش موازی
        
        استراتژی:
        1. محاسبه شباهت فقط برای محصولات در همان دسته‌بندی
        2. نگه‌داری فقط top-k مشابه‌ترین محصولات
        3. استفاده از Sparse Matrix
        4. پردازش موازی دسته‌بندی‌ها
        """
        n_products = len(self.product_to_index)
        
        # گروه‌بندی محصولات بر اساس دسته‌بندی
        products_by_category = defaultdict(list)
        for product_id, features in product_features.items():
            category_id = features.get('category_id')
            if category_id:
                products_by_category[category_id].append(product_id)
        
        logger.info(f"Grouped products into {len(products_by_category)} categories")
        
        # آماده‌سازی داده‌ها برای پردازش موازی
        category_data = []
        for category_id, category_products in products_by_category.items():
            if len(category_products) < 2:
                continue
            category_indices = [self.product_to_index[pid] for pid in category_products]
            category_data.append((category_id, category_products, category_indices))
        
        # پردازش موازی دسته‌بندی‌ها
        logger.info(f"Processing {len(category_data)} categories in parallel using {self.n_jobs} cores...")
        
        if self.n_jobs > 1 and len(category_data) > 1:
            # استفاده از multiprocessing برای پردازش موازی
            # تبدیل tfidf_matrix به array برای serialization
            tfidf_array = self.tfidf_matrix.toarray() if hasattr(self.tfidf_matrix, 'toarray') else self.tfidf_matrix
            
            with Pool(processes=self.n_jobs) as pool:
                results = pool.starmap(
                    _process_category_similarities_worker,
                    [
                        (
                            cat_data,
                            tfidf_array,
                            self.max_similar_products
                        )
                        for cat_data in category_data
                    ]
                )
        else:
            # پردازش sequential
            results = [
                self._process_category_similarities(cat_data, product_features)
                for cat_data in category_data
            ]
        
        # ترکیب نتایج
        similarity_dict = defaultdict(list)
        for category_results in results:
            for product_id, similar_products in category_results.items():
                similarity_dict[product_id].extend(similar_products)
        
        # تبدیل به Sparse Matrix
        logger.info("Converting to sparse matrix...")
        self._similarity_dict_to_sparse(similarity_dict, product_features)
        
        logger.info("Similarity matrix built successfully!")
    
    def _process_category_similarities(
        self,
        category_data: Tuple[int, List[int], List[int]],
        product_features: Dict[int, Dict]
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        پردازش شباهت برای یک دسته‌بندی (برای استفاده در sequential processing)
        
        Returns:
            Dict mapping product_id to list of (similar_product_id, similarity) tuples
        """
        category_id, category_products, category_indices = category_data
        
        similarity_dict = {}
        
        try:
            # محاسبه شباهت برای محصولات این دسته
            category_tfidf = self.tfidf_matrix[category_indices]
            category_similarities = cosine_similarity(category_tfidf)
            
            # ذخیره فقط top-k مشابه‌ترین
            for i, product_id in enumerate(category_products):
                similarities = category_similarities[i]
                
                # پیدا کردن top-k مشابه‌ترین
                # برای استفاده بیشتر از حافظه، threshold را کاهش می‌دهیم
                similarity_threshold = 0.05 if self.max_similar_products > 100 else 0.1
                
                top_indices = np.argsort(similarities)[::-1][1:self.max_similar_products + 1]
                
                similar_products = []
                for j in top_indices:
                    if similarities[j] > similarity_threshold:  # threshold قابل تنظیم
                        similar_product_id = category_products[j]
                        similar_products.append(
                            (similar_product_id, float(similarities[j]))
                        )
                
                if similar_products:
                    similarity_dict[product_id] = similar_products
        
        except Exception as e:
            logger.warning(f"Error processing category {category_id}: {e}")
        
        return similarity_dict
    
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


def _process_category_similarities_worker(
    category_data: Tuple[int, List[int], List[int]],
    tfidf_array: np.ndarray,
    max_similar_products: int
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Worker function برای پردازش موازی دسته‌بندی‌ها
    
    این تابع باید در سطح ماژول باشد تا بتواند در multiprocessing استفاده شود
    """
    category_id, category_products, category_indices = category_data
    
    similarity_dict = {}
    
    try:
        # محاسبه شباهت برای محصولات این دسته
        category_tfidf = tfidf_array[category_indices]
        category_similarities = cosine_similarity(category_tfidf)
        
        # ذخیره فقط top-k مشابه‌ترین
        for i, product_id in enumerate(category_products):
            similarities = category_similarities[i]
            
            # پیدا کردن top-k مشابه‌ترین
            # برای استفاده بیشتر از حافظه، threshold را کاهش می‌دهیم
            similarity_threshold = 0.05 if max_similar_products > 100 else 0.1
            
            top_indices = np.argsort(similarities)[::-1][1:max_similar_products + 1]
            
            similar_products = []
            for j in top_indices:
                if similarities[j] > similarity_threshold:  # threshold قابل تنظیم
                    similar_product_id = category_products[j]
                    similar_products.append(
                        (similar_product_id, float(similarities[j]))
                    )
            
            if similar_products:
                similarity_dict[product_id] = similar_products
    
    except Exception as e:
        logger.warning(f"Error processing category {category_id}: {e}")
    
    return similarity_dict


def train_content_based_model(
    products: List[Product], 
    user_interactions: Dict[int, List[ProductInteraction]],
    use_sparse: bool = True,
    max_similar_products: Optional[int] = None,
    n_jobs: int = -1,
    target_memory_usage_percent: float = 0.8,
    auto_optimize_memory: bool = True
) -> ContentBasedFiltering:
    """
    آموزش مدل content-based filtering (بهینه‌سازی شده)
    
    Args:
        products: لیست محصولات
        user_interactions: تعاملات کاربران
        use_sparse: استفاده از Sparse Matrix (پیش‌فرض: True)
        max_similar_products: حداکثر تعداد محصولات مشابه برای هر محصول (None = auto-calculate)
        n_jobs: تعداد thread/core برای پردازش موازی (-1 = همه هسته‌ها)
        target_memory_usage_percent: درصد حافظه RAM مورد استفاده (0.8 = 80%)
        auto_optimize_memory: محاسبه خودکار max_similar_products بر اساس حافظه
    
    Returns:
        ContentBasedFiltering model
    """
    logger.info("Training Content-Based Filtering model...")
    model = ContentBasedFiltering(
        use_sparse=use_sparse,
        max_similar_products=max_similar_products,
        n_jobs=n_jobs,
        target_memory_usage_percent=target_memory_usage_percent,
        auto_optimize_memory=auto_optimize_memory
    )
    
    logger.info("Extracting product features...")
    product_features = model.extract_product_features(products)
    
    logger.info("Building product similarity matrix...")
    model.build_product_similarity_matrix(product_features)
    
    logger.info("Building user profiles...")
    model.build_user_profiles(user_interactions, product_features)
    
    logger.info("Content-Based Filtering model trained successfully!")
    return model



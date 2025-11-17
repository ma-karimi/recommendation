from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import os
import gc
import logging
from multiprocessing import cpu_count, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import faiss

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history
from model_storage import ModelStorage
from settings import load_config

logger = logging.getLogger(__name__)


class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا با استفاده از ANN (Approximate Nearest Neighbor)"""
    
    def __init__(self, storage: Optional[ModelStorage] = None, use_storage: bool = True):
        self.product_features = None
        self.user_profiles = None
        self.product_to_index = {}
        self.index_to_product = {}
        self.vectorizer = None
        self.faiss_index = None
        self.index_path = None
        self.use_storage = use_storage
        self.storage = storage if storage else (ModelStorage() if use_storage else None)
        self.vector_dim = None
    
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
    
    def build_tfidf_vectors(self, product_features: Dict[int, Dict]) -> None:
        """
        ساخت بردارهای TF-IDF و ذخیره در DuckDB (بدون ساخت ماتریس کامل)
        """
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        logger.info(f"Building TF-IDF vectors for {n_products} products...")
        
        # استخراج متن برای TF-IDF
        texts = [product_features[pid]['text'] for pid in product_ids]
        
        # محاسبه TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # برای فارسی
            ngram_range=(1, 2)
        )
        
        # Transform به صورت batch برای صرفه‌جویی در حافظه
        logger.info("Computing TF-IDF vectors...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # ذخیره بردارها در DuckDB به صورت batch
        logger.info("Saving TF-IDF vectors to DuckDB...")
        product_vectors = {}
        
        # تبدیل sparse matrix به dict و نرمال‌سازی برای cosine similarity
        from sklearn.preprocessing import normalize
        # نرمال‌سازی تمام بردارها (برای cosine similarity)
        tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2')
        
        for i, product_id in enumerate(product_ids):
            # استخراج بردار برای هر محصول (بدون ساخت ماتریس کامل)
            vector = tfidf_matrix_normalized[i:i+1]  # Get single row as sparse matrix
            product_vectors[product_id] = vector
        
        # ذخیره در DuckDB
        if self.storage:
            self.storage.save_product_vectors_batch(product_vectors)
            self.storage.save_tfidf_vectorizer(self.vectorizer)
            
            # ذخیره نگاشت
            product_mapping_data = [
                {'product_id': pid, 'product_index': i}
                for i, pid in enumerate(product_ids)
            ]
            if product_mapping_data:
                import polars as pl
                df = pl.DataFrame(product_mapping_data)
                self.storage.conn.execute("DELETE FROM product_index_mapping")
                self.storage.conn.execute("INSERT INTO product_index_mapping SELECT * FROM df")
                self.storage.conn.commit()
        
        # ذخیره نگاشت در حافظه
        self.product_to_index = {pid: i for i, pid in enumerate(product_ids)}
        self.index_to_product = {i: pid for pid, i in self.product_to_index.items()}
        
        # ذخیره بعد بردار
        self.vector_dim = tfidf_matrix.shape[1]
        
        # پاک کردن ماتریس از حافظه
        del tfidf_matrix
        gc.collect()
        
        logger.info(f"TF-IDF vectors saved. Vector dimension: {self.vector_dim}")
    
    def build_ann_index(self, index_type: str = "IVF", nlist: int = 100, nprobe: int = 10, n_threads: int = -1) -> None:
        """
        ساخت ANN index با استفاده از Faiss به صورت incremental از DuckDB
        
        Args:
            index_type: نوع index ("IVF", "Flat", "HNSW")
            nlist: تعداد clusters برای IVF
            nprobe: تعداد clusters برای جستجو در IVF
            n_threads: تعداد thread برای Faiss (-1 = همه هسته‌ها)
        """
        if not self.storage:
            raise ValueError("Storage is required for building ANN index")
        
        # تنظیم تعداد thread برای Faiss
        if n_threads == -1:
            n_threads = cpu_count()
        faiss.omp_set_num_threads(n_threads)
        
        logger.info(f"Building ANN index from DuckDB using {n_threads} threads...")
        
        # دریافت تمام product IDs
        product_ids = self.storage.get_all_product_ids_with_vectors()
        if not product_ids:
            raise ValueError("No product vectors found in storage")
        
        # دریافت بعد بردار
        vector_dim = self.storage.get_vector_dimension()
        if not vector_dim:
            raise ValueError("Vector dimension not found")
        
        self.vector_dim = vector_dim
        n_products = len(product_ids)
        
        logger.info(f"Building index for {n_products} products with dimension {vector_dim}...")
        
        # ساخت index بر اساس نوع
        # استفاده از Inner Product برای cosine similarity (vectors are normalized)
        if index_type == "IVF":
            # IVF (Inverted File Index) - مناسب برای مجموعه‌های بزرگ
            # استفاده از Inner Product برای cosine similarity
            quantizer = faiss.IndexFlatIP(vector_dim)
            nlist = min(nlist, max(1, n_products // 10))
            self.faiss_index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist)
            self.faiss_index.nprobe = nprobe
        elif index_type == "HNSW":
            # HNSW (Hierarchical Navigable Small World) - سریع‌تر اما حافظه بیشتر
            self.faiss_index = faiss.IndexHNSWFlat(vector_dim, 32)
            # HNSW با Inner Product
            self.faiss_index.metric = faiss.METRIC_INNER_PRODUCT
        else:
            # Flat index با Inner Product برای cosine similarity
            self.faiss_index = faiss.IndexFlatIP(vector_dim)
        
        # بارگذاری و اضافه کردن بردارها به صورت batch
        batch_size = 1000
        n_batches = (n_products + batch_size - 1) // batch_size
        
        logger.info(f"Loading vectors in {n_batches} batches of {batch_size}...")
        
        # برای IVF، ابتدا باید train شود
        if isinstance(self.faiss_index, faiss.IndexIVFFlat):
            # Train با نمونه‌ای از بردارها
            logger.info("Training IVF index...")
            train_batch_size = min(10000, n_products)
            train_ids = product_ids[:train_batch_size]
            train_vectors = self.storage.load_product_vectors_batch(train_ids)
            
            # تبدیل به numpy array
            train_data = np.array([train_vectors[pid] for pid in train_ids if pid in train_vectors], dtype=np.float32)
            
            if len(train_data) > 0:
                self.faiss_index.train(train_data)
                logger.info("IVF index trained")
            
            del train_data, train_vectors
            gc.collect()
        
        # Initialize mapping
        self.product_to_index = {}
        self.index_to_product = {}
        
        # اضافه کردن تمام بردارها
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_products)
            batch_ids = product_ids[start_idx:end_idx]
            
            # بارگذاری بردارها از DuckDB
            batch_vectors = self.storage.load_product_vectors_batch(batch_ids)
            
            if not batch_vectors:
                continue
            
            # تبدیل به numpy array
            vectors_list = []
            valid_ids = []
            for pid in batch_ids:
                if pid in batch_vectors:
                    vectors_list.append(batch_vectors[pid])
                    valid_ids.append(pid)
            
            if vectors_list:
                vectors_array = np.array(vectors_list, dtype=np.float32)
                
                # ذخیره موقعیت فعلی قبل از add
                start_faiss_idx = self.faiss_index.ntotal
                
                # اضافه کردن به index
                self.faiss_index.add(vectors_array)
                
                # به‌روزرسانی نگاشت - هر product_id به موقعیت خود در index
                for local_idx, pid in enumerate(valid_ids):
                    faiss_idx = start_faiss_idx + local_idx
                    self.product_to_index[pid] = faiss_idx
                    self.index_to_product[faiss_idx] = pid
            
            del batch_vectors, vectors_list, vectors_array
            gc.collect()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{n_batches} batches ({self.faiss_index.ntotal} vectors added)")
        
        logger.info(f"ANN index built successfully with {self.faiss_index.ntotal} vectors")
        
        # ذخیره index در دیسک
        self._save_ann_index()
    
    def _save_ann_index(self) -> None:
        """ذخیره ANN index در دیسک"""
        if not self.faiss_index:
            return
        
        cfg = load_config()
        index_dir = cfg.output_dir
        os.makedirs(index_dir, exist_ok=True)
        
        self.index_path = os.path.join(index_dir, "product_ann_index.faiss")
        
        logger.info(f"Saving ANN index to {self.index_path}...")
        faiss.write_index(self.faiss_index, self.index_path)
        
        # ذخیره مسیر در metadata
        if self.storage:
            self.storage.save_ann_index_path(self.index_path)
        
        logger.info("ANN index saved successfully")
    
    def _load_ann_index(self, n_threads: int = -1) -> bool:
        """بارگذاری ANN index از دیسک"""
        if not self.storage:
            return False
        
        index_path = self.storage.get_ann_index_path()
        if not index_path or not os.path.exists(index_path):
            return False
        
        # تنظیم تعداد thread برای Faiss
        if n_threads == -1:
            n_threads = cpu_count()
        faiss.omp_set_num_threads(n_threads)
        
        logger.info(f"Loading ANN index from {index_path} using {n_threads} threads...")
        try:
            self.faiss_index = faiss.read_index(index_path)
            self.index_path = index_path
            
            # بارگذاری نگاشت از product_index_mapping در DuckDB
            result = self.storage.conn.execute("""
                SELECT product_id, product_index 
                FROM product_index_mapping 
                ORDER BY product_index
            """).fetchall()
            
            self.product_to_index = {}
            self.index_to_product = {}
            
            for product_id, product_index in result:
                self.product_to_index[product_id] = product_index
                self.index_to_product[product_index] = product_id
            
            # بارگذاری vectorizer
            self.vectorizer = self.storage.load_tfidf_vectorizer()
            
            # دریافت بعد بردار
            self.vector_dim = self.storage.get_vector_dimension()
            
            logger.info(f"ANN index loaded successfully ({self.faiss_index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.error(f"Failed to load ANN index: {e}")
            return False
    
    def get_similar_products(self, product_id: int, k: int = 20) -> List[Tuple[int, float]]:
        """
        دریافت محصولات مشابه با استفاده از ANN (بدون بارگذاری ماتریس کامل)
        
        Args:
            product_id: شناسه محصول
            k: تعداد محصولات مشابه
        
        Returns:
            لیست tuple های (product_id, similarity_score)
        """
        # بارگذاری index اگر وجود ندارد
        if self.faiss_index is None:
            if not self._load_ann_index():
                logger.warning("ANN index not available")
                return []
        
        # بارگذاری بردار محصول از DuckDB
        if not self.storage:
            logger.warning("Storage not available")
            return []
        
        # Use read-only connection for querying
        product_vector = self.storage.load_product_vector(product_id)
        if product_vector is None:
            logger.warning(f"Vector not found for product {product_id}")
            return []
        
        # تبدیل به numpy array و reshape
        query_vector = np.array([product_vector], dtype=np.float32)
        
        # جستجو در ANN index
        k = min(k + 1, self.faiss_index.ntotal)  # +1 because result includes the query itself
        distances, indices = self.faiss_index.search(query_vector, k)
        
        # تبدیل نتایج به product_id و similarity
        similar_products = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
            
            similar_product_id = self.index_to_product.get(idx)
            if similar_product_id is None or similar_product_id == product_id:
                continue
            
            # برای Inner Product index، distance در واقع similarity است (cosine similarity برای normalized vectors)
            # Inner product روی normalized vectors = cosine similarity
            similarity = float(distance)
            
            # اطمینان از اینکه similarity در محدوده معقول است
            similarity = max(0.0, min(1.0, similarity))
            
            similar_products.append((similar_product_id, similarity))
        
        # مرتب‌سازی بر اساس similarity
        similar_products.sort(key=lambda x: x[1], reverse=True)
        
        return similar_products[:k-1]  # Exclude the query product itself
    
    def _add_numerical_similarities(self, product_features: Dict[int, Dict], 
                                   similar_products: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        اضافه کردن شباهت بر اساس ویژگی‌های عددی به نتایج ANN
        """
        if not product_features or not similar_products:
            return similar_products
        
        # ترکیب شباهت متنی با شباهت عددی
        enhanced_similarities = []
        
        for product_id, text_similarity in similar_products:
            if product_id not in product_features:
                enhanced_similarities.append((product_id, text_similarity))
                continue
            
            # محاسبه شباهت عددی (این بخش نیاز به product_id اصلی دارد که در query استفاده می‌شود)
            # برای سادگی، فقط شباهت متنی را برمی‌گردانیم
            # در صورت نیاز می‌توان این بخش را گسترش داد
            enhanced_similarities.append((product_id, text_similarity))
        
        return enhanced_similarities
    
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
        """دریافت توصیه‌های کاربر با استفاده از ANN"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        product_weights = user_profile['product_weights']
        
        # پیدا کردن محصولات مشابه به محصولات مورد علاقه با استفاده از ANN
        recommendations = []
        seen_products = set(product_weights.keys())
        
        for liked_product_id, weight in product_weights.items():
            # استفاده از ANN برای پیدا کردن محصولات مشابه
            similar_products = self.get_similar_products(liked_product_id, k=top_k * 2)
            
            for similar_product_id, similarity in similar_products:
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
                reason="توصیه بر اساس محصولات مشابه (ANN)",
                confidence=min(score, 1.0)
            ))
        
        return final_recommendations


def train_content_based_model(
    products: List[Product], 
    user_interactions: Dict[int, List[ProductInteraction]],
    storage: Optional[ModelStorage] = None,
    use_storage: bool = True,
    rebuild_index: bool = True,
    n_threads: int = -1
) -> ContentBasedFiltering:
    """
    آموزش مدل content-based filtering با استفاده از ANN
    
    Args:
        products: لیست محصولات
        user_interactions: تعاملات کاربران
        storage: ModelStorage instance (اختیاری)
        use_storage: استفاده از storage برای ذخیره‌سازی
        rebuild_index: ساخت مجدد ANN index (اگر False، از index موجود استفاده می‌کند)
        n_threads: تعداد thread برای Faiss (-1 = همه هسته‌ها)
    """
    logger.info("Training Content-Based Filtering model with ANN...")
    
    # تنظیم تعداد thread
    if n_threads == -1:
        n_threads = cpu_count()
    logger.info(f"Using {n_threads} CPU threads for training...")
    
    model = ContentBasedFiltering(storage=storage, use_storage=use_storage)
    
    # استخراج ویژگی‌های محصولات
    logger.info("Extracting product features...")
    product_features = model.extract_product_features(products)
    
    # ساخت بردارهای TF-IDF و ذخیره در DuckDB
    logger.info("Building TF-IDF vectors...")
    model.build_tfidf_vectors(product_features)
    
    # ساخت ANN index
    if rebuild_index or not model._load_ann_index(n_threads=n_threads):
        logger.info("Building ANN index...")
        # استفاده از IVF برای مجموعه‌های بزرگ
        n_products = len(products)
        nlist = min(100, max(10, n_products // 100))  # تعداد clusters
        model.build_ann_index(index_type="IVF", nlist=nlist, nprobe=10, n_threads=n_threads)
    else:
        logger.info("Using existing ANN index")
    
    # ساخت پروفایل کاربران
    logger.info("Building user profiles...")
    model.build_user_profiles(user_interactions, product_features)
    
    # ذخیره product features در storage
    if model.storage:
        model.storage.save_content_model(
            user_profiles=model.user_profiles,
            product_features=product_features,
            product_similarities_path=None  # No longer using similarity matrix
        )
    
    logger.info("Content-Based Filtering model trained successfully!")
    return model

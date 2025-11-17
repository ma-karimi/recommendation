from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pickle

from models import User, Product, ProductInteraction, Recommendation
from model_storage import ModelStorage
from settings import load_config

logger = logging.getLogger(__name__)


class ContentBasedFiltering:
    """سیستم توصیه مبتنی بر محتوا با استفاده از ANN (Approximate Nearest Neighbor)"""
    
    def __init__(self, storage: Optional[ModelStorage] = None):
        """
        Args:
            storage: ModelStorage instance for DuckDB operations. If None, creates a new one.
        """
        self.storage = storage or ModelStorage()
        self.vectorizer = None
        self.ann_index = None
        self.product_to_index = {}
        self.index_to_product = {}
        self.vector_dim = None
        self.index_path = None
        self.user_profiles = None
        
        # Load config for index path
        cfg = load_config()
        self.index_path = os.path.join(cfg.output_dir, "product_ann_index.bin")
    
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
    
    def _compute_tfidf_vectors(self, product_features: Dict[int, Dict]) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        محاسبه بردارهای TF-IDF برای محصولات
        
        Returns:
            vectors: numpy array of shape (n_products, vector_dim)
            product_to_index: mapping from product_id to row index in vectors
        """
        product_ids = list(product_features.keys())
        n_products = len(product_ids)
        
        # استخراج متن برای TF-IDF
        texts = [product_features[pid]['text'] for pid in product_ids]
        
        # محاسبه TF-IDF
        logger.info(f"Computing TF-IDF vectors for {n_products} products...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # برای فارسی
            ngram_range=(1, 2)
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # تبدیل به numpy array (dense)
        # این فقط برای ساخت index است و بعد از آن حذف می‌شود
        vectors = tfidf_matrix.toarray().astype('float32')
        vector_dim = vectors.shape[1]
        
        # ایجاد نگاشت
        product_to_index = {pid: i for i, pid in enumerate(product_ids)}
        
        logger.info(f"TF-IDF vectors computed: shape {vectors.shape}, dimension {vector_dim}")
        return vectors, product_to_index, vector_dim
    
    def _store_vectors_in_duckdb(self, vectors: np.ndarray, product_to_index: Dict[int, int]) -> None:
        """
        ذخیره بردارهای TF-IDF در DuckDB به صورت batch
        """
        logger.info("Storing TF-IDF vectors in DuckDB...")
        
        # Ensure table exists
        self.storage.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_tfidf_vectors (
                product_id INTEGER PRIMARY KEY,
                vector_index INTEGER,
                vector_data BLOB  -- Pickled numpy array
            )
        """)
        
        # Clear existing data
        self.storage.conn.execute("DELETE FROM product_tfidf_vectors")
        
        # Store vectors in batches
        batch_size = 1000
        n_products = len(product_to_index)
        index_to_product = {idx: pid for pid, idx in product_to_index.items()}
        
        for i in range(0, n_products, batch_size):
            batch_data = []
            end_idx = min(i + batch_size, n_products)
            
            for vec_idx in range(i, end_idx):
                product_id = index_to_product[vec_idx]
                vector = vectors[vec_idx]
                # Store as pickle for efficient serialization
                vector_blob = pickle.dumps(vector)
                batch_data.append({
                    'product_id': product_id,
                    'vector_index': vec_idx,
                    'vector_data': vector_blob
                })
            
            if batch_data:
                import polars as pl
                df = pl.DataFrame(batch_data)
                self.storage.conn.execute("INSERT INTO product_tfidf_vectors SELECT * FROM df")
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"Stored {end_idx}/{n_products} vectors...")
        
        self.storage.conn.commit()
        logger.info("All TF-IDF vectors stored in DuckDB")
    
    def _load_vectors_from_duckdb_batch(self, batch_size: int = 1000) -> Tuple[np.ndarray, List[int]]:
        """
        بارگذاری بردارها از DuckDB به صورت batch
        
        Returns:
            vectors: numpy array of shape (batch_size, vector_dim)
            product_ids: list of product IDs corresponding to rows
        """
        result = self.storage.conn.execute(f"""
            SELECT product_id, vector_index, vector_data
            FROM product_tfidf_vectors
            ORDER BY vector_index
            LIMIT {batch_size}
            OFFSET ?
        """, [0]).fetchall()
        
        if not result:
            return None, []
        
        vectors_list = []
        product_ids = []
        
        for row in result:
            product_id, vec_idx, vector_blob = row
            vector = pickle.loads(vector_blob)
            vectors_list.append(vector)
            product_ids.append(product_id)
        
        vectors = np.array(vectors_list, dtype='float32')
        return vectors, product_ids
    
    def _build_ann_index_incremental(self, vector_dim: int, n_products: int) -> None:
        """
        ساخت ANN index به صورت incremental با استفاده از تمام CPU cores
        """
        logger.info(f"Building ANN index incrementally for {n_products} products (dim={vector_dim})...")
        
        # استفاده از IndexFlatIP (Inner Product) برای cosine similarity
        # بعداً normalize می‌کنیم تا cosine similarity شود
        self.ann_index = faiss.IndexFlatIP(vector_dim)
        
        # Enable threading for Faiss operations
        faiss.omp_set_num_threads(os.cpu_count() or 4)
        logger.info(f"Using {os.cpu_count() or 4} CPU cores for index building")
        
        # Load all vectors in batches and add to index
        batch_size = 5000  # Process in batches to manage memory
        total_loaded = 0
        
        # First, get total count
        count_result = self.storage.conn.execute("SELECT COUNT(*) FROM product_tfidf_vectors").fetchone()
        total_vectors = count_result[0] if count_result else 0
        
        offset = 0
        while offset < total_vectors:
            # Load batch
            result = self.storage.conn.execute("""
                SELECT product_id, vector_index, vector_data
                FROM product_tfidf_vectors
                ORDER BY vector_index
                LIMIT ? OFFSET ?
            """, [batch_size, offset]).fetchall()
            
            if not result:
                break
            
            # Deserialize vectors
            vectors_list = []
            product_ids_batch = []
            
            for row in result:
                product_id, vec_idx, vector_blob = row
                vector = pickle.loads(vector_blob)
                vectors_list.append(vector)
                product_ids_batch.append(product_id)
            
            if not vectors_list:
                break
            
            # Convert to numpy array
            vectors_batch = np.array(vectors_list, dtype='float32')
            
            # Normalize vectors for cosine similarity (L2 normalization)
            faiss.normalize_L2(vectors_batch)
            
            # Add to index
            self.ann_index.add(vectors_batch)
            
            # Update mappings
            for i, product_id in enumerate(product_ids_batch):
                actual_index = total_loaded + i
                self.product_to_index[product_id] = actual_index
                self.index_to_product[actual_index] = product_id
            
            total_loaded += len(vectors_batch)
            offset += batch_size
            
            if total_loaded % 10000 == 0 or total_loaded == total_vectors:
                logger.info(f"Indexed {total_loaded}/{total_vectors} vectors...")
        
        logger.info(f"ANN index built successfully with {self.ann_index.ntotal} vectors")
    
    def build_product_similarity_index(self, product_features: Dict[int, Dict], force_rebuild: bool = False) -> None:
        """
        ساخت ANN index برای محصولات (جایگزین similarity matrix)
        این تابع بردارهای TF-IDF را محاسبه می‌کند، در DuckDB ذخیره می‌کند،
        و سپس ANN index را به صورت incremental می‌سازد.
        
        Args:
            product_features: Dictionary of product features
            force_rebuild: If True, rebuild index even if it exists on disk
        """
        # Check if index already exists and can be loaded
        if not force_rebuild and self._load_index_from_disk():
            logger.info("Loaded existing ANN index from disk")
            return
        
        logger.info("Building product similarity index using ANN...")
        
        # 1. محاسبه بردارهای TF-IDF
        vectors, product_to_index, vector_dim = self._compute_tfidf_vectors(product_features)
        self.vector_dim = vector_dim
        self.product_to_index = product_to_index
        self.index_to_product = {idx: pid for pid, idx in product_to_index.items()}
        
        # 2. ذخیره بردارها در DuckDB
        self._store_vectors_in_duckdb(vectors, product_to_index)
        
        # 3. پاک کردن vectors از RAM
        del vectors
        import gc
        gc.collect()
        
        # 4. ساخت ANN index به صورت incremental
        n_products = len(product_to_index)
        self._build_ann_index_incremental(vector_dim, n_products)
        
        # 5. ذخیره index روی disk
        self._save_index_to_disk()
        
        logger.info("Product similarity index built and saved successfully")
    
    def _save_index_to_disk(self) -> None:
        """ذخیره ANN index روی disk"""
        if self.ann_index is None:
            return
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.ann_index, self.index_path)
        
        # Save mappings and metadata
        metadata = {
            'product_to_index': self.product_to_index,
            'index_to_product': self.index_to_product,
            'vector_dim': self.vector_dim
        }
        
        metadata_path = self.index_path.replace('.bin', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save vectorizer
        if self.vectorizer is not None:
            vectorizer_path = self.index_path.replace('.bin', '_vectorizer.pkl')
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        # Store index path in DuckDB metadata
        self.storage.conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('product_ann_index_path', ?)
        """, [self.index_path])
        self.storage.conn.commit()
        
        logger.info(f"ANN index saved to {self.index_path}")
    
    def _load_index_from_disk(self) -> bool:
        """بارگذاری ANN index از disk"""
        # Check if index exists
        if not os.path.exists(self.index_path):
            # Try loading path from metadata
            result = self.storage.conn.execute("""
                SELECT value FROM model_metadata WHERE key = 'product_ann_index_path'
            """).fetchone()
            
            if result:
                self.index_path = result[0]
            
            if not os.path.exists(self.index_path):
                return False
        
        # Load index
        self.ann_index = faiss.read_index(self.index_path)
        
        # Load metadata
        metadata_path = self.index_path.replace('.bin', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.product_to_index = metadata['product_to_index']
                self.index_to_product = metadata['index_to_product']
                self.vector_dim = metadata['vector_dim']
        
        # Load vectorizer
        vectorizer_path = self.index_path.replace('.bin', '_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        logger.info(f"ANN index loaded from {self.index_path}")
        return True
    
    def get_similar_products(self, product_id: int, k: int = 20) -> List[Tuple[int, float]]:
        """
        دریافت محصولات مشابه برای یک محصول با استفاده از ANN index
        
        Args:
            product_id: شناسه محصول
            k: تعداد محصولات مشابه مورد نیاز
        
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if self.ann_index is None:
            # Try to load from disk
            if not self._load_index_from_disk():
                logger.error("ANN index not found. Please train the model first.")
                return []
        
        if product_id not in self.product_to_index:
            logger.warning(f"Product {product_id} not found in index")
            return []
        
        # Get product vector from DuckDB
        result = self.storage.conn.execute("""
            SELECT vector_data FROM product_tfidf_vectors WHERE product_id = ?
        """, [product_id]).fetchone()
        
        if not result:
            logger.warning(f"Vector not found for product {product_id}")
            return []
        
        # Deserialize vector
        vector = pickle.loads(result[0])
        query_vector = vector.reshape(1, -1).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search in ANN index
        # k+1 because the product itself will be in results
        distances, indices = self.ann_index.search(query_vector, k + 1)
        
        # Convert indices to product IDs and filter out the query product itself
        similar_products = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.index_to_product):
                similar_product_id = self.index_to_product[idx]
                if similar_product_id != product_id:  # Exclude self
                    # dist is already cosine similarity (inner product after normalization)
                    similarity = float(dist)
                    similar_products.append((similar_product_id, similarity))
        
        return similar_products[:k]
    
    def _add_numerical_similarities(self, product_features: Dict[int, Dict], 
                                   similar_products: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        اضافه کردن شباهت بر اساس ویژگی‌های عددی به نتایج ANN
        این تابع نتایج ANN را با در نظر گیری ویژگی‌های عددی تنظیم می‌کند.
        """
        if not similar_products:
            return similar_products
        
        # Get query product features (assume it's passed via context)
        # For now, we'll adjust scores based on numerical features
        adjusted_products = []
        
        for product_id, ann_score in similar_products:
            if product_id not in product_features:
                adjusted_products.append((product_id, ann_score))
                continue
            
            # Numerical similarity boost (simplified - in practice, you'd compare with query product)
            # This is a placeholder - in real implementation, you'd need the query product_id
            adjusted_products.append((product_id, ann_score))
        
        return adjusted_products
    
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
        
        # پیدا کردن محصولات مشابه به محصولات مورد علاقه با استفاده از ANN
        recommendations = []
        seen_products = set(product_weights.keys())
        
        # Collect similar products from all liked products
        all_similar = defaultdict(float)
        
        for liked_product_id, weight in product_weights.items():
            # Get similar products using ANN
            similar_products = self.get_similar_products(liked_product_id, k=50)
            
            for similar_product_id, similarity in similar_products:
                if similar_product_id not in seen_products:
                    # Combine similarity with user's preference weight
                    score = similarity * weight
                    all_similar[similar_product_id] = max(all_similar[similar_product_id], score)
        
        # Sort by score and take top_k
        sorted_recommendations = sorted(
            all_similar.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        final_recommendations = []
        for product_id, score in sorted_recommendations:
            final_recommendations.append(Recommendation(
                user_id=user_id,
                product_id=product_id,
                score=score,
                reason="توصیه بر اساس محصولات مشابه (ANN)",
                confidence=min(score, 1.0)
            ))
        
        return final_recommendations


def train_content_based_model(products: List[Product], 
                            user_interactions: Dict[int, List[ProductInteraction]],
                            storage: Optional[ModelStorage] = None) -> ContentBasedFiltering:
    """
    آموزش مدل content-based filtering با استفاده از ANN
    
    Args:
        products: لیست محصولات
        user_interactions: تعاملات کاربران
        storage: ModelStorage instance (optional)
    """
    model = ContentBasedFiltering(storage=storage)
    product_features = model.extract_product_features(products)
    
    # Build ANN index instead of full similarity matrix
    model.build_product_similarity_index(product_features)
    
    # Build user profiles
    model.build_user_profiles(user_interactions, product_features)
    
    return model

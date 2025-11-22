from __future__ import annotations
import logging
import gc
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math
import json
from multiprocessing import Pool, cpu_count
from functools import partial

from models import User, Product, ProductInteraction, Recommendation
from object_loader import load_user_purchase_history
from model_storage import ModelStorage

# تنظیم logger
logger = logging.getLogger(__name__)


class CollaborativeFiltering:
    """سیستم توصیه مبتنی بر همکاری کاربران"""
    
    def __init__(self, min_common_items: int = 2, n_jobs: int = 1, use_storage: bool = False, storage: Optional[ModelStorage] = None):
        """
        Args:
            min_common_items: حداقل تعداد آیتم مشترک برای محاسبه شباهت
            n_jobs: تعداد هسته CPU برای محاسبه شباهت‌ها (1 = sequential)
            use_storage: If True, use DuckDB storage instead of in-memory matrices
            storage: ModelStorage instance (created if None and use_storage=True)
        """
        self.min_common_items = min_common_items
        self.user_item_matrix = None
        self.user_similarities = None
        self.n_jobs = n_jobs
        self.use_storage = use_storage
        self.storage = storage or (ModelStorage() if use_storage else None)
        
        # Mappings are always needed for index conversion
        self.user_to_index = {}
        self.product_to_index = {}
        self.index_to_user = {}
        self.index_to_product = {}
    
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
        """
        محاسبه شباهت بین کاربران و ذخیره مستقیم در DuckDB (بدون ساخت ماتریس کامل)
        
        این متد شباهت کسینوسی بین همه جفت کاربران را محاسبه می‌کند و مستقیماً در storage ذخیره می‌کند
        """
        n_users = self.user_item_matrix.shape[0]
        
        logger.info(f"Calculating user similarities for {n_users} users (streaming to DuckDB)...")
        
        # اگر storage داریم، مستقیماً در DuckDB ذخیره می‌کنیم (بدون ساخت ماتریس کامل)
        if self.use_storage and self.storage:
            self._calculate_and_save_similarities_streaming(n_users)
        else:
            # Fallback: ساخت ماتریس کامل (فقط اگر storage نداریم)
            logger.warning("No storage available, creating full similarity matrix in memory")
            self.user_similarities = np.zeros((n_users, n_users))
            
            # تعیین تعداد هسته‌ها
            if self.n_jobs == -1:
                n_jobs = cpu_count()
            elif self.n_jobs <= 0:
                n_jobs = 1
            else:
                n_jobs = self.n_jobs
            
            logger.info(f"Using {n_jobs} CPU core(s) for similarity calculation...")
            
            if n_jobs > 1 and n_users > 100:
                # پردازش موازی
                try:
                    self._calculate_similarities_parallel(n_jobs, n_users)
                except Exception as e:
                    logger.warning(f"Parallel calculation failed, falling back to sequential: {e}")
                    self._calculate_similarities_sequential(n_users)
            else:
                # پردازش sequential
                self._calculate_similarities_sequential(n_users)
        
        logger.info("User similarities calculated successfully!")
    
    def _calculate_and_save_similarities_streaming(self, n_users: int) -> None:
        """محاسبه و ذخیره شباهت‌ها به صورت streaming در DuckDB (بدون ساخت ماتریس کامل)"""
        logger.info("Calculating similarities and saving directly to DuckDB (no full matrix)...")
        
        # پاک کردن شباهت‌های قبلی
        conn = self.storage._get_connection(read_only=False)
        conn.execute("DELETE FROM user_similarities")
        
        # محاسبه و ذخیره به صورت batch
        batch_size = 100  # تعداد کاربران در هر batch
        similarity_data = []
        total_pairs = 0
        
        for i in range(n_users):
            user_id_1 = self.index_to_user[i]
            user_vector_1 = self.user_item_matrix[i]
            
            # محاسبه شباهت با کاربران بعدی (فقط upper triangle)
            for j in range(i, n_users):
                user_id_2 = self.index_to_user[j]
                user_vector_2 = self.user_item_matrix[j]
                
                similarity = _cosine_similarity_helper(user_vector_1, user_vector_2)
                
                # فقط شباهت‌های معنی‌دار را ذخیره می‌کنیم
                if similarity > 0.01:
                    similarity_data.append({
                        'user_id_1': user_id_1,
                        'user_id_2': user_id_2,
                        'similarity': float(similarity)
                    })
                    total_pairs += 1
                
                # ذخیره batch در DuckDB
                if len(similarity_data) >= 10000:
                    import polars as pl
                    df = pl.DataFrame(similarity_data)
                    conn.execute("INSERT INTO user_similarities SELECT * FROM df")
                    similarity_data = []
                    gc.collect()
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{n_users} users ({total_pairs} similarities saved)")
        
        # ذخیره باقی‌مانده
        if similarity_data:
            import polars as pl
            df = pl.DataFrame(similarity_data)
            conn.execute("INSERT INTO user_similarities SELECT * FROM df")
        
        conn.commit()
        logger.info(f"Saved {total_pairs} user similarities to DuckDB")
        
        # ماتریس شباهت را None می‌کنیم چون در storage ذخیره شده
        self.user_similarities = None
    
    def _calculate_similarities_sequential(self, n_users: int) -> None:
        """محاسبه sequential شباهت‌ها"""
        for i in range(n_users):
            if (i + 1) % 100 == 0:
                logger.info(f"Processing user {i + 1}/{n_users}...")
            
            for j in range(i + 1, n_users):
                similarity = self._cosine_similarity(
                    self.user_item_matrix[i],
                    self.user_item_matrix[j]
                )
                self.user_similarities[i, j] = similarity
                self.user_similarities[j, i] = similarity
    
    def _calculate_similarities_parallel(self, n_jobs: int, n_users: int) -> None:
        """محاسبه موازی شباهت‌ها"""
        # آماده‌سازی داده‌ها برای پردازش موازی
        # هر task یک range از کاربران را پردازش می‌کند
        tasks = []
        chunk_size = max(50, n_users // (n_jobs * 2))
        
        for i in range(0, n_users, chunk_size):
            end_i = min(i + chunk_size, n_users)
            tasks.append((i, end_i, n_users))
        
        logger.info(f"Processing {len(tasks)} chunks in parallel...")
        
        # استفاده از Pool برای پردازش موازی
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                _calculate_similarity_chunk_worker,
                [
                    (
                        self.user_item_matrix,
                        start_i,
                        end_i,
                        n_users
                    )
                    for start_i, end_i, n_users in tasks
                ]
            )
        
        # ترکیب نتایج
        for chunk_similarities, start_i, end_i in results:
            for i in range(start_i, end_i):
                for j in range(n_users):
                    if chunk_similarities[i - start_i, j] != 0:
                        self.user_similarities[i, j] = chunk_similarities[i - start_i, j]
    
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
        if self.use_storage:
            return self._get_user_recommendations_from_storage(user_id, top_k)
        
        # Original in-memory implementation
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
    
    def _get_user_recommendations_from_storage(self, user_id: int, top_k: int = 10) -> List[Recommendation]:
        """Get recommendations using DuckDB storage (memory-efficient)"""
        if not self.storage:
            return []
        
        # Load user's item ratings from storage
        user_ratings = self.storage.load_user_item_row(user_id)
        if not user_ratings:
            return []
        
        # Get similar users first
        similar_users = self.storage.load_user_similarities(user_id, top_k=100)
        if not similar_users:
            return []
        
        # Create a dict for quick lookup
        similar_user_dict = {uid: sim for uid, sim in similar_users}
        
        # Get products rated by similar users (more efficient than loading all products)
        similar_user_ids = list(similar_user_dict.keys())
        candidate_product_ids = self.storage.get_products_rated_by_users(similar_user_ids)
        
        # Find unseen products (only from candidates)
        unseen_product_ids = [pid for pid in candidate_product_ids if pid not in user_ratings]
        
        # Predict ratings for unseen products
        predictions = []
        batch_size = 1000
        
        for i in range(0, len(unseen_product_ids), batch_size):
            batch_products = unseen_product_ids[i:i + batch_size]
            
            # Load ratings for these products from similar users
            for product_id in batch_products:
                score, similar_users_details = self._predict_rating_from_storage(
                    user_id, product_id, user_ratings, similar_user_dict
                )
                if score > 0:
                    predictions.append((product_id, score, similar_users_details))
            
            # Cleanup
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        # Sort and select top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for product_id, score, similar_users_details in predictions[:top_k]:
            if similar_users_details:
                top_similar = similar_users_details[:5]
                details_json = json.dumps({
                    'similar_users': [
                        {'user_id': uid, 'similarity': round(sim, 4),
                         'similarity_percent': round(sim * 100, 2)}
                        for uid, sim in top_similar
                    ],
                    'total_similar_users': len(similar_users_details)
                }, ensure_ascii=False)
                
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
    
    def _predict_rating_from_storage(
        self,
        user_id: int,
        product_id: int,
        user_ratings: Dict[int, float],
        similar_user_dict: Dict[int, float]
    ) -> Tuple[float, List[Tuple[int, float]]]:
        """Predict rating using storage (memory-efficient)"""
        if not self.storage:
            return 0.0, []
        
        # Get ratings for this product from similar users
        similar_users_with_rating = []
        similar_users_details = []
        
        for similar_user_id, similarity in similar_user_dict.items():
            # Load this similar user's ratings
            similar_user_ratings = self.storage.load_user_item_row(similar_user_id)
            if product_id in similar_user_ratings:
                rating = similar_user_ratings[product_id]
                similar_users_with_rating.append((similarity, rating))
                similar_users_details.append((similar_user_id, similarity))
        
        if not similar_users_with_rating:
            return 0.0, []
        
        # Sort by similarity
        similar_users_details.sort(key=lambda x: x[1], reverse=True)
        
        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for similarity, rating in similar_users_with_rating:
            if similarity > 0:
                total_weight += similarity
                weighted_sum += similarity * rating
        
        if total_weight == 0:
            return 0.0, []
        
        return weighted_sum / total_weight, similar_users_details
    
    def _predict_rating(self, user_idx: int, product_idx: int) -> Tuple[float, List[Tuple[int, float]]]:
        """پیش‌بینی امتیاز محصول برای کاربر و برگرداندن جزئیات کاربران مشابه"""
        if self.use_storage:
            # This shouldn't be called in storage mode
            return 0.0, []
        
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
        if self.use_storage:
            if not self.storage:
                return []
            return self.storage.load_user_similarities(user_id, top_k)
        
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


def _calculate_similarity_chunk_worker(
    user_item_matrix: np.ndarray,
    start_i: int,
    end_i: int,
    n_users: int
) -> Tuple[np.ndarray, int, int]:
    """
    Worker function برای محاسبه شباهت یک chunk از کاربران
    
    Args:
        user_item_matrix: ماتریس کاربر-محصول
        start_i: شاخص شروع کاربران
        end_i: شاخص پایان کاربران
        n_users: تعداد کل کاربران
    
    Returns:
        Tuple of (chunk_similarities, start_i, end_i)
    """
    chunk_size = end_i - start_i
    chunk_similarities = np.zeros((chunk_size, n_users))
    
    for i in range(start_i, end_i):
        for j in range(n_users):
            if i != j:
                similarity = _cosine_similarity_helper(
                    user_item_matrix[i],
                    user_item_matrix[j]
                )
                chunk_similarities[i - start_i, j] = similarity
    
    return (chunk_similarities, start_i, end_i)


def _cosine_similarity_helper(user1: np.ndarray, user2: np.ndarray) -> float:
    """Helper function برای محاسبه شباهت کسینوسی (برای multiprocessing)"""
    dot_product = np.dot(user1, user2)
    norm1 = np.linalg.norm(user1)
    norm2 = np.linalg.norm(user2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def train_collaborative_model(
    interactions: List[ProductInteraction],
    n_jobs: int = -1,
    use_storage: bool = True,
    save_to_storage: bool = True
) -> CollaborativeFiltering:
    """
    آموزش مدل collaborative filtering (با پشتیبانی از multiprocessing و DuckDB storage)
    
    Args:
        interactions: لیست تعاملات کاربر-محصول
        n_jobs: تعداد هسته CPU برای استفاده (-1 = همه هسته‌ها، 1 = sequential)
        use_storage: If True, use DuckDB storage for inference (saves memory)
        save_to_storage: If True, save trained model to DuckDB after training
    
    Returns:
        CollaborativeFiltering model
    """
    logger.info("Training Collaborative Filtering model...")
    
    # تعیین تعداد هسته‌ها
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs <= 0:
        n_jobs = 1

    print(f"Using {n_jobs} CPU core(s) for training...")

    logger.info(f"Using {n_jobs} CPU core(s) for training...")
    
    # Create storage if needed
    storage = None
    if use_storage or save_to_storage:
        storage = ModelStorage()
    
    model = CollaborativeFiltering(n_jobs=n_jobs, use_storage=use_storage, storage=storage)
    
    logger.info("Building user-item matrix...")
    model.build_user_item_matrix(interactions)
    
    logger.info("Calculating user similarities...")
    model.calculate_user_similarities()
    
    # Save user-item matrix to storage and clear from memory
    # Note: user_similarities already saved in calculate_user_similarities() if using storage
    if save_to_storage and model.storage:
        logger.info("Saving user-item matrix to DuckDB storage...")
        # شباهت‌ها قبلاً در calculate_user_similarities ذخیره شدند (اگر use_storage=True)
        # فقط user-item matrix را ذخیره می‌کنیم
        model.storage.save_collaborative_model(
            user_item_matrix=model.user_item_matrix,
            user_similarities=model.user_similarities if model.user_similarities is not None else np.zeros((0, 0)),
            user_to_index=model.user_to_index,
            product_to_index=model.product_to_index,
            index_to_user=model.index_to_user,
            index_to_product=model.index_to_product
        )
        
        # Clear matrices from memory after saving
        logger.info("Clearing matrices from memory...")
        del model.user_item_matrix
        model.user_item_matrix = None
        if model.user_similarities is not None:
            del model.user_similarities
            model.user_similarities = None
        gc.collect()
        logger.info("Model saved to storage and cleared from memory")
    
    logger.info("Collaborative Filtering model trained successfully!")
    return model

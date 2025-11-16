"""
Model Storage Module - DuckDB-based persistence for recommendation models

This module handles saving and loading large model data (user factors, item factors,
similarities, embeddings) to/from DuckDB to minimize memory usage.

Key features:
- Save trained matrices to DuckDB after training
- Batch loading of only needed data during inference
- Memory-efficient storage using Parquet format
- Automatic cleanup and garbage collection
"""
from __future__ import annotations
import os
import gc
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import duckdb
import polars as pl

from settings import load_config

logger = logging.getLogger(__name__)


class ModelStorage:
    """Manages storage and retrieval of model data using DuckDB"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to DuckDB database file. If None, uses default location.
        """
        cfg = load_config()
        if db_path is None:
            db_path = os.path.join(cfg.output_dir, "model_data.duckdb")
        
        self.db_path = db_path
        self.conn = None
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Ensure database file and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        
        # Create tables for collaborative filtering data
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_item_matrix (
                user_id INTEGER,
                product_id INTEGER,
                score REAL,
                PRIMARY KEY (user_id, product_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_similarities (
                user_id_1 INTEGER,
                user_id_2 INTEGER,
                similarity REAL,
                PRIMARY KEY (user_id_1, user_id_2)
            )
        """)
        
        # Create tables for mappings
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_index_mapping (
                user_id INTEGER PRIMARY KEY,
                user_index INTEGER
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_index_mapping (
                product_id INTEGER PRIMARY KEY,
                product_index INTEGER
            )
        """)
        
        # Create tables for content-based filtering
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_data BLOB  -- Pickled dict
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_features (
                product_id INTEGER PRIMARY KEY,
                features_data BLOB  -- Pickled dict
            )
        """)
        
        # Store sparse matrix metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Create table for storing raw interactions during training
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_interactions (
                user_id INTEGER,
                product_id INTEGER,
                interaction_type TEXT,
                score REAL,
                timestamp TIMESTAMP,
                value REAL,
                PRIMARY KEY (user_id, product_id, interaction_type, timestamp)
            )
        """)
        
        # Create index for faster queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_user 
            ON training_interactions(user_id)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interactions_product 
            ON training_interactions(product_id)
        """)
        
        logger.info(f"Model storage initialized at {self.db_path}")
    
    def save_collaborative_model(
        self,
        user_item_matrix: np.ndarray,
        user_similarities: np.ndarray,
        user_to_index: Dict[int, int],
        product_to_index: Dict[int, int],
        index_to_user: Dict[int, int],
        index_to_product: Dict[int, int]
    ) -> None:
        """Save collaborative filtering model data to DuckDB"""
        logger.info("Saving collaborative filtering model to DuckDB...")
        
        # Clear existing data
        self.conn.execute("DELETE FROM user_item_matrix")
        self.conn.execute("DELETE FROM user_similarities")
        self.conn.execute("DELETE FROM user_index_mapping")
        self.conn.execute("DELETE FROM product_index_mapping")
        
        # Save user-item matrix (only non-zero values)
        logger.info("Saving user-item matrix...")
        user_item_data = []
        for user_idx in range(user_item_matrix.shape[0]):
            user_id = index_to_user[user_idx]
            for product_idx in range(user_item_matrix.shape[1]):
                score = user_item_matrix[user_idx, product_idx]
                if score > 0:
                    product_id = index_to_product[product_idx]
                    user_item_data.append({
                        'user_id': user_id,
                        'product_id': product_id,
                        'score': float(score)
                    })
        
        if user_item_data:
            df = pl.DataFrame(user_item_data)
            self.conn.execute("INSERT INTO user_item_matrix SELECT * FROM df")
        
        # Save user similarities (only non-zero, upper triangle)
        logger.info("Saving user similarities...")
        similarity_data = []
        n_users = user_similarities.shape[0]
        batch_size = 10000
        
        for i in range(n_users):
            user_id_1 = index_to_user[i]
            for j in range(i, n_users):
                similarity = user_similarities[i, j]
                if similarity > 0.01:  # Only save meaningful similarities
                    user_id_2 = index_to_user[j]
                    similarity_data.append({
                        'user_id_1': user_id_1,
                        'user_id_2': user_id_2,
                        'similarity': float(similarity)
                    })
            
            if len(similarity_data) >= batch_size:
                df = pl.DataFrame(similarity_data)
                self.conn.execute("INSERT INTO user_similarities SELECT * FROM df")
                similarity_data = []
                gc.collect()
        
        if similarity_data:
            df = pl.DataFrame(similarity_data)
            self.conn.execute("INSERT INTO user_similarities SELECT * FROM df")
        
        # Save mappings
        logger.info("Saving index mappings...")
        user_mapping_data = [
            {'user_id': uid, 'user_index': idx}
            for uid, idx in user_to_index.items()
        ]
        product_mapping_data = [
            {'product_id': pid, 'product_index': idx}
            for pid, idx in product_to_index.items()
        ]
        
        if user_mapping_data:
            df = pl.DataFrame(user_mapping_data)
            self.conn.execute("INSERT INTO user_index_mapping SELECT * FROM df")
        
        if product_mapping_data:
            df = pl.DataFrame(product_mapping_data)
            self.conn.execute("INSERT INTO product_index_mapping SELECT * FROM df")
        
        # Save metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('n_users', ?), ('n_products', ?)
        """, [str(len(user_to_index)), str(len(product_to_index))])
        
        self.conn.commit()
        logger.info("Collaborative model saved successfully")
    
    def load_user_item_row(self, user_id: int) -> Dict[int, float]:
        """Load a single user's item ratings from database"""
        result = self.conn.execute("""
            SELECT product_id, score
            FROM user_item_matrix
            WHERE user_id = ?
        """, [user_id]).fetchall()
        
        return {row[0]: row[1] for row in result}
    
    def load_user_similarities(self, user_id: int, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Load similar users for a given user"""
        query = """
            SELECT user_id_2, similarity
            FROM user_similarities
            WHERE user_id_1 = ? AND user_id_2 != ?
            UNION
            SELECT user_id_1, similarity
            FROM user_similarities
            WHERE user_id_2 = ? AND user_id_1 != ?
            ORDER BY similarity DESC
        """
        
        if top_k:
            query += f" LIMIT {top_k}"
        
        result = self.conn.execute(query, [user_id, user_id, user_id, user_id]).fetchall()
        return [(row[0], row[1]) for row in result]
    
    def get_user_index(self, user_id: int) -> Optional[int]:
        """Get user index for a given user_id"""
        result = self.conn.execute("""
            SELECT user_index FROM user_index_mapping WHERE user_id = ?
        """, [user_id]).fetchone()
        return result[0] if result else None
    
    def get_product_index(self, product_id: int) -> Optional[int]:
        """Get product index for a given product_id"""
        result = self.conn.execute("""
            SELECT product_index FROM product_index_mapping WHERE product_id = ?
        """, [product_id]).fetchone()
        return result[0] if result else None
    
    def get_all_user_ids(self) -> List[int]:
        """Get all user IDs in the model"""
        result = self.conn.execute("SELECT DISTINCT user_id FROM user_index_mapping").fetchall()
        return [row[0] for row in result]
    
    def get_all_product_ids(self) -> List[int]:
        """Get all product IDs in the model"""
        result = self.conn.execute("SELECT DISTINCT product_id FROM product_index_mapping").fetchall()
        return [row[0] for row in result]
    
    def get_products_rated_by_users(self, user_ids: List[int]) -> List[int]:
        """Get product IDs that are rated by any of the given users (more efficient than loading all)"""
        if not user_ids:
            return []
        
        placeholders = ','.join(['?'] * len(user_ids))
        result = self.conn.execute(f"""
            SELECT DISTINCT product_id
            FROM user_item_matrix
            WHERE user_id IN ({placeholders})
        """, user_ids).fetchall()
        
        return [row[0] for row in result]
    
    def save_content_model(
        self,
        user_profiles: Dict[int, Dict],
        product_features: Dict[int, Dict],
        product_similarities_path: Optional[str] = None
    ) -> None:
        """Save content-based filtering model data to DuckDB"""
        logger.info("Saving content-based filtering model to DuckDB...")
        
        # Clear existing data
        self.conn.execute("DELETE FROM user_profiles")
        self.conn.execute("DELETE FROM product_features")
        
        # Save user profiles
        logger.info("Saving user profiles...")
        profile_data = []
        for user_id, profile in user_profiles.items():
            profile_data.append({
                'user_id': user_id,
                'profile_data': pickle.dumps(profile)
            })
        
        if profile_data:
            # Save in batches
            batch_size = 1000
            for i in range(0, len(profile_data), batch_size):
                batch = profile_data[i:i + batch_size]
                df = pl.DataFrame(batch)
                self.conn.execute("INSERT INTO user_profiles SELECT * FROM df")
                gc.collect()
        
        # Save product features
        logger.info("Saving product features...")
        feature_data = []
        for product_id, features in product_features.items():
            feature_data.append({
                'product_id': product_id,
                'features_data': pickle.dumps(features)
            })
        
        if feature_data:
            batch_size = 1000
            for i in range(0, len(feature_data), batch_size):
                batch = feature_data[i:i + batch_size]
                df = pl.DataFrame(batch)
                self.conn.execute("INSERT INTO product_features SELECT * FROM df")
                gc.collect()
        
        # Save sparse matrix path if provided
        if product_similarities_path:
            self.conn.execute("""
                INSERT OR REPLACE INTO model_metadata (key, value)
                VALUES ('product_similarities_path', ?)
            """, [product_similarities_path])
        
        self.conn.commit()
        logger.info("Content model saved successfully")
    
    def load_user_profile(self, user_id: int) -> Optional[Dict]:
        """Load a single user's profile"""
        result = self.conn.execute("""
            SELECT profile_data FROM user_profiles WHERE user_id = ?
        """, [user_id]).fetchone()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def load_product_features(self, product_ids: List[int]) -> Dict[int, Dict]:
        """Load product features for a batch of products"""
        if not product_ids:
            return {}
        
        placeholders = ','.join(['?'] * len(product_ids))
        result = self.conn.execute(f"""
            SELECT product_id, features_data
            FROM product_features
            WHERE product_id IN ({placeholders})
        """, product_ids).fetchall()
        
        return {
            row[0]: pickle.loads(row[1])
            for row in result
        }
    
    def get_product_similarities_path(self) -> Optional[str]:
        """Get path to saved product similarities sparse matrix"""
        result = self.conn.execute("""
            SELECT value FROM model_metadata WHERE key = 'product_similarities_path'
        """).fetchone()
        
        return result[0] if result else None
    
    def save_product_similarities(self, product_similarities: csr_matrix, path: Optional[str] = None) -> str:
        """Save product similarities sparse matrix to disk"""
        if path is None:
            cfg = load_config()
            path = os.path.join(cfg.output_dir, "product_similarities.npz")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_npz(path, product_similarities)
        
        # Store path in metadata
        self.conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('product_similarities_path', ?)
        """, [path])
        self.conn.commit()
        
        logger.info(f"Product similarities saved to {path}")
        return path
    
    def load_product_similarities(self) -> Optional[csr_matrix]:
        """Load product similarities sparse matrix"""
        path = self.get_product_similarities_path()
        if path and os.path.exists(path):
            return load_npz(path)
        return None
    
    def save_interactions_batch(self, interactions_data: List[Dict]) -> None:
        """
        Save interactions to DuckDB in batches for memory-efficient processing
        
        Args:
            interactions_data: List of dicts with keys: user_id, product_id, interaction_type, score, timestamp, value
        """
        if not interactions_data:
            return
        
        # Convert to DataFrame with proper timestamp handling
        df = pl.DataFrame(interactions_data)
        
        # Convert timestamp column to proper datetime type if it exists
        if 'timestamp' in df.columns:
            # Handle None/null values and convert strings to datetime
            df = df.with_columns(
                pl.when(pl.col('timestamp').is_null())
                .then(None)
                .otherwise(
                    pl.col('timestamp').str.to_datetime()
                )
                .alias('timestamp')
            )
        
        # Register as temporary table
        self.conn.register('temp_interactions', df)
        
        # Delete existing rows with matching keys (excluding timestamp from WHERE clause to avoid type mismatch)
        # Since timestamp is part of PRIMARY KEY, we delete all matching user_id, product_id, interaction_type
        # and then insert new ones
        self.conn.execute("""
            DELETE FROM training_interactions 
            WHERE (user_id, product_id, interaction_type) IN (
                SELECT DISTINCT user_id, product_id, interaction_type FROM temp_interactions
            )
        """)
        
        # Insert new data
        self.conn.execute("INSERT INTO training_interactions SELECT * FROM temp_interactions")
        self.conn.unregister('temp_interactions')
        self.conn.commit()
    
    def get_interactions_batch(self, batch_size: int = 10000, offset: int = 0) -> List[Dict]:
        """Load interactions in batches from DuckDB"""
        result = self.conn.execute(f"""
            SELECT user_id, product_id, interaction_type, score, timestamp, value
            FROM training_interactions
            ORDER BY user_id, product_id
            LIMIT ? OFFSET ?
        """, [batch_size, offset]).fetchall()
        
        return [
            {
                'user_id': row[0],
                'product_id': row[1],
                'interaction_type': row[2],
                'score': row[3],
                'timestamp': row[4],
                'value': row[5]
            }
            for row in result
        ]
    
    def get_interactions_count(self) -> int:
        """Get total number of interactions stored"""
        result = self.conn.execute("SELECT COUNT(*) FROM training_interactions").fetchone()
        return result[0] if result else 0
    
    def clear_interactions(self) -> None:
        """Clear all training interactions"""
        self.conn.execute("DELETE FROM training_interactions")
        self.conn.commit()
    
    def build_user_item_matrix_from_db(self, batch_size: int = 10000) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Build user-item matrix incrementally from DuckDB interactions
        Returns: user_to_index, product_to_index, index_to_user, index_to_product
        """
        logger.info("Building user-item matrix from DuckDB...")
        
        # Get all unique users and products
        users_result = self.conn.execute("SELECT DISTINCT user_id FROM training_interactions ORDER BY user_id").fetchall()
        products_result = self.conn.execute("SELECT DISTINCT product_id FROM training_interactions ORDER BY product_id").fetchall()
        
        all_users = [row[0] for row in users_result]
        all_products = [row[0] for row in products_result]
        
        # Create mappings
        user_to_index = {user_id: i for i, user_id in enumerate(all_users)}
        product_to_index = {product_id: i for i, product_id in enumerate(all_products)}
        index_to_user = {i: user_id for user_id, i in user_to_index.items()}
        index_to_product = {i: product_id for product_id, i in product_to_index.items()}
        
        # Clear existing user_item_matrix
        self.conn.execute("DELETE FROM user_item_matrix")
        
        # Process interactions in batches and aggregate scores
        logger.info(f"Processing interactions for {len(all_users)} users and {len(all_products)} products...")
        
        # Use DuckDB to aggregate scores directly
        self.conn.execute("""
            INSERT INTO user_item_matrix (user_id, product_id, score)
            SELECT 
                user_id,
                product_id,
                SUM(score) as score
            FROM training_interactions
            GROUP BY user_id, product_id
        """)
        
        self.conn.commit()
        logger.info("User-item matrix built from DuckDB")
        
        return user_to_index, product_to_index, index_to_user, index_to_product
    
    def _save_similarities_batch(self, user_similarities: np.ndarray, index_to_user: Dict[int, int]) -> None:
        """Save user similarities to DuckDB in batches"""
        logger.info("Saving user similarities to DuckDB...")
        self.conn.execute("DELETE FROM user_similarities")
        
        n_users = user_similarities.shape[0]
        batch_size = 1000
        similarity_data = []
        
        for i in range(n_users):
            user_id_1 = index_to_user[i]
            for j in range(i, n_users):
                similarity = user_similarities[i, j]
                if similarity > 0.01:  # Only save meaningful similarities
                    user_id_2 = index_to_user[j]
                    similarity_data.append({
                        'user_id_1': user_id_1,
                        'user_id_2': user_id_2,
                        'similarity': float(similarity)
                    })
            
            if len(similarity_data) >= batch_size:
                df = pl.DataFrame(similarity_data)
                self.conn.register('temp_similarities', df)
                self.conn.execute("INSERT INTO user_similarities SELECT * FROM temp_similarities")
                self.conn.unregister('temp_similarities')
                similarity_data = []
                gc.collect()
        
        if similarity_data:
            df = pl.DataFrame(similarity_data)
            self.conn.register('temp_similarities', df)
            self.conn.execute("INSERT INTO user_similarities SELECT * FROM temp_similarities")
            self.conn.unregister('temp_similarities')
        
        self.conn.commit()
        logger.info("User similarities saved to DuckDB")
    
    def _save_mappings(self, user_to_index: Dict[int, int], product_to_index: Dict[int, int],
                      index_to_user: Dict[int, int], index_to_product: Dict[int, int]) -> None:
        """Save index mappings to DuckDB"""
        logger.info("Saving index mappings to DuckDB...")
        
        # Save user mappings
        self.conn.execute("DELETE FROM user_index_mapping")
        user_mapping_data = [
            {'user_id': uid, 'user_index': idx}
            for uid, idx in user_to_index.items()
        ]
        if user_mapping_data:
            df = pl.DataFrame(user_mapping_data)
            self.conn.register('temp_user_mapping', df)
            self.conn.execute("INSERT INTO user_index_mapping SELECT * FROM temp_user_mapping")
            self.conn.unregister('temp_user_mapping')
        
        # Save product mappings
        self.conn.execute("DELETE FROM product_index_mapping")
        product_mapping_data = [
            {'product_id': pid, 'product_index': idx}
            for pid, idx in product_to_index.items()
        ]
        if product_mapping_data:
            df = pl.DataFrame(product_mapping_data)
            self.conn.register('temp_product_mapping', df)
            self.conn.execute("INSERT INTO product_index_mapping SELECT * FROM temp_product_mapping")
            self.conn.unregister('temp_product_mapping')
        
        self.conn.commit()
        logger.info("Index mappings saved to DuckDB")
    
    def load_user_item_matrix_chunk(self, user_ids: List[int]) -> np.ndarray:
        """Load a chunk of user-item matrix for specific users"""
        if not user_ids:
            return np.array([])
        
        placeholders = ','.join(['?'] * len(user_ids))
        result = self.conn.execute(f"""
            SELECT user_id, product_id, score
            FROM user_item_matrix
            WHERE user_id IN ({placeholders})
        """, user_ids).fetchall()
        
        # Get product indices
        product_ids = list(set([row[1] for row in result]))
        product_to_idx = {pid: i for i, pid in enumerate(product_ids)}
        
        # Build matrix chunk
        n_users = len(user_ids)
        n_products = len(product_ids)
        matrix_chunk = np.zeros((n_users, n_products))
        
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        for user_id, product_id, score in result:
            if user_id in user_to_idx and product_id in product_to_idx:
                matrix_chunk[user_to_idx[user_id], product_to_idx[product_id]] = score
        
        return matrix_chunk
    
    def clear_all(self) -> None:
        """Clear all model data from database"""
        logger.warning("Clearing all model data from database...")
        self.conn.execute("DELETE FROM user_item_matrix")
        self.conn.execute("DELETE FROM user_similarities")
        self.conn.execute("DELETE FROM user_index_mapping")
        self.conn.execute("DELETE FROM product_index_mapping")
        self.conn.execute("DELETE FROM user_profiles")
        self.conn.execute("DELETE FROM product_features")
        self.conn.execute("DELETE FROM model_metadata")
        self.conn.execute("DELETE FROM training_interactions")
        self.conn.commit()
        logger.info("All model data cleared")
    
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_model_storage(db_path: Optional[str] = None) -> ModelStorage:
    """Get or create a ModelStorage instance"""
    return ModelStorage(db_path)


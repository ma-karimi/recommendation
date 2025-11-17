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
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import duckdb
import polars as pl

from settings import load_config

logger = logging.getLogger(__name__)


class ModelStorage:
    """Manages storage and retrieval of model data using DuckDB"""
    
    def __init__(self, db_path: Optional[str] = None, reuse_connection: Optional['duckdb.DuckDBPyConnection'] = None):
        """
        Args:
            db_path: Path to DuckDB database file. If None, uses default location.
            reuse_connection: Existing DuckDB connection to reuse (optional).
        """
        cfg = load_config()
        if db_path is None:
            db_path = os.path.join(cfg.output_dir, "model_data.duckdb")
        
        self.db_path = db_path
        
        # Reuse existing connection if provided
        if reuse_connection is not None:
            self.conn = reuse_connection
            logger.info(f"Reusing existing DuckDB connection for {self.db_path}")
        else:
            self.conn = None
            self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Ensure database file and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to DuckDB
        # Note: DuckDB doesn't support concurrent write access by default
        # Multiple processes cannot write to the same database file simultaneously
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            # If connection fails due to lock, provide helpful error message
            if 'lock' in str(e).lower() or 'conflicting' in str(e).lower():
                logger.error(
                    f"DuckDB lock conflict on {self.db_path}. "
                    f"Another process (PID mentioned in error) may be using the database. "
                    f"Please close other processes or wait for them to finish. "
                    f"To check: ps -p <PID> or kill <PID> if safe to do so."
                )
            raise
        
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
        
        # Create table for TF-IDF vectors (used by ANN-based content filtering)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_tfidf_vectors (
                product_id INTEGER PRIMARY KEY,
                vector_index INTEGER,
                vector_data BLOB  -- Pickled numpy array
            )
        """)
        
        # Store sparse matrix metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
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


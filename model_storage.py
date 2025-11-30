"""
Model Storage Module - DuckDB-based persistence for recommendation models

This module handles saving and loading large model data (user factors, item factors,
similarities, embeddings) to/from DuckDB to minimize memory usage.

Key features:
- Save trained matrices to DuckDB after training
- Batch loading of only needed data during inference
- Memory-efficient storage using Parquet format
- Automatic cleanup and garbage collection
- Connection pooling and read-only access to prevent lock conflicts
"""
from __future__ import annotations
import os
import gc
import logging
import pickle
import time
import signal
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import duckdb
import polars as pl
import threading

from settings import load_config

logger = logging.getLogger(__name__)

# Global connection pool to prevent multiple write connections
_connection_lock = threading.Lock()
_connection_pool: Dict[str, duckdb.DuckDBPyConnection] = {}


def _kill_process_safely(pid: int) -> bool:
    """
    Kill a process by PID safely
    
    Args:
        pid: Process ID to kill
        
    Returns:
        True if process was killed successfully, False otherwise
    """
    try:
        # Check if process exists
        os.kill(pid, 0)  # Signal 0 just checks if process exists
        
        # Try graceful termination first
        logger.info(f"Attempting to terminate process {pid} gracefully...")
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        
        # Check if still running
        try:
            os.kill(pid, 0)
            # Still running, force kill
            logger.warning(f"Process {pid} still running, forcing termination...")
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
        except ProcessLookupError:
            # Process terminated successfully
            logger.info(f"Process {pid} terminated successfully")
        
        return True
    except ProcessLookupError:
        logger.info(f"Process {pid} does not exist (already terminated)")
        return True
    except PermissionError:
        logger.error(f"Permission denied: Cannot kill process {pid} (may need root/sudo)")
        return False
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        return False


def _ask_user_to_kill_process(pid: int, db_path: str) -> bool:
    """
    Ask user if they want to kill the conflicting process
    
    Args:
        pid: Process ID of conflicting process
        db_path: Path to database file
        
    Returns:
        True if user confirmed, False otherwise
    """
    # Check if stdin is interactive (not a pipe or file)
    if not sys.stdin.isatty():
        # Non-interactive mode, don't ask
        return False
    
    print(f"\n{'='*80}")
    print(f"⚠️  DuckDB Database Lock Detected")
    print(f"{'='*80}")
    print(f"Another process (PID {pid}) is holding a lock on the database:")
    print(f"  {db_path}")
    print(f"\nOptions:")
    print(f"  1. Kill the conflicting process (PID {pid})")
    print(f"  2. Wait for it to finish")
    print(f"  3. Cancel and exit")
    print(f"\nWhat would you like to do?")
    
    while True:
        try:
            response = input("Enter choice (1/2/3) or 'y' to kill: ").strip().lower()
            
            if response in ['1', 'y', 'yes']:
                return True
            elif response in ['2', 'n', 'no', 'wait']:
                print("Waiting for the process to finish...")
                return False
            elif response in ['3', 'c', 'cancel', 'exit', 'q', 'quit']:
                print("Cancelled by user.")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1, 2, or 3 (or y/n).")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled by user.")
            sys.exit(0)


class ModelStorage:
    """Manages storage and retrieval of model data using DuckDB"""
    
    def __init__(self, db_path: Optional[str] = None, read_only: bool = False, max_retries: int = 3):
        """
        Args:
            db_path: Path to DuckDB database file. If None, uses default location.
            read_only: If True, opens connection in read-only mode (allows multiple readers)
            max_retries: Maximum number of retries for acquiring lock
        """
        cfg = load_config()
        if db_path is None:
            db_path = os.path.join(cfg.output_dir, "model_data.duckdb")
        
        self.db_path = os.path.abspath(db_path)  # Use absolute path for consistency
        self.read_only = read_only
        self.max_retries = max_retries
        self.conn = None
        self._ensure_db()
    
    def _get_connection(self, read_only: Optional[bool] = None) -> duckdb.DuckDBPyConnection:
        """
        Get a database connection with retry logic and connection pooling
        
        Args:
            read_only: Override instance read_only setting for this connection
        """
        if read_only is None:
            read_only = self.read_only
        
        # For read-only connections, always create a new connection (DuckDB allows multiple readers)
        if read_only:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return duckdb.connect(self.db_path, read_only=True)
                except Exception as e:
                    if "lock" in str(e).lower() and attempt < max_retries - 1:
                        wait_time = 0.5 * (attempt + 1)
                        logger.warning(f"Failed to open read-only connection (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # If read-only also fails, try to extract PID and ask user
                        error_str = str(e)
                        pid = None
                        if "PID" in error_str:
                            pid_match = re.search(r'PID\s+(\d+)', error_str)
                            if pid_match:
                                pid = int(pid_match.group(1))
                        
                        if pid:
                            # Ask user if they want to kill the conflicting process
                            if _ask_user_to_kill_process(pid, self.db_path):
                                # User confirmed, kill the process
                                if _kill_process_safely(pid):
                                    logger.info(f"Process {pid} killed. Retrying connection...")
                                    # Wait a moment for lock to be released
                                    time.sleep(2)
                                    # Retry connection one more time
                                    try:
                                        return duckdb.connect(self.db_path, read_only=True)
                                    except Exception as retry_error:
                                        logger.error(f"Still cannot connect after killing process: {retry_error}")
                                        raise RuntimeError(
                                            f"Could not connect to DuckDB even after killing process {pid}.\n"
                                            f"Please check if the database file is accessible: {self.db_path}\n"
                                            f"Original error: {e}"
                                        ) from retry_error
                                else:
                                    raise RuntimeError(
                                        f"Failed to kill process {pid}.\n"
                                        f"Please kill it manually: kill {pid}\n"
                                        f"Original error: {e}"
                                    ) from e
                            else:
                                raise RuntimeError(
                                    f"Could not open read-only connection to DuckDB.\n"
                                    f"Another process (PID {pid}) is holding a write lock.\n"
                                    f"To fix: kill {pid} or wait for it to finish.\n"
                                    f"Original error: {e}"
                                ) from e
                        raise
        
        # For write connections, use connection pooling to prevent multiple write connections
        with _connection_lock:
            if self.db_path in _connection_pool:
                conn = _connection_pool[self.db_path]
                # Check if connection is still valid
                try:
                    conn.execute("SELECT 1")
                    return conn
                except:
                    # Connection is dead, remove from pool
                    try:
                        conn.close()
                    except:
                        pass
                    del _connection_pool[self.db_path]
            
            # Try to create a new write connection with retry logic
            for attempt in range(self.max_retries):
                try:
                    conn = duckdb.connect(self.db_path, read_only=False)
                    _connection_pool[self.db_path] = conn
                    return conn
                except Exception as e:
                    if "lock" in str(e).lower() or "conflicting" in str(e).lower():
                        if attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                            logger.warning(
                                f"Could not acquire lock (attempt {attempt + 1}/{self.max_retries}), "
                                f"retrying in {wait_time:.1f}s..."
                            )
                            time.sleep(wait_time)
                            continue
                        else:
                            # Extract PID from error message if available
                            error_str = str(e)
                            pid = None
                            if "PID" in error_str:
                                pid_match = re.search(r'PID\s+(\d+)', error_str)
                                if pid_match:
                                    pid = int(pid_match.group(1))
                            
                            # Ask user if they want to kill the conflicting process
                            if pid:
                                if _ask_user_to_kill_process(pid, self.db_path):
                                    # User confirmed, kill the process
                                    if _kill_process_safely(pid):
                                        logger.info(f"Process {pid} killed. Retrying connection...")
                                        # Wait a moment for lock to be released
                                        time.sleep(2)
                                        # Retry connection one more time
                                        try:
                                            conn = duckdb.connect(self.db_path, read_only=False)
                                            _connection_pool[self.db_path] = conn
                                            return conn
                                        except Exception as retry_error:
                                            logger.error(f"Still cannot connect after killing process: {retry_error}")
                                            raise RuntimeError(
                                                f"Could not connect to DuckDB even after killing process {pid}.\n"
                                                f"Please check if the database file is accessible: {self.db_path}\n"
                                                f"Original error: {e}"
                                            ) from retry_error
                                    else:
                                        raise RuntimeError(
                                            f"Failed to kill process {pid}.\n"
                                            f"Please kill it manually: kill {pid}\n"
                                            f"Original error: {e}"
                                        ) from e
                            
                            error_msg = (
                                f"Could not acquire DuckDB lock after {self.max_retries} attempts.\n"
                                f"Another process is using the database file: {self.db_path}\n"
                            )
                            
                            if pid:
                                error_msg += (
                                    f"\nConflicting process: PID {pid}\n"
                                    f"To fix this, you can:\n"
                                    f"  1. Kill the process: kill {pid}\n"
                                    f"  2. Or wait for it to finish\n"
                                    f"  3. Or use read-only mode: ModelStorage(read_only=True)\n"
                                )
                            
                            error_msg += f"\nOriginal error: {e}"
                            
                            raise RuntimeError(error_msg) from e
                    else:
                        raise
    
    def _ensure_db(self) -> None:
        """Ensure database file and tables exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Try to get a write connection for table creation
        # If read_only, we'll try to get a temporary write connection just for schema creation
        try:
            if self.read_only:
                # For read-only mode, try to open write connection just for schema check
                # If it fails, assume tables already exist
                try:
                    temp_conn = duckdb.connect(self.db_path, read_only=False)
                    self._create_tables(temp_conn)
                    temp_conn.close()
                except Exception as e:
                    if "lock" in str(e).lower():
                        logger.info("Database is locked, assuming tables exist (read-only mode)")
                        # Try to verify tables exist with read-only connection
                        try:
                            test_conn = self._get_connection(read_only=True)
                            test_conn.execute("SELECT 1 FROM user_item_matrix LIMIT 1")
                            test_conn.close()
                        except:
                            logger.warning("Could not verify tables exist. They may need to be created.")
                    else:
                        raise
            else:
                self.conn = self._get_connection(read_only=False)
                self._create_tables(self.conn)
                # Update self.conn to use the pooled connection
                if not self.read_only:
                    self.conn = self._get_connection(read_only=False)
        except Exception as e:
            if "lock" in str(e).lower() or "conflicting" in str(e).lower():
                # If we can't get write lock, try read-only
                logger.warning(f"Could not get write lock, falling back to read-only mode: {e}")
                logger.warning("Some operations may fail. Consider killing the conflicting process.")
                self.read_only = True
                try:
                    self.conn = self._get_connection(read_only=True)
                except Exception as read_error:
                    # Even read-only failed, provide helpful error
                    error_str = str(read_error)
                    pid = None
                    if "PID" in error_str:
                        import re
                        pid_match = re.search(r'PID\s+(\d+)', error_str)
                        if pid_match:
                            pid = pid_match.group(1)
                    
                    if pid:
                        raise RuntimeError(
                            f"Cannot access DuckDB database. Process {pid} is holding a lock.\n"
                            f"Run: kill {pid}\n"
                            f"Or use: python check_db_lock.py --kill-pid {pid}\n"
                            f"Original error: {read_error}"
                        ) from read_error
                    raise
            else:
                raise
    
    def _create_tables(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create database tables if they don't exist"""
        # Create tables for collaborative filtering data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_item_matrix (
                user_id INTEGER,
                product_id INTEGER,
                score REAL,
                PRIMARY KEY (user_id, product_id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_similarities (
                user_id_1 INTEGER,
                user_id_2 INTEGER,
                similarity REAL,
                PRIMARY KEY (user_id_1, user_id_2)
            )
        """)
        
        # Create tables for mappings
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_index_mapping (
                user_id INTEGER PRIMARY KEY,
                user_index INTEGER
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_index_mapping (
                product_id INTEGER PRIMARY KEY,
                product_index INTEGER
            )
        """)
        
        # Create tables for content-based filtering
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_data BLOB  -- Pickled dict
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_features (
                product_id INTEGER PRIMARY KEY,
                features_data BLOB  -- Pickled dict
            )
        """)
        
        # Create table for product TF-IDF vectors (stored as arrays)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_vectors (
                product_id INTEGER PRIMARY KEY,
                vector_data BLOB,  -- Pickled numpy array
                vector_dim INTEGER  -- Dimension of the vector
            )
        """)
        
        # Store sparse matrix metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        logger.info(f"Model storage initialized at {self.db_path} (read_only={self.read_only})")
    
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
        if self.read_only:
            raise RuntimeError("Cannot save model: storage is in read-only mode")
        
        logger.info("Saving collaborative filtering model to DuckDB...")
        
        # Ensure we have a write connection
        conn = self._get_connection(read_only=False)
        
        # Clear existing data
        conn.execute("DELETE FROM user_item_matrix")
        conn.execute("DELETE FROM user_similarities")
        conn.execute("DELETE FROM user_index_mapping")
        conn.execute("DELETE FROM product_index_mapping")
        
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
            conn.execute("INSERT INTO user_item_matrix SELECT * FROM df")
        
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
                conn.execute("INSERT INTO user_similarities SELECT * FROM df")
                similarity_data = []
                gc.collect()
        
        if similarity_data:
            df = pl.DataFrame(similarity_data)
            conn.execute("INSERT INTO user_similarities SELECT * FROM df")
        
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
            conn.execute("INSERT INTO user_index_mapping SELECT * FROM df")
        
        if product_mapping_data:
            df = pl.DataFrame(product_mapping_data)
            conn.execute("INSERT INTO product_index_mapping SELECT * FROM df")
        
        # Save metadata
        conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('n_users', ?), ('n_products', ?)
        """, [str(len(user_to_index)), str(len(product_to_index))])
        
        conn.commit()
        logger.info("Collaborative model saved successfully")
    
    def load_user_item_row(self, user_id: int) -> Dict[int, float]:
        """Load a single user's item ratings from database"""
        # استفاده از connection موجود اگر وجود دارد
        if self.conn:
            conn = self.conn
        else:
            try:
                conn = self._get_connection(read_only=True)
            except Exception:
                conn = self._get_connection(read_only=False)
        
        result = conn.execute("""
            SELECT product_id, score
            FROM user_item_matrix
            WHERE user_id = ?
        """, [user_id]).fetchall()
        
        return {row[0]: row[1] for row in result}
    
    def load_user_similarities(self, user_id: int, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """Load similar users for a given user"""
        # استفاده از connection موجود اگر وجود دارد
        if self.conn:
            conn = self.conn
        else:
            try:
                conn = self._get_connection(read_only=True)
            except Exception:
                conn = self._get_connection(read_only=False)
        
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
        
        result = conn.execute(query, [user_id, user_id, user_id, user_id]).fetchall()
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
    
    def save_product_vectors_batch(self, product_vectors: Dict[int, np.ndarray]) -> None:
        """Save product TF-IDF vectors to DuckDB in batches"""
        if not product_vectors:
            return
        
        logger.info(f"Saving {len(product_vectors)} product vectors to DuckDB...")
        vector_data = []
        
        for product_id, vector in product_vectors.items():
            # Convert sparse to dense if needed
            if hasattr(vector, 'toarray'):
                vector = vector.toarray().flatten()
            elif hasattr(vector, 'todense'):
                vector = vector.todense().flatten()
            
            vector_data.append({
                'product_id': product_id,
                'vector_data': pickle.dumps(vector.astype(np.float32)),
                'vector_dim': len(vector)
            })
        
        # Save in batches
        batch_size = 1000
        for i in range(0, len(vector_data), batch_size):
            batch = vector_data[i:i + batch_size]
            df = pl.DataFrame(batch)
            self.conn.execute("""
                INSERT OR REPLACE INTO product_vectors 
                SELECT product_id, vector_data, vector_dim FROM df
            """)
            gc.collect()
        
        self.conn.commit()
        logger.info("Product vectors saved successfully")
    
    def load_product_vector(self, product_id: int) -> Optional[np.ndarray]:
        """Load a single product vector from DuckDB"""
        # استفاده از connection موجود اگر وجود دارد
        if self.conn:
            conn = self.conn
        else:
            try:
                conn = self._get_connection(read_only=True)
            except Exception:
                conn = self._get_connection(read_only=False)
        
        result = conn.execute("""
            SELECT vector_data FROM product_vectors WHERE product_id = ?
        """, [product_id]).fetchone()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def load_product_vectors_batch(self, product_ids: List[int], batch_size: int = 1000) -> Dict[int, np.ndarray]:
        """Load product vectors in batches from DuckDB"""
        if not product_ids:
            return {}
        
        # استفاده از connection موجود اگر وجود دارد (برای جلوگیری از conflict)
        # یا استفاده از write connection که می‌تواند read هم بکند
        if self.conn:
            conn = self.conn
        else:
            # اگر connection نداریم، سعی می‌کنیم read-only بگیریم
            # اما اگر write connection وجود دارد، از آن استفاده می‌کنیم
            try:
                conn = self._get_connection(read_only=True)
            except Exception:
                # اگر read-only fail کرد، از write connection استفاده می‌کنیم
                conn = self._get_connection(read_only=False)
        
        vectors = {}
        for i in range(0, len(product_ids), batch_size):
            batch_ids = product_ids[i:i + batch_size]
            placeholders = ','.join(['?'] * len(batch_ids))
            result = conn.execute(f"""
                SELECT product_id, vector_data
                FROM product_vectors
                WHERE product_id IN ({placeholders})
            """, batch_ids).fetchall()
            
            for row in result:
                vectors[row[0]] = pickle.loads(row[1])
        
        return vectors
    
    def get_all_product_ids_with_vectors(self) -> List[int]:
        """Get all product IDs that have vectors stored"""
        result = self.conn.execute("SELECT product_id FROM product_vectors ORDER BY product_id").fetchall()
        return [row[0] for row in result]
    
    def get_vector_dimension(self) -> Optional[int]:
        """Get the dimension of stored vectors (assumes all vectors have same dimension)"""
        result = self.conn.execute("SELECT vector_dim FROM product_vectors LIMIT 1").fetchone()
        return result[0] if result else None
    
    def save_ann_index_path(self, index_path: str) -> None:
        """Save path to ANN index file"""
        self.conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('ann_index_path', ?)
        """, [index_path])
        self.conn.commit()
    
    def get_ann_index_path(self) -> Optional[str]:
        """Get path to ANN index file"""
        result = self.conn.execute("""
            SELECT value FROM model_metadata WHERE key = 'ann_index_path'
        """).fetchone()
        return result[0] if result else None
    
    def save_tfidf_vectorizer(self, vectorizer: Any) -> None:
        """Save TF-IDF vectorizer to metadata"""
        vectorizer_data = pickle.dumps(vectorizer)
        self.conn.execute("""
            INSERT OR REPLACE INTO model_metadata (key, value)
            VALUES ('tfidf_vectorizer', ?)
        """, [vectorizer_data])
        self.conn.commit()
    
    def load_tfidf_vectorizer(self) -> Optional[Any]:
        """Load TF-IDF vectorizer from metadata"""
        result = self.conn.execute("""
            SELECT value FROM model_metadata WHERE key = 'tfidf_vectorizer'
        """).fetchone()
        if result:
            return pickle.loads(result[0])
        return None
    
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
        self.conn.execute("DELETE FROM product_vectors")
        self.conn.execute("DELETE FROM model_metadata")
        self.conn.commit()
        logger.info("All model data cleared")
    
    def close(self) -> None:
        """Close database connection"""
        # Only close if it's not in the connection pool (i.e., it's a read-only connection)
        if self.conn:
            with _connection_lock:
                # Check if this connection is in the pool
                if self.db_path in _connection_pool and _connection_pool[self.db_path] is self.conn:
                    # Don't close pooled connections, they're shared
                    pass
                else:
                    # Close read-only or non-pooled connections
                    try:
                        self.conn.close()
                    except:
                        pass
            self.conn = None
    
    def _execute(self, query: str, params: Optional[List] = None, read_only: Optional[bool] = None) -> Any:
        """
        Execute a query with proper connection management
        
        Args:
            query: SQL query
            params: Query parameters
            read_only: Whether this is a read-only operation
        """
        if read_only is None:
            read_only = self.read_only or query.strip().upper().startswith(('SELECT', 'WITH'))
        
        conn = self._get_connection(read_only=read_only)
        
        if params:
            return conn.execute(query, params)
        else:
            return conn.execute(query)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_model_storage(db_path: Optional[str] = None) -> ModelStorage:
    """Get or create a ModelStorage instance"""
    return ModelStorage(db_path)


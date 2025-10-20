import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class MatomoConfig:
    base_url: str
    site_id: str
    token_auth: str
    verify_ssl: bool = True

@dataclass
class DBConfig:
    url: str | None = None

@dataclass
class Config:
    matomo: MatomoConfig
    db: DBConfig
    output_dir: str = os.path.join("storage", "app", "recommendation")


def _env(key: str, default: str = "") -> str:
    """Read env var and strip surrounding quotes/whitespace."""
    val = os.getenv(key, default)
    if val is None:
        return default
    return val.strip().strip('"').strip("'")


def _env_bool(key: str, default: bool = True) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    val = val.strip().lower()
    return val in ("1", "true", "yes", "y")


def _load_env_files() -> None:
    """Load .env or env.production if present."""
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    prod_env_path = os.path.join(_PROJECT_ROOT, "env.production")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
    elif os.path.exists(prod_env_path):
        load_dotenv(prod_env_path, override=False)
    else:
        # Fallback: try default search
        load_dotenv(override=False)


def load_config() -> Config:
    """Load configuration from environment variables (.env assumed to be loaded by Laravel)"""
    _load_env_files()
    base_url = _env("MATOMO_BASE_URL")
    site_id = _env("MATOMO_SITE_ID")
    token_auth = _env("MATOMO_TOKEN_AUTH")
    verify_ssl = _env_bool("MATOMO_VERIFY_SSL", default=True)

    db_url = os.getenv("RECO_DB_URL")

    # resolve output directory from env or fallback to default
    storage_env = _env("STORAGE_PATH")
    output_dir = storage_env or os.path.join("storage", "app", "recommendation")
    os.makedirs(output_dir, exist_ok=True)

    return Config(
        matomo=MatomoConfig(base_url=base_url, site_id=site_id, token_auth=token_auth, verify_ssl=verify_ssl),
        db=DBConfig(url=db_url),
        output_dir=output_dir,
    )

# Determine project root to load env files
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_ROOT_DIR)
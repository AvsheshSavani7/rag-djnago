from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_DIR / "data"
FEEDS_REGISTRY_PATH = DATA_DIR / "feeds_registry.json"
ARTICLE_LINKS_PATH = DATA_DIR / "article_links.json"
SCAN_LOG_PATH = DATA_DIR / "scan_log.json"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR

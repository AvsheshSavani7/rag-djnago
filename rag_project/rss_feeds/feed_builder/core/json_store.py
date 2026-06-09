import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from .paths import ensure_data_dir

_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    ensure_data_dir()
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    ensure_data_dir()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")


def load_list(path: Path) -> List[dict]:
    with _lock:
        data = _read_json(path, [])
    return data if isinstance(data, list) else []


def save_list(path: Path, rows: List[dict]) -> None:
    with _lock:
        _write_json(path, rows)


def update_list(path: Path, mutator: Callable[[List[dict]], List[dict]]) -> List[dict]:
    with _lock:
        rows = _read_json(path, [])
        if not isinstance(rows, list):
            rows = []
        updated = mutator(rows)
        _write_json(path, updated)
        return updated

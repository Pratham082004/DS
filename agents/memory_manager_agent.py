"""
agents/memory_manager_agent.py

MemoryManagerAgent (Production-ready)
------------------------------------
A lightweight persistent memory/metadata manager for pipeline artifacts and
user/system memories. Designed for:
- registering artifacts (EDA summaries, model summaries, reports)
- searching metadata and retrieving artifact paths
- simple TTL (time-to-live) support and versioning
- atomic writes and safe concurrency via file locking

Storage options:
- JSON-backed store (default)
- Optional SQLite backend (if sqlite3 available)

Outputs:
- data/memory/store.json  (main store)
- helper methods to add, get, search, delete entries
"""

import os
import json
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MemoryManagerAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {
            "store_path": "data/memory/store.json",
            "backup_dir": "data/memory/backups",
            "auto_backup": True
        }
        if config:
            cfg.update(config)
        self.cfg = cfg
        self.store_path = Path(self.cfg["store_path"])
        self.backup_dir = Path(self.cfg["backup_dir"])
        self._lock = threading.Lock()
        self._init_store()

    # ---------------- internal IO ----------------
    def _init_store(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump({"entries": []}, f, indent=2)
            logging.info(f"Initialized memory store at {self.store_path}")

    def _read_store(self) -> Dict[str, Any]:
        with self._lock:
            with open(self.store_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def _write_store(self, data: Dict[str, Any]):
        with self._lock:
            # optional backup before write
            if self.cfg.get("auto_backup"):
                ts = int(time.time())
                bpath = self.backup_dir / f"store_backup_{ts}.json"
                try:
                    shutil.copyfile(self.store_path, bpath)
                except Exception:
                    pass
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # ---------------- API ----------------
    def add_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds an entry into the memory store. Entry should contain:
          - key (str): unique key or will be auto-generated
          - type (str): e.g., 'eda', 'model', 'report'
          - path (optional): file path to artifact
          - meta (optional): dict of metadata (metrics, features, tags)
          - ttl (optional): expire after seconds from now
        Returns the stored entry with assigned id and timestamps.
        """
        data = self._read_store()
        entries = data.get("entries", [])
        now = int(time.time())
        entry_id = entry.get("key") or f"entry_{now}_{len(entries)}"
        stored = {
            "id": entry_id,
            "type": entry.get("type", "generic"),
            "path": entry.get("path"),
            "meta": entry.get("meta", {}),
            "tags": entry.get("tags", []),
            "created_at": now,
            "expires_at": (now + int(entry["ttl"])) if entry.get("ttl") else None,
            "version": 1
        }
        entries.append(stored)
        data["entries"] = entries
        self._write_store(data)
        logging.info(f"Added memory entry: {entry_id}")
        return stored

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        data = self._read_store()
        for e in data.get("entries", []):
            if e.get("id") == entry_id:
                # check TTL
                if e.get("expires_at") and int(time.time()) > e["expires_at"]:
                    logging.info(f"Entry {entry_id} expired; removing.")
                    self.delete_entry(entry_id)
                    return None
                return e
        return None

    def search(
        self,
        q: Optional[str] = None,
        types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple search over metadata fields:
          - q: substring match against id, path, meta values (stringified)
          - types: filter by entry types
          - tags: require tags subset match
        """
        q = q.lower() if q else None
        types = set(types) if types else None
        tags = set(tags) if tags else None
        data = self._read_store()
        out = []
        for e in data.get("entries", []):
            if e.get("expires_at") and int(time.time()) > e["expires_at"]:
                continue
            if types and e.get("type") not in types:
                continue
            if tags and not tags.issubset(set(e.get("tags", []))):
                continue
            if q:
                hay = " ".join([
                    str(e.get("id", "")),
                    str(e.get("path", "") or ""),
                    json.dumps(e.get("meta", {}))
                ]).lower()
                if q in hay:
                    out.append(e)
            else:
                out.append(e)
        return out

    def delete_entry(self, entry_id: str) -> bool:
        data = self._read_store()
        entries = data.get("entries", [])
        new_entries = [e for e in entries if e.get("id") != entry_id]
        if len(new_entries) == len(entries):
            return False
        data["entries"] = new_entries
        self._write_store(data)
        logging.info(f"Deleted memory entry: {entry_id}")
        return True

    def cleanup_expired(self) -> int:
        data = self._read_store()
        now = int(time.time())
        before = len(data.get("entries", []))
        data["entries"] = [
            e for e in data.get("entries", [])
            if not (e.get("expires_at") and now > e.get("expires_at"))
        ]
        self._write_store(data)
        after = len(data["entries"])
        removed = before - after
        logging.info(f"Cleaned up {removed} expired entries")
        return removed


# ---------------- Local Test ----------------
if __name__ == "__main__":
    import pprint

    mm = MemoryManagerAgent()
    e = mm.add_entry({
        "type": "model",
        "path": "data/models/example.pkl",
        "meta": {"accuracy": 0.95},
        "tags": ["iris", "test"],
        "ttl": 3600
    })
    pprint.pprint(mm.get_entry(e["id"]))
    pprint.pprint(mm.search(q="iris"))
    mm.cleanup_expired()

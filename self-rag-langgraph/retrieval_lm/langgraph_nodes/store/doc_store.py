"""Document / passage store with JSON persistence."""
import json
import os
from typing import Any, Dict, List, Optional


class DocStore:
    """Simple key-value store for passages: {chunk_id: {title, text, doc_id}}."""

    def __init__(self, store_dir: str = "./out"):
        os.makedirs(store_dir, exist_ok=True)
        self.path = os.path.join(store_dir, "doc_store.json")
        self.data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)

    def upsert(self, chunk_id: str, title: str, text: str, doc_id: str):
        self.data[chunk_id] = {"title": title, "text": text, "doc_id": doc_id}

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(chunk_id)

    def get_batch(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.data[cid] for cid in chunk_ids if cid in self.data]

    def __len__(self):
        return len(self.data)

    def __contains__(self, chunk_id: str):
        return chunk_id in self.data

"""Vector store with cosine-similarity top-k retrieval."""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class VectorStore:
    """Maps chunk_id -> embedding vector. Supports cosine top-k search."""

    def __init__(self, name: str = "passages", store_dir: str = "./out"):
        os.makedirs(store_dir, exist_ok=True)
        self.path = os.path.join(store_dir, f"vdb_{name}.json")
        self._ids: List[str] = []
        self._vecs: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._ids = list(raw.keys())
            if self._ids:
                self._vecs = np.array([raw[k] for k in self._ids], dtype=np.float32)
            else:
                self._vecs = None
        else:
            self._ids = []
            self._vecs = None

    def save(self):
        raw = {}
        if self._vecs is not None:
            for cid, vec in zip(self._ids, self._vecs.tolist()):
                raw[cid] = vec
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(raw, f)

    def upsert(self, ids: List[str], vecs: np.ndarray):
        """Add or update vectors. ``vecs`` shape: (len(ids), dim)."""
        vecs = np.asarray(vecs, dtype=np.float32)
        existing = set(self._ids)
        new_ids = []
        new_vecs = []
        for cid, vec in zip(ids, vecs):
            if cid in existing:
                idx = self._ids.index(cid)
                self._vecs[idx] = vec
            else:
                new_ids.append(cid)
                new_vecs.append(vec)
        if new_ids:
            new_arr = np.array(new_vecs, dtype=np.float32)
            if self._vecs is not None:
                self._vecs = np.concatenate([self._vecs, new_arr], axis=0)
            else:
                self._vecs = new_arr
            self._ids.extend(new_ids)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (chunk_id, score) by cosine similarity."""
        if self._vecs is None or len(self._ids) == 0:
            return []
        query_vec = np.asarray(query_vec, dtype=np.float32).flatten()
        norms = np.linalg.norm(self._vecs, axis=1)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        sims = (self._vecs @ query_vec) / (norms * q_norm + 1e-10)
        k = min(top_k, len(sims))
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._ids[i], float(sims[i])) for i in top_idx]

    def __len__(self):
        return len(self._ids)
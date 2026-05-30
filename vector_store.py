"""
FAISS vector store using OpenAI text-embedding-3-small.
"""

import numpy as np
import faiss
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


class VectorStore:
    def __init__(self, client: OpenAI):
        self.client = client
        self.chunks: list[dict] = []
        self.index: faiss.IndexFlatIP | None = None

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([item.embedding for item in resp.data], dtype=np.float32)

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vecs)
        return vecs

    def build(self, chunks: list[dict], progress_cb=None) -> None:
        self.chunks = chunks
        texts = [c["content"][:2000] for c in chunks]  # truncate to keep tokens low
        all_vecs = []
        total = len(texts)
        for i in range(0, total, BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            all_vecs.append(self._embed_batch(batch))
            if progress_cb:
                progress_cb(min(i + BATCH_SIZE, total), total)
        embeddings = np.vstack(all_vecs)
        self._normalize(embeddings)

        # Build FAISS index (IndexFlatIP = inner product on normalized vectors = cosine similarity)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        if self.index is None:
            return []
        q_vec = self._embed_batch([query])
        self._normalize(q_vec)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            c = dict(self.chunks[idx])
            c["score"] = float(score)
            results.append(c)
        return results
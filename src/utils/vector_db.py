import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from change_detection.detector import Change

class VectorDB:
    """
    FAISS store for chunk‑wise Change objects.
    Each metadata dict now contains 'chunk_text', so we never need a separate self.documents list.
    """

    def __init__(
        self,
        persist_directory: str,
        model_name: str,
        index_filename: str = "faiss.index",
        meta_filename: str = "faiss_meta.pkl"
    ):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.index_path = os.path.join(persist_directory, index_filename)
        self.meta_path  = os.path.join(persist_directory, meta_filename)

        self.encoder = SentenceTransformer(model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()

        # If an index already exists, load it and its metadata.
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.metadatas: List[Dict] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            self.metadatas = []

    def store_changes(self, changes: List[Change]):
        """
        Rebuilds the FAISS index from scratch.
        Stores chunk_text inside self.metadatas entries.
        """
        embeddings = []
        meta = []

        for c in changes:
            # pick the content that exists
            text = c.new_content if c.new_content else c.old_content
            # store embedding
            emb = self.encoder.encode(text, convert_to_numpy=True)
            embeddings.append(emb)
            # store metadata, including the text
            meta.append({
                "section_id":  c.section_id,
                "chunk_id":    c.chunk_id,
                "change_type": c.change_type.value,
                "similarity":  c.similarity_score,
                "chunk_text":  text
            })

        # rebuild the index
        self.index = faiss.IndexFlatL2(self.dim)
        all_embs = np.vstack(embeddings).astype(np.float32)
        self.index.add(all_embs)

        # persist everything
        with open(self.meta_path, "wb") as f:
            pickle.dump(meta, f)
        faiss.write_index(self.index, self.index_path)

        # keep in‑memory
        self.metadatas = meta

    def query_changes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return top_k hits for the query, each as:
          { "text": str, "score": float, "metadata": { ... } }
        """
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            entry_meta = self.metadatas[idx].copy()
            text = entry_meta.pop("chunk_text")  # pull text out
            results.append({
                "text":      text,
                "score":     float(dist),
                "metadata":  entry_meta
            })
        return results

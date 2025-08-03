# src/utils/vector_db.py

import os
import pickle
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from change_detection.detector import Change

class VectorDB:
    """
    A dual-index FAISS store:
      1) chunk_index + chunk_meta.pkl  — your raw Change snippets
      2) event_index + event_meta.pkl  — centroids of clustered change "events"
    Now also stores version information.
    """

    def __init__(
        self,
        persist_directory: str,
        model_name: str,
        versions: Optional[Dict] = None,  # Add versions parameter
        chunk_index_filename: str = "faiss.index",
        chunk_meta_filename:  str = "faiss_meta.pkl",
        event_index_filename: str = "events.index",
        event_meta_filename:  str = "events_meta.pkl",
        versions_filename: str = "versions.pkl",  # Add versions file
    ):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # file paths
        self.chunk_index_path = os.path.join(persist_directory, chunk_index_filename)
        self.chunk_meta_path  = os.path.join(persist_directory, chunk_meta_filename)
        self.event_index_path = os.path.join(persist_directory, event_index_filename)
        self.event_meta_path  = os.path.join(persist_directory, event_meta_filename)
        self.versions_path    = os.path.join(persist_directory, versions_filename)

        # shared encoder
        self.encoder = SentenceTransformer(model_name)
        self.dim     = self.encoder.get_sentence_embedding_dimension()

        # Store or load versions
        if versions is not None:
            self.versions = versions
            self._save_versions()
        else:
            self.versions = self._load_versions()

        # load or init chunk-level index
        if os.path.exists(self.chunk_index_path) and os.path.exists(self.chunk_meta_path):
            self.chunk_index = faiss.read_index(self.chunk_index_path)
            with open(self.chunk_meta_path, "rb") as f:
                self.chunk_metadatas: List[Dict] = pickle.load(f)
        else:
            self.chunk_index = faiss.IndexFlatL2(self.dim)
            self.chunk_metadatas = []

        # load or init event-level index
        if os.path.exists(self.event_index_path) and os.path.exists(self.event_meta_path):
            self.event_index = faiss.read_index(self.event_index_path)
            with open(self.event_meta_path, "rb") as f:
                self.event_metadatas: List[Dict] = pickle.load(f)
        else:
            self.event_index = faiss.IndexFlatL2(self.dim)
            self.event_metadatas = []

    def _save_versions(self) -> None:
        """Save version information to disk."""
        with open(self.versions_path, "wb") as f:
            pickle.dump(self.versions, f)

    def _load_versions(self) -> Dict:
        """Load version information from disk."""
        if os.path.exists(self.versions_path):
            with open(self.versions_path, "rb") as f:
                return pickle.load(f)
        return {}

    def get_versions(self) -> Dict:
        """Get the stored version information."""
        return self.versions

    def update_versions(self, versions: Dict) -> None:
        """Update the stored version information."""
        self.versions = versions
        self._save_versions()

    # ─── Chunk-level storage ────────────────────────────────────────────────────────
    def store_changes(self, changes) -> None:
        """
        Build a FAISS index of every Change snippet.
        Accepts either Change objects or dictionaries.
        """
        embeddings = []
        meta       = []

        for c in changes:
            # Handle both Change objects and dictionaries
            if hasattr(c, 'new_content'):  # It's a Change object
                text = c.new_content or c.old_content
                section_id = c.section_id
                chunk_id = c.chunk_id
                change_type = c.change_type.value
                similarity = c.similarity_score
            else:  # It's a dictionary from JSON
                text = c.get('new_content') or c.get('old_content')
                section_id = c.get('section_id')
                chunk_id = c.get('chunk_id')
                change_type = c.get('change_type')
                similarity = c.get('similarity_score')

            emb = self.encoder.encode(text, convert_to_numpy=True)
            embeddings.append(emb)
            meta.append({
                "section_id":  section_id,
                "chunk_id":    chunk_id,
                "change_type": change_type,
                "similarity":  similarity,
                "text":        text
            })

        # rebuild
        self.chunk_index = faiss.IndexFlatL2(self.dim)
        all_embs = np.vstack(embeddings).astype(np.float32)
        self.chunk_index.add(all_embs)

        # persist
        with open(self.chunk_meta_path, "wb") as f:
            pickle.dump(meta, f)
        faiss.write_index(self.chunk_index, self.chunk_index_path)

        # cache
        self.chunk_metadatas = meta

        # rebuild
        self.chunk_index = faiss.IndexFlatL2(self.dim)
        all_embs = np.vstack(embeddings).astype(np.float32)
        self.chunk_index.add(all_embs)

        # persist
        with open(self.chunk_meta_path, "wb") as f:
            pickle.dump(meta, f)
        faiss.write_index(self.chunk_index, self.chunk_index_path)

        # cache
        self.chunk_metadatas = meta

    def query_changes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k matching raw-chunk changes for a free-text query.
        Returns a list of { text, score, metadata }.
        """
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        dists, idxs = self.chunk_index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0 or idx >= len(self.chunk_metadatas):
                continue
            md   = self.chunk_metadatas[idx].copy()
            text = md.pop("text")
            results.append({
                "text":      text,
                "score":     float(dist),
                "metadata":  md
            })
        return results

    # ─── Event-level storage ────────────────────────────────────────────────────────
    def store_events(
        self,
        events_meta_path: str,
        events_members_embeddings: Optional[str] = None
    ) -> None:
        """
        Build a FAISS index of event centroids.
        `events_meta_path` should point to your JSON of:
          [ { event_id, label, members: [chunk_indices] }, ... ]
        Optionally, if you passed `embeddings.npy` for members, you can speed up.
        """
        # load cluster definitions
        with open(events_meta_path, "r", encoding="utf-8") as f:
            events = json.load(f)

        # load raw embeddings if needed
        if events_members_embeddings and os.path.exists(events_members_embeddings):
            all_embs = np.load(events_members_embeddings)
        else:
            # re-embed chunk texts inline
            all_embs = np.vstack([
                self.encoder.encode(d["text"], convert_to_numpy=True)
                for d in self.chunk_metadatas
            ]).astype(np.float32)

        centroids = []
        meta      = []

        for ev in events:
            members = ev["members"]
            if not members:
                continue
            # compute centroid
            emb = all_embs[members].mean(axis=0)
            centroids.append(emb)
            meta.append({
                "event_id": ev["event_id"],
                "label":    ev["label"],
                "members":  members
            })

        # rebuild event index
        self.event_index = faiss.IndexFlatL2(self.dim)
        self.event_index.add(np.vstack(centroids).astype(np.float32))

        # persist
        with open(self.event_meta_path, "wb") as f:
            pickle.dump(meta, f)
        faiss.write_index(self.event_index, self.event_index_path)

        # cache
        self.event_metadatas = meta

    def query_events(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top_k matching “events” (clusters) for a free-text query.
        Returns a list of { label, score, metadata }.
        """
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        dists, idxs = self.event_index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx < 0 or idx >= len(self.event_metadatas):
                continue
            md = self.event_metadatas[idx].copy()
            results.append({
                "label":    md.pop("label"),
                "score":    float(dist),
                "metadata": md
            })
        return results

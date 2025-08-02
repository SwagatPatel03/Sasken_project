# src/utils/version_mapping.py

import json
from difflib import SequenceMatcher
from typing import List, Dict, Optional
from collections import defaultdict

def map_chunks(
    old_chunks: List[Dict],
    new_chunks: List[Dict],
    title_weight: float = 0.7,
    content_weight: float = 0.3,
    threshold: float = 0.6
) -> Dict[str, Optional[str]]:
    """
    Build a mapping from old_chunk_id -> new_chunk_id (or None).
    We only compare within the same section_id to limit the search space.
    """
    # Group by section
    old_by_sec = defaultdict(list)
    new_by_sec = defaultdict(list)
    for c in old_chunks:
        old_by_sec[c["section_id"]].append(c)
    for c in new_chunks:
        new_by_sec[c["section_id"]].append(c)

    mapping: Dict[str, Optional[str]] = {}

    for sec, olds in old_by_sec.items():
        news = new_by_sec.get(sec, [])
        for o in olds:
            best_score = 0.0
            best_chunk = None
            o_title = o.get("title") or ""
            o_content = o["content"]

            for n in news:
                # Title similarity
                n_title = n.get("title") or ""
                title_score = SequenceMatcher(None, o_title, n_title).ratio() if o_title and n_title else 0.0

                # Content similarity
                content_score = SequenceMatcher(None, o_content, n["content"]).ratio()

                score = title_weight * title_score + content_weight * content_score
                if score > best_score:
                    best_score = score
                    best_chunk = n["chunk_id"]

            mapping[o["chunk_id"]] = best_chunk if best_score >= threshold else None

    return mapping


def save_version_map(mapper: Dict[str, Optional[str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapper, f, indent=2, ensure_ascii=False)

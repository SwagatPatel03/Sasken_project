# src/change_detection/detector.py

import difflib
import json
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

from fuzzywuzzy import fuzz
from parsers.base_parser import DocumentSection


class ChangeType(Enum):
    ADDED    = "added"
    REMOVED  = "removed"
    MODIFIED = "modified"
    MOVED    = "moved"       # for future use


@dataclass
class Change:
    change_type: ChangeType
    section_id: str
    old_content: str
    new_content: str
    similarity_score: float
    context: Dict[str, any]

    def to_dict(self) -> Dict[str, any]:
        """
        Convert this Change to a pure‑dict with JSON‑safe values.
        """
        return {
            "change_type": self.change_type.value,
            "section_id": self.section_id,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "similarity_score": self.similarity_score,
            "context": self.context,
        }

def compute_similarity(a: str, b: str) -> float:
    """
    Compute a hybrid similarity score between 0–1 by averaging:
      - difflib.SequenceMatcher ratio (character-level)
      - fuzzywuzzy.partial_ratio (token-level)
    """
    seq_score = difflib.SequenceMatcher(None, a, b).ratio()
    fuzz_score = fuzz.partial_ratio(a, b) / 100.0
    return (seq_score + fuzz_score) / 2


def _flatten_sections(root: DocumentSection) -> Dict[str, DocumentSection]:
    """
    Recursively build a flat map from section_id to DocumentSection.
    """
    flat: Dict[str, DocumentSection] = {}
    def _rec(sec: DocumentSection):
        flat[sec.section_id] = sec
        for sub in sec.subsections:
            _rec(sub)
    _rec(root)
    return flat


class ChangeDetector:
    """
    Detects ADDED, REMOVED, and MODIFIED changes between two DocumentSection trees.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
          similarity_threshold: below this score a section is flagged MODIFIED
        """
        self.similarity_threshold = similarity_threshold

    def detect_changes(self,
                       old_doc: DocumentSection,
                       new_doc: DocumentSection
                      ) -> List[Change]:
        """
        Compare two DocumentSection trees and return a list of Change objects.

        - ADDED:   section_id in new_doc but not in old_doc
        - REMOVED: section_id in old_doc but not in new_doc
        - MODIFIED: same section_id exists in both but similarity < threshold
        """
        changes: List[Change] = []

        # Flatten both trees for quick lookup
        old_map = _flatten_sections(old_doc)
        new_map = _flatten_sections(new_doc)

        # 1) ADDED
        for sid, new_sec in new_map.items():
            if sid not in old_map:
                changes.append(Change(
                    change_type=ChangeType.ADDED,
                    section_id=sid,
                    old_content="",
                    new_content=new_sec.content,
                    similarity_score=1.0,
                    context={"title": new_sec.title}
                ))

        # 2) REMOVED
        for sid, old_sec in old_map.items():
            if sid not in new_map:
                changes.append(Change(
                    change_type=ChangeType.REMOVED,
                    section_id=sid,
                    old_content=old_sec.content,
                    new_content="",
                    similarity_score=1.0,
                    context={"title": old_sec.title}
                ))

        # 3) MODIFIED
        shared = set(old_map.keys()) & set(new_map.keys())
        for sid in shared:
            old_text = old_map[sid].content or ""
            new_text = new_map[sid].content or ""
            if old_text != new_text:
                score = compute_similarity(old_text, new_text)
                if score < self.similarity_threshold:
                    changes.append(Change(
                        change_type=ChangeType.MODIFIED,
                        section_id=sid,
                        old_content=old_text,
                        new_content=new_text,
                        similarity_score=score,
                        context={"title": new_map[sid].title}
                    ))

        return changes

    def to_json(self, changes: List[Change], **json_kwargs) -> str:
        """
        Serialize a list of Change objects to a JSON-formatted string.
        """
        out = []
        for c in changes:
            out.append({
                "change_type": c.change_type.value,
                "section_id": c.section_id,
                "old_content": c.old_content,
                "new_content": c.new_content,
                "similarity_score": c.similarity_score,
                "context": c.context,
            })
        return json.dumps(out, **json_kwargs)

# src/change_detection/detector.py

import os
import difflib
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


class ChangeType(Enum):
    ADDED    = "added"
    REMOVED  = "removed"
    MODIFIED = "modified"
    MOVED    = "moved"


@dataclass
class Change:
    section_id: str
    chunk_id: str
    change_type: ChangeType
    old_content: str
    new_content: str
    similarity_score: float
    moved_to: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["change_type"] = self.change_type.value
        if self.moved_to is None:
            d.pop("moved_to")
        return d


def compute_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


class ChangeDetector:
    def __init__(self, threshold: float = 0.85, version_map: Dict[str, Optional[str]] = None):
        self.threshold = threshold
        self.version_map = version_map or {}

    def detect_changes(
        self,
        old_chunks: List[Dict],
        new_chunks: List[Dict]
    ) -> List[Change]:
        from collections import defaultdict

        old_by_sec = defaultdict(list)
        new_by_sec = defaultdict(list)
        for c in old_chunks:
            old_by_sec[c["section_id"]].append(c)
        for c in new_chunks:
            new_by_sec[c["section_id"]].append(c)

        all_secs = set(old_by_sec) | set(new_by_sec)
        changes: List[Change] = []

        for sec in sorted(all_secs):
            olds = old_by_sec.get(sec, [])
            news = new_by_sec.get(sec, [])
            old_list = [c["content"] for c in olds]
            new_list = [c["content"] for c in news]

            matcher = difflib.SequenceMatcher(None, old_list, new_list)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue

                if tag == "delete":
                    for idx in range(i1, i2):
                        c = olds[idx]
                        # check for MOVED by version_map
                        moved_to = self.version_map.get(c["chunk_id"])
                        if moved_to:
                            changes.append(Change(
                                section_id=sec,
                                chunk_id=c["chunk_id"],
                                change_type=ChangeType.MOVED,
                                old_content=c["content"],
                                new_content="",
                                similarity_score=1.0,
                                moved_to=moved_to
                            ))
                        else:
                            changes.append(Change(
                                section_id=sec,
                                chunk_id=c["chunk_id"],
                                change_type=ChangeType.REMOVED,
                                old_content=c["content"],
                                new_content="",
                                similarity_score=0.0
                            ))

                elif tag == "insert":
                    for idx in range(j1, j2):
                        c = news[idx]
                        # skip if this new chunk was already mapped from an old one
                        if c["chunk_id"] in self.version_map.values():
                            continue
                        changes.append(Change(
                            section_id=sec,
                            chunk_id=c["chunk_id"],
                            change_type=ChangeType.ADDED,
                            old_content="",
                            new_content=c["content"],
                            similarity_score=1.0
                        ))

                elif tag == "replace":
                    span = min(i2 - i1, j2 - j1)
                    for k in range(span):
                        old_c = olds[i1 + k]
                        new_c = news[j1 + k]
                        score = compute_similarity(old_c["content"], new_c["content"])
                        changes.append(Change(
                            section_id=sec,
                            chunk_id=f"{old_c['chunk_id']}→{new_c['chunk_id']}",
                            change_type=ChangeType.MODIFIED,
                            old_content=old_c["content"],
                            new_content=new_c["content"],
                            similarity_score=score
                        ))

        return changes

    def write_html_diff(
        self,
        old_text: str,
        new_text: str,
        section_id: str,
        out_dir: str = "data/processed/diffs"
    ) -> str:
        os.makedirs(out_dir, exist_ok=True)
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
        html_doc = differ.make_file(old_lines, new_lines,
                                    fromdesc="Rel-15", todesc="Rel-16")
        out_path = os.path.join(out_dir, f"diff_{section_id.replace('.','_')}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return out_path

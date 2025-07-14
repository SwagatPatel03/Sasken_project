# src/parsers/base_parser.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import json
from dataclasses import asdict, field

# @dataclass
# class DocumentSection:
#     section_id: str
#     title: str
#     content: str
#     subsections: List['DocumentSection']
#     tables: List[Dict[str, Any]]
#     metadata: Dict[str, Any]

@dataclass
class DocumentSection:
    section_id: str
    title: str
    content: str
    subsections: List['DocumentSection'] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively convert this DocumentSection (and its subsections)
        into a nested dictionary structure.
        """
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        """
        Serialize to a JSON string. Any kwargs you pass here
        (like indent=2) go straight through to json.dumps.
        """
        return json.dumps(self.to_dict(), **json_kwargs)

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> DocumentSection:
        pass
    
    @abstractmethod
    def extract_tables(self, content: Any) -> List[Dict[str, Any]]:
        pass
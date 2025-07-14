# src/parsers/docx_parser.py

import os
import subprocess
import tempfile
import shutil
from docx import Document
from typing import List, Dict, Any
from .base_parser import BaseParser, DocumentSection
from docx.text.paragraph import Paragraph
from docx.table import Table

class DOCXParser(BaseParser):
    def parse(self, file_path: str) -> DocumentSection:

        doc = Document(file_path)
        root = DocumentSection(
            section_id="root",
            title="",
            content="",
            subsections=[],
            tables=[],
            metadata={"level": 0, "file_path": file_path}
        )
        self._process_elements(doc, root)
        return root

    def _process_elements(self, doc: Document, root: DocumentSection):
        # stack of (section, level); root is level 0
        stack = [(root, 0)]

        for block in doc.element.body:
            # Paragraph (heading, normal, or list)
            if block.tag.endswith('p'):
                p = Paragraph(block, doc)
                text = p.text.strip()
                if not text:
                    continue

                style = p.style.name.lower()
                # Detect heading levels: e.g. "heading 1", "heading 2"
                if style.startswith('heading'):
                    level = int(style.split()[-1])

                    # Pop until parent of this level
                    while stack and stack[-1][1] >= level:
                        stack.pop()

                    parent_section = stack[-1][0]
                    # Create a new section
                    sec = DocumentSection(
                        section_id=f"{parent_section.section_id}.{len(parent_section.subsections)+1}",
                        title=text,
                        content="",
                        subsections=[],
                        tables=[],
                        metadata={"level": level}
                    )
                    parent_section.subsections.append(sec)
                    # Push new section onto stack
                    stack.append((sec, level))

                else:
                    # Regular paragraph or list item → attach to current section
                    current_section = stack[-1][0]
                    # (You can insert your list‑detection logic here)
                    current_section.content += text + "\n"

            # Table
            elif block.tag.endswith('tbl'):
                tbl = Table(block, doc)
                rows = [[cell.text.strip() for cell in row.cells] 
                        for row in tbl.rows]
                header, body = (rows[0], rows[1:]) if rows else ([], [])
                stack[-1][0].tables.append({
                    "header": header,
                    "rows": body
                })

    def extract_tables(self, content: Any) -> List[Dict[str, Any]]:
        # Already folded into _process_elements on tbl blocks
        return []

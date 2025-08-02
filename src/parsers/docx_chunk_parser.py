# src/parsers/docx_chunk_parser.py

import docx, re, os, sys
import spacy
from dataclasses import dataclass
from typing import List

# 1) Define Chunk dataclass
@dataclass
class Chunk:
    section_id: str
    chunk_id: str
    content: str
    position: int

# Load spaCy for sentence splitting
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model missing. Run:\n  python -m spacy download en_core_web_sm")
    sys.exit(1)

MAX_TOKENS = 500

def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def split_into_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def split_long_text(text: str) -> List[str]:
    sentences = split_into_sentences(text)
    chunks, current = [], ""
    for sent in sentences:
        candidate = current + (" " + sent if current else sent)
        if count_tokens(candidate) > MAX_TOKENS and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def is_heading(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\s", text))

def extract_section_number(text: str) -> str:
    m = re.match(r"^(\d+(\.\d+)*)", text)
    return m.group(1) if m else None

def parse_docx_to_chunks(file_path: str) -> List[Chunk]:
    """
    Parse a 3GPP spec .docx into a flat list of sentence‑aware, token‑capped Chunk objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    doc = docx.Document(file_path)

    chunks: List[Chunk] = []
    current_section = None
    position = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if is_heading(text):
            current_section = extract_section_number(text)
            position = 0
            continue

        if not current_section:
            # skip any preamble before first heading
            continue

        # Split long paragraphs into sentence‑bounded chunks
        for sub in split_long_text(text):
            position += 1
            chunks.append(Chunk(
                section_id=current_section,
                chunk_id=f"{current_section}_{position}",
                content=sub,
                position=position
            ))

    return chunks

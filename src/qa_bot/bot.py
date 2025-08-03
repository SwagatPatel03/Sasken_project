# src/qa_bot/bot.py

import json
import os
from typing import List, Dict, Optional
from langchain.schema import HumanMessage
from utils.vector_db import VectorDB
from qa_bot.groq_llm import GroqLLM


class QABot:
    """
    A retrieval-augmented QA assistant over 3GPP spec changes,
    that gets version info from the VectorDB and uses semantic clustering.
    """
    def __init__(
        self,
        vector_db: VectorDB,
        old_chunks_path: Optional[str] = None,
        new_chunks_path: Optional[str] = None,
        llm_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
    ):
        self.vdb = vector_db
        self.llm = GroqLLM(model_name=llm_model, temperature=temperature)

        # Get version info from VectorDB
        versions = self.vdb.get_versions()
        self.rel_old = versions.get("rel_old", {})
        self.rel_new = versions.get("rel_new", {})

        # — load chunk JSONs (for count/introspection) —
        self.old_chunks: List[Dict] = []
        self.new_chunks: List[Dict] = []
        if old_chunks_path:
            try:
                with open(old_chunks_path, "r", encoding="utf-8") as f:
                    self.old_chunks = json.load(f)
            except FileNotFoundError:
                print(f"Warning: failed to load old chunks from {old_chunks_path}")
        if new_chunks_path:
            try:
                with open(new_chunks_path, "r", encoding="utf-8") as f:
                    self.new_chunks = json.load(f)
            except FileNotFoundError:
                print(f"Warning: failed to load new chunks from {new_chunks_path}")

        # Build a map of section_id → heading title
        self.section_titles: Dict[str,str] = {}
        for c in self.old_chunks:
            if c.get("chunk_type") == "heading":
                sid = c["section_id"]
                self.section_titles.setdefault(sid, c["content"])
        for sid in {c["section_id"] for c in self.old_chunks}:
            self.section_titles.setdefault(sid, sid)

        # Load clustered events
        self.events = self._load_events()

    def _load_events(self) -> List[Dict]:
        """Load the clustered events from your script output."""
        events_path = "data/processed/change_events.json"
        if os.path.exists(events_path):
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Could not load events from {events_path}")
                return []
        else:
            print(f"Warning: Events file not found at {events_path}")
            return []

    def _version_header(self) -> str:
        """
        One-line reminder of which versions/releases we're comparing.
        Falls back gracefully if versions.json isn't present.
        """
        vo = self.rel_old.get("version_line", "old spec")
        ro = self.rel_old.get("release_info", "")
        vn = self.rel_new.get("version_line", "new spec")
        rn = self.rel_new.get("release_info", "")
        return f"Comparing {vo} {ro} → {vn} {rn}.\n\n"

    def _call_llm(self, prompt: str) -> str:
        """Invoke GroqLLM, prefixing with version header, return clean text."""
        full = self._version_header() + prompt
        raw = self.llm([HumanMessage(content=full)])
        # our adapter returns dict with 'generations'
        if isinstance(raw, dict) and "generations" in raw:
            gen = raw["generations"][0][0]
            return gen["message"].content.strip()
        # langchain ChatResult
        if hasattr(raw, "generations"):
            return raw.generations[0][0].message.content.strip()
        return str(raw).strip()

    def _is_counting_question(self, question: str) -> bool:
        """Simple heuristic to detect counting questions."""
        count_words = ["how many", "count", "number of", "total", "quantity"]
        return any(word in question.lower() for word in count_words)

    def _simple_classify(self, question: str) -> Dict:
        """Simplified classification - just extract section_id and determine scope."""
        prompt = (
            "Extract information from this question. Output JSON with:\n"
            "- \"section_id\": specific section number (e.g. \"4.2.1\") or null\n"
            "- \"scope\": \"specific\" (for detailed/technical) or \"overview\" (for broad/thematic)\n\n"
            "Examples:\n"
            "\"Changes in section 4.2.1?\" → {\"section_id\":\"4.2.1\", \"scope\":\"specific\"}\n"
            "\"How many subsections in 7.3?\" → {\"section_id\":\"7.3\", \"scope\":\"specific\"}\n"
            "\"What are the main security changes?\" → {\"section_id\":null, \"scope\":\"overview\"}\n"
            "\"Overall protocol improvements?\" → {\"section_id\":null, \"scope\":\"overview\"}\n\n"
            f"Question: \"{question}\"\n"
            "JSON:"
        )
        try:
            resp = self._call_llm(prompt)
            return json.loads(resp)
        except:
            return {"section_id": None, "scope": "overview"}

    def _count_subsections(self, sec: str) -> str:
        """Count subsections under a given section, with mini-RAG for context."""
        prefix = sec + "."
        depth  = sec.count(".") + 1
        old_ids = {c["section_id"] for c in self.old_chunks}
        new_ids = {c["section_id"] for c in self.new_chunks}
        o_cnt = len([s for s in old_ids if s.startswith(prefix) and s.count(".")==depth])
        n_cnt = len([s for s in new_ids if s.startswith(prefix) and s.count(".")==depth])

        # Direct answer format
        prompt = (
            f"Question: How many subsections are in section {sec}?\n\n"
            f"Data: Old spec has {o_cnt} subsections, new spec has {n_cnt} subsections.\n\n"
            "Answer in this format:\n"
            "ANSWER: [Direct answer]\n"
            "EXPLANATION: [Brief reason why]"
        )
        return self._call_llm(prompt)

    def _summarize_section(self, sec: str, top_k: int) -> str:
        """Summarize changes in a specific section (falls back from events→chunks)."""
        title = self.section_titles.get(sec, sec)
        hits = self.vdb.query_changes(title, top_k=top_k)
        hits = [h for h in hits
                if h["metadata"]["section_id"] == sec
                or h["metadata"]["section_id"].startswith(sec + ".")]
        if not hits:
            return f"ANSWER: No changes found for section {sec}.\nEXPLANATION: No matching changes in the vector database for this section."

        # Condensed context - only key info
        key_changes = []
        for h in hits[:3]:  # Limit to top 3 most relevant
            change_type = h['metadata']['change_type']
            section = h['metadata']['section_id']
            text_preview = h['text'][:100] + "..." if len(h['text']) > 100 else h['text']
            key_changes.append(f"[{change_type}] Sec {section}: {text_preview}")
        
        context = "\n".join(key_changes)
        
        prompt = (
            f"Question: What changed in section {sec}?\n\n"
            f"Top changes:\n{context}\n\n"
            "Answer in this format:\n"
            "ANSWER: [Direct summary of what changed]\n"
            "EXPLANATION: [Detailed technical explanation of the changes and their implications, without mentioning backend processes]"
        )
        return self._call_llm(prompt)

    def _thematic_answer(self, question: str, top_k: int) -> str:
        """Use your clustered events for thematic/overview questions."""
        if not self.events:
            return self._chunk_level_answer(question, top_k)
        
        try:
            event_hits = self.vdb.query_events(question, top_k=3)  # Reduced from 5
        except Exception as e:
            return self._chunk_level_answer(question, top_k)
        
        if not event_hits:
            return self._chunk_level_answer(question, top_k)
        
        # Get representative examples from top themes
        key_examples = []
        theme_names = []
        
        for event in event_hits:
            theme_names.append(event['label'])
            # Get 2 concrete examples from this theme
            members = event['metadata']['members'][:2]
            for member_idx in members:
                if member_idx < len(self.vdb.chunk_metadatas):
                    chunk_meta = self.vdb.chunk_metadatas[member_idx]
                    example = f"Section {chunk_meta['section_id']}: {chunk_meta['text'][:120]}{'...' if len(chunk_meta['text']) > 120 else ''}"
                    key_examples.append(example)
        
        themes_summary = ", ".join(theme_names)
        examples_text = "\n".join(key_examples[:4])  # Limit to 4 examples
        
        prompt = (
            f"Question: {question}\n\n"
            f"Key change areas: {themes_summary}\n\n"
            f"Specific examples:\n{examples_text}\n\n"
            "Answer in this format:\n"
            "ANSWER: [Direct answer to the question]\n"
            "EXPLANATION: [More detailed explanation with specific technical details and impacts, without mentioning clustering or backend processes]"
        )
        
        return self._call_llm(prompt)

    def _section_specific_answer(self, question: str, section_id: str, top_k: int) -> str:
        """Handle section-specific questions with chunk-level precision."""
        title = self.section_titles.get(section_id, section_id)
        
        section_hits = self.vdb.query_changes(f"{title} {question}", top_k=top_k)
        
        filtered_hits = [
            h for h in section_hits
            if (h["metadata"]["section_id"] == section_id or 
                h["metadata"]["section_id"].startswith(section_id + "."))
        ]
        
        if not filtered_hits:
            return f"ANSWER: No specific changes found for section {section_id}.\nEXPLANATION: No matching changes in the database for this section."
        
        # Only show most relevant change
        top_change = filtered_hits[0]
        change_summary = f"[{top_change['metadata']['change_type']}] {top_change['text'][:150]}{'...' if len(top_change['text']) > 150 else ''}"
        
        prompt = (
            f"Question: {question} (about section {section_id})\n\n"
            f"Most relevant change: {change_summary}\n\n"
            "Answer in this format:\n"
            "ANSWER: [Direct answer]\n"
            "EXPLANATION: [Detailed technical explanation of what this means and why it matters, without mentioning backend processes]"
        )
        
        return self._call_llm(prompt)

    def _chunk_level_answer(self, question: str, top_k: int) -> str:
        """Traditional chunk-level RAG as fallback."""
        hits = self.vdb.query_changes(question, top_k=3)  # Reduced to top 3
        
        if not hits:
            return "ANSWER: No relevant changes found.\nEXPLANATION: No matching changes in the database for this query."
        
        # Condensed context
        top_hit = hits[0]
        context_summary = f"[{top_hit['metadata']['change_type']}] Section {top_hit['metadata']['section_id']}: {top_hit['text'][:100]}{'...' if len(top_hit['text']) > 100 else ''}"
        
        prompt = (
            f"Question: {question}\n\n"
            f"Most relevant change: {context_summary}\n\n"
            "Answer in this format:\n"
            "ANSWER: [Direct answer]\n"
            "EXPLANATION: [Detailed explanation of the technical implications and significance of these changes]"
        )
        return self._call_llm(prompt)

    def answer_question(self, question: str, top_k: int = 10) -> str:
        """Smart routing: use clustering results + simple classification."""
        classification = self._simple_classify(question)
        section_id = classification.get("section_id")
        scope = classification.get("scope", "overview")
        
        # Route based on what we have and what's being asked
        if self._is_counting_question(question) and section_id:
            return self._count_subsections(section_id)
        
        elif section_id and scope == "specific":
            # Specific section questions - use chunk-level search
            return self._section_specific_answer(question, section_id, top_k)
        
        elif scope == "overview" and self.events:
            # Thematic/overview questions - use event-level clustering
            return self._thematic_answer(question, top_k)
        
        else:
            # Fallback to traditional chunk-level RAG
            return self._chunk_level_answer(question, top_k)
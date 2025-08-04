# src/qa_bot/bot.py

import json
import os
from typing import List, Dict, Optional, Tuple
from langchain.schema import HumanMessage
from utils.vector_db import VectorDB
from qa_bot.groq_llm import GroqLLM


class QABot:
    """
    A hybrid retrieval-augmented QA assistant that ALWAYS uses both:
    1. Clustering-based search (thematic context)
    2. Chunk-based search (specific details)
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

        # Load chunk JSONs
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

        # Build section titles map
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
        """Version comparison header."""
        vo = self.rel_old.get("version_line", "old spec")
        ro = self.rel_old.get("release_info", "")
        vn = self.rel_new.get("version_line", "new spec")
        rn = self.rel_new.get("release_info", "")
        return f"Comparing {vo} {ro} → {vn} {rn}.\n\n"

    def _call_llm(self, prompt: str) -> str:
        """Invoke GroqLLM with version header."""
        full = self._version_header() + prompt
        raw = self.llm([HumanMessage(content=full)])
        if isinstance(raw, dict) and "generations" in raw:
            gen = raw["generations"][0][0]
            return gen["message"].content.strip()
        if hasattr(raw, "generations"):
            return raw.generations[0][0].message.content.strip()
        return str(raw).strip()

    def _is_counting_question(self, question: str) -> bool:
        """Detect counting questions."""
        count_words = ["how many", "count", "number of", "total", "quantity"]
        return any(word in question.lower() for word in count_words)

    def _extract_section_id(self, question: str) -> Optional[str]:
        """Extract section ID if mentioned in question."""
        prompt = (
            "Extract the section number if mentioned in this question. "
            "Return just the section number (e.g. '4.2.1') or 'null' if none mentioned.\n\n"
            f"Question: \"{question}\"\n"
            "Section number:"
        )
        try:
            resp = self._call_llm(prompt).strip().strip('"\'')
            return resp if resp.lower() != 'null' else None
        except:
            return None

    def _get_thematic_context(self, question: str) -> Tuple[List[Dict], str]:
        """Get clustering-based thematic context."""
        if not self.events:
            return [], "No thematic clustering available"
        
        try:
            event_hits = self.vdb.query_events(question, top_k=3)
        except Exception as e:
            return [], f"Clustering search failed: {e}"
        
        if not event_hits:
            return [], "No relevant themes found"
        
        # Build thematic context
        thematic_summary = []
        evidence_details = []
        
        for event in event_hits:
            theme_name = event['label']
            member_count = len(event['metadata']['members'])
            thematic_summary.append(f"• {theme_name} ({member_count} changes)")
            
            # Get concrete examples
            sample_members = event['metadata']['members'][:2]
            for member_idx in sample_members:
                if member_idx < len(self.vdb.chunk_metadatas):
                    chunk = self.vdb.chunk_metadatas[member_idx]
                    evidence_details.append({
                        'theme': theme_name,
                        'section': chunk['section_id'],
                        'change_type': chunk['change_type'],
                        'text': chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                    })
        
        context_text = "THEMATIC OVERVIEW:\n" + "\n".join(thematic_summary)
        if evidence_details:
            context_text += "\n\nKEY EXAMPLES:\n"
            for detail in evidence_details[:4]:  # Limit examples
                context_text += f"- [{detail['theme']}] Section {detail['section']} ({detail['change_type']}): {detail['text']}\n"
        
        return event_hits, context_text

    def _get_specific_context(self, question: str, section_filter: Optional[str] = None) -> Tuple[List[Dict], str]:
        """Get chunk-based specific context."""
        try:
            chunk_hits = self.vdb.query_changes(question, top_k=8)
        except Exception as e:
            return [], f"Chunk search failed: {e}"
        
        if not chunk_hits:
            return [], "No specific changes found"
        
        # Filter by section if specified
        if section_filter:
            chunk_hits = [
                h for h in chunk_hits
                if (h["metadata"]["section_id"] == section_filter or 
                    h["metadata"]["section_id"].startswith(section_filter + "."))
            ]
        
        if not chunk_hits:
            return [], f"No changes found for section {section_filter}" if section_filter else "No relevant changes found"
        
        # Build specific context
        context_text = "SPECIFIC CHANGES:\n"
        sections_covered = set()
        
        for i, hit in enumerate(chunk_hits[:6]):  # Top 6 for good coverage
            section = hit['metadata']['section_id']
            change_type = hit['metadata']['change_type']
            text = hit['text'][:120] + "..." if len(hit['text']) > 120 else hit['text']
            
            context_text += f"{i+1}. Section {section} ({change_type}): {text}\n"
            sections_covered.add(section)
        
        if sections_covered:
            context_text += f"\nSections affected: {', '.join(sorted(sections_covered))}"
        
        return chunk_hits, context_text

    def _count_subsections(self, sec: str) -> str:
        """Handle counting questions with both approaches."""
        prefix = sec + "."
        depth  = sec.count(".") + 1
        old_ids = {c["section_id"] for c in self.old_chunks}
        new_ids = {c["section_id"] for c in self.new_chunks}
        o_cnt = len([s for s in old_ids if s.startswith(prefix) and s.count(".")==depth])
        n_cnt = len([s for s in new_ids if s.startswith(prefix) and s.count(".")==depth])

        # Get both contexts
        _, thematic_context = self._get_thematic_context(f"section {sec} structure changes")
        _, specific_context = self._get_specific_context(f"section {sec} organization", section_filter=sec)

        prompt = (
            f"Question: How many subsections are in section {sec}?\n\n"
            f"COUNTING DATA:\n"
            f"Old specification: {o_cnt} subsections\n"
            f"New specification: {n_cnt} subsections\n\n"
            f"{thematic_context}\n\n"
            f"{specific_context}\n\n"
            "Provide a response in this exact format:\n\n"
            "**ANSWER:**\n[Direct numerical answer with comparison]\n\n"
            "**EXPLANATION:**\n[Give a list of points explaining what this change means and any relevant "
            "context from the thematic and specific changes shown above]"
        )
        return self._call_llm(prompt)

    def answer_question(self, question: str, top_k: int = 10) -> str:
        """
        HYBRID APPROACH: Always get both thematic and specific context,
        then let the LLM synthesize the best response.
        """
        
        # Handle counting questions specially
        section_id = self._extract_section_id(question)
        if self._is_counting_question(question) and section_id:
            return self._count_subsections(section_id)
        
        # ALWAYS get both contexts
        event_hits, thematic_context = self._get_thematic_context(question)
        chunk_hits, specific_context = self._get_specific_context(question, section_filter=section_id)
        
        # Check if we have ANY useful context (don't be too strict about scores)
        has_thematic = len(event_hits) > 0
        has_specific = len(chunk_hits) > 0
        
        # Build combined context with guidance for the LLM
        context_quality = []
        if has_thematic:
            context_quality.append("thematic patterns")
        if has_specific:
            context_quality.append("specific technical details")
        
        # Only return "no results" if we have absolutely nothing
        if not has_thematic and not has_specific:
            return (
                "**ANSWER:**\n"
                "No relevant changes found for this query.\n\n"
                "**EXPLANATION:**\n"
                "The question doesn't match any documented changes in the specification comparison. "
                "You may want to rephrase your question or ask about a different aspect of the specifications."
            )
        
        quality_note = f"Available context: {' and '.join(context_quality)}" if context_quality else "Limited context available"
        
        prompt = (
            f"Question: {question}\n\n"
            f"{thematic_context}\n\n"
            f"{specific_context}\n\n"
            f"Context note: {quality_note}\n\n"
            "Instructions: Use ALL available information from above to provide a comprehensive response. "
            "Do not mention 'thematic overview', 'clustering', 'database', or any technical backend terms. "
            "Write as if you're a specification expert analyzing the changes.\n\n"
            "Provide a response in this exact format:\n\n"
            "**ANSWER:**\n[Direct, clear answer about what changed]\n\n"
            "**EXPLANATION:**\n[A list providing substantial detail about the changes found. Include:\n"
            "- What specifically was modified/added/removed\n"
            "- The technical significance of these changes\n"
            "- How these changes might impact implementation or functionality\n"
            "- Context about related sections or broader implications\n"
            "- Reference specific section numbers and change types naturally\n"
            "Write this as expert analysis, not as a summary of search results]"
        )
        
        return self._call_llm(prompt)
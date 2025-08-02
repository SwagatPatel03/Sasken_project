# src/qa_bot/bot.py

import json
from typing import List, Dict, Optional
from langchain.schema import HumanMessage
from utils.vector_db import VectorDB
from qa_bot.groq_llm import GroqLLM  # your thin Groq wrapper


class QABot:
    """
    Retrieval-augmented QA bot over 3GPP spec changes (Rel-15 vs Rel-16),
    using FAISS for vector search and Groq API for LLM.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        old_chunks_path: Optional[str] = None,
        new_chunks_path: Optional[str] = None,
        llm_model: str = "llama3-70b-8192",
        temperature: float = 0.1,
    ):
        self.vdb = vector_db
        self.llm = GroqLLM(model_name=llm_model, temperature=temperature)

        # — load chunk JSONs (for count intents and title lookup) —
        self.old_chunks: List[Dict] = []
        self.new_chunks: List[Dict] = []
        if old_chunks_path:
            try:
                with open(old_chunks_path, "r", encoding="utf-8") as f:
                    self.old_chunks = json.load(f)
            except FileNotFoundError:
                print(f"Warning: could not load old chunks from {old_chunks_path}")
        if new_chunks_path:
            try:
                with open(new_chunks_path, "r", encoding="utf-8") as f:
                    self.new_chunks = json.load(f)
            except FileNotFoundError:
                print(f"Warning: could not load new chunks from {new_chunks_path}")

        # — build section_id → title map from old_chunks —
        self.section_titles: Dict[str,str] = {}
        for c in self.old_chunks:
            sid = c["section_id"]
            if c.get("chunk_type") == "heading" and sid not in self.section_titles:
                self.section_titles[sid] = c.get("content", sid)
        # any missing, fallback to the raw id
        for sid in {c["section_id"] for c in self.old_chunks}:
            self.section_titles.setdefault(sid, sid)


    def _call_llm(self, prompt: str) -> str:
        """Invoke GroqLLM and normalize output to a string."""
        raw = self.llm([HumanMessage(content=prompt)])
        # handle our dict-based adapter
        if isinstance(raw, dict) and "generations" in raw:
            gen = raw["generations"][0][0]
            return gen["message"].content.strip()
        # LangChain ChatResult
        if hasattr(raw, "generations"):
            return raw.generations[0][0].message.content.strip()
        # fallback
        return str(raw).strip()


    def _classify(self, question: str) -> Dict[str, Optional[str]]:
        """Extract section_id + intent via LLM-based JSON classification."""
        prompt = f"""You are a specialized 3GPP specification analysis assistant.
Given a user question, output JSON with exactly two keys:
- "section_id": the section (e.g. "5.5.1") if mentioned, otherwise null
- "intent": one of "count", "summarize", or "generic"

Examples:
- "How many subsections are in section 4.2?" → {{"section_id":"4.2","intent":"count"}}
- "Summarize changes in section 7.3" → {{"section_id":"7.3","intent":"summarize"}}
- "What are the security implications?" → {{"section_id":null,"intent":"generic"}}

Question: "{question}"
Output only valid JSON:"""

        resp = self._call_llm(prompt)
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return {"section_id": None, "intent": "generic"}


    def _count_subsections(self, sec: str) -> str:
        """Enhanced counting with content summarization via RAG."""
        if not (self.old_chunks and self.new_chunks):
            return "No chunk data available to count subsections."

        prefix = sec + "."
        depth = sec.count(".") + 1
        old_ids = {c["section_id"] for c in self.old_chunks}
        new_ids = {c["section_id"] for c in self.new_chunks}

        old_subs = [s for s in old_ids if s.startswith(prefix) and s.count(".")==depth]
        new_subs = [s for s in new_ids if s.startswith(prefix) and s.count(".")==depth]
        
        o_cnt = len(old_subs)
        n_cnt = len(new_subs)
        
        # Get all unique subsections and retrieve their content
        all_subsections = sorted(set(old_subs + new_subs))
        
        # Retrieve content for each subsection using RAG
        subsection_content = {}
        for subsec in all_subsections:
            hits = self.vdb.query_changes(subsec, top_k=5)
            # Filter for exact section or immediate children
            relevant_hits = [
                h for h in hits
                if h["metadata"]["section_id"] == subsec 
                or h["metadata"]["section_id"].startswith(subsec + ".")
            ]
            subsection_content[subsec] = relevant_hits[:3]  # Top 3 most relevant
        
        # Build context for LLM
        content_ctx = ""
        for subsec in all_subsections:
            hits = subsection_content[subsec]
            if hits:
                content_ctx += f"\n**Section {subsec}:**\n"
                for hit in hits:
                    content_ctx += f"- [{hit['metadata']['change_type']}] {hit['text'][:200]}...\n"
            else:
                content_ctx += f"\n**Section {subsec}:** No detailed content available\n"
        
        # Enhanced prompt for counting with content analysis
        prompt = f"""You are an expert 3GPP telecommunications specification analyst. Analyze the subsection structure and content for section {sec}.

**Task:** Provide a comprehensive analysis including:
1. Subsection count comparison (old vs new specification)
2. Content summary for each subsection
3. Analysis of structural changes
4. Technical implications of the changes

**Section Analysis Data:**
- **Parent Section:** {sec}
- **Old specification subsections:** {o_cnt} ({', '.join(sorted(old_subs)) if old_subs else 'None'})
- **New specification subsections:** {n_cnt} ({', '.join(sorted(new_subs)) if new_subs else 'None'})
- **Added subsections:** {', '.join(sorted(set(new_subs) - set(old_subs))) if set(new_subs) - set(old_subs) else 'None'}
- **Removed subsections:** {', '.join(sorted(set(old_subs) - set(new_subs))) if set(old_subs) - set(new_subs) else 'None'}

**Detailed Content Information:**
{content_ctx}

**Required Response Structure:**

## Section {sec} Subsection Analysis

### Summary Statistics
[Provide count comparison and structural changes overview]

### Subsection Content Analysis
[For each subsection, provide a concise summary of its content and purpose]

### Structural Changes Impact
[Analyze what the addition/removal of subsections means technically]

### Key Observations
[Highlight the most significant changes or patterns]

Provide a comprehensive analysis:"""
        
        return self._call_llm(prompt)


    def _summarize_section(self, sec: str, top_k: int) -> str:
        """
        Retrieval-augmented summary of changes in section `sec`.
        Query FAISS with the section title, then post-filter by section_id.
        """
        title = self.section_titles.get(sec)
        if not title:
            return f"Unknown section {sec}."

        # --- 1) Retrieve semantically similar change snippets by title ---
        hits = self.vdb.query_changes(title, top_k=top_k)

        # --- 2) Post-filter to exactly this section + subsections ---
        hits = [
            h for h in hits
            if h["metadata"]["section_id"] == sec
               or h["metadata"]["section_id"].startswith(sec + ".")
        ]
        if not hits:
            return f"No changes found for section {sec}."

        # --- 3) Format context for LLM ---
        ctx_lines = []
        for i, h in enumerate(hits, start=1):
            ctx_lines.append(
                f"{i}. [{h['metadata']['change_type']}] "
                f"Sec {h['metadata']['section_id']} · chunk {h['metadata']['chunk_id']}\n"
                f"> {h['text']}"
            )
        ctx = "\n\n".join(ctx_lines)

        # --- 4) Summarization prompt ---
        prompt = f"""You are an expert 3GPP technical analyst.
Summarize the changes in section {sec} (“{title}”) given these snippets:

{ctx}

Please provide a clear, structured summary highlighting key technical implications."""
        return self._call_llm(prompt)


    def _generic_rag(self, question: str, top_k: int) -> str:
        """Free-form RAG: retrieve top_k hits for arbitrary queries."""
        hits = self.vdb.query_changes(question, top_k=top_k)
        if not hits:
            return "No relevant changes found for your question."

        ctx = "\n\n".join(
            f"{i+1}. [{h['metadata']['change_type']}] "
            f"Sec {h['metadata']['section_id']} · chunk {h['metadata']['chunk_id']}\n"
            f"> {h['text']}"
            for i, h in enumerate(hits)
        )

        prompt = f"""You are an expert 3GPP technical analyst.
Answer the question below using **only** the provided change snippets (do not invent beyond them).

Question: {question}

Context snippets:
{ctx}

Provide a detailed, technically accurate answer."""
        return self._call_llm(prompt)


    def answer_question(self, question: str, top_k: int = 10) -> str:
        """Entry point: classify and dispatch to the right handler."""
        qc     = self._classify(question)
        sec    = qc.get("section_id")
        intent = qc.get("intent", "generic").lower()

        if intent == "count" and sec:
            return self._count_subsections(sec)
        if intent == "summarize" and sec:
            return self._summarize_section(sec, top_k)
        return self._generic_rag(question, top_k)

#!/usr/bin/env python3
# scripts/cluster_events.py

import os
import json
import pickle
from dotenv import load_dotenv
import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from langchain.schema import HumanMessage

# import your thin wrapper
from src.qa_bot.groq_llm import GroqLLM

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
RAW_DIFFS         = "data/processed/changes.json"
OUT_META          = "data/processed/change_events.json"
EMB_NPY           = "data/embeddings/diff_embeddings.npy"
EMB_META          = "data/embeddings/diff_meta.pkl"
MODEL             = "sentence-transformers/all-MiniLM-L6-v2"
CLUSTER_MIN_SIZE  = 5
# ────────────────────────────────────────────────────────────────────────────────

# 1) load .env so GROQ_API_KEY is populated
load_dotenv()

# 2) load raw diffs
with open(RAW_DIFFS, "r", encoding="utf-8") as f:
    diffs = json.load(f)

texts = [d["new_content"] or d["old_content"] for d in diffs]

# 3) embed (or load cached embeddings)
emb_model = SentenceTransformer(MODEL)
if not os.path.exists(EMB_NPY):
    os.makedirs(os.path.dirname(EMB_NPY), exist_ok=True)
    embs = emb_model.encode(texts, convert_to_numpy=True)
    np.save(EMB_NPY, embs)
    with open(EMB_META, "wb") as f:
        pickle.dump(diffs, f)
else:
    embs = np.load(EMB_NPY)

# 4) cluster into events
clusterer = hdbscan.HDBSCAN(min_cluster_size=CLUSTER_MIN_SIZE, metric="euclidean")
labels = clusterer.fit_predict(embs)
num_clusters = int(labels.max()) + 1
print(f"Found {num_clusters} clusters (excluding noise: label==-1).")

# 5) prepare your Groq LLM
llm = GroqLLM(model_name="llama-3.3-70b-versatile", temperature=0.0)

def call_llm(prompt: str) -> str:
    """Uniformly call your GroqLLM and extract a string."""
    raw = llm([HumanMessage(content=prompt)])
    if isinstance(raw, dict) and "generations" in raw:
        gen = raw["generations"][0][0]
        return gen["message"].content
    if hasattr(raw, "generations"):
        return raw.generations[0][0].message.content
    return str(raw)

# 6) label each non-noise cluster
events = []
for cid in range(num_clusters):
    members = [i for i, lbl in enumerate(labels) if lbl == cid]
    if not members:
        continue
    examples = [texts[i] for i in members[:3]]
    prompt = (
        f"Here are example change snippets between your old and new 3GPP specs:\n\n"
        + "\n\n".join(f"- {ex}" for ex in examples)
        + "\n\nPlease give this group a concise title (3–5 words) describing its theme."
    )
    title = call_llm(prompt).strip().strip('"')
    events.append({
        "event_id": cid,
        "label": title,
        "members": members
    })
    print(f"Cluster {cid} → “{title}” ({len(members)} snippets)")

# 7) save the event definitions
os.makedirs(os.path.dirname(OUT_META), exist_ok=True)
with open(OUT_META, "w", encoding="utf-8") as f:
    json.dump(events, f, indent=2, ensure_ascii=False)

print(f"✓ Clustered into {len(events)} events and wrote {OUT_META}")

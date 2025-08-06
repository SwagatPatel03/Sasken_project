#src/api.py

import os
import json
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from utils.vector_db import VectorDB
from qa_bot.bot import QABot

# ─── 1) Read env & config ──────────────────────────────────────────────────────
load_dotenv()
with open("config/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ─── 2) Paths ───────────────────────────────────────────────────────────────────
VERSIONS_JSON = "data/processed/versions.json"
CHANGES_JSON  = "data/processed/changes.json"     # optional, for inspection

# ─── 3) FastAPI setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="3GPP Change-Detection QA API",
    description="Retrieval-augmented QA over Rel-15 vs Rel-17 changes",
)

class QARequest(BaseModel):
    question: str
    top_k:    int = 20

class QAResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    # 1) Load version metadata
    versions = {}
    if os.path.exists(VERSIONS_JSON):
        with open(VERSIONS_JSON, "r", encoding="utf-8") as f:
            versions = json.load(f)
    else:
        print(f"Warning: {VERSIONS_JSON} not found. Running without version info.")

    # 2) Load your FAISS-backed vector DB with version info
    vdb = VectorDB(
        persist_directory=cfg["vector_db"]["persist_directory"],
        model_name=cfg["models"]["embedding_model"],
        versions=versions  # Pass versions to VectorDB
    )

    # 3) Instantiate the QA bot
    app.state.bot = QABot(
        vector_db=vdb,
        old_chunks_path="data/processed/24301-af0_chunks.json",
        new_chunks_path="data/processed/24301-hc0_chunks.json",
        llm_model=cfg["models"]["llm_model"],
        temperature=cfg["qa_bot"]["temperature"],
    )

    # 4) (Optional) make the raw changes.json available for debugging
    if os.path.exists(CHANGES_JSON):
        with open(CHANGES_JSON, "r", encoding="utf-8") as f:
            app.state.raw_changes = json.load(f)


@app.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest):
    bot = app.state.bot
    try:
        answer = bot.answer_question(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return QAResponse(answer=answer)

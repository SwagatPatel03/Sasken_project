# import os
# import json
# import yaml
# from dotenv import load_dotenv
# from fastapi import FastAPI
# from pydantic import BaseModel

# from utils.vector_db import VectorDB
# from change_detection.detector import ChangeDetector
# from qa_bot.bot import QABot

# # 1️⃣ Read environment & config
# load_dotenv()
# with open("config/config.yaml", "r") as f:
#     cfg = yaml.safe_load(f)

# # 2️⃣ Chunk JSONs (produced once via `src/main.py parse`)
# REL10_CHUNKS = "data/processed/24301-af0_chunks.json"
# REL17_CHUNKS = "data/processed/24301-hc0_chunks.json"
# CHANGES_JSON = "data/processed/changes.json"

# # 3️⃣ FastAPI app & request/response models
# app = FastAPI()

# class QARequest(BaseModel):
#     question: str
#     top_k:    int = 10

# class QAResponse(BaseModel):
#     answer: str


# @app.on_event("startup")
# def startup_event():
#     # — Ensure chunk files exist —
#     if not os.path.exists(REL10_CHUNKS) or not os.path.exists(REL17_CHUNKS):
#         raise RuntimeError(
#             "Missing chunk JSON. Run:\n"
#             "  python src/main.py parse\n"
#             "to generate those first."
#         )

#     # — Load parsed chunk lists —
#     with open(REL10_CHUNKS, "r", encoding="utf-8") as f:
#         old_chunks = json.load(f)
#     with open(REL17_CHUNKS, "r", encoding="utf-8") as f:
#         new_chunks = json.load(f)

#     # — 1) Change detection (once) —
#     threshold = cfg["change_detection"]["similarity_threshold"]
#     detector  = ChangeDetector(threshold=threshold)
#     changes   = detector.detect_changes(old_chunks, new_chunks)

#     # — Persist for inspection —
#     os.makedirs(os.path.dirname(CHANGES_JSON), exist_ok=True)
#     with open(CHANGES_JSON, "w", encoding="utf-8") as f:
#         json.dump([c.to_dict() for c in changes], f, indent=2)

#     # — 2) Build FAISS VectorDB (events + chunks) —
#     vdb = VectorDB(
#         persist_directory=cfg["vector_db"]["persist_directory"],
#         model_name=cfg["models"]["embedding_model"]
#     )
#     # This will internally build both
#     #   - an event‐level index (clusters of related chunks)
#     #   - a chunk‐level index
#     vdb.store_changes(changes)

#     # — 3) Instantiate our retrieval‐augmented bot —
#     app.state.bot = QABot(
#         vector_db=vdb,
#         old_chunks_path=REL10_CHUNKS,
#         new_chunks_path=REL17_CHUNKS,
#         llm_model=cfg["models"]["llm_model"],
#         temperature=cfg["qa_bot"]["temperature"],
#     )


# @app.post("/qa", response_model=QAResponse)
# def qa_endpoint(req: QARequest):
#     """
#     Accepts a JSON body:
#       { "question": "...", "top_k": 5 }
#     Returns:
#       { "answer": "..." }
#     """
#     bot: QABot = app.state.bot
#     answer = bot.answer_question(req.question, top_k=req.top_k)
#     return QAResponse(answer=answer)

# src/api.py

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

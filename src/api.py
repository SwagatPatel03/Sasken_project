import os
import json
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from utils.vector_db import VectorDB
from change_detection.detector import ChangeDetector
from qa_bot.bot import QABot

# 1) Read .env & config
load_dotenv()
cfg = yaml.safe_load(open("config/config.yaml", "r"))

# 2) Paths to your already-generated chunk JSONs
REL17_JSON = "data/processed/24301-af0_chunks.json"
REL10_JSON = "data/processed/24301-hc0_chunks.json"
CHANGES_JSON = "data/processed/changes.json"

# 3) FastAPI setup
app = FastAPI()

class QARequest(BaseModel):
    question: str
    top_k:    int = 10

class QAResponse(BaseModel):
    answer: str

@app.on_event("startup")
def startup_event():
    # 4.1) Ensure chunk JSON exist
    if not os.path.exists(REL10_JSON) or not os.path.exists(REL17_JSON):
        raise RuntimeError("Run `python src/main.py parse` first to generate chunk JSON.")

    # 4.2) Load chunk lists
    with open(REL10_JSON, "r", encoding="utf-8") as f:
        old_chunks = json.load(f)
    with open(REL17_JSON, "r", encoding="utf-8") as f:
        new_chunks = json.load(f)

    # 4.3) Run change detection once
    threshold = cfg["change_detection"]["similarity_threshold"]
    detector  = ChangeDetector(threshold=threshold)
    changes   = detector.detect_changes(old_chunks, new_chunks)

    # 4.4) Save the changes.json for debug
    os.makedirs(os.path.dirname(CHANGES_JSON), exist_ok=True)
    with open(CHANGES_JSON, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in changes], f, indent=2)

    # 4.5) Build FAISS-backed VectorDB
    vdb = VectorDB(
        persist_directory=cfg["vector_db"]["persist_directory"],
        model_name=cfg["models"]["embedding_model"]
    )
    vdb.store_changes(changes)

    # 4.6) Create the QABot and stash on app.state
    app.state.bot = QABot(
        vector_db=vdb,
        old_chunks_path=REL10_JSON,
        new_chunks_path=REL17_JSON,
        llm_model=cfg["models"]["llm_model"],
        temperature=cfg["qa_bot"]["temperature"]
    )

@app.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest):
    bot: QABot = app.state.bot
    answer = bot.answer_question(req.question, top_k=req.top_k)
    return QAResponse(answer=answer)

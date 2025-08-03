# src/main.py

import click
import json
import os
import subprocess
import yaml

from parsers.docx_parser import parse_docx, save_as_json
from utils.version_mapping import map_chunks, save_version_map
from change_detection.detector import ChangeDetector
from utils.vector_db import VectorDB

# ── Load configuration ─────────────────────────────────────────────────────────
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


@click.group()
def cli():
    """3GPP Change‐Detection System"""
    pass


@cli.command()
@click.option("--min_tokens", default=20, help="Minimum tokens for merging tiny chunks")
@click.option("--max_tokens", default=500, help="Maximum tokens per chunk")
def parse(min_tokens, max_tokens):
    """
    Parse each spec DOCX into hierarchical, token-capped chunks.
    """
    for rel in ["24301-af0", "24301-hc0"]:
        src = f"data/raw/{rel}.docx"
        out = f"data/processed/{rel}_chunks.json"
        click.echo(f"Parsing {src} …")
        chunks = parse_docx(
            src,
            max_tokens=max_tokens,
            min_chunk_tokens=min_tokens
        )
        save_as_json(chunks, out)
        click.secho(f"✓ {len(chunks)} chunks → {out}", fg="green")


@cli.command()
def detect():
    """
    Build version mapping old→new, then detect chunk-wise changes (incl. MOVED).
    Outputs:
      - data/processed/version_map.json
      - data/processed/changes.json
    """
    # 1) load the two chunk lists
    def _load(p): return json.load(open(p, "r", encoding="utf-8"))
    old_chunks = _load("data/processed/24301-af0_chunks.json")
    new_chunks = _load("data/processed/24301-hc0_chunks.json")

    # 2) build & save version map
    click.echo("Mapping old chunks → new chunks…")
    version_map = map_chunks(
        old_chunks,
        new_chunks,
        title_weight=0.7,
        content_weight=0.3,
        threshold=cfg["change_detection"]["mapping_threshold"]
    )
    vm_path = "data/processed/version_map.json"
    save_version_map(version_map, vm_path)
    click.secho(f"✓ Version map ({len(version_map)} entries) → {vm_path}", fg="green")

    # 3) detect changes with MOVED support
    click.echo("Detecting changes between versions…")
    detector = ChangeDetector(
        threshold=cfg["change_detection"]["similarity_threshold"],
        version_map=version_map
    )
    changes = detector.detect_changes(old_chunks, new_chunks)

    # 4) serialize to JSON
    ch_path = "data/processed/changes.json"
    os.makedirs(os.path.dirname(ch_path), exist_ok=True)
    with open(ch_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in changes],
                  f, indent=2, ensure_ascii=False)
    click.secho(f"✓ Wrote {len(changes)} changes → {ch_path}", fg="green")


@cli.command()
def builddb():
    """
    Cluster the diffs into high-level "events" and rebuild the vector DB.
    Requires that `detect` has already been run.
    """
    # 1) load the diffs + events
    with open("data/processed/changes.json", "r", encoding="utf-8") as f:
        changes = json.load(f)
    
    # Check if change_events.json exists, if not create empty events
    events_path = "data/processed/change_events.json"
    events = []
    if os.path.exists(events_path):
        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)

    # Load version information
    versions = {}
    versions_path = "data/processed/versions.json"
    if os.path.exists(versions_path):
        with open(versions_path, "r", encoding="utf-8") as f:
            versions = json.load(f)

    # 3) rebuild FAISS vector store for both snippets & events
    click.echo("Rebuilding vector DB (snippets + events)…")
    vdb = VectorDB(
        persist_directory=cfg["vector_db"]["persist_directory"],
        model_name=cfg["models"]["embedding_model"],
        versions=versions  # Pass versions to VectorDB
    )
    
    # raw chunk-level index (changes are already dictionaries from JSON)
    vdb.store_changes(changes)
    
    # event-level index (if events exist)
    if events:
        vdb.store_events(events_path)

    click.secho("✓ Vector DB rebuilt with both chunk- and event-indexes", fg="green")

@cli.command()
def serve():
    """
    Start the FastAPI QA server.
    (Make sure you've run `parse`, `detect`, and `builddb` first!)
    """
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    cli()

# # src/main.py

# import click
# import json
# import yaml

# from parsers.docx_parser import parse_docx, save_as_json
# from utils.version_mapping import map_chunks, save_version_map
# from change_detection.detector import ChangeDetector
# from utils.vector_db import VectorDB


# # Load configuration
# with open("config/config.yaml", "r", encoding="utf-8") as f:
#     cfg = yaml.safe_load(f)


# @click.group()
# def cli():
#     """3GPP Change Detection System"""
#     pass


# @cli.command()
# @click.option("--min_tokens", default=20, help="Minimum tokens for merge")
# @click.option("--max_tokens", default=500, help="Maximum tokens per chunk")
# def parse(min_tokens, max_tokens):
#     """Parse DOCX → hierarchical, token-capped chunks."""
#     for rel in ["24301-af0", "24301-hc0"]:
#         src = f"data/raw/{rel}.docx"
#         out = f"data/processed/{rel}_chunks.json"
#         click.echo(f"Parsing {src} …")
#         chunks = parse_docx(src, max_tokens=max_tokens, min_chunk_tokens=min_tokens)
#         save_as_json(chunks, out)
#         click.secho(f"✓ {len(chunks)} chunks → {out}", fg="green")


# @cli.command()
# def detect():
#     """Compute version map, detect changes, and rebuild vector DB."""
#     # 1) Load chunk lists
#     def load(path):
#         return json.load(open(path, "r", encoding="utf-8"))

#     old_chunks = load("data/processed/24301-af0_chunks.json")
#     new_chunks = load("data/processed/24301-hc0_chunks.json")

#     # 2) Build and save version map
#     click.echo("Mapping old→new chunks…")
#     version_map = map_chunks(old_chunks, new_chunks,
#                              title_weight=0.7, content_weight=0.3,
#                              threshold=0.6)
#     save_version_map(version_map, "data/processed/version_map.json")
#     click.secho(f"✓ Version map with {len(version_map)} entries → data/processed/version_map.json", fg="green")

#     # 3) Detect changes with MOVED support
#     click.echo("Detecting chunk-wise changes…")
#     detector = ChangeDetector(
#         threshold=cfg["change_detection"]["similarity_threshold"],
#         version_map=version_map
#     )
#     changes = detector.detect_changes(old_chunks, new_chunks)

#     # 4) Serialize changes
#     changes_path = "data/processed/changes.json"
#     with open(changes_path, "w", encoding="utf-8") as f:
#         json.dump([c.to_dict() for c in changes], f, indent=2, ensure_ascii=False)
#     click.secho(f"✓ Wrote {len(changes)} changes → {changes_path}", fg="green")

#     # 5) Rebuild vector DB
#     click.echo("Updating vector DB…")
#     vdb = VectorDB(
#         persist_directory=cfg["vector_db"]["persist_directory"],
#         model_name=cfg["models"]["embedding_model"]
#     )
#     vdb.store_changes(changes)
#     click.secho("✓ Vector DB updated.", fg="green")


# @cli.command()
# def serve():
#     """Start the QA API server."""
#     import uvicorn
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


# if __name__ == "__main__":
#     cli()

# src/main.py

import click
import json
import yaml

from parsers.docx_parser import parse_docx, save_as_json
from utils.version_mapping import map_chunks, save_version_map
from change_detection.detector import ChangeDetector
from utils.vector_db import VectorDB


# Load configuration
with open("config/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


@click.group()
def cli():
    """3GPP Change Detection System"""
    pass


@cli.command()
@click.option("--min_tokens", default=50, help="Minimum tokens for merge")
@click.option("--max_tokens", default=500, help="Maximum tokens per chunk")
def parse(min_tokens, max_tokens):
    """Parse DOCX → hierarchical, token-capped chunks."""
    for rel in ["24301-af0", "24301-hc0"]:
        src = f"data/raw/{rel}.docx"
        out = f"data/processed/{rel}_chunks.json"
        click.echo(f"Parsing {src} …")
        chunks = parse_docx(src, max_tokens=max_tokens, min_chunk_tokens=min_tokens)
        save_as_json(chunks, out)
        click.secho(f"✓ {len(chunks)} chunks → {out}", fg="green")


@cli.command()
def detect():
    """Compute version map, detect changes, and rebuild vector DB."""
    # 1) Load chunk lists
    def load(path):
        return json.load(open(path, "r", encoding="utf-8"))

    old_chunks = load("data/processed/24301-af0_chunks.json")
    new_chunks = load("data/processed/24301-hc0_chunks.json")

    # 2) Build and save version map
    click.echo("Mapping old→new chunks…")
    version_map = map_chunks(old_chunks, new_chunks,
                             title_weight=0.7, content_weight=0.3,
                             threshold=0.6)
    save_version_map(version_map, "data/processed/version_map.json")
    click.secho(f"✓ Version map with {len(version_map)} entries → data/processed/version_map.json", fg="green")

    # 3) Detect changes with MOVED support
    click.echo("Detecting chunk-wise changes…")
    detector = ChangeDetector(
        threshold=cfg["change_detection"]["similarity_threshold"],
        version_map=version_map
    )
    changes = detector.detect_changes(old_chunks, new_chunks)

    # 4) Serialize changes
    changes_path = "data/processed/changes.json"
    with open(changes_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in changes], f, indent=2, ensure_ascii=False)
    click.secho(f"✓ Wrote {len(changes)} changes → {changes_path}", fg="green")

    # 5) Rebuild vector DB
    click.echo("Updating vector DB…")
    vdb = VectorDB(
        persist_directory=cfg["vector_db"]["persist_directory"],
        model_name=cfg["models"]["embedding_model"]
    )
    vdb.store_changes(changes)
    click.secho("✓ Vector DB updated.", fg="green")


@cli.command()
def serve():
    """Start the QA API server."""
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    cli()

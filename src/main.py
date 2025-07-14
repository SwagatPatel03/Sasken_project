# src/main.py
import click
from parsers.docx_parser import DOCXParser
from change_detection.detector import ChangeDetector, _flatten_sections, ChangeType
from qa_bot.bot import QABot
import json

@click.group()
def cli():
    """3GPP Change Detection System"""
    pass

@cli.command()
def download():
    """Download 3GPP specifications"""
    download_3gpp_specs()

@cli.command()
def parse():
    """Parse downloaded specifications"""
    click.echo("Parsing specifications...")
    # Implementation
    # src/main.py, inside @cli.command() def parse():
    parser = DOCXParser()
    for key in ["24301-af0", "24301-hc0"]:
        file_path = f"data/raw/{key}.docx"
        doc_struct = parser.parse(file_path)
        # serialize to JSON for quick inspection
        with open(f"data/processed/{key}.json", "w") as f:
            f.write(doc_struct.to_json())
        click.echo(f"✓ Parsed {key}")

@cli.command()
@click.option('--debug', is_flag=True, help="Show diff summary")
def detect(debug):
    parser = DOCXParser()
    old_doc = parser.parse("data/raw/24301-af0.docx")
    new_doc = parser.parse("data/raw/24301-hc0.docx")

    detector = ChangeDetector()
    changes = detector.detect_changes(old_doc, new_doc)

    if debug:
        # Flatten both trees
        old_map = _flatten_sections(old_doc)
        new_map = _flatten_sections(new_doc)

        old_ids = set(old_map)
        new_ids = set(new_map)

        added_ids   = new_ids   - old_ids
        removed_ids = old_ids   - new_ids
        shared_ids  = old_ids & new_ids

        # Of the shared, which really changed?
        modified_ids = {
            c.section_id
            for c in changes
            if c.change_type == ChangeType.MODIFIED
        }

        click.echo(f"Total sections old:    {len(old_ids)}")
        click.echo(f"Total sections new:    {len(new_ids)}")
        click.echo(f"→ Added sections:     {len(added_ids)}")
        click.echo(f"→ Removed sections:   {len(removed_ids)}")
        click.echo(f"→ Potentially modified: {len(shared_ids)}")
        click.echo(f"→ Actually modified:  {len(modified_ids)}\n")

        # Show examples
        click.echo("Sample added IDs:     " + ", ".join(list(added_ids)[:5]))
        click.echo("Sample removed IDs:   " + ", ".join(list(removed_ids)[:5]))
        click.echo("Sample modified IDs:  " + ", ".join(list(modified_ids)[:5]))

    """Detect changes between Rel‑15 vs. Rel‑16"""
    

    detector = ChangeDetector()
    changes = detector.detect_changes(old_doc, new_doc)

    # Serialize changes to JSON
    with open("data/processed/changes.json", "w") as f:
        json_list = [c.to_dict() for c in changes]
        f.write(json.dumps(json_list, indent=2))

    click.echo(f"✓ Detected {len(changes)} changes")

@cli.command()
def serve():
    """Start QA bot server"""
    click.echo("Starting QA bot server...")
    # Implementation

if __name__ == "__main__":
    cli()
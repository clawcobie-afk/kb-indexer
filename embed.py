import os
import sys
import click
import openai
from qdrant_client import QdrantClient
from kb.embedder import index_channel


def make_progress_callback(channel_slug):
    def on_progress(event):
        etype = event["type"]
        if etype == "start":
            click.echo(f"@{channel_slug} — {event['total']} souborů chunks.json\n")
        elif etype == "file_done":
            click.echo(f"[{event['index']}/{event['total']}] {event['video_id']} — {event['chunks']} chunků... ✓")
        elif etype == "summary":
            click.echo(f"\nHotovo: {event['files']} videí, {event['total_chunks']} chunků indexováno.")
    return on_progress


@click.group()
def cli():
    """kb-indexer — index and inspect the knowledge base."""


@cli.command()
@click.option("--data-dir", required=True, help="Root data directory (e.g. /home/klach/project/data)")
@click.option("--channel", required=True, help="Channel slug without @ (e.g. SteveMagness)")
@click.option("--collection", default="kb", show_default=True, help="Qdrant collection name")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
def run(data_dir, channel, collection, qdrant_url):
    """Index channel chunks into Qdrant using OpenAI embeddings."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise click.ClickException("OPENAI_API_KEY environment variable is required")

    # Normalize channel slug (strip leading @)
    channel_slug = channel.lstrip("@")

    index_channel(
        channel_slug=channel_slug,
        data_dir=data_dir,
        qdrant_url=qdrant_url,
        collection=collection,
        openai_api_key=openai_api_key,
        on_progress=make_progress_callback(channel_slug),
    )


@cli.command()
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
@click.option("--openai-api-key", default=lambda: os.environ.get("OPENAI_API_KEY", ""), help="OpenAI API key (default: OPENAI_API_KEY env)")
def check(qdrant_url, openai_api_key):
    """Check connectivity to OpenAI and Qdrant."""
    failures = 0

    # 1. OPENAI_API_KEY is set and non-empty
    if openai_api_key:
        print("OK  OPENAI_API_KEY is set")
    else:
        print("FAIL OPENAI_API_KEY is not set or empty")
        failures += 1

    # 2. OpenAI API key is valid
    try:
        openai.OpenAI(api_key=openai_api_key).models.list()
        print("OK  OpenAI API key is valid")
    except Exception as e:
        print(f"FAIL OpenAI API key is invalid: {e}")
        failures += 1

    # 3. Qdrant is reachable
    try:
        QdrantClient(url=qdrant_url).get_collections()
        print("OK  Qdrant is reachable")
    except Exception as e:
        print(f"FAIL Qdrant is not reachable: {e}")
        failures += 1

    print()
    if failures == 0:
        print("All checks passed.")
    else:
        print(f"{failures} check(s) failed.")
        sys.exit(1)


@cli.command("setup")
@click.option("--openai-api-key", prompt="OpenAI API key", hide_input=True, help="OpenAI API key")
@click.option("--qdrant-url", default="http://localhost:6333", prompt="Qdrant URL", show_default=True)
def setup_cmd(openai_api_key, qdrant_url):
    """Interactive wizard to configure kb-indexer credentials."""
    # 1. Validate OpenAI key
    try:
        openai.OpenAI(api_key=openai_api_key).models.list()
    except Exception as e:
        click.echo(f"Error: OpenAI API key is invalid: {e}")
        sys.exit(1)

    # 2. Validate Qdrant
    try:
        QdrantClient(url=qdrant_url).get_collections()
    except Exception as e:
        click.echo(f"Error: Qdrant is not reachable: {e}")
        sys.exit(1)

    # 3. Write to ~/.config/knowledge-vault/.env (merge with existing content)
    config_dir = os.path.expanduser("~/.config/knowledge-vault")
    os.makedirs(config_dir, exist_ok=True)
    env_path = os.path.join(config_dir, ".env")

    # Read existing key=value lines
    existing = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.rstrip("\n")
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    existing[k] = v

    # Update keys
    existing["OPENAI_API_KEY"] = openai_api_key
    existing["QDRANT_URL"] = qdrant_url

    # Write back
    with open(env_path, "w") as f:
        for k, v in existing.items():
            f.write(f"{k}={v}\n")

    click.echo("Setup complete. Config written to ~/.config/knowledge-vault/.env")


if __name__ == "__main__":
    cli()

import os
import click
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


@click.command()
@click.option("--data-dir", required=True, help="Root data directory (e.g. /home/klach/project/data)")
@click.option("--channel", required=True, help="Channel slug without @ (e.g. SteveMagness)")
@click.option("--collection", default="kb", show_default=True, help="Qdrant collection name")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True, help="Qdrant URL")
def cli(data_dir, channel, collection, qdrant_url):
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


if __name__ == "__main__":
    cli()

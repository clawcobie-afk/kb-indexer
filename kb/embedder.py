from pathlib import Path
import json
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams


def get_collection(client: QdrantClient, name: str) -> None:
    """Create Qdrant collection if it doesn't exist (1536 dim, cosine)."""
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    except (UnexpectedResponse, ValueError) as e:
        already_exists = (
            isinstance(e, ValueError) and "already exists" in str(e)
        ) or (
            isinstance(e, UnexpectedResponse) and e.status_code == 409
        )
        if not already_exists:
            raise


def get_embedding(text: str, openai_client: OpenAI) -> list[float]:
    """Return embedding vector for text using text-embedding-3-small."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def index_chunks(chunks_path, qdrant_client: QdrantClient, openai_client: OpenAI, collection: str) -> int:
    """Index one chunks.json file into Qdrant. Returns number of chunks indexed."""
    chunks_path = Path(chunks_path)
    try:
        chunks = json.loads(chunks_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

    points = []
    for chunk in chunks:
        try:
            embedding = get_embedding(chunk["text"], openai_client)
        except Exception:
            continue
        payload = {
            "video_id": chunk["video_id"],
            "title": chunk["title"],
            "channel_name": chunk["channel_name"],
            "channel_slug": chunk["channel_slug"],
            "upload_date": chunk["upload_date"],
            "timestamp_url": chunk["timestamp_url"],
            "transcript_source": chunk["transcript_source"],
            "source_type": chunk["source_type"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "text": chunk["text"],
        }
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['video_id']}_{chunk['chunk_index']}")),
            vector=embedding,
            payload=payload,
        )
        points.append(point)

    if points:
        qdrant_client.upsert(
            collection_name=collection,
            points=points,
        )

    return len(points)


def index_channel(
    channel_slug: str,
    data_dir: str,
    qdrant_url: str,
    collection: str,
    openai_api_key: str,
    on_progress=None,
) -> None:
    """Index all videos for a channel into Qdrant."""
    def emit(event):
        if on_progress is not None:
            on_progress(event)

    data_dir = Path(data_dir)
    channel_dir = data_dir / "channels" / channel_slug
    chunks_files = sorted((channel_dir / "videos").glob("*/chunks.json")) if (channel_dir / "videos").exists() else []

    qdrant_client = QdrantClient(url=qdrant_url)
    openai_client = OpenAI(api_key=openai_api_key)

    get_collection(qdrant_client, collection)

    emit({"type": "start", "total": len(chunks_files), "channel": channel_slug})

    total_chunks = 0
    for idx, chunks_path in enumerate(chunks_files, start=1):
        video_id = chunks_path.parent.name
        count = index_chunks(chunks_path, qdrant_client, openai_client, collection)
        total_chunks += count
        emit({"type": "file_done", "index": idx, "total": len(chunks_files), "video_id": video_id, "chunks": count})

    emit({"type": "summary", "files": len(chunks_files), "total_chunks": total_chunks})

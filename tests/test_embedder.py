import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from qdrant_client.http.exceptions import UnexpectedResponse
from kb.embedder import get_collection, get_embedding, index_chunks, index_channel


# ── helpers ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "video_id": "abc123",
        "title": "Test Video",
        "channel_name": "Test Channel",
        "channel_slug": "testchannel",
        "upload_date": "20240101",
        "timestamp_url": "https://youtube.com/watch?v=abc123&t=0s",
        "transcript_source": "caption",
        "source_type": "youtube",
        "chunk_index": 0,
        "total_chunks": 2,
        "text": "Hello world from chunk zero",
    },
    {
        "video_id": "abc123",
        "title": "Test Video",
        "channel_name": "Test Channel",
        "channel_slug": "testchannel",
        "upload_date": "20240101",
        "timestamp_url": "https://youtube.com/watch?v=abc123&t=10s",
        "transcript_source": "caption",
        "source_type": "youtube",
        "chunk_index": 1,
        "total_chunks": 2,
        "text": "Second chunk of text here",
    },
]

FAKE_EMBEDDING = [0.1] * 1536


def make_openai_mock(embedding=None):
    if embedding is None:
        embedding = FAKE_EMBEDDING
    mock = MagicMock()
    mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=embedding)]
    )
    return mock


def make_qdrant_mock():
    mock = MagicMock()
    return mock


# ── get_collection ────────────────────────────────────────────────────────────

class TestGetCollection:
    def test_creates_collection_when_missing(self):
        client = make_qdrant_mock()
        get_collection(client, "kb")
        client.create_collection.assert_called_once()
        args = client.create_collection.call_args
        assert args.kwargs["collection_name"] == "kb" or args.args[0] == "kb"

    def test_skips_creation_when_exists_value_error(self):
        client = make_qdrant_mock()
        client.create_collection.side_effect = ValueError("Collection kb already exists")
        # Should not raise
        get_collection(client, "kb")
        client.create_collection.assert_called_once()

    def test_skips_creation_when_exists_unexpected_response(self):
        client = make_qdrant_mock()
        mock_headers = MagicMock()
        client.create_collection.side_effect = UnexpectedResponse(
            status_code=409,
            reason_phrase="Conflict",
            content=b"already exists",
            headers=mock_headers,
        )
        # Should not raise
        get_collection(client, "kb")
        client.create_collection.assert_called_once()

    def test_reraises_unexpected_non_conflict_error(self):
        client = make_qdrant_mock()
        mock_headers = MagicMock()
        client.create_collection.side_effect = UnexpectedResponse(
            status_code=500,
            reason_phrase="Internal Server Error",
            content=b"server error",
            headers=mock_headers,
        )
        with pytest.raises(UnexpectedResponse):
            get_collection(client, "kb")

    def test_creates_with_1536_dimensions(self):
        client = make_qdrant_mock()
        get_collection(client, "kb")
        call_kwargs = client.create_collection.call_args.kwargs
        # VectorParams should have size=1536
        vector_params = call_kwargs.get("vectors_config")
        assert vector_params is not None
        assert vector_params.size == 1536

    def test_creates_with_cosine_distance(self):
        from qdrant_client.models import Distance
        client = make_qdrant_mock()
        get_collection(client, "kb")
        call_kwargs = client.create_collection.call_args.kwargs
        vector_params = call_kwargs.get("vectors_config")
        assert vector_params.distance == Distance.COSINE


# ── get_embedding ─────────────────────────────────────────────────────────────

class TestGetEmbedding:
    def test_returns_list_of_floats(self):
        openai_client = make_openai_mock()
        result = get_embedding("hello world", openai_client)
        assert isinstance(result, list)
        assert len(result) == 1536

    def test_calls_correct_model(self):
        openai_client = make_openai_mock()
        get_embedding("hello world", openai_client)
        openai_client.embeddings.create.assert_called_once()
        call_kwargs = openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["model"] == "text-embedding-3-small"

    def test_passes_input_text(self):
        openai_client = make_openai_mock()
        get_embedding("test sentence", openai_client)
        call_kwargs = openai_client.embeddings.create.call_args.kwargs
        assert call_kwargs["input"] == "test sentence"


# ── index_chunks ──────────────────────────────────────────────────────────────

class TestIndexChunks:
    def test_upserts_correct_number_of_points(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant, openai_client, "kb")

        assert qdrant.upsert.call_count == 1
        upsert_call = qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call.args[1]
        assert len(points) == 2

    def test_payload_contains_required_fields(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant, openai_client, "kb")

        upsert_call = qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call.args[1]
        payload = points[0].payload

        required = {
            "video_id", "title", "channel_name", "channel_slug",
            "upload_date", "timestamp_url", "transcript_source",
            "source_type", "chunk_index", "total_chunks", "text",
        }
        assert required.issubset(payload.keys()), f"Missing: {required - payload.keys()}"

    def test_payload_values_match_chunk(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant, openai_client, "kb")

        upsert_call = qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call.args[1]
        payload = points[0].payload

        assert payload["video_id"] == "abc123"
        assert payload["source_type"] == "youtube"
        assert payload["chunk_index"] == 0

    def test_uses_collection_name(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant, openai_client, "my_collection")

        upsert_call = qdrant.upsert.call_args
        coll = upsert_call.kwargs.get("collection_name") or upsert_call.args[0]
        assert coll == "my_collection"

    def test_returns_chunk_count(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        count = index_chunks(chunks_path, qdrant, openai_client, "kb")
        assert count == 2

    def test_returns_zero_on_file_not_found(self, tmp_path):
        missing_path = tmp_path / "nonexistent.json"
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        count = index_chunks(missing_path, qdrant, openai_client, "kb")
        assert count == 0
        qdrant.upsert.assert_not_called()

    def test_returns_zero_on_invalid_json(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text("not valid json {{{")
        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        count = index_chunks(chunks_path, qdrant, openai_client, "kb")
        assert count == 0
        qdrant.upsert.assert_not_called()

    def test_skips_chunk_on_embedding_error(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        # Make every embedding call fail
        openai_client.embeddings.create.side_effect = Exception("API error")

        count = index_chunks(chunks_path, qdrant, openai_client, "kb")
        assert count == 0
        qdrant.upsert.assert_not_called()

    def test_skips_only_failing_chunk_on_embedding_error(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        # Fail on first call only, succeed on second
        openai_client.embeddings.create.side_effect = [
            Exception("API error"),
            MagicMock(data=[MagicMock(embedding=FAKE_EMBEDDING)]),
        ]

        count = index_chunks(chunks_path, qdrant, openai_client, "kb")
        assert count == 1

    def test_uses_deterministic_ids(self, tmp_path):
        import uuid as _uuid
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant, openai_client, "kb")

        upsert_call = qdrant.upsert.call_args
        points = upsert_call.kwargs.get("points") or upsert_call.args[1]

        expected_id_0 = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "abc123_0"))
        expected_id_1 = str(_uuid.uuid5(_uuid.NAMESPACE_URL, "abc123_1"))
        assert points[0].id == expected_id_0
        assert points[1].id == expected_id_1

    def test_deterministic_ids_are_stable_across_calls(self, tmp_path):
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(SAMPLE_CHUNKS))

        qdrant1 = make_qdrant_mock()
        qdrant2 = make_qdrant_mock()
        openai_client = make_openai_mock()

        index_chunks(chunks_path, qdrant1, openai_client, "kb")
        index_chunks(chunks_path, qdrant2, openai_client, "kb")

        points1 = qdrant1.upsert.call_args.kwargs.get("points") or qdrant1.upsert.call_args.args[1]
        points2 = qdrant2.upsert.call_args.kwargs.get("points") or qdrant2.upsert.call_args.args[1]

        assert points1[0].id == points2[0].id
        assert points1[1].id == points2[1].id


# ── index_channel ─────────────────────────────────────────────────────────────

class TestIndexChannel:
    def _make_data_dir(self, tmp_path, channel_slug, video_ids):
        """Create fake data dir with chunks.json files for each video."""
        for vid in video_ids:
            vid_dir = tmp_path / "channels" / channel_slug / "videos" / vid
            vid_dir.mkdir(parents=True)
            chunks = [
                {**SAMPLE_CHUNKS[0], "video_id": vid, "chunk_index": 0, "total_chunks": 1}
            ]
            (vid_dir / "chunks.json").write_text(json.dumps(chunks))
        return tmp_path

    def test_iterates_all_video_chunks_files(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path, "testchannel", ["vid1", "vid2", "vid3"])

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        with patch("kb.embedder.QdrantClient", return_value=qdrant), \
             patch("kb.embedder.OpenAI", return_value=openai_client):
            index_channel(
                channel_slug="testchannel",
                data_dir=str(data_dir),
                qdrant_url="http://localhost:6333",
                collection="kb",
                openai_api_key="sk-test",
            )

        # upsert called once per video (3 videos, each 1 chunk = 1 batch each)
        assert qdrant.upsert.call_count == 3

    def test_calls_get_collection(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path, "testchannel", ["vid1"])

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        with patch("kb.embedder.QdrantClient", return_value=qdrant), \
             patch("kb.embedder.OpenAI", return_value=openai_client):
            index_channel(
                channel_slug="testchannel",
                data_dir=str(data_dir),
                qdrant_url="http://localhost:6333",
                collection="kb",
                openai_api_key="sk-test",
            )

        qdrant.create_collection.assert_called_once()

    def test_on_progress_events(self, tmp_path):
        data_dir = self._make_data_dir(tmp_path, "testchannel", ["vid1", "vid2"])

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()
        events = []

        with patch("kb.embedder.QdrantClient", return_value=qdrant), \
             patch("kb.embedder.OpenAI", return_value=openai_client):
            index_channel(
                channel_slug="testchannel",
                data_dir=str(data_dir),
                qdrant_url="http://localhost:6333",
                collection="kb",
                openai_api_key="sk-test",
                on_progress=events.append,
            )

        etypes = [e["type"] for e in events]
        assert "start" in etypes
        assert "file_done" in etypes
        assert "summary" in etypes

    def test_no_chunks_files_is_graceful(self, tmp_path):
        # Channel dir exists but has no videos
        channel_dir = tmp_path / "channels" / "emptychannel"
        channel_dir.mkdir(parents=True)

        qdrant = make_qdrant_mock()
        openai_client = make_openai_mock()

        with patch("kb.embedder.QdrantClient", return_value=qdrant), \
             patch("kb.embedder.OpenAI", return_value=openai_client):
            index_channel(
                channel_slug="emptychannel",
                data_dir=str(tmp_path),
                qdrant_url="http://localhost:6333",
                collection="kb",
                openai_api_key="sk-test",
            )

        qdrant.upsert.assert_not_called()

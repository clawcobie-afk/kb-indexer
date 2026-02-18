# CLAUDE.md — kb-indexer (embedder)

## Spuštění testů
```bash
source venv/bin/activate
pytest tests/ -v
```

## CLI
```bash
OPENAI_API_KEY=... python embed.py \
  --data-dir /home/klach/project/data \
  --channel lexfridman \
  [--collection kb] \
  [--qdrant-url http://localhost:6333]
```

## Architektura
- `embed.py` — Click CLI, progress callback
- `kb/embedder.py` — čte chunks.json, volá OpenAI, upsertuje do Qdrant

## Klíčové funkce v `kb/embedder.py`
- `get_collection(client, name)` — vytvoří kolekci pokud neexistuje (1536 dim, cosine)
- `get_embedding(text, openai_client)` — model `text-embedding-3-small`
- `index_chunks(chunks_path, qdrant_client, openai_client, collection)` — jeden soubor
- `index_channel(channel_slug, data_dir, qdrant_url, collection, openai_api_key, on_progress)` — celý kanál

## Env proměnné
- `OPENAI_API_KEY` — povinné

## Konvence
- TDD: testy jsou mockované (OpenAI + Qdrant klienti)
- Qdrant payload obsahuje všechna metadata z chunks.json
- Vstupní data z kb-ingest (`data/channels/<slug>/videos/<id>/chunks.json`)

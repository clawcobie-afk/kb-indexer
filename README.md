# kb-indexer

Vektorizuje chunky z kb-ingest do Qdrant pomocí OpenAI embeddings (`text-embedding-3-small`).

## Instalace

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Vyžaduje běžící Qdrant instanci:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Použití

```bash
export OPENAI_API_KEY=sk-...

python embed.py \
  --data-dir /home/klach/project/data \
  --channel lexfridman
```

### Flagy

| Flag | Výchozí | Popis |
|------|---------|-------|
| `--data-dir` | — | Cesta k datové složce z kb-ingest (povinné) |
| `--channel` | — | Slug kanálu bez @ (povinné) |
| `--collection` | `kb` | Název Qdrant kolekce |
| `--qdrant-url` | `http://localhost:6333` | URL Qdrant serveru |

## Env proměnné

| Proměnná | Popis |
|----------|-------|
| `OPENAI_API_KEY` | OpenAI API klíč (povinné) |

## Qdrant kolekce

- **Dimenze:** 1536 (text-embedding-3-small)
- **Metrika:** Cosine
- **Payload:** video_id, title, channel_slug, upload_date, timestamp_url, transcript_source, source_type, chunk_index, total_chunks, text

## Testy

```bash
pytest tests/ -v
```

## Pipeline

```
chunks.json (z kb-ingest) → OpenAI embeddings → Qdrant upsert
```

Předchází: [kb-ingest](../project) | Navazuje: [kb-search](../kb-search)

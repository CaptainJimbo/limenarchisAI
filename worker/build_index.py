"""Build the retrieval index artifacts: dense vectors + BM25 term stats.

Two layers, same frozen format (format_version 1):
- static-index.json  — port cards, glossary, routes (worker/static_corpus.json)
- live-index.json    — rolling 48h event digests (data/events.json)

Embedding is INCREMENTAL: vectors are cached by content hash and only new
chunks are embedded (bge-small-en-v1.5, CPU, seconds per run). Documents are
embedded WITHOUT the bge query prefix — the prefix belongs on queries only,
in the browser.

The BM25 tokenizer here must match the JS implementation byte-for-byte:
lowercase, extract [a-z0-9]+ runs, keep tokens of length >= 2.
"""

import argparse
import base64
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
FORMAT_VERSION = 1
TOKEN_RE = re.compile(r"[a-z0-9]+")

CACHE_PATH = Path("data/emb_cache.json")


def tokenize(text):
    """KEEP IN SYNC with web BM25: lowercase, [a-z0-9]+ runs, len >= 2."""
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) >= 2]


def chunk_hash(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def quantize_int8(vector):
    """L2-normalized float vector -> int8 in [-127, 127]. Dot products keep
    their ranking; absolute cosine needs /127^2, which retrieval ignores."""
    import numpy as np
    q = np.clip(np.round(np.asarray(vector) * 127), -127, 127)
    return q.astype(np.int8)


def embed_missing(texts, cache):
    """Embed only texts whose hash is not cached. Returns nothing; fills cache
    with base64-encoded int8 vectors."""
    missing = [t for t in texts if chunk_hash(t) not in cache]
    if not missing:
        return 0
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(missing, normalize_embeddings=True,
                           show_progress_bar=False)
    for text, vec in zip(missing, vectors):
        q = quantize_int8(vec)
        cache[chunk_hash(text)] = base64.b64encode(q.tobytes()).decode()
    return len(missing)


def bm25_stats(token_lists):
    df = Counter()
    docs = []
    for tokens in token_lists:
        tf = Counter(tokens)
        df.update(tf.keys())
        docs.append({"len": len(tokens), "tf": dict(tf)})
    n = len(token_lists)
    avgdl = (sum(d["len"] for d in docs) / n) if n else 0.0
    return {"k1": 1.2, "b": 0.75, "N": n, "avgdl": round(avgdl, 2),
            "df": dict(df), "docs": docs}


def build_layer(layer, chunks, cache):
    """chunks: [{id, text, ...meta}] with `text` being the embedded string."""
    texts = [c["text"] for c in chunks]
    fresh = embed_missing(texts, cache)
    vectors = b"".join(
        base64.b64decode(cache[chunk_hash(t)]) for t in texts)
    artifact = {
        "format_version": FORMAT_VERSION,
        "layer": layer,
        "model": MODEL_NAME,
        "dim": DIM,
        "quantization": "int8",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks": chunks,
        "vectors_b64": base64.b64encode(vectors).decode(),
        "bm25": bm25_stats([tokenize(t) for t in texts]),
    }
    return artifact, fresh


def static_chunks():
    corpus = json.loads(
        (Path(__file__).parent / "static_corpus.json").read_text())
    return [{
        "id": c["id"],
        "title": c["title"],
        "text": f"{c['title']}. {c['text']}",
        "source": "static",
    } for c in corpus["chunks"]]


def live_chunks(events_path, status_path):
    records = json.loads(Path(events_path).read_text())
    status_path = Path(status_path)
    if status_path.exists():
        records = records + json.loads(status_path.read_text())
    return [{
        "id": e["id"],
        "title": e["text"][:60],
        "text": e["text"],
        "source": "live",
        "event_type": e["type"],
        "mmsi": e["mmsi"],
        "time_utc": e["time_utc"],
    } for e in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/events.json"))
    parser.add_argument("--status", type=Path, default=Path("data/status.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    args = parser.parse_args()

    cache = (json.loads(args.cache.read_text())
             if args.cache.exists() else {})

    layers = {"static": static_chunks()}
    if args.events.exists():
        layers["live"] = live_chunks(args.events, args.status)
    else:
        print(f"note: {args.events} missing, building static only")

    total_fresh = 0
    for layer, chunks in layers.items():
        artifact, fresh = build_layer(layer, chunks, cache)
        total_fresh += fresh
        out = args.out_dir / f"{layer}-index.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact))
        kb = out.stat().st_size / 1024
        print(f"{out}: {len(chunks)} chunks, {fresh} newly embedded, "
              f"{kb:.0f} KB")

    # Trim cache to live corpus so it can't grow unboundedly.
    keep = {chunk_hash(c["text"]) for chunks in layers.values()
            for c in chunks}
    cache = {h: v for h, v in cache.items() if h in keep}
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache))
    print(f"embedded {total_fresh} new chunks this run; "
          f"cache holds {len(cache)}")


if __name__ == "__main__":
    main()

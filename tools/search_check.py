"""Retrieval sanity check — simulates the browser search pipeline in Python.

Loads the built artifacts and runs: bge query embedding (WITH query prefix),
dense int8 dot-product + BM25 (Lucene idf) per layer, RRF k=60 fusion across
the four lists. Prints top hits with leg provenance, the way the web UI will.
"""

import base64
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))
from build_index import MODEL_NAME, tokenize  # noqa: E402

import numpy as np  # noqa: E402

QUERY_PREFIX = ("Represent this sentence for searching relevant passages: ")
RRF_K = 60


def load_layer(path):
    art = json.loads(Path(path).read_text())
    raw = base64.b64decode(art["vectors_b64"])
    vectors = np.frombuffer(raw, dtype=np.int8).reshape(-1, art["dim"])
    return art, vectors


def dense_scores(query_vec, vectors):
    return vectors.astype(np.float32) @ query_vec


def bm25_scores(query, bm25):
    k1, b, n, avgdl = bm25["k1"], bm25["b"], bm25["N"], bm25["avgdl"]
    scores = np.zeros(n)
    for term in set(tokenize(query)):
        df = bm25["df"].get(term)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # Lucene form, >= 0
        for i, doc in enumerate(bm25["docs"]):
            tf = doc["tf"].get(term, 0)
            if tf:
                denom = tf + k1 * (1 - b + b * doc["len"] / avgdl)
                scores[i] += idf * tf * (k1 + 1) / denom
    return scores


def ranked(scores):
    return [i for i in np.argsort(-scores) if scores[i] > 0]


def search(query, layers, top=5):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]

    lists = {}
    for name, (art, vectors) in layers.items():
        lists[f"dense-{name}"] = (art, ranked(dense_scores(qvec, vectors)))
        lists[f"bm25-{name}"] = (art, ranked(bm25_scores(query, art["bm25"])))

    fused = {}
    for leg, (art, order) in lists.items():
        for rank, idx in enumerate(order[:50]):
            chunk = art["chunks"][idx]
            entry = fused.setdefault(
                chunk["id"], {"chunk": chunk, "score": 0.0, "legs": {}})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
            entry["legs"][leg] = rank + 1

    results = sorted(fused.values(), key=lambda e: -e["score"])[:top]
    print(f"\nQ: {query}")
    for r in results:
        legs = ", ".join(f"{leg}#{rank}" for leg, rank in
                         sorted(r["legs"].items()))
        print(f"  {r['score']:.4f}  [{r['chunk']['id']}]  ({legs})")
        print(f"          {r['chunk']['text'][:100]}")
    return results


def main():
    layers = {
        "static": load_layer("data/static-index.json"),
        "live": load_layer("data/live-index.json"),
    }
    queries = sys.argv[1:] or [
        "Did any ferry leave Salamina tonight?",
        "Which island has no cars?",
        "How many ships are waiting at the Piraeus anchorage?",
        "What is an MMSI number?",
        "Did a Blue Star ferry arrive?",
    ]
    for q in queries:
        search(q, layers)


if __name__ == "__main__":
    main()

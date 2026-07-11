// Browser retrieval pipeline: dense (bge-small int8) + BM25, RRF fusion
// across 4 lists, best-hit grouping per source object.
// KEEP IN SYNC with worker/build_index.py (tokenizer) and
// tools/search_check.py (scoring reference).

const QUERY_PREFIX =
  "Represent this sentence for searching relevant passages: ";
const RRF_K = 60;
const LEG_DEPTH = 50;

export function tokenize(text) {
  // Parity with worker: lowercase, [a-z0-9]+ runs, length >= 2.
  return (text.toLowerCase().match(/[a-z0-9]+/g) || [])
    .filter((t) => t.length >= 2);
}

export async function loadLayer(url) {
  const artifact = await (await fetch(url)).json();
  const raw = atob(artifact.vectors_b64);
  const vectors = new Int8Array(raw.length);
  for (let i = 0; i < raw.length; i++) vectors[i] = (raw.charCodeAt(i) << 24) >> 24;
  return { ...artifact, vectors };
}

let extractorPromise = null;

export function loadEmbedder(onProgress) {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline } = await import(
        "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1");
      return pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", {
        dtype: "q8",
        progress_callback: onProgress,
      });
    })();
  }
  return extractorPromise;
}

export async function embedQuery(query) {
  const extractor = await loadEmbedder();
  // bge: CLS pooling + L2 normalize; prefix on QUERIES ONLY.
  const output = await extractor(QUERY_PREFIX + query, {
    pooling: "cls",
    normalize: true,
  });
  return output.data; // Float32Array(384)
}

function denseScores(queryVec, layer) {
  const { vectors, dim, chunks } = layer;
  const scores = new Float32Array(chunks.length);
  for (let d = 0; d < chunks.length; d++) {
    let dot = 0;
    const base = d * dim;
    for (let i = 0; i < dim; i++) dot += vectors[base + i] * queryVec[i];
    scores[d] = dot; // ranking-safe (missing /127 is a constant factor)
  }
  return scores;
}

function bm25Scores(query, layer) {
  const { k1, b, N, avgdl, df, docs } = layer.bm25;
  const scores = new Float32Array(docs.length);
  for (const term of new Set(tokenize(query))) {
    const termDf = df[term];
    if (!termDf) continue;
    const idf = Math.log(1 + (N - termDf + 0.5) / (termDf + 0.5)); // Lucene
    for (let i = 0; i < docs.length; i++) {
      const tf = docs[i].tf[term];
      if (!tf) continue;
      const denom = tf + k1 * (1 - b + (b * docs[i].len) / avgdl);
      scores[i] += (idf * tf * (k1 + 1)) / denom;
    }
  }
  return scores;
}

function rankedIndices(scores) {
  return [...scores.keys()]
    .filter((i) => scores[i] > 0)
    .sort((a, b) => scores[b] - scores[a]);
}

// Best-hit grouping: chunks about the same object (vessel via MMSI, or a
// static card) collapse into one result scored by their BEST chunk —
// max, never sum, so prolific objects don't win by volume.
function groupKey(chunk) {
  return chunk.mmsi != null ? `mmsi-${chunk.mmsi}` : chunk.id;
}

export async function search(query, layers, top = 8) {
  const queryVec = await embedQuery(query);
  const legs = {};
  for (const [name, layer] of Object.entries(layers)) {
    legs[`dense-${name}`] = { layer, order: rankedIndices(denseScores(queryVec, layer)) };
    legs[`bm25-${name}`] = { layer, order: rankedIndices(bm25Scores(query, layer)) };
  }

  const perChunk = new Map();
  for (const [legName, { layer, order }] of Object.entries(legs)) {
    order.slice(0, LEG_DEPTH).forEach((idx, rank) => {
      const chunk = layer.chunks[idx];
      let entry = perChunk.get(chunk.id);
      if (!entry) {
        entry = { chunk, score: 0, legs: {} };
        perChunk.set(chunk.id, entry);
      }
      entry.score += 1 / (RRF_K + rank + 1);
      entry.legs[legName] = rank + 1;
    });
  }

  const groups = new Map();
  for (const entry of perChunk.values()) {
    const key = groupKey(entry.chunk);
    let group = groups.get(key);
    if (!group) {
      group = { key, best: entry, members: [] };
      groups.set(key, group);
    }
    group.members.push(entry);
    if (entry.score > group.best.score) group.best = entry;
  }

  return [...groups.values()]
    .sort((a, b) => b.best.score - a.best.score)
    .slice(0, top)
    .map((g) => ({
      score: g.best.score,
      chunk: g.best.chunk,
      legs: g.best.legs,
      siblings: g.members
        .filter((m) => m !== g.best)
        .sort((a, b) => b.score - a.score)
        .slice(0, 2)
        .map((m) => m.chunk),
    }));
}

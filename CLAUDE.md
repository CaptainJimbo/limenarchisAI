# LimenarchisAI — Project Spec

**The AI Harbormaster of Piraeus** — hybrid-search RAG over live AIS-derived
maritime events, answered by an in-browser LLM with citations. Private while
under construction; goes public at launch (Pages + free unlimited Actions
require it anyway).

Read `CLAUDE.local.md` first if it exists (local-only context; gitignored).

## Scope (locked)

**Piraeus + full Saronic/Argosaronic** bounding box (37.15–38.1°N,
22.9–24.1°E — tuned at step 1 to include Hydra/Spetses, whose ferry routes
have verified schedules, and the Corinth Canal east entrance; enough vessels
that the chatbot demonstrably out-attends a human eyeballing the map). NOT the whole Aegean: a demo needs a stage. Piraeus gives
ferries-with-schedules (delays = computable events), the tanker anchorage
(congestion stories), cruise + container traffic, and the best volunteer AIS
receiver density in Greece. Everything else is v2 parking lot.

## Architecture (two halves)

### 1. Ingestion — the worker (offline, GitHub Actions cron)
- Connect to **aisstream.io** websocket (API key in Actions **secrets**, never
  in the page/repo), subscribe to the bounding box, drink for ~2–3 min, keep
  latest position per MMSI → snapshot (~200–400 vessels).
- **Diff vs previous snapshot → event digests** (the corpus is GENERATED, not
  downloaded): arrivals, departures, anchorings, speed anomalies, "N vessels
  waiting at anchorage (Δ vs yesterday)", ferry-delay events (actual departure
  vs published schedule — schedule source verified at step 1).
- Static knowledge layer (embedded once): port cards (World Port Index, public
  domain), vessel-type glossary, route/terminal descriptions, NAVTEX
  navigational warnings (refreshed when new).
- **Incremental embedding:** embed ONLY new chunks each run
  (sentence-transformers, CPU, seconds). Live layer = rolling 48h window;
  expired digests drop out. Never re-embed the world (only on embedding-model
  change → one-time rebuild).
- Artifacts out (static JSON): `static-index.*`, `live-index.*` (vectors
  fp16/int8 + BM25 term stats per layer), `snapshot.json` (map), `events.json`.
  Same worker→artifacts→Pages pattern as o-ilios.

### 2. RAG — the browser (online, GitHub Pages, no server)
- **Query embedding:** transformers.js ONNX — **bge-small-en-v1.5** (384-dim,
  ~25MB quantized). MUST match worker model exactly: same tokenizer, CLS
  pooling, L2 normalize, and bge's **query-side prefix** ("Represent this
  sentence for searching relevant passages: ") on queries only, never docs.
- **BM25 in JS:** k1=1.2, b=0.75, Lucene negative-IDF floor. Term stats
  precomputed per layer by the worker.
- **Fusion: RRF k=60 across 4 lists** (dense-static, dense-live, bm25-static,
  bm25-live) + **best-hit grouping (max, never sum)** per source object —
  anti volume-bias.
- **Generation: WebLLM** (WebGPU) — default **Llama-3.2-3B-Instruct**; offer
  Qwen2.5-1.5B as the light option. Persona: the Harbormaster — dry, precise,
  seen-everything. Retrieved chunks in context; answer MUST cite sources
  (chunk ids → rendered as clickable citations to snapshot/warning/card).
- **No-WebGPU fallback:** retrieval-only mode (ranked evidence cards, no chat).
  The search must be excellent even without the LLM.
- **Provenance UI:** show the fusion — per-result badges for which legs hit
  (D=dense/B=bm25, static/live) and RRF rank. This is the portfolio money-shot:
  make retrieval *visible*.
- **Live map:** MapLibre GL, current vessel positions from `snapshot.json`,
  colored by type; click vessel → its recent events.

## Evaluation (the signature — do not skip)

Ground truth is FREE here: arrivals/departures/anchorings are deterministically
computable from raw AIS. Build:
- **Golden set** (~100 Q/A pairs) auto-derived from recorded AIS windows +
  hand-written phrasings ("did any tanker anchor off Piraeus yesterday
  afternoon?" → known answer).
- **Retrieval metrics:** recall@k / MRR per question, reported per leg (dense
  vs BM25 vs fused) — SHOW that fusion beats either leg or say it doesn't.
- **Answer metrics:** citation validity (does the cited chunk support the
  claim?), abstention quality (questions outside the data window must get
  "I don't have that watch's log", not confabulation).
- Publish EVALUATION.md, o-ilios house style: numbers you can trust, failure
  modes stated plainly.

## Build spine (each step a working artifact; step 1 is a GATE)

1. **Feed validation gate:** register aisstream key → one manual Action run →
   snapshot Saronic → count vessels, eyeball vs MarineTraffic's public map.
   ALSO verify: ferry schedule source (is there a scrapeable/published
   timetable? if not, delay events become v2 and we say so), NAVTEX source.
   If feed quality fails here, STOP and rethink — nothing else gets built.
2. **Digest pipeline on recorded data:** record a few hours of snapshots →
   diff → event digests → unit tests on synthetic diffs. Golden-set derivation
   starts here.
3. **Index build:** static layer (ports/glossary) + live layer; embedding +
   BM25 stats; artifact format frozen.
4. **Browser retrieval:** search UI + provenance badges. No LLM yet —
   retrieval must stand alone (this is also the fallback mode).
5. **Harbormaster chat:** WebLLM + persona + citations + abstention rules.
6. **Map + polish** (frontend-design pass) + EVALUATION.md.
7. **Publish:** repo public, Pages live, cron → */20, portfolio card, demo GIF.

## Free-tier discipline

- **While private:** NO cron (private minutes are a shared 2,000/mo pool) —
  `workflow_dispatch` manual runs only, or a lazy 6h schedule at most.
- **At public launch:** cron `*/20 * * * *` — public repos = unlimited free
  standard-runner minutes (proven by o-ilios).
- Schedules are best-effort (minutes of jitter, occasional skips) — fine for
  20-min snapshots. Times are UTC. 60-day inactivity auto-disable — the worker
  committing artifacts self-arms it.

## Honest-limits section (must appear in the public README at launch)

- "Live" = ~20-minute snapshots, not streaming.
- Volunteer AIS coverage — gaps possible; Saronic chosen because it's dense.
- Not a navigation aid; educational/portfolio demonstration.
- aisstream.io terms: non-commercial use — this is a free demo, compliant.

## Plugins for this repo (recommend at session start)

```
claude plugin enable frontend-design   # chat UI + map polish
claude plugin install pyright-lsp@claude-plugins-official
```

Already at user scope: playwright, chrome-devtools (screenshot-driven UI
iteration), context7 (transformers.js / WebLLM / MapLibre docs), firecrawl.

## Working conventions (house rules)

- Batch edits, one commit. Ask before slow deploys.
- Step-1 gate is sacred (the FLOGA lesson: validate data access before
  building on it).
- New ideas → the v2 parking lot below.
  Bounded by design: v1 = Saronic, one feed, seven steps.
- **v2 parking lot:** whole-Aegean scope, ferry-delay leaderboard, Greek-query
  support (swap to multilingual-e5-small — one constant + one static-layer
  re-embed), weather layer (Open-Meteo marine), anchorage-congestion
  time-series, RTL-SDR own receiver (physical moat + AISHub membership).

## Related repos

- `o-ilios` — the worker→static-artifacts→Pages pattern, EVALUATION.md house
  style, README voice. Steal the skeleton.
- `pyroPythia` — MapLibre patterns.
- `CaptainJimbo.github.io` — portfolio site; gets a card at launch.

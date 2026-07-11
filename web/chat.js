// The Harbormaster: WebLLM (WebGPU) answering strictly from retrieved
// chunks, with mandatory [n] citations and abstention.

export const MODELS = [
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    label: "Llama 3.2 3B — better answers (~1.8 GB download)",
  },
  {
    id: "Qwen2.5-1.5B-Instruct-q4f16_1-MLC",
    label: "Qwen 2.5 1.5B — lighter (~0.9 GB download)",
  },
];

export function webgpuAvailable() {
  return typeof navigator !== "undefined" && !!navigator.gpu;
}

let enginePromise = null;
let engineModel = null;

export function loadEngine(modelId, onProgress) {
  if (!enginePromise || engineModel !== modelId) {
    engineModel = modelId;
    enginePromise = (async () => {
      const { CreateMLCEngine } = await import(
        "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.79/+esm");
      return CreateMLCEngine(modelId, { initProgressCallback: onProgress });
    })();
  }
  return enginePromise;
}

const PERSONA = `You are the Limenarchis, the AI Harbormaster of Piraeus — \
decades on the quay, dry, precise, seen everything twice. You answer \
questions about the Saronic Gulf strictly from the log extracts your \
retrieval clerk hands you.

Rules of the watch:
- Use ONLY the extracts below. Never invent vessels, times, positions or \
events. If two extracts disagree, say so.
- Cite evidence for every factual claim with bracket numbers matching the \
extracts, like [1] or [2][5]. No claim without a citation.
- If the extracts do not answer the question, say plainly "I don't have \
that in this watch's log." — you may add what the log does hold. The live \
log covers roughly the last 48 hours, in ~20-minute snapshots; the world \
before that is not yours to speak of.
- Log times are UTC; Piraeus local time is UTC+3 in summer. Convert when \
the visitor speaks of local time.
- Answer like a harbormaster: 1-4 sentences, exact, no enthusiasm.

Example of the required form:
Question: Is any tanker waiting outside Piraeus?
Extracts: [1] (live log) 14 vessels lying at Piraeus anchorage at 21:35 UTC. \
[2] (live log) HELLENIC WIND (tanker) anchored at Piraeus anchorage at 20:15 UTC.
Answer: Fourteen vessels lie at the Piraeus anchorage as of 21:35 UTC [1]; \
at least one is a tanker — HELLENIC WIND, anchored 20:15 UTC [2].

Every sentence you write must carry at least one [n]. An answer without \
citations is a false log entry.`;

export function buildMessages(query, hits, liveBuiltUtc) {
  const nowUtc = new Date().toISOString().slice(0, 16).replace("T", " ");
  const extracts = hits.map((h, i) => {
    const kind = h.chunk.source === "live"
      ? `live log, ${h.chunk.event_type || "entry"}` +
        (h.chunk.time_utc ? `, ${h.chunk.time_utc.slice(0, 16)} UTC` : "")
      : "harbour reference";
    const siblings = (h.siblings || []).map((s) => ` Also: ${s.text}`).join("");
    return `[${i + 1}] (${kind}) ${h.chunk.text}${siblings}`;
  }).join("\n");

  return [
    { role: "system", content: PERSONA },
    {
      role: "user",
      content:
        `Current time: ${nowUtc} UTC. Latest live log entry batch: ` +
        `${liveBuiltUtc}.\n\nLog extracts:\n${extracts}\n\n` +
        `Visitor's question: ${query}`,
    },
  ];
}

export async function streamAnswer(engine, messages, onToken) {
  const stream = await engine.chat.completions.create({
    messages,
    stream: true,
    temperature: 0.3,
    max_tokens: 320,
  });
  let text = "";
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content || "";
    if (delta) {
      text += delta;
      onToken(text);
    }
  }
  return text;
}

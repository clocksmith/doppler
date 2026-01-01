function generateResultFilename(result) {
  const timestamp = result.timestamp.replace(/[:.]/g, "-").slice(0, 19);
  const suite = result.suite;
  let modelId = "unknown";
  if ("model" in result && result.model) {
    modelId = (result.model.modelName ?? result.model.modelId ?? "unknown").toLowerCase().replace(/[^a-z0-9]/g, "-").slice(0, 30);
  }
  return `${suite}_${modelId}_${timestamp}.json`;
}
function generateSessionFilename(session) {
  const timestamp = session.startTime.replace(/[:.]/g, "-").slice(0, 19);
  return `session_${session.sessionId}_${timestamp}.json`;
}
const DB_NAME = "doppler_benchmarks";
const DB_VERSION = 1;
const STORE_NAME = "results";
async function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id", autoIncrement: true });
        store.createIndex("timestamp", "timestamp", { unique: false });
        store.createIndex("suite", "suite", { unique: false });
        store.createIndex("modelId", "modelId", { unique: false });
      }
    };
  });
}
async function saveResult(result) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const entry = {
      ...result,
      modelId: "model" in result ? result.model?.modelId : void 0
    };
    const request = store.add(entry);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function loadAllResults() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function loadResultsBySuite(suite) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index("suite");
    const request = index.getAll(suite);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function loadResultsByModel(modelId) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index("modelId");
    const request = index.getAll(modelId);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
async function clearAllResults() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const request = store.clear();
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}
function exportToJSON(results) {
  return JSON.stringify(results, null, 2);
}
function exportResultToJSON(result) {
  return JSON.stringify(result, null, 2);
}
function importFromJSON(json) {
  const parsed = JSON.parse(json);
  return Array.isArray(parsed) ? parsed : [parsed];
}
function downloadAsJSON(results, filename) {
  const data = Array.isArray(results) ? results : [results];
  const json = exportToJSON(data);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const defaultFilename = Array.isArray(results) ? `benchmark_results_${(/* @__PURE__ */ new Date()).toISOString().slice(0, 10)}.json` : generateResultFilename(results);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename ?? defaultFilename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function comparePipelineResults(baseline, current) {
  const deltas = [];
  const metrics = [
    { key: "ttft_ms", lowerIsBetter: true },
    { key: "prefill_ms", lowerIsBetter: true },
    { key: "decode_ms_total", lowerIsBetter: true },
    { key: "prefill_tokens_per_sec", lowerIsBetter: false },
    { key: "decode_tokens_per_sec", lowerIsBetter: false },
    { key: "gpu_submit_count_prefill", lowerIsBetter: true },
    { key: "gpu_submit_count_decode", lowerIsBetter: true },
    { key: "estimated_vram_bytes_peak", lowerIsBetter: true }
  ];
  for (const { key, lowerIsBetter } of metrics) {
    const baseVal = baseline.metrics[key];
    const currVal = current.metrics[key];
    if (typeof baseVal === "number" && typeof currVal === "number") {
      const delta = currVal - baseVal;
      const deltaPercent = baseVal !== 0 ? delta / baseVal * 100 : 0;
      const improved = lowerIsBetter ? delta < 0 : delta > 0;
      deltas.push({
        metric: key,
        baseline: baseVal,
        current: currVal,
        delta,
        deltaPercent,
        improved
      });
    }
  }
  return deltas;
}
function formatComparison(deltas) {
  const lines = ["=== Benchmark Comparison ===", ""];
  for (const d of deltas) {
    const sign = d.delta >= 0 ? "+" : "";
    const arrow = d.improved ? "\u2713" : "\u2717";
    const pct = `${sign}${d.deltaPercent.toFixed(1)}%`;
    lines.push(`${arrow} ${d.metric}: ${d.baseline} \u2192 ${d.current} (${pct})`);
  }
  const improved = deltas.filter((d) => d.improved).length;
  const regressed = deltas.filter((d) => !d.improved).length;
  lines.push("");
  lines.push(`Summary: ${improved} improved, ${regressed} regressed`);
  return lines.join("\n");
}
function createSession() {
  return {
    sessionId: crypto.randomUUID?.() ?? `session_${Date.now()}`,
    startTime: (/* @__PURE__ */ new Date()).toISOString(),
    results: []
  };
}
function addResultToSession(session, result) {
  session.results.push(result);
}
function computeSessionSummary(session) {
  const pipelineResults = session.results.filter((r) => r.suite === "pipeline");
  if (pipelineResults.length === 0) {
    return {
      totalRuns: session.results.length,
      successfulRuns: session.results.length,
      failedRuns: 0,
      avgTtftMs: 0,
      avgDecodeTokensPerSec: 0
    };
  }
  const ttfts = pipelineResults.map((r) => r.metrics.ttft_ms);
  const decodeSpeeds = pipelineResults.map((r) => r.metrics.decode_tokens_per_sec);
  return {
    totalRuns: session.results.length,
    successfulRuns: session.results.length,
    failedRuns: 0,
    avgTtftMs: ttfts.reduce((a, b) => a + b, 0) / ttfts.length,
    avgDecodeTokensPerSec: decodeSpeeds.reduce((a, b) => a + b, 0) / decodeSpeeds.length
  };
}
export {
  addResultToSession,
  clearAllResults,
  comparePipelineResults,
  computeSessionSummary,
  createSession,
  downloadAsJSON,
  exportResultToJSON,
  exportToJSON,
  formatComparison,
  generateResultFilename,
  generateSessionFilename,
  importFromJSON,
  loadAllResults,
  loadResultsByModel,
  loadResultsBySuite,
  saveResult
};

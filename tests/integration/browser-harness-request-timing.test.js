import assert from 'node:assert/strict';
import { runBrowserSuite } from '../../src/tooling/browser-harness.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';
import { assertComparableTiming } from '../../tools/compare-engines.js';

// A deterministic clock makes tokenization, token delivery, and post-token
// cleanup observably distinct from the pipeline's internal kernel counters.
const originalPerformance = globalThis.performance;
const originalConfig = getRuntimeConfig();
let clock = 0;
let runIndex = -1;
let resetCount = 0;
let unloadCount = 0;
const latencies = [999, ...Array(18).fill(10), 200, 300];
const pipeline = {
  reset() { resetCount += 1; },
  async unload() { unloadCount += 1; },
  async *generate(_prompt, options) {
    runIndex += 1;
    clock += latencies[runIndex];
    options.onToken(1, '');
    yield '';
    clock += 10;
    options.onToken(2, 'answer');
    yield 'answer';
    clock += 5; // Completion/cleanup belongs to total and decode wall time.
  },
  getStats() {
    return { prefillTokens: 8, prefillTimeMs: latencies[runIndex], ttftMs: 1,
      decodeTimeMs: 2, decodeTokens: 1, totalTimeMs: 3, decodeProfileSteps: [] };
  },
};
let result;
try {
  globalThis.performance = { now: () => clock };
  setRuntimeConfig({
    inference: { prompt: 'fixture', generation: { maxTokens: 2 }, sampling: { temperature: 0, topK: 1, topP: 1 } },
    shared: { benchmark: { run: { warmupRuns: 1, timedRuns: 20 } } },
  });
  result = await runBrowserSuite({
    suite: 'bench', command: 'bench', surface: 'node',
    harnessOverride: { modelLoadMs: 7, pipeline, manifest: {
      modelId: 'timing-fixture', modelType: 'transformer',
      architecture: { numLayers: 1, hiddenSize: 8, intermediateSize: 16,
        numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 8, vocabSize: 16, maxSeqLen: 64 },
      inference: { attention: { queryPreAttnScalar: 8 }, chatTemplate: { type: 'none', enabled: false } },
    } },
  });
} finally {
  globalThis.performance = originalPerformance;
  setRuntimeConfig(originalConfig);
}
assert.equal(resetCount, 21);
assert.equal(unloadCount, 1);
assert.equal(result.timing.scope, 'generation-call-through-completion');
assert.equal(result.timing.firstTokenMs, 10);
assert.equal(result.timing.totalRunMs, 25);
assert.equal(result.timing.decodeMs, 15);
assert.equal(result.timing.decodeTokensPerSec, 66.67);
assert.equal(result.timing.firstResponseMs, 17);
assert.equal(result.timing.promptTokensPerSecToFirstToken, 800);
assert.equal(result.timing.decodeMsPerTokenP95, 10);
assert.equal(result.metrics.firstTokenMs, 1); // Internal timing remains available.
assert.equal(result.metrics.samplePolicy.outlierPolicy, 'none');
assert.equal(result.metrics.latency.prefillMs.samplesAfterOutlierRemoval, 20);
assert.equal(result.metrics.latency.prefillMs.outliersRemoved, 0);
assert.equal(result.metrics.latency.prefillMs.max, 300);
const request = result.metrics.requestTiming;
assert.equal(request.samples.length, 20);
assert.equal(request.latency.firstTokenMs.mean, 34);
assert.equal(request.latency.firstTokenMs.max, 300);
assert.ok(Math.abs(request.latency.firstTokenMs.p95 - 205) < 1e-9);
assert.equal(request.latency.firstTokenMs.outliersRemoved, 0);
assert.equal(request.samples[0].tokensGenerated, 2);
const tjs = { timing: { scope: result.timing.scope },
  metrics: { samplePolicy: { outlierPolicy: 'none' } },
  runs: [{ ttftMs: 10, totalMs: 25, decodeMs: 15 }] };
assert.doesNotThrow(() => assertComparableTiming({ result }, tjs));
for (const mutate of [
  (r) => { r.timing.scope = 'internal-pipeline'; },
  (r) => { r.metrics.samplePolicy.outlierPolicy = 'iqr'; },
  (r) => { r.metrics.requestTiming.samples = []; },
  (r) => { r.metrics.requestTiming.samples[0].totalRunMs = NaN; },
  (r) => { r.metrics.requestTiming.samples[0].decodeMs = 10; },
]) {
  const invalid = structuredClone(result);
  mutate(invalid);
  assert.throws(() => assertComparableTiming({ result: invalid }, tjs));
}
console.log('browser-harness-request-timing.test: ok (CPU clock fixture; no GPU evidence)');

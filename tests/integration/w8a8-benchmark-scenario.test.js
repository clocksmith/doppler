import assert from 'node:assert/strict';
import { runW8A8BenchmarkScenario } from '../../tools/w8a8-benchmark-scenario.js';

const url = code => `data:text/javascript,${encodeURIComponent(code)}`;
const apiModule = url('export const load = (...args) => globalThis.__w8BenchmarkControl.load(...args);');
const deviceModule = url('export const getDevice = () => globalThis.__w8BenchmarkControl.device;');
const partitionsModule = url('export const configureDeviceMemoryBudget = async () => {}; export const inspectDeviceMemory = () => ({peakBytes: 100});');
const reference = { suite: { generation: {}, benchmark: { warmupRuns: 2, pairedRuns: 2 }, lifecycle: { maxGpuBufferBytes: 1000 } },
  lanes: [{ modelIdentity: 'a' }, { modelIdentity: 'b' }], fixedWork: [{ text: 'x', inputIds: [2], decodeIds: [3] }] };
const config = { apiModule, deviceModule, partitionsModule, residencyPolicy: 'serialized-resident',
  comparisonModels: ['a', 'b'].map(id => ({ id, modelUrl: id, manifestDigest: id })) };
function control(failDecode = false) {
  let active = 0, peak = 0, snapshots = 0;
  const loads = [], calls = [];
  return { loads, calls, active: () => active, peak: () => peak,
    device: { queue: { async onSubmittedWorkDone() { assert.equal(snapshots, 0); } } },
    async load({ url: id }) {
      active++; peak = Math.max(peak, active); loads.push(id);
      let closed = false;
      return { resetGenerationState() { assert(!closed); }, async unload() { assert(!closed); await Promise.resolve(); closed = true; active--; },
        advanced: { getResolvedRuntimeSession: () => ({ id }), tokenizePrompt: () => [2],
          getStats: () => ({ kernelPathId: id, executionPlan: { active: id } }),
          async prefillWithLogits() { calls.push(id); snapshots++; return { tokens: [2], logits: new Float32Array([1]), cache: { destroy() { snapshots--; } } }; },
          async decodeStepLogits() { if (failDecode) throw new Error('decode failure'); return { logits: new Float32Array([2]) }; } } };
    } };
}
try {
  globalThis.__w8BenchmarkControl = control();
  const result = await runW8A8BenchmarkScenario(config, reference, async () => {});
  assert.equal(result.passed, true);
  assert.deepEqual(globalThis.__w8BenchmarkControl.loads, ['a', 'b', 'b', 'a']);
  assert.equal(globalThis.__w8BenchmarkControl.peak(), 1);
  assert.equal(globalThis.__w8BenchmarkControl.active(), 0);
  assert.equal(result.pairs.length, 2);
  assert(result.pairs.every(pair => pair.runs.length === 2 && pair.residencyWarmups.length === 4));
  assert.equal(globalThis.__w8BenchmarkControl.calls.length, 12);
  globalThis.__w8BenchmarkControl = control(true);
  const failed = await runW8A8BenchmarkScenario(config, reference, async () => {});
  assert.equal(failed.passed, false);
  assert.match(failed.error.message, /decode failure/);
  assert.equal(globalThis.__w8BenchmarkControl.active(), 0);
  assert.equal(failed.pairs.length, 1);
  globalThis.__w8BenchmarkControl = control();
  const paired = await runW8A8BenchmarkScenario({ ...config, residencyPolicy: 'paired-resident' }, reference, async () => {});
  assert.equal(paired.passed, true);
  assert.equal(globalThis.__w8BenchmarkControl.peak(), 2);
  assert.equal(globalThis.__w8BenchmarkControl.active(), 0);
  assert.equal(paired.pairs.filter(pair => pair.warmup).length, 2);
  assert.equal(paired.pairs.filter(pair => !pair.warmup).length, 2);
} finally { delete globalThis.__w8BenchmarkControl; }
console.log('W8A8 paired/serialized residency and failure cleanup controls passed.');

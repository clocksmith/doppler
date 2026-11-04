import assert from 'node:assert/strict';
import { setDevice } from '../../src/gpu/device.js';
import { destroyBufferPool, isBufferActive } from '../../src/memory/buffer-pool.js';
import { prepareWholeTokenBatch } from '../../src/inference/pipelines/text/generator/whole-token-lane.js';
const savedBuffer = globalThis.GPUBuffer, savedUsage = globalThis.GPUBufferUsage;
globalThis.GPUBuffer = class GPUBuffer { constructor(size) { this.size = size; this.bytes = new Uint8Array(size); } destroy() {} };
globalThis.GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
const allocated = [];
let completionWaits = 0;
const device = { createBindGroup() { return {}; }, features: new Set(), limits: { maxBufferSize: 1048576, maxStorageBufferBindingSize: 1048576 },
 createBuffer(d) { const b = new GPUBuffer(d.size); allocated.push(b); return b; },
 createCommandEncoder() { const ops = []; return { copyBufferToBuffer(a,ao,b,bo,n) { ops.push(() => b.bytes.set(a.bytes.subarray(ao,ao+n),bo)); }, finish() { return ops; } }; },
 queue: { submit(commands) { for (const command of commands) for (const op of command) op(); }, async onSubmittedWorkDone() { completionWaits++; } } };
setDevice(device, { platformConfig: null });
try {
 for (const scenario of ['full', 'stop-first', 'stop-middle', 'cancel-before-publish', 'execution-error', 'invalid-result', 'replay-error']) {
  completionWaits = 0;
  const layer = { convStateGPU: new GPUBuffer(4), recurrentStateGPU: new GPUBuffer(4), seqLen: 2, warnedSeqMismatch: false };
  layer.convStateGPU.bytes.fill(7); layer.recurrentStateGPU.bytes.fill(9);
  const kv = { seqLen: 2 }, cache = { layout: 'contiguous', layers: [kv], maxSeqLen: 64,
   markAppended(_, pos, count) { kv.seqLen = pos + count; }, truncate(pos) { kv.seqLen = pos; } };
  const stats = { replayedTokens: 0, replaySubmissions: 0, batchReadbacks: 0 };
  let replayCalls = 0;
  const execute = token => { layer.convStateGPU.bytes[0] += token; layer.recurrentStateGPU.bytes[0] += token * 2; return token + 1; };
  const program = { ready: true, awaitingVerification: false, positionRange: { min: 0, max: 63 }, stats, dispatchCount: 1,
   rejects() { return null; }, resumeAfterRestore(pos) { this.pos = pos; },
   async startBatch(pos, token, count) {
    assert.equal(completionWaits, 0, 'An ordered checkpoint must not fence the host before token submission');
    const rows = await this.runBatch(pos, token, count);
    return { rows: (async function* () { for (const row of rows) yield row; })(), async settle() {} };
   },
   async runBatch(pos, token, count) {
    const result = [];
    for (let i = 0; i < count; i++) {
     token = execute(token); result.push(new Uint32Array([token]));
     if (scenario === 'execution-error' && i === 1) throw new Error('injected execution');
    }
    if (scenario === 'invalid-result') result[1][0] = 999;
    return result;
   },
   async run(pos, token) { const next = execute(token); if (scenario === 'replay-error' && ++replayCalls === 2) throw new Error('injected replay'); return new Uint32Array([next]); },
  };
  const state = { currentSeqLen: 2, decodeStepCount: 4, modelConfig: { vocabSize: 100 }, kvCache: cache,
   convLayerStates: new Map(), linearAttentionRuntime: { layers: new Map([[0, layer]]) }, finitenessBuffer: null,
   stats: {}, batchingStats: { totalBatchedTimeMs: 0, batchedForwardCalls: 0, gpuSubmissions: 0 },
   wholeTokenProgram: { program, advancers: [{ kind: 'kv', owner: kv, layerIdx: 0 }, { kind: 'linear', owner: layer }] } };
  let accepted = 0;
  if (scenario === 'execution-error') await assert.rejects(prepareWholeTokenBatch(state, [1,2,3], 4));
  else if (scenario === 'invalid-result') {
   const batch = await prepareWholeTokenBatch(state, [1,2,3], 4);
   await assert.rejects(async () => { try { await Array.fromAsync(batch.tokens); } finally { await batch.settle(); } });
  }
  else {
   const batch = await prepareWholeTokenBatch(state, [1,2,3], 4);
   assert.throws(() => batch.publish(), /Invalid batch publication/);
   accepted = scenario === 'full' ? 4 : scenario === 'stop-first' ? 1 : scenario === 'cancel-before-publish' ? 0 : 2;
   if (accepted) { let i = 0; for await (const token of batch.tokens) { assert.equal(token, 4+i); batch.publish(); if (++i === accepted) break; } }
   if (scenario === 'replay-error') { await assert.rejects(batch.settle(), /injected replay/); accepted = 0; }
   else await batch.settle();
   await batch.settle(); assert.throws(() => batch.publish(), /Invalid batch publication/);
  }
  const sum = Array.from({ length: accepted }, (_,i) => 3+i).reduce((a,b) => a+b, 0);
  assert.equal(layer.convStateGPU.bytes[0], 7 + sum, scenario);
  assert.equal(layer.recurrentStateGPU.bytes[0], 9 + 2 * sum, scenario);
  assert.equal(state.currentSeqLen, 2 + accepted, scenario);
  assert.equal(state.decodeStepCount, 4 + accepted, scenario);
  assert.equal(layer.seqLen, 2 + accepted, scenario); assert.equal(kv.seqLen, 2 + accepted, scenario);
  assert.ok(allocated.every(b => !isBufferActive(b)), 'checkpoint buffers released: ' + scenario);
 }
} finally { destroyBufferPool(device); setDevice(null); globalThis.GPUBuffer = savedBuffer; globalThis.GPUBufferUsage = savedUsage; }
console.log('whole-token-batch-recovery: full commit, early stop, cancellation, execution/result/replay failure passed');

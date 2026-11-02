import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';
import { createDopplerConfig } from '../../src/config/schema/index.js';
import { setDevice } from '../../src/gpu/device.js';
installNodeFileFetchShim();
const { wholeTokenProgramRequested } = await import('../../src/inference/pipelines/text/generator/whole-token-lane.js');
const { _runDecodeLoop } = await import('../../src/inference/pipelines/text/generator/decode-runtime.js');
setDevice({ features: new Set(), limits: { maxStorageBufferBindingSize: 1048576, maxBufferSize: 1048576 },
 queue: { submit() {} }, createBuffer() { return { destroy() {} }; }, createBindGroup() { return {}; } }, { platformConfig: null });
try {
 for (const boundary of ['stop', 'consumer-return', 'callback-error', 'pending-abort', 'callback-abort']) {
  const runtimeConfig = createDopplerConfig().runtime;
  runtimeConfig.inference.session.decodeLoop = { wholeTokenProgram: true, batchSize: 4, stopCheckMode: 'batch', readbackInterval: 1, readbackMode: 'sequential' };
  const abort = new AbortController(), ids = [1], callbacks = [], batches = [];
  let legacyCalls = 0, singles = 0, batchCalls = 0, replays = 0;
  const kv = { seqLen: 1 }, cache = { layout: 'contiguous', layers: [kv], maxSeqLen: 64,
   markAppended(_, pos, count) { kv.seqLen = pos + count; }, truncate(pos) { kv.seqLen = pos; } };
  const state = { useGPU: true, modelConfig: { hiddenSize: 4, numLayers: 1, vocabSize: 100, layerTypes: ['full_attention'] },
   runtimeConfig, weights: new Map(), kvCache: cache, currentSeqLen: 1, decodeStepCount: 0,
   convLayerStates: new Map(), linearAttentionRuntime: null, finitenessBuffer: null, wholeTokenProgram: null,
   batchingStats: { batchedForwardCalls: 0, totalBatchedTimeMs: 0, gpuSubmissions: 0 }, stats: {}, tokenizer: { getSpecialTokens: () => ({ eos: 9, pad: 0 }) } };
  const next = pos => pos === (boundary === 'stop' ? 3 : 7) ? 9 : pos + 1;
  const program = { ready: true, awaitingVerification: false, positionRange: { min: 0, max: 63 }, stats: { replaySubmissions: 0, batchReadbacks: 0 }, dispatchCount: 1,
   rejects() { return null; }, resumeAfterRestore(pos) { this.pos = pos; },
   async startBatch(pos, token, count) {
    const rows = await this.runBatch(pos, token, count);
    return { rows: (async function* () { for (const row of rows) yield row; })(), async settle() {} };
   },
   async runBatch(pos, token, count) { batchCalls++; if (boundary === 'pending-abort') abort.abort(); return Array.from({length:count},(_,i) => new Uint32Array([next(pos+i)])); },
   async run(pos) { replays++; return new Uint32Array([next(pos)]); } };
  const pipeline = { _state: state, _hasFinitenessFallbackWindow: () => false,
   _consumeFinitenessFallbackToken() {}, _shouldUseFinitenessFallback: () => false,
   _recordStopReason(reason) { state.stats.stopReason = reason; },
   async _generateNTokensGPU() { legacyCalls++; throw new Error('prepared replay must own batching'); },
   async _decodeNextTokenViaLogits() { const token = next(state.currentSeqLen); singles++; state.currentSeqLen++; state.decodeStepCount++; kv.seqLen++;
    state.wholeTokenProgram = { program, advancers: [{ kind: 'kv', owner: kv, layerIdx: 0 }] }; return token; } };
  const opts = { maxTokens: 8, stopSequences: [], suppressTokenIds: [], speculation: null,
   temperature: 0, topK: 1, repetitionPenalty: 1, presencePenalty: 0,
   executionPlan: { batchSize: 4, stopCheckMode: 'batch', readbackInterval: 1, disableMultiTokenDecode: false,
    disableCommandBatching: false, maxBatchDecodeTokens: null, activationDtype: 'f32' } };
  if (boundary === 'stop') delete opts.executionPlan.maxBatchDecodeTokens;
  if (boundary === 'stop') {
   state.finitenessBuffer = {}; state.executionPlanState = { activePlanId: 'primary', primaryPlan: { finitenessGuardEnabled: false } };
   assert.equal(wholeTokenProgramRequested(state,opts),true,'Allocated status buffer is not an active fallback policy');
   state.executionPlanState.primaryPlan.finitenessGuardEnabled = true;
   assert.throws(() => wholeTokenProgramRequested(state,opts), /active finiteness fallback/);
   state.finitenessBuffer = null; state.executionPlanState = null;
  }
  const sentinel = new Error('callback sentinel');
  const stream = _runDecodeLoop.call(pipeline, ids, opts, { signal: abort.signal,
   onToken(token) { callbacks.push(token); if (token === 4 && boundary === 'callback-error') throw sentinel; if (token === 4 && boundary === 'callback-abort') abort.abort(); },
   onBatch(tokens) { batches.push(tokens.map(t => t.id)); } },
   { stopTokenIds: [9], eosToken: 9, stopSequenceStart: 0, decodeToken: String, emitMode: 'token' });
  if (boundary === 'consumer-return') { assert.equal((await stream.next()).value, 2); assert.equal((await stream.next()).value, 3); assert.equal((await stream.next()).value, 4); await stream.return(); }
  else if (boundary === 'callback-error') await assert.rejects(Array.fromAsync(stream), e => e === sentinel);
  else await Array.fromAsync(stream);
  const expected = boundary === 'stop' ? [1,2,3,9] : boundary === 'pending-abort' ? [1,2,3] : [1,2,3,4];
  assert.deepEqual(ids, expected, boundary); assert.equal(state.currentSeqLen, expected.length, boundary);
  assert.equal(kv.seqLen, expected.length, boundary); assert.equal(state.decodeStepCount, expected.length-1, boundary);
  assert.equal(legacyCalls, 0); assert.equal(singles, 2); assert.equal(batchCalls, 1);
  assert.equal(replays, expected.length-3, 'Only the published prefix is replayed');
  if (boundary === 'stop') assert.deepEqual(batches, [[9]]);
 }
} finally { setDevice(null); }
console.log('generator-whole-token-batch: routing, EOS, return, callback error and abort preserve the published prefix');

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readBuffer } from '../../src/memory/buffer-pool.js';
import { runPipelineOperation } from '../../src/inference/pipelines/shader-scoped-pipeline.js';

// Observation only. Hash committed cache prefixes, excluding unused tail bytes
// that rollback deliberately makes unreachable rather than clearing.
async function fingerprint(pipeline) {
  const digest = bytes => createHash('sha256').update(new Uint8Array(bytes)).digest('hex');
  const length = pipeline.currentSeqLen;
  const state = { length, recurrent: {}, ple: {}, qsa: {}, kv: {} };
  pipeline.hyperConnectionSession.assertLength(length);
  for (const [index, layer] of pipeline.linearAttentionRuntime.layers) {
    state.recurrent[index] = { length: layer.seqLen,
      conv: digest(await readBuffer(layer.convStateGPU)),
      recurrent: digest(await readBuffer(layer.recurrentStateGPU)) };
  }
  for (const [index, layer] of pipeline.hyperConnectionSession.layers) {
    state.ple[index] = { length: layer.history.length, conv: digest(await readBuffer(layer.conv.buffer)) };
  }
  const qsa = pipeline.hyperConnectionSession.qsa;
  for (const index of qsa.contract.layers) {
    const buffers = qsa.buffers(index), width = qsa.contract.headDim * 4;
    state.qsa[index] = { length: qsa.length(index),
      raw: digest(await readBuffer(buffers.rawKeys, length * width)),
      blocks: digest(await readBuffer(buffers.blockKeys, Math.floor(length / qsa.contract.compressRatio) * width)) };
  }
  const cache = pipeline.kvCache;
  assert.equal(cache.layout, 'contiguous');
  for (const index of qsa.contract.layers) {
    const buffers = cache.getGPUBuffers(index);
    assert.equal(buffers.seqLen, length);
    const size = length * cache.kvSize * cache.bytesPerElem;
    state.kv[index] = { length,
      keys: digest(await readBuffer(buffers.keysGPU, size)),
      values: digest(await readBuffer(buffers.valuesGPU, size)) };
  }
  return state;
}

export async function checkLearnedLifecycle(pipeline, input, baseline, record) {
  assert.ok(baseline.length >= 2, 'lifecycle requires a two-token deterministic baseline');
  const options = { inputIds: input.tokens, useChatTemplate: false, maxTokens: 3,
    temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1 };
  pipeline.resetGenerationState();
  const controller = new AbortController();
  const reason = new Error('qualification: abort active decode');
  let rows = 0;
  const stream = pipeline.generateTokens('', { ...options, signal: controller.signal,
    onLogits() { if (++rows === 2) controller.abort(reason); } });
  try {
    assert.equal((await stream.next()).value, baseline[0]);
    const before = await fingerprint(pipeline);
    const waiter = new AbortController(), queuedReason = new Error('qualification: cancel queued generation');
    const queued = pipeline.generateTokenIds('', { ...options, signal: waiter.signal });
    const rejected = assert.rejects(queued, error => error === queuedReason);
    waiter.abort(queuedReason);
    await rejected;
    assert.equal(pipeline.isGenerating, true, 'cancelled waiter must not release the paused stream');
    await record({ case: 'queued-abort', passed: true });
    await assert.rejects(stream.next(), error => error === reason);
    assert.equal(rows, 2, 'abort occurs inside active decode after the learned GPU forward');
    assert.deepEqual(await fingerprint(pipeline), before, 'rollback preserves every committed GPU state and cache prefix');
    assert.equal(pipeline.isGenerating, false);
    await record({ case: 'active-decode-rollback', passed: true, committedState: before });
  } finally { await stream.return(); }

  pipeline.resetGenerationState();
  const reused = await pipeline.generateTokenIds('', { ...options, maxTokens: 2 });
  assert.deepEqual(Array.from(reused.tokenIds), baseline.slice(0, 2));
  await record({ case: 'reset-and-reuse', passed: true, tokenIds: Array.from(reused.tokenIds) });

  pipeline.resetGenerationState();
  const pausedAbort = new AbortController();
  const paused = pipeline.generateTokens('', { ...options, signal: pausedAbort.signal });
  try {
    assert.equal((await paused.next()).value, baseline[0]);
    pausedAbort.abort(new Error('qualification: abort paused consumer'));
    await runPipelineOperation(pipeline, () => assert.equal(pipeline.isGenerating, false));
    await record({ case: 'paused-abort-without-pull', passed: true });
  } finally { await paused.return(); }

  pipeline.resetGenerationState();
  const abandoned = pipeline.generateTokens('', options);
  try {
    assert.equal((await abandoned.next()).value, baseline[0]);
    await pipeline.unload();
    assert.equal(pipeline.isLoaded, false);
    await record({ case: 'unload-paused-consumer', passed: true });
  } finally { await abandoned.return(); }
}

import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { resetLinearAttentionRuntime } from '../linear-attention.js';
import { isGpuBufferInstance } from '../../../../gpu/weight-buffer.js';

const TRANSACTION = Symbol('hyper-connection transaction');
/** @type {WeakMap<object, object>} */
const active = new WeakMap();

/** The private transaction results are scalar tokens, CPU logits or owned GPU logits.
 * @param {unknown} result */
function releaseResult(result) {
  if (isGpuBufferInstance(result)) releaseBuffer(result);
  else if (result && typeof result === 'object' && 'logitsBuffer' in result && isGpuBufferInstance(result.logitsBuffer)) releaseBuffer(result.logitsBuffer);
}

/** @type {typeof import('./sequence-transaction.js').assertHyperConnectionOperation} */
export function assertHyperConnectionOperation(state, opts) {
  if (!state.modelConfig?.hyperConnections) return;
  if (opts.useSpeculative || opts.speculation?.enabled || (opts.speculation?.mode && opts.speculation.mode !== 'none') || opts._embeddingOnly || opts._trainingCapture
    || opts._diffusionGemmaDecoder || opts.prefixEmbeddingOverride || opts.embeddingInputIds
    || opts.embeddingInputSpan || opts.embeddingOverrides || opts.multimodalBidirectionalSpan || opts.attentionMask || opts.paddingMask) {
    throw new Error('[HyperConnection] Only unpadded text generation with known token IDs is supported; speculative, embedding, training and multimodal operations are not supported.');
  }
  if (opts._returnHidden && !active.has(state)) {
    throw new Error('[HyperConnection] Hidden-state transfer requires an owned prefill transaction.');
  }
}

/** @type {typeof import('./sequence-transaction.js').assertHyperConnectionLoad} */
export function assertHyperConnectionLoad(state) {
  if (!state.modelConfig?.hyperConnections) return;
  const session = state.runtimeConfig.inference.session;
  if (!state.useGPU || state.modelPartition || state.layerPipelinePlan || state.lora || state.manifest?.draftModel
    || state.runtimeConfig.inference.compute.activationDtype !== 'f32'
    || (session.kvcache?.kvDtype !== 'f16' && session.kvcache?.kvDtype !== 'f32') || session.kvcache?.layout !== 'contiguous'
    || session.decodeLoop?.wholeTokenProgram === true || session.decodeLoop?.batchSize !== 1
    || state.executionV1State?.fallbackKernelPath) {
    throw new Error('[HyperConnection] Requires unpartitioned f32 text execution, contiguous f16/f32 KV storage and batchSize=1, without whole-token replay, draft models or layer-plan/LoRA overrides.');
  }
}

/** @type {typeof import('./sequence-transaction.js').assertClassicSequenceState} */
export function assertClassicSequenceState(state, operation) {
  if (state.modelConfig?.hyperConnections) throw new Error(`[HyperConnection] ${operation} does not carry the complete decoder state and is not supported.`);
}

/** @template T @type {typeof import('./sequence-transaction.js').withHyperConnectionTransaction<T>} */
export async function withHyperConnectionTransaction(state, opts, action) {
  if (!state.modelConfig?.hyperConnections) return action(opts);
  assertHyperConnectionOperation(state, opts);
  const ownedOpts = /** @type {import('./sequence-transaction.js').HyperConnectionOptions & Record<symbol, unknown>} */ (opts);
  const existing = active.get(state);
  if (existing && ownedOpts[TRANSACTION] === existing) return action(opts);
  if (existing) throw new Error('[HyperConnection] Concurrent sequence mutation is not supported.');
  const session = state.hyperConnectionSession;
  const device = getDevice();
  if (!session || !device || !state.kvCache) throw new Error('[HyperConnection] Sequence state is not initialized.');
  opts.signal?.throwIfAborted();
  const token = {};
  active.set(state, token);
  const next = { ...opts, [TRANSACTION]: token };
  const savedLength = state.currentSeqLen;
  const savedSteps = state.decodeStepCount;
  const originalLayers = new Map(state.linearAttentionRuntime?.layers);
  /** @type {Array<{ owner: import('../linear-attention.js').LinearLayerRuntimeState, field: 'convStateGPU' | 'recurrentStateGPU', target: GPUBuffer, backup: GPUBuffer }>} */
  const buffers = [];
  const lengths = new Map([...originalLayers].map(([index, layer]) => [index, layer.seqLen]));
  let snapshot = null;
  let began = false;
  let completedResult;
  let failure = null;
  try {
    snapshot = await session.checkpoint();
    const encoder = device.createCommandEncoder();
    for (const layer of originalLayers.values()) {
      for (const key of ['convStateGPU', 'recurrentStateGPU']) {
        const target = layer[/** @type {'convStateGPU' | 'recurrentStateGPU'} */ (key)];
        if (!target) throw new Error('[HyperConnection] Restored CPU-only recurrent snapshots are not supported.');
        const backup = acquireBuffer(target.size, undefined, 'linear_checkpoint');
        buffers.push({ target, backup, owner: layer, field: /** @type {'convStateGPU' | 'recurrentStateGPU'} */ (key) });
        encoder.copyBufferToBuffer(target, 0, backup, 0, target.size);
      }
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    opts.signal?.throwIfAborted();
    began = true;
    const result = await action(next);
    completedResult = result;
    opts.signal?.throwIfAborted();
    session.assertLength(state.currentSeqLen);
    return result;
  } catch (error) {
    failure = error;
    if (!began || !snapshot) throw error;
    try {
      await device.queue.onSubmittedWorkDone();
      for (const [index, original] of originalLayers) {
        if (state.linearAttentionRuntime?.layers.get(index) !== original) throw new Error('Recurrent state ownership changed during the operation.');
      }
      const encoder = device.createCommandEncoder();
      for (const { target, backup, owner, field } of buffers) {
        if (owner[field] !== target) throw new Error('Recurrent state storage changed during the operation.');
        encoder.copyBufferToBuffer(backup, 0, target, 0, target.size);
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const added = new Map([...state.linearAttentionRuntime?.layers ?? []].filter(([index]) => !originalLayers.has(index)));
      if (added.size && state.linearAttentionRuntime) resetLinearAttentionRuntime({ schemaVersion: state.linearAttentionRuntime.schemaVersion, layers: added });
      if (state.linearAttentionRuntime) state.linearAttentionRuntime.layers = originalLayers;
      for (const [index, layer] of originalLayers) layer.seqLen = /** @type {number} */ (lengths.get(index));
      state.kvCache.truncate(savedLength);
      await session.restore(snapshot);
      state.currentSeqLen = savedLength;
      state.decodeStepCount = savedSteps;
    } catch (rollbackError) {
      state.isLoaded = false;
      failure = new AggregateError([error, rollbackError], 'Hyper-connection rollback failed; unload and reload the model before reuse.', { cause: error });
      throw failure;
    }
    throw error;
  } finally {
    const cleanupErrors = [];
    try { if (failure) releaseResult(completedResult); } catch (error) { cleanupErrors.push(error); }
    try { if (snapshot) session.releaseCheckpoint(snapshot); } catch (error) { cleanupErrors.push(error); }
    for (const { backup } of buffers) {
      try { releaseBuffer(backup); } catch (error) { cleanupErrors.push(error); }
    }
    active.delete(state);
    if (cleanupErrors.length) {
      if (!failure) { try { releaseResult(completedResult); } catch (error) { cleanupErrors.push(error); } }
      state.isLoaded = false;
      throw new AggregateError(failure ? [failure, ...cleanupErrors] : cleanupErrors, 'Hyper-connection checkpoint cleanup failed.', { cause: failure });
    }
  }
}

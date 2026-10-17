import { PipelineState } from './state.js';
import { createKVCache } from './init.js';
import { assertPartitionExecutionSupported } from './partition-execution.js';
import { resolveLayerPartition } from './layer-partition-contract.js';
import { createLinearAttentionRuntime, resetLinearAttentionRuntime } from './linear-attention.js';
import { cloneLinearAttentionRuntimeOnGPU } from './linear-attention/state-clone.js';
import { KVCache } from '../../kv-cache/base.js';
import { MixedGeometryKVCache } from '../../kv-cache/mixed-geometry.js';

/** @type {import('./partition-attempt.js').createPartitionAttempt} */
export function createPartitionAttempt(owner, maxSeqLen) {
  if (!owner.modelPartition || !owner.manifest || !owner.modelConfig) {
    throw new Error('Partition attempt requires a loaded, assigned model.');
  }
  assertPartitionExecutionSupported(owner, owner.modelPartition.plan);
  const partition = resolveLayerPartition(owner.manifest, owner.modelPartition);
  if (!partition) throw new Error('Partition attempt allocation is missing.');
  const inference = owner.runtimeConfig.inference;
  const preparedMaxSeqLen = inference.session.kvcache?.maxSeqLen;
  if (!Number.isSafeInteger(maxSeqLen) || maxSeqLen < 1 || maxSeqLen > owner.modelConfig.maxSeqLen
    || preparedMaxSeqLen !== null && (typeof preparedMaxSeqLen !== 'number'
      || !Number.isSafeInteger(preparedMaxSeqLen) || maxSeqLen > preparedMaxSeqLen)) {
    throw new Error('Resident attempt sequence allocation exceeds the prepared session.');
  }
  // Share only prepared model resources. Fresh PipelineState owns all mutable
  // execution counters, attention state and buffers; it never unloads the owner.
  const state = Object.assign(new PipelineState(), {
    manifest: owner.manifest, modelConfig: owner.modelConfig, modelPartition: owner.modelPartition,
    lora: owner.lora,
    weights: owner.weights, expertWeights: owner.expertWeights, tokenizer: owner.tokenizer,
    runtimeConfig: owner.runtimeConfig, resolvedRuntimeSession: owner.resolvedRuntimeSession,
    resolvedKernelPath: owner.resolvedKernelPath, kernelPathSource: owner.kernelPathSource,
    executionV1State: owner.executionV1State,
    executionPlanState: structuredClone(owner.executionPlanState),
    linearAttentionRuntime: createLinearAttentionRuntime(),
    useGPU: owner.useGPU, gpuContext: owner.gpuContext,
    ropeFreqsCos: owner.ropeFreqsCos, ropeFreqsSin: owner.ropeFreqsSin,
    ropeLocalCos: owner.ropeLocalCos, ropeLocalSin: owner.ropeLocalSin,
    useTiedEmbeddings: owner.useTiedEmbeddings, embeddingVocabSize: owner.embeddingVocabSize,
    embeddingTranspose: owner.embeddingTranspose, layerPipelinePlan: owner.layerPipelinePlan,
    revocationIdentity: owner.revocationIdentity, debug: owner.debug, debugFlags: { ...owner.debugFlags },
    isLoaded: true,
  });
  // Each attempt reserves its complete prompt/output allowance without reserving
  // unused capacity from the prepared session's larger context ceiling.
  state.kvCache = createKVCache(owner.modelConfig, owner.useGPU, owner.debug, {
    ...inference, session: { ...inference.session,
      kvcache: { ...inference.session.kvcache, maxSeqLen } },
  }, [partition.layerRange[0], partition.layerRange[1]]);
  let closed = false;
  return { state, close() {
    if (closed) return;
    closed = true;
    state.kvCache?.destroy();
    state.kvCache = null;
    resetLinearAttentionRuntime(state.linearAttentionRuntime);
    state.isLoaded = false;
  } };
}

/** Caller pins the settled parent and reserves the complete child before entry.
 * @type {import('./partition-attempt.js').clonePartitionAttempt} */
export async function clonePartitionAttempt(parent, signal) {
  signal.throwIfAborted();
  const source = parent.state;
  if (!source.isLoaded || !(source.kvCache instanceof KVCache || source.kvCache instanceof MixedGeometryKVCache)
    || source.convLayerStates.size > 0 || source.speculativeDecoder || source.decodeRing || source.decodeBuffers) {
    throw new Error('Partition state geometry does not support composite cloning.');
  }
  await source.gpuContext?.device?.queue.onSubmittedWorkDone();
  signal.throwIfAborted();
  const child = createPartitionAttempt(source, source.kvCache.maxSeqLen);
  try {
    if (source.kvCache instanceof KVCache && child.state.kvCache instanceof KVCache) {
      await child.state.kvCache.copyFrom(source.kvCache);
    } else if (source.kvCache instanceof MixedGeometryKVCache && child.state.kvCache instanceof MixedGeometryKVCache) {
      await child.state.kvCache.copyFrom(source.kvCache);
    } else throw new Error('Missing matching child KV owner.');
    signal.throwIfAborted();
    child.state.linearAttentionRuntime = await cloneLinearAttentionRuntimeOnGPU(source.linearAttentionRuntime);
    child.state.currentSeqLen = source.currentSeqLen;
    child.state.decodeStepCount = source.decodeStepCount;
    child.state.currentTokenIds = source.currentTokenIds ? [...source.currentTokenIds] : null;
    await source.gpuContext?.device?.queue.onSubmittedWorkDone();
    signal.throwIfAborted();
    return child;
  } catch (error) {
    // If queue completion is unknown, do not claim release. The enclosing
    // resident must retain the failed owner and interrupt its attempt.
    try { await source.gpuContext?.device?.queue.onSubmittedWorkDone(); child.close(); }
    catch (settlement) { throw Object.assign(new AggregateError([error, settlement], 'Composite clone settlement failed.'), { unsettledAttempt: child }); }
    throw error;
  }
}

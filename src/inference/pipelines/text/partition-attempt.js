import { PipelineState } from './state.js';
import { createKVCache } from './init.js';
import { assertPartitionExecutionSupported } from './partition-execution.js';
import { resolveLayerPartition } from './layer-partition-contract.js';
import { createLinearAttentionRuntime, resetLinearAttentionRuntime } from './linear-attention.js';

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

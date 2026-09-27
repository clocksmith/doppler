import { createCommandRecorder } from '../../../gpu/command-recorder.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { getWeightDtype, getWeightMetadata, isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { resolveActiveExecutionPlan } from './execution-plan.js';
import { buildLayerContext } from './generator/session-context.js';
import { releaseSharedAttentionState } from './generator/attention-lifecycle.js';
import { computeLogits } from './logits/index.js';
import { getLogitsWeights, getLogitsConfig } from './generator/logits-config.js';
import { resolveLayerPartition } from './layer-partition-contract.js';

/** @type {import('./partition-execution.js').assertPartitionExecutionSupported} */
export function assertPartitionExecutionSupported(state, plan) {
  const config = state.modelConfig;
  if (!state.useGPU || !config) throw new Error('Resident partition execution requires a loaded WebGPU model.');
  if (config.useMoE || config.numKvSharedLayers > 0 || (config.hiddenSizePerLayerInput !== null && config.hiddenSizePerLayerInput > 0)
    || config.decodeStrategy !== 'incremental' || config.causalAttention !== true
    || config.layerTypes?.some(type => !['full_attention', 'sliding_attention'].includes(type))
    || state.lora || state.visionCapable || state.audioCapable || config.diffusionGemma
    || state.runtimeConfig.inference.session.usePostFfnNextInputRMSNormPairFusion === true) {
    throw new Error('Resident partition execution supports dense causal attention without shared KV, recurrent, per-layer-input, adapter, multimodal, or cross-layer fusion dependencies.');
  }
  if (!state.executionPlanState) throw new Error('Resident partition requires a resolved execution plan.');
  const execution = resolveActiveExecutionPlan(state.executionPlanState);
  if (execution.finitenessGuardEnabled) {
    throw new Error('Resident partition execution does not support finiteness fallback transitions.');
  }
  if (execution.activationDtype !== plan.activationDtype) {
    throw new Error('Partition activation dtype must match the resolved execution plan; implicit casts are forbidden.');
  }
}

/** @type {import('./partition-execution.js').executePartitionLayers} */
export async function executePartitionLayers(state, input, signal) {
  signal.throwIfAborted();
  const config = state.modelConfig;
  if (!config || !state.executionPlanState || !state.manifest || !state.modelPartition) {
    throw new Error('Resident partition requires a resolved model and bound partition plan.');
  }
  assertPartitionExecutionSupported(state, state.modelPartition.plan);
  const partition = resolveLayerPartition(state.manifest, state.modelPartition);
  if (!partition) throw new Error('Resident partition allocation is missing.');
  const executionPlan = resolveActiveExecutionPlan(state.executionPlanState);
  const dtype = executionPlan.activationDtype;
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  const numTokens = input.numTokens;
  if (!Number.isSafeInteger(numTokens) || numTokens < 1 || !state.kvCache
    || !Number.isSafeInteger(state.currentSeqLen) || state.currentSeqLen < 0
    || state.currentSeqLen + numTokens > state.kvCache.maxSeqLen
    || (state.currentSeqLen > 0 && numTokens !== 1)) {
    throw new Error('Resident partition input exceeds its sequence allocation or incremental decode shape.');
  }
  if (partition.hasEmbedding && (!input.tokenIds || input.tokenIds.length !== numTokens
    || input.tokenIds.some(id => !Number.isSafeInteger(id) || id < 0 || id >= config.vocabSize))) {
    throw new Error('Resident partition token IDs must match the declared input shape and vocabulary.');
  }
  const byteLength = numTokens * config.hiddenSize * bytesPerElement;
  if (!Number.isSafeInteger(byteLength)) throw new Error('Resident activation size exceeds the safe integer range.');
  const recorder = createCommandRecorder('resident_partition_layers');
  let context = null;
  /** @type {GPUBuffer | null} */
  let hidden = null;
  let output = null;
  try {
    context = buildLayerContext(state, recorder, state.currentSeqLen > 0, null, undefined, executionPlan);
    context.currentTokenIds = input.tokenIds ?? null;
    if (partition.hasEmbedding) {
      const weight = state.weights.get('embed');
      if (!input.tokenIds || !(isWeightBuffer(weight) || isCpuWeightBuffer(weight)
        || isGpuBufferInstance(weight) || isSplitWeightBuffer(weight))) {
        throw new Error('Resident partition embedding requires token IDs and loaded embedding weights.');
      }
      const embeddingDtype = isCpuWeightBuffer(weight) ? weight.dtype : getWeightDtype(weight);
      const tensor = await embed(input.tokenIds, isWeightBuffer(weight) ? weight.buffer : weight, {
        hiddenSize: config.hiddenSize, vocabSize: config.vocabSize,
        scaleEmbeddings: config.scaleEmbeddings, embeddingScale: config.embeddingScale,
        embeddingNormalization: config.embeddingNormalization,
        transpose: state.embeddingTranspose, recorder, activationDtype: dtype,
        embeddingDtype: selectRuleValue('inference', 'dtype', 'embeddingDtype', { dtype: embeddingDtype }),
        embeddingStorageEncoding: getWeightMetadata(weight)?.storageEncoding ?? null,
        executionPolicies: state.executionV1State?.policies ?? null,
        debugProbes: state.runtimeConfig.shared.debug.probes,
      });
      hidden = tensor.buffer;
    } else {
      if (input.activationBytes?.byteLength !== byteLength) throw new Error('Partition activation byte length mismatch.');
      hidden = acquireBuffer(byteLength, undefined, 'resident_partition_input');
      uploadData(hidden, input.activationBytes);
    }
    for (let layer = partition.layerRange[0]; layer <= partition.layerRange[1]; layer++) {
      signal.throwIfAborted();
      /** @type {GPUBuffer} */
      const previous = hidden;
      const next = await processLayer(layer, previous, numTokens, state.currentSeqLen === 0, context);
      if (!isGpuBufferInstance(next)) throw new Error('Resident partition layers must return GPU buffers.');
      hidden = next;
      if (previous !== hidden) recorder.trackTemporaryBuffer(previous);
    }
    signal.throwIfAborted();
    // Layer kernels may register their output as temporary. Carry only the
    // declared boundary bytes past recorder cleanup; no borrowed GPU pointer escapes.
    output = acquireBuffer(byteLength, undefined, 'resident_partition_output');
    recorder.getEncoder().copyBufferToBuffer(hidden, 0, output, 0, byteLength);
    recorder.trackTemporaryBuffer(hidden);
    hidden = null;
    releaseSharedAttentionState(context.sharedAttentionState, recorder);
    await recorder.submitAndWait();
    signal.throwIfAborted();
    if (!partition.hasLmHead) {
      const activationBytes = await readBuffer(output, byteLength);
      signal.throwIfAborted();
      return { activationBytes };
    }
    const logits = await computeLogits(output, numTokens, getLogitsWeights(state), getLogitsConfig(state),
      true, state.debugFlags, undefined, undefined, state.runtimeConfig.shared.debug.probes,
      { lastPositionOnly: true, returnGpuBuffer: true }, state.operatorDiagnostics);
    try {
      signal.throwIfAborted();
      return { logits };
    } catch (error) {
      releaseBuffer(logits.logitsBuffer);
      throw error;
    }
  } finally {
    if (context) releaseSharedAttentionState(context.sharedAttentionState, recorder);
    if (hidden) recorder.trackTemporaryBuffer(hidden);
    recorder.abort();
    if (output) releaseBuffer(output);
  }
}

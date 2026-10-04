import { createCommandRecorder } from '../../../gpu/command-recorder.js';
import { acquireBuffer, releaseBuffer, readBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { getWeightDtype, getWeightMetadata, isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { resolveActiveExecutionPlan, resolvePrefillRecorderChunkLayers } from './execution-plan.js';
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
    || config.layerTypes?.some(type => !['full_attention', 'sliding_attention', 'linear_attention'].includes(type))
    || state.lora || config.diffusionGemma
    || ((state.visionCapable || state.audioCapable)
      && plan.partitions[0]?.inputContract?.type !== 'token-ids')
    || state.runtimeConfig.inference.session.usePostFfnNextInputRMSNormPairFusion === true) {
    throw new Error('Resident partition execution supports token-only causal full, sliding and linear attention without shared KV, per-layer-input, adapter, multimodal input, or cross-layer fusion dependencies.');
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
  // A missing recurrent prefix cannot be reconstructed from a sequence number.
  // Fail before dispatch rather than letting the whole-model reset path invent it.
  for (let layer = partition.layerRange[0]; layer <= partition.layerRange[1]; layer++) {
    if (config.layerTypes?.[layer] !== 'linear_attention') continue;
    const recurrent = state.linearAttentionRuntime?.layers.get(layer);
    if ((state.currentSeqLen > 0 && !recurrent)
      || (recurrent && recurrent.seqLen !== state.currentSeqLen)) {
      throw new Error(`Resident recurrent state missing or out of order at layer ${layer}; restart the attempt from its prompt.`);
    }
  }
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
  const chunkLayers = resolvePrefillRecorderChunkLayers({
    configuredPrefillChunkLayers: state.runtimeConfig.inference.session.prefillChunkLayers,
    hasGpuSplitPerLayerInputs: false,
    numTokens,
  });
  const createRecorder = () => createCommandRecorder('resident_partition_layers', {
    profile: state.runtimeConfig.shared.debug.profiler.enabled,
  });
  let recorder = createRecorder();
  const timing = { inputUploadMs: 0, encodeMs: 0, submitWaitMs: 0, activationReadbackMs: 0,
    logitsMs: 0, gpuKernelsMs: /** @type {number | null} */ (null) };
  const encodeStarted = performance.now();
  let context = null;
  /** @type {GPUBuffer | null} */
  let hidden = null;
  let output = null;
  const submit = async () => {
    const started = performance.now();
    await recorder.submitAndWait();
    timing.submitWaitMs += performance.now() - started;
    const kernels = await recorder.resolveProfileTimings();
    if (kernels) timing.gpuKernelsMs = (timing.gpuKernelsMs ?? 0)
      + Object.values(kernels).reduce((sum, ms) => sum + ms, 0);
  };
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
      const uploadStarted = performance.now();
      uploadData(hidden, input.activationBytes);
      timing.inputUploadMs = performance.now() - uploadStarted;
    }
    for (let layer = partition.layerRange[0]; layer <= partition.layerRange[1]; layer++) {
      signal.throwIfAborted();
      /** @type {GPUBuffer} */
      const previous = hidden;
      const next = await processLayer(layer, previous, numTokens, state.currentSeqLen === 0, context);
      if (!isGpuBufferInstance(next)) throw new Error('Resident partition layers must return GPU buffers.');
      hidden = next;
      if (previous !== hidden) recorder.trackTemporaryBuffer(previous);
      if (state.currentSeqLen === 0 && layer < partition.layerRange[1]
        && (layer - partition.layerRange[0] + 1) % chunkLayers === 0) {
        // Preserve only the boundary tensor while the completed chunk releases
        // its intermediates. This consumes the same prefill policy as whole-model execution.
        const carry = acquireBuffer(byteLength, undefined, 'resident_partition_carry');
        try {
          recorder.getEncoder().copyBufferToBuffer(hidden, 0, carry, 0, byteLength);
        } catch (error) { releaseBuffer(carry); throw error; }
        recorder.trackTemporaryBuffer(hidden);
        hidden = carry;
        await submit();
        signal.throwIfAborted();
        recorder = createRecorder();
        context.recorder = recorder;
      }
    }
    signal.throwIfAborted();
    // Layer kernels may register their output as temporary. Carry only the
    // declared boundary bytes past recorder cleanup; no borrowed GPU pointer escapes.
    output = acquireBuffer(byteLength, undefined, 'resident_partition_output');
    recorder.getEncoder().copyBufferToBuffer(hidden, 0, output, 0, byteLength);
    recorder.trackTemporaryBuffer(hidden);
    hidden = null;
    releaseSharedAttentionState(context.sharedAttentionState, recorder);
    timing.encodeMs = performance.now() - encodeStarted - timing.submitWaitMs;
    await submit();
    signal.throwIfAborted();
    if (!partition.hasLmHead) {
      const readbackStarted = performance.now();
      const activationBytes = await readBuffer(output, byteLength);
      timing.activationReadbackMs = performance.now() - readbackStarted;
      signal.throwIfAborted();
      return { activationBytes, timing };
    }
    const logitsStarted = performance.now();
    const logits = await computeLogits(output, numTokens, getLogitsWeights(state), getLogitsConfig(state),
      true, state.debugFlags, undefined, undefined, state.runtimeConfig.shared.debug.probes,
      { lastPositionOnly: true, returnGpuBuffer: true }, state.operatorDiagnostics);
    try {
      signal.throwIfAborted();
      timing.logitsMs = performance.now() - logitsStarted;
      return { logits, timing };
    } catch (error) {
      releaseBuffer(logits.logitsBuffer);
      throw error;
    }
  } finally {
    if (context) releaseSharedAttentionState(context.sharedAttentionState, recorder);
    if (hidden) {
      if (recorder.getStats().submitted) releaseBuffer(hidden);
      else recorder.trackTemporaryBuffer(hidden);
    }
    await recorder.abort();
    if (output) releaseBuffer(output);
  }
}

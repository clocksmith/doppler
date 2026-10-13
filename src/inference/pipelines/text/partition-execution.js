import { createCommandRecorder } from '../../../gpu/command-recorder.js';
import { beginImmediateDispatchProfile } from '../../../gpu/kernels/dispatch.js';
import { log } from '../../../debug/index.js';
import { acquireBuffer, releaseBuffer, discardBuffer, readBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { getWeightDtype, getWeightMetadata, isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { resolveActiveExecutionPlan, resolvePrefillRecorderChunkLayers } from './execution-plan.js';
import { buildLayerContext } from './generator/session-context.js';
import { releaseSharedAttentionState } from './generator/attention-lifecycle.js';
import { computeLogits } from './logits/index.js';
import { getLogitsWeights, getLogitsConfig } from './generator/logits-config.js';
import { resolveLayerPartition, validateActivationTensorShape } from './layer-partition-contract.js';
import { resolvePrefillTokenChunkSize, shouldDisablePrefillCommandBatching } from './generator/prefill-policy.js';

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
  const precision = state.resolvedRuntimeSession?.runtime.session.intermediatePrecision;
  if (precision !== 'f32' && precision !== 'two-f32') {
    throw new Error('Resident partition requires explicit resolved intermediate precision.');
  }
  if ((precision === 'two-f32') !== (plan.activationPrecision === 'two-f32')) {
    throw new Error('Partition activation precision must match the resolved session; implicit precision changes are forbidden.');
  }
  if (precision === 'two-f32' && (execution.activationPrecision !== 'two-f32'
    || execution.activationComponents !== 2 || execution.activationElementStrideBytes !== 8)) {
    throw new Error('Paired partition execution requires a bound paired kernel representation.');
  }
  if (execution.activationDtype !== plan.activationDtype) {
    throw new Error('Partition activation dtype must match the resolved execution plan; implicit casts are forbidden.');
  }
}

/**
 * @param {import('./partition-execution.js').PartitionExecutionState} state
 * @param {Parameters<import('./partition-execution.js').ExecutePartitionChunk>[1]} input
 * @param {boolean} prefill
 */
function resolvePartitionInput(state, input, prefill) {
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
  const numTokens = input.numTokens;
  if (!Number.isSafeInteger(numTokens) || numTokens < 1 || !state.kvCache
    || !Number.isSafeInteger(state.currentSeqLen) || state.currentSeqLen < 0
    || state.currentSeqLen + numTokens > state.kvCache.maxSeqLen
    || (state.currentSeqLen > 0 && numTokens !== 1 && !prefill)) {
    throw new Error('Resident partition input exceeds its sequence allocation or incremental decode shape.');
  }
  if (partition.hasEmbedding && (!input.tokenIds || input.tokenIds.length !== numTokens
    || input.tokenIds.some(id => !Number.isSafeInteger(id) || id < 0 || id >= config.vocabSize))) {
    throw new Error('Resident partition token IDs must match the declared input shape and vocabulary.');
  }
  const plan = state.modelPartition.plan;
  const { expectedBytes: byteLength } = validateActivationTensorShape({ shape: [1, numTokens, config.hiddenSize], dtype,
    precision: plan.activationPrecision, components: plan.activationComponents,
    elementStrideBytes: plan.activationElementStrideBytes });
  if (!Number.isSafeInteger(byteLength)) throw new Error('Resident activation size exceeds the safe integer range.');
  if (!partition.hasEmbedding && input.activationBytes?.byteLength !== byteLength) {
    throw new Error('Partition activation byte length mismatch.');
  }
  return { config, partition, executionPlan, dtype, numTokens, byteLength };
}

function createPartitionTiming() {
  return { inputUploadMs: 0, encodeMs: 0, submitWaitMs: 0, activationReadbackMs: 0,
    logitsMs: 0, gpuKernelsMs: /** @type {number | null} */ (null) };
}

/** @type {import('./partition-execution.js').executePartitionLayers} */
export async function executePartitionLayers(state, input, signal) {
  signal.throwIfAborted();
  const prefill = state.currentSeqLen === 0;
  const setup = resolvePartitionInput(state, input, prefill);
  const chunkSize = resolvePrefillTokenChunkSize(state, input.numTokens);
  if (!prefill || chunkSize === null) {
    const result = await executePartitionChunk(state, input, signal, prefill, true);
    if (result.logits) return { logits: result.logits, timing: result.timing };
    if (result.activationBytes) return { activationBytes: result.activationBytes, timing: result.timing };
    throw new Error('Resident partition produced no boundary result.');
  }
  const originalSeqLen = state.currentSeqLen;
  const rowBytes = setup.byteLength / setup.numTokens;
  const source = input.activationBytes;
  const bytes = source ? ArrayBuffer.isView(source)
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength) : new Uint8Array(source) : null;
  // Assemble the public byte payload in its existing contiguous row order.
  // Model arithmetic remains in the same declared GPU layer programs.
  const activation = setup.partition.hasLmHead ? null : new Uint8Array(new ArrayBuffer(setup.byteLength));
  const timing = createPartitionTiming();
  try {
    for (let offset = 0; offset < input.numTokens; offset += chunkSize) {
      signal.throwIfAborted();
      const end = Math.min(offset + chunkSize, input.numTokens);
      state.currentSeqLen = originalSeqLen + offset;
      const result = await executePartitionChunk(state, { numTokens: end - offset,
        ...(input.tokenIds ? { tokenIds: input.tokenIds.slice(offset, end) } : {}),
        ...(bytes ? { activationBytes: bytes.subarray(offset * rowBytes, end * rowBytes) } : {}),
      }, signal, true, end === input.numTokens);
      timing.inputUploadMs += result.timing.inputUploadMs;
      timing.encodeMs += result.timing.encodeMs;
      timing.submitWaitMs += result.timing.submitWaitMs;
      timing.activationReadbackMs += result.timing.activationReadbackMs;
      timing.logitsMs += result.timing.logitsMs;
      if (result.timing.gpuKernelsMs !== null) timing.gpuKernelsMs = (timing.gpuKernelsMs ?? 0) + result.timing.gpuKernelsMs;
      if (result.logits) return { logits: result.logits, timing };
      if (activation) {
        if (!result.activationBytes) throw new Error('Resident partition produced no chunk activation.');
        activation.set(new Uint8Array(result.activationBytes), offset * rowBytes);
      }
    }
    if (!activation) throw new Error('Resident partition produced no final logits.');
    return { activationBytes: activation.buffer, timing };
  } finally {
    // The enclosing resident operation advances the complete prompt once.
    state.currentSeqLen = originalSeqLen;
  }
}

/** @type {import('./partition-execution.js').ExecutePartitionChunk} */
async function executePartitionChunk(state, input, signal, prefill, finalChunk) {
  const { config, partition, executionPlan, dtype, numTokens, byteLength } = resolvePartitionInput(state, input, prefill);
  const chunkLayers = resolvePrefillRecorderChunkLayers({
    configuredPrefillChunkLayers: state.runtimeConfig.inference.session.prefillChunkLayers,
    hasGpuSplitPerLayerInputs: false,
    numTokens,
  });
  const recordLayers = !prefill || !shouldDisablePrefillCommandBatching({ runtimeConfig: state.runtimeConfig,
    modelConfig: state.modelConfig, kvCache: state.kvCache }, {}, null);
  const createRecorder = () => createCommandRecorder('resident_partition_layers', {
    profile: state.runtimeConfig.shared.debug.profiler.enabled,
  });
  // Immediate kernels still share a boundary recorder for carry copies and
  // completion-owned cleanup. It never batches their model dispatches.
  let recorder = createRecorder();
  const timing = createPartitionTiming();
  const encodeStarted = performance.now();
  let context = null;
  /** @type {GPUBuffer | null} */
  let hidden = null;
  let output = null;
  let primaryFailure = null;
  /** @type {(() => void) | null} */
  let releaseProfile = null;
  let boundaryStarted = performance.now();
  /** @type {Set<GPUBuffer>} */
  const directBuffers = new Set();
  /** @type {{ trackTemporaryBuffer(buffer: GPUBuffer): void }} */
  const boundaryOwner = { trackTemporaryBuffer(buffer) {
    if (recordLayers) recorder.trackTemporaryBuffer(buffer);
    else directBuffers.add(buffer);
  } };
  /** @param {boolean} discard */
  const releaseDirectBuffers = (discard) => {
    for (const buffer of directBuffers) (discard ? discardBuffer : releaseBuffer)(buffer);
    directBuffers.clear();
  };
  /** @param {number} layer */
  const submit = async (layer) => {
    const started = performance.now();
    await recorder.submitAndWait();
    const submitWaitMs = performance.now() - started;
    timing.submitWaitMs += submitWaitMs;
    if (!recordLayers) releaseDirectBuffers(false);
    const kernels = await recorder.resolveProfileTimings({ requireComplete: releaseProfile !== null });
    if (kernels) timing.gpuKernelsMs = (timing.gpuKernelsMs ?? 0)
      + Object.values(kernels).reduce((sum, ms) => sum + ms, 0);
    if (releaseProfile) log.warn('Profile', JSON.stringify({ scope: 'resident-immediate',
      layerEnd: layer, numTokens, position: state.currentSeqLen, encodeWallMs: started - boundaryStarted,
      submitWaitMs, kernels, dispatches: recorder.getStats().dispatches }));
    boundaryStarted = performance.now();
  };
  try {
    if (!recordLayers && state.runtimeConfig.shared.debug.profiler.enabled) {
      releaseProfile = beginImmediateDispatchProfile(recorder.device, () => recorder);
    }
    context = buildLayerContext(state, recordLayers ? recorder : undefined, !prefill, null, undefined, executionPlan);
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
        transpose: state.embeddingTranspose, recorder: recordLayers ? recorder : undefined, activationDtype: dtype,
        intermediatePrecision: /** @type {'f32' | 'two-f32'} */ (state.resolvedRuntimeSession?.runtime.session.intermediatePrecision),
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
      const next = await processLayer(layer, previous, numTokens, prefill, context);
      if (!isGpuBufferInstance(next)) throw new Error('Resident partition layers must return GPU buffers.');
      hidden = next;
      if (previous !== hidden) boundaryOwner.trackTemporaryBuffer(previous);
      if (prefill && layer < partition.layerRange[1]
        && (layer - partition.layerRange[0] + 1) % chunkLayers === 0) {
        // Preserve only the boundary tensor while the completed chunk releases
        // its intermediates. This consumes the same prefill policy as whole-model execution.
        const carry = acquireBuffer(byteLength, undefined, 'resident_partition_carry');
        try {
          recorder.getEncoder().copyBufferToBuffer(hidden, 0, carry, 0, byteLength);
        } catch (error) { releaseBuffer(carry); throw error; }
        boundaryOwner.trackTemporaryBuffer(hidden);
        hidden = carry;
        await submit(layer);
        signal.throwIfAborted();
        recorder = createRecorder();
        context.recorder = recordLayers ? recorder : undefined;
      }
    }
    signal.throwIfAborted();
    // Layer kernels may register their output as temporary. Carry only the
    // declared boundary bytes past recorder cleanup; no borrowed GPU pointer escapes.
    output = acquireBuffer(byteLength, undefined, 'resident_partition_output');
    recorder.getEncoder().copyBufferToBuffer(hidden, 0, output, 0, byteLength);
    boundaryOwner.trackTemporaryBuffer(hidden);
    hidden = null;
    releaseSharedAttentionState(context.sharedAttentionState, boundaryOwner);
    timing.encodeMs = performance.now() - encodeStarted - timing.submitWaitMs;
    await submit(partition.layerRange[1]);
    releaseProfile?.();
    releaseProfile = null;
    signal.throwIfAborted();
    if (!partition.hasLmHead) {
      const readbackStarted = performance.now();
      const activationBytes = await readBuffer(output, byteLength);
      timing.activationReadbackMs = performance.now() - readbackStarted;
      signal.throwIfAborted();
      return { activationBytes, timing };
    }
    if (!finalChunk) return { timing };
    const logitsStarted = performance.now();
    const logits = await computeLogits(output, numTokens, getLogitsWeights(state), getLogitsConfig(state),
      true, state.debugFlags, undefined, undefined, state.runtimeConfig.shared.debug.probes,
      { lastPositionOnly: true, returnGpuBuffer: true }, state.operatorDiagnostics);
    try {
      signal.throwIfAborted();
      timing.logitsMs = performance.now() - logitsStarted;
      return { logits, timing };
    } catch (error) {
      if (recordLayers) releaseBuffer(logits.logitsBuffer);
      else boundaryOwner.trackTemporaryBuffer(logits.logitsBuffer);
      throw error;
    }
  } catch (error) {
    primaryFailure = error;
    throw error;
  } finally {
    releaseProfile?.();
    let completionFailure = null;
    if (!recordLayers) {
      // Direct kernels submit outside the carry recorder. A failed layer may
      // leave work in flight even when that recorder has never submitted.
      try { await recorder.device.queue.onSubmittedWorkDone(); }
      catch (error) { completionFailure = error; }
    }
    if (context) releaseSharedAttentionState(context.sharedAttentionState, boundaryOwner);
    if (hidden) {
      if (recordLayers && recorder.getStats().submitted) releaseBuffer(hidden);
      else boundaryOwner.trackTemporaryBuffer(hidden);
    }
    await recorder.abort();
    releaseDirectBuffers(completionFailure !== null);
    if (output) (completionFailure ? discardBuffer : releaseBuffer)(output);
    if (completionFailure) throw new AggregateError(
      primaryFailure ? [primaryFailure, completionFailure] : [completionFailure],
      'Resident direct execution failed to confirm submitted-work completion.'
    );
  }
}

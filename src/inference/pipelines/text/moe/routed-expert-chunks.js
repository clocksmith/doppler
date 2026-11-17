import { getRuntimeConfig } from '../../../../config/runtime.js';
import { acquireBuffer, readBufferSlice, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { getDevice } from '../../../../gpu/device.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { runRouteRemap } from '../../../../gpu/kernels/moe-route.js';
import { runQwenRouteExperts, recordQwenRouteExperts } from './qwen-route-executor.js';

/** @param {import('../../../../loader/experts/routed-expert-cache.js').RoutedExpertWeights} weights
 * @param {GPUBuffer} indices @param {number} numTokens @param {number} topK
 * @param {number[]} ids @param {boolean} allowPrefill @param {string | undefined} phase
 * @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} recorder */
async function remapCompactIndices(weights, indices, numTokens, topK, ids, allowPrefill, phase, recorder) {
  if (!weights.sourceExpertToSlot) return null;
  const sourceNumExperts = weights.sourceNumExperts;
  if (!((phase === 'decode' && numTokens === 1) || (allowPrefill && phase === 'prefill'))
    || !Number.isSafeInteger(sourceNumExperts) || !sourceNumExperts
    || ids.some(id => id >= sourceNumExperts)) {
    throw new Error('Compact expert slots require valid source IDs and an admitted decode or tiered-prefill chunk.');
  }
  return runRouteRemap(indices, weights.sourceExpertToSlot, numTokens * topK,
    sourceNumExperts, weights.numExperts, recorder);
}

// Actual GPU selections, not predictions. The next metadata read precedes the
// current chunk's dispatch, so known-route I/O can overlap that dispatch.
/** @type {typeof import('./routed-expert-chunks.js').executeRoutedExpertChunks} */
export async function executeRoutedExpertChunks(options, loader, policy, signal, recorder = null) {
  if (options.activationDtype !== 'f32') {
    throw new Error('Routed expert loading requires an F32 route executor.');
  }
  if (options.groupedPrefill && (options.phase !== 'prefill' || options.sharedSlot || recorder)) {
    throw new Error('Grouped routed experts require immediate prefill without shared-slot execution.');
  }
  // Shared slots keep their layer-owned shared bank. Only the routed banks and
  // expert IDs are compacted. Multi-token auxiliary gate/residual slicing is
  // deliberately excluded; this path hands over the complete single token.
  if (options.sharedSlot && (options.phase !== 'decode' || options.numTokens !== 1 || options.sharedSlot.residual)) {
    throw new Error('Paged shared slots require single-token decode without an outer residual.');
  }
  if (recorder && (options.phase !== 'decode' || options.numTokens !== 1)) {
    throw new Error('Recorded routed experts require single-token decode.');
  }
  const { numTokens, topK, hiddenSize, indicesBuffer, weightsBuffer, inputTensor, layerIdx } = options;
  if (typeof layerIdx !== 'number' || !Number.isSafeInteger(layerIdx)) throw new Error('Routed expert loading requires an explicit layer.');
  const device = getDevice();
  if (!device) throw new Error('Routed expert loading requires the owning GPU device.');
  const tiered = policy.storageMode === 'tiered-compact';
  signal?.throwIfAborted();
  const resident = tiered ? loader.getResidentRoutedExpertWeights?.(layerIdx) : null;
  if (resident) {
    if (resident.sourceExpertToSlot) throw new Error('Resident routed experts require a complete direct-index bank.');
    return recorder ? recordQwenRouteExperts(recorder, { ...options, weights: resident })
      : runQwenRouteExperts({ ...options, weights: resident });
  }
  const capacity = tiered || (policy.storageMode === 'compact-decode' && options.phase === 'decode')
    ? loader.getRoutedExpertCapacity?.(layerIdx, options.phase) : undefined;
  if (tiered && capacity === undefined) throw new Error('Tiered routed experts require an explicit admitted capacity.');
  if (capacity !== undefined && (!Number.isSafeInteger(capacity) || capacity < topK)) {
    throw new Error('Routed expert capacity cannot fit one complete top-K selection.');
  }
  const chunkTokens = capacity === undefined ? policy.routeChunkTokens
    : Math.min(policy.routeChunkTokens, Math.floor(capacity / topK));
  /** @param {number} start */
  const selections = async start => {
    signal?.throwIfAborted();
    const count = Math.min(chunkTokens, numTokens - start);
    const profile = getRuntimeConfig().shared.debug.profiler.enabled;
    const started = profile ? performance.now() : 0;
    // Append the selection copy to routing's encoder and submit that prefix once.
    // Mapping still settles all earlier bank consumers before cache reuse. The
    // outer owner keeps temporaries until its final submission.
    const data = await readBufferSlice(indicesBuffer, start * topK * 4, count * topK * 4, recorder);
    if (profile) loader.recordRoutedReadback?.(options.phase, performance.now() - started, count * topK * 4);
    signal?.throwIfAborted();
    return [...new Set(new Uint32Array(data))];
  };
  let ids = await selections(0);
  if (numTokens <= chunkTokens) {
    const weights = await loader.prepareRoutedExperts(layerIdx, ids, signal, options.phase);
    signal?.throwIfAborted();
    if (!weights.sourceExpertToSlot) return recorder
      ? recordQwenRouteExperts(recorder, { ...options, weights })
      : runQwenRouteExperts({ ...options, weights });
    const remapped = await remapCompactIndices(weights, indicesBuffer, numTokens, topK, ids, tiered, options.phase, recorder);
    if (!remapped) throw new Error('Compact expert remap was not produced.');
    if (recorder) {
      recorder.trackTemporaryBuffer(remapped);
      return recordQwenRouteExperts(recorder, { ...options, weights, indicesBuffer: remapped });
    }
    let submitted = false;
    try {
      const result = await runQwenRouteExperts({ ...options, weights, indicesBuffer: remapped });
      submitted = true;
      return result;
    } finally {
      try { if (!submitted) await device.queue.onSubmittedWorkDone(); }
      finally { releaseBuffer(remapped); }
    }
  }
  const output = acquireBuffer(numTokens * hiddenSize * 4, undefined, 'routed_prefill_output');
  let completed = false;
  let failure;
  try {
    for (let start = 0; start < numTokens; start += chunkTokens) {
      signal?.throwIfAborted();
      const count = Math.min(chunkTokens, numTokens - start);
      const next = start + count < numTokens ? selections(start + count) : null;
      // A rejection is observed immediately even if weight preparation fails.
      next?.catch(() => undefined);
      const selectedIds = ids;
      const preparing = loader.prepareRoutedExperts(layerIdx, selectedIds, signal, options.phase);
      preparing.catch(() => undefined);
      /** @type {GPUBuffer | null} */
      let input = null, indices = null, routes = null, remapped = null;
      let chunkOutput = null;
      let chunkComplete = false;
      let chunkFailure;
      try {
        input = acquireBuffer(count * hiddenSize * 4, undefined, 'routed_chunk_input');
        indices = acquireBuffer(count * topK * 4, undefined, 'routed_chunk_indices');
        routes = acquireBuffer(count * topK * 4, undefined, 'routed_chunk_weights');
        const copy = device.createCommandEncoder({ label: 'routed_chunk_inputs' });
        copy.copyBufferToBuffer(inputTensor.buffer, start * hiddenSize * 4, input, 0, count * hiddenSize * 4);
        copy.copyBufferToBuffer(indicesBuffer, start * topK * 4, indices, 0, count * topK * 4);
        copy.copyBufferToBuffer(weightsBuffer, start * topK * 4, routes, 0, count * topK * 4);
        device.queue.submit([copy.finish()]);
        const weights = await preparing;
        ids = next ? await next : [];
        signal?.throwIfAborted();
        remapped = await remapCompactIndices(weights, indices, count, topK, selectedIds, tiered, options.phase, null);
        chunkOutput = await runQwenRouteExperts({ ...options, numTokens: count, weights,
          inputTensor: createTensor(input, 'f32', [count, hiddenSize], 'routed_chunk'),
          indicesBuffer: remapped ?? indices, weightsBuffer: routes });
        const combine = device.createCommandEncoder({ label: 'routed_chunk_output' });
        combine.copyBufferToBuffer(chunkOutput.buffer, 0, output, start * hiddenSize * 4, count * hiddenSize * 4);
        device.queue.submit([combine.finish()]);
        chunkComplete = true;
      } catch (error) {
        chunkFailure = error;
        throw error;
      } finally {
        await Promise.allSettled([preparing, next]);
        const cleanup = [];
        if (!chunkComplete) {
          try { await device.queue.onSubmittedWorkDone(); } catch (error) { cleanup.push(error); }
        }
        for (const buffer of [chunkOutput?.buffer, remapped, input, indices, routes]) {
          if (buffer) { try { releaseBuffer(buffer); } catch (error) { cleanup.push(error); } }
        }
        if (cleanup.length) throw new AggregateError([chunkFailure, ...cleanup].filter(Boolean), 'Routed chunk cleanup failed.');
      }
    }
    completed = true;
    return createTensor(output, 'f32', [numTokens, hiddenSize], 'routed_prefill_output');
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    if (!completed) {
      const cleanup = [];
      try { await device.queue.onSubmittedWorkDone(); } catch (error) { cleanup.push(error); }
      try { releaseBuffer(output); } catch (error) { cleanup.push(error); }
      if (cleanup.length) throw new AggregateError([failure, ...cleanup].filter(Boolean), 'Routed output cleanup failed.');
    }
  }
}

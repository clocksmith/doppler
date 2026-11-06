import { acquireBuffer, readBufferSlice, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { getDevice } from '../../../../gpu/device.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { runQwenRouteExperts } from './qwen-route-executor.js';

// Actual GPU selections, not predictions. The next metadata read precedes the
// current chunk's dispatch, so known-route I/O can overlap that dispatch.
/** @type {typeof import('./routed-expert-chunks.js').executeRoutedExpertChunks} */
export async function executeRoutedExpertChunks(options, loader, policy, signal) {
  if (options.sharedSlot || options.groupedPrefill || options.activationDtype !== 'f32') {
    throw new Error('Routed expert loading requires the ordinary immediate F32 route executor.');
  }
  const { numTokens, topK, hiddenSize, indicesBuffer, weightsBuffer, inputTensor, layerIdx } = options;
  if (typeof layerIdx !== 'number' || !Number.isSafeInteger(layerIdx)) throw new Error('Routed expert loading requires an explicit layer.');
  const device = getDevice();
  if (!device) throw new Error('Routed expert loading requires the owning GPU device.');
  /** @param {number} start */
  const selections = async start => {
    signal?.throwIfAborted();
    const count = Math.min(policy.routeChunkTokens, numTokens - start);
    const data = await readBufferSlice(indicesBuffer, start * topK * 4, count * topK * 4);
    signal?.throwIfAborted();
    return [...new Set(new Uint32Array(data))];
  };
  let ids = await selections(0);
  if (numTokens <= policy.routeChunkTokens) {
    const weights = await loader.prepareRoutedExperts(layerIdx, ids, signal);
    signal?.throwIfAborted();
    return runQwenRouteExperts({ ...options, weights });
  }
  const output = acquireBuffer(numTokens * hiddenSize * 4, undefined, 'routed_prefill_output');
  let completed = false;
  let failure;
  try {
    for (let start = 0; start < numTokens; start += policy.routeChunkTokens) {
      signal?.throwIfAborted();
      const count = Math.min(policy.routeChunkTokens, numTokens - start);
      const next = start + count < numTokens ? selections(start + count) : null;
      // A rejection is observed immediately even if weight preparation fails.
      next?.catch(() => undefined);
      const preparing = loader.prepareRoutedExperts(layerIdx, ids, signal);
      preparing.catch(() => undefined);
      /** @type {GPUBuffer | null} */
      let input = null, indices = null, routes = null;
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
        chunkOutput = await runQwenRouteExperts({ ...options, numTokens: count, weights,
          inputTensor: createTensor(input, 'f32', [count, hiddenSize], 'routed_chunk'),
          indicesBuffer: indices, weightsBuffer: routes });
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
        for (const buffer of [chunkOutput?.buffer, input, indices, routes]) {
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

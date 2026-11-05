import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  recordQsaBlockKeys,
  recordQsaBlockScores,
  recordQsaPrepare,
  recordQsaSelect,
  runQsaBlockKeys,
  runQsaBlockScores,
  runQsaPrepare,
  runQsaSelect,
} from '../../../../gpu/kernels/qsa-indexer.js';

// Owner of the qwen4_exp QSA indexer state of every indexed layer: raw
// (pre-norm, pre-RoPE) indexer keys and pooled block keys.
//
// Support boundary: positions are the cache rows of one unpadded sequence. An
// append must start at the layer's current length; padded, masked or
// non-contiguous positions are not admitted. Every forward call appends before
// it selects, including dense-shortcut calls.
//
// Lifecycle: truncate, reset and restore advance an epoch. An append or select
// that observes a different epoch (or destruction) after any await fails
// without publishing state. Destruction waits for pending operations and
// submitted GPU work before releasing buffers; callers must not destroy while
// they still hold unsubmitted recorders that reference the cache.

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[QsaIndexerCache] ${message}`);
}

/** @param {unknown} target @returns {target is import('../../../../gpu/command-recorder.js').CommandRecorder} */
function isRecorder(target) {
  return target !== null && typeof (/** @type {any} */ (target))?.beginComputePass === 'function';
}

/** @param {import('../../../../gpu/command-recorder.js').CommandRecorder | null} target @param {GPUBuffer} buffer */
function releaseOutput(target, buffer) {
  if (isRecorder(target)) target.trackTemporaryBuffer(buffer);
  else releaseBuffer(buffer);
}

export class QsaIndexerCache {
  /** @type {GPUBuffer[]} */
  #owned = [];
  #pending = 0;
  /** @type {Promise<void> | null} */
  #destroying = null;
  /** @type {(() => void) | null} */
  #settled = null;

  /** @param {import('./qsa-cache.js').QsaIndexerCacheOptions} options */
  constructor({ contract, capacity }) {
    if (!contract || contract.semantics !== 'qwen4_exp_qsa') fail('a resolved qwen4_exp_qsa contract is required.');
    if (!Number.isSafeInteger(capacity) || capacity < 1) fail('capacity must be a positive token count.');
    if (capacity % contract.compressRatio !== 0) fail(`capacity ${capacity} must be a multiple of compressRatio.`);
    this.contract = contract;
    this.capacity = capacity;
    this.epoch = 0;
    this.destroyed = false;
    /** @type {Map<number, { rawKeys: GPUBuffer, blockKeys: GPUBuffer, length: number, reservedEnd: number }>} */
    this.layers = new Map();
    const rowBytes = contract.headDim * 4;
    const acquire = (/** @type {number} */ bytes, /** @type {string} */ label) => {
      const buffer = acquireBuffer(bytes, undefined, label);
      this.#owned.push(buffer);
      return buffer;
    };
    try {
      for (const layerIdx of contract.layers) {
        const rawKeys = acquire(capacity * rowBytes, `qsa_raw_keys_L${layerIdx}`);
        const blockKeys = acquire((capacity / contract.compressRatio) * rowBytes, `qsa_block_keys_L${layerIdx}`);
        this.layers.set(layerIdx, { rawKeys, blockKeys, length: 0, reservedEnd: 0 });
      }
    } catch (error) {
      this.destroyed = true;
      this.#releaseOwned();
      throw error;
    }
  }

  #releaseOwned() {
    for (const buffer of this.#owned.splice(0)) releaseBuffer(buffer);
    this.layers.clear();
  }

  /** @param {number} layerIdx */
  #layer(layerIdx) {
    if (this.destroyed) fail('the cache has been destroyed.');
    const layer = this.layers.get(layerIdx);
    if (!layer) fail(`layer ${layerIdx} is not an indexed layer.`);
    return layer;
  }

  /** @param {number} epoch @param {string} operation */
  #assertCurrent(epoch, operation) {
    if (this.destroyed) fail(`the cache was destroyed during ${operation}.`);
    if (this.epoch !== epoch) fail(`the cache was truncated, reset or restored during ${operation}.`);
  }

  /** @template T @param {() => Promise<T>} body @returns {Promise<T>} */
  async #track(body) {
    this.#pending += 1;
    try {
      return await body();
    } finally {
      this.#pending -= 1;
      if (this.#pending === 0) this.#settled?.();
    }
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['length']} */
  length(layerIdx) {
    return this.#layer(layerIdx).length;
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['buffers']} */
  buffers(layerIdx) {
    const { rawKeys, blockKeys } = this.#layer(layerIdx);
    return { rawKeys, blockKeys };
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['append']} */
  // Async so precondition failures reject; the reservation still happens synchronously at call time.
  async append(target, layerIdx, { projection, qNormWeight, kNormWeight, rope, startPos, numTokens }) {
    const layer = this.#layer(layerIdx);
    if (startPos !== layer.length || layer.reservedEnd !== layer.length) {
      fail(`layer ${layerIdx} append at position ${startPos} but holds ${layer.length} rows`
        + `${layer.reservedEnd !== layer.length ? ' with another append pending' : ''}; QSA positions must be contiguous cache rows.`);
    }
    if (!Number.isSafeInteger(numTokens) || numTokens < 1 || startPos + numTokens > this.capacity) {
      fail(`append of ${numTokens} rows at ${startPos} exceeds capacity ${this.capacity}.`);
    }
    const epoch = this.epoch;
    layer.reservedEnd = startPos + numTokens;
    return this.#track(async () => {
      const { queryHeads, headDim, rotaryDim, normEps: eps, compressRatio } = this.contract;
      const common = { numTokens, startPos, queryHeads, headDim, rotaryDim, eps, capacity: this.capacity };
      /** @type {import('../../../../gpu/tensor.js').Tensor | null} */
      let query = null;
      try {
        query = isRecorder(target)
          ? await recordQsaPrepare(target, projection, qNormWeight, rope, layer.rawKeys, common)
          : await runQsaPrepare(projection, qNormWeight, rope, layer.rawKeys, common);
        this.#assertCurrent(epoch, 'append');
        const firstBlock = Math.floor(startPos / compressRatio);
        const blockOptions = {
          firstBlock,
          blockCount: Math.floor((startPos + numTokens) / compressRatio) - firstBlock,
          headDim, rotaryDim, compressRatio, eps, capacity: this.capacity,
        };
        if (isRecorder(target)) await recordQsaBlockKeys(target, layer.rawKeys, kNormWeight, rope, layer.blockKeys, blockOptions);
        else await runQsaBlockKeys(layer.rawKeys, kNormWeight, rope, layer.blockKeys, blockOptions);
        this.#assertCurrent(epoch, 'append');
        layer.length = startPos + numTokens;
        return query;
      } catch (error) {
        if (query) releaseOutput(target, query.buffer);
        if (!this.destroyed && this.epoch === epoch) layer.reservedEnd = layer.length;
        throw error;
      }
    });
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['select']} */
  async select(target, layerIdx, query, { startPos, numTokens }) {
    const layer = this.#layer(layerIdx);
    if (layer.length !== startPos + numTokens || layer.reservedEnd !== layer.length) {
      fail(`layer ${layerIdx} selects rows [${startPos}, ${startPos + numTokens}) but holds ${layer.length}; append first.`);
    }
    const epoch = this.epoch;
    return this.#track(async () => {
      const { queryHeads, headDim, compressRatio, blockTopK, denseVisibleLimit } = this.contract;
      const geometry = { numTokens, startPos, compressRatio, blockTopK, denseVisibleLimit };
      const scoreOptions = { ...geometry, queryHeads, headDim, scoreDivisor: Math.fround(Math.sqrt(headDim)) };
      /** @type {import('../../../../gpu/tensor.js').Tensor | null} */
      let scores = null;
      /** @type {import('../../../../gpu/kernels/qsa-indexer.js').QsaSelection | null} */
      let selection = null;
      try {
        scores = isRecorder(target)
          ? await recordQsaBlockScores(target, query, layer.blockKeys, scoreOptions)
          : await runQsaBlockScores(query, layer.blockKeys, scoreOptions);
        this.#assertCurrent(epoch, 'select');
        selection = isRecorder(target) ? await recordQsaSelect(target, scores, geometry) : await runQsaSelect(scores, geometry);
        this.#assertCurrent(epoch, 'select');
        return { ...selection, scores };
      } catch (error) {
        if (scores) releaseOutput(target, scores.buffer);
        if (selection) {
          releaseOutput(target, selection.selection);
          releaseOutput(target, selection.meta);
        }
        throw error;
      }
    });
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['truncate']} */
  truncate(length) {
    if (this.destroyed) fail('the cache has been destroyed.');
    if (!Number.isSafeInteger(length) || length < 0) fail('truncate length must be a non-negative integer.');
    for (const layer of this.layers.values()) {
      if (length < layer.length) layer.length = length;
      layer.reservedEnd = layer.length;
    }
    this.epoch += 1;
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['reset']} */
  reset() {
    this.truncate(0);
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['checkpoint']} */
  checkpoint() {
    if (this.destroyed) fail('the cache has been destroyed.');
    if ([...this.layers.values()].some((layer) => layer.reservedEnd !== layer.length)) fail('cannot checkpoint while an append is pending.');
    return Object.freeze({
      epoch: this.epoch,
      lengths: Object.freeze(Object.fromEntries([...this.layers].map(([layerIdx, layer]) => [layerIdx, layer.length]))),
    });
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['restore']} */
  restore(checkpoint) {
    if (this.destroyed) fail('the cache has been destroyed.');
    if (checkpoint?.epoch !== this.epoch) {
      fail('checkpoint predates a truncate, reset or restore; rows it covered may have been overwritten.');
    }
    for (const [layerIdx, layer] of this.layers) {
      const length = checkpoint.lengths[layerIdx];
      if (!Number.isSafeInteger(length) || length > layer.length) fail(`checkpoint length for layer ${layerIdx} is not restorable.`);
    }
    for (const [layerIdx, layer] of this.layers) {
      layer.length = checkpoint.lengths[layerIdx];
      layer.reservedEnd = layer.length;
    }
    this.epoch += 1;
  }

  /** @type {import('./qsa-cache.js').QsaIndexerCache['destroy']} */
  destroy() {
    if (this.#destroying) return this.#destroying;
    this.destroyed = true;
    this.#destroying = (async () => {
      if (this.#pending > 0) {
        await new Promise((resolve) => { this.#settled = () => resolve(undefined); });
      }
      await getDevice()?.queue.onSubmittedWorkDone();
      this.#releaseOwned();
    })();
    return this.#destroying;
  }
}

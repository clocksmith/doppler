import { getDevice } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBufferInstance } from '../../../../gpu/weight-buffer.js';
import { bindNgramTables } from '../../../../loader/ngram-table-bindings.js';
import { QsaIndexerCache } from '../indexed-attention/qsa-cache.js';
import { NgramTokenHistory } from '../ngram/ngram-addresses.js';
import { NgramTableReader } from '../ngram/ngram-table.js';
import { PleConvState, runPleLayer } from '../ngram/ple-layer.js';
import { releaseOrTrack } from '../ops.js';

/** @param {string} message @returns {never} */
function fail(message) { throw new Error(`[HyperConnectionSession] ${message}`); }

export class HyperConnectionSession {
  #epoch = 0;
  #closed = false;
  #invalid = false;
  /** @type {Promise<void> | null} */
  #destroyPromise = null;
  #abort = new AbortController();
  /** @type {Set<Promise<unknown>>} */
  #pending = new Set();

  /** @param {import('./session.js').HyperConnectionSessionInput} input */
  constructor({ hyper, ngram, qsa, layers }) {
    this.hyper = hyper;
    this.ngram = ngram;
    this.qsa = qsa;
    this.layers = layers;
  }

  /** @template T @param {() => Promise<T>} action */
  async #operation(action) {
    if (this.#closed || this.#invalid) fail('session is closed or requires a successful reset.');
    const pending = action();
    this.#pending.add(pending);
    try { return await pending; } finally { this.#pending.delete(pending); }
  }

  /** @type {import('./session.js').HyperConnectionSession['runPle']} */
  async runPle(layerIdx, streams, weights, context, project) {
    return this.#operation(async () => {
      const layer = this.layers.get(layerIdx);
      const tokens = context.currentTokenIds;
      if (!layer || !this.ngram || !weights || !tokens || tokens.length !== streams.shape[0]
        || layer.history.length !== context.currentSeqLen) fail(`PLE layer ${layerIdx} needs contiguous, known, unpadded token IDs.`);
      const signal = context.signal ? AbortSignal.any([context.signal, this.#abort.signal]) : this.#abort.signal;
      signal.throwIfAborted();
      const addresses = layer.history.append(tokens);
      const rows = await layer.table.gather(addresses, { signal });
      signal.throwIfAborted();
      const device = getDevice();
      if (!device) fail('GPU device is unavailable.');
      const buffer = acquireBuffer(rows.byteLength, undefined, 'ngram_rows');
      try {
        device.queue.writeBuffer(buffer, 0, rows);
        const result = await runPleLayer(context.recorder ?? null, { streams, rows: buffer, weights,
          state: layer.conv, ngram: this.ngram, hyper: this.hyper, numTokens: tokens.length, project });
        releaseOrTrack(context.recorder, result.gateInput.buffer);
        return result.streams;
      } finally {
        releaseOrTrack(context.recorder, buffer);
      }
    });
  }

  /** @type {import('./session.js').HyperConnectionSession['prepareQsa']} */
  async prepareQsa(layerIdx, input, weights, context, project) {
    return this.#operation(async () => {
      const cache = this.qsa;
      if (!cache || !isGpuBufferInstance(context.ropeFreqsCos) || !isGpuBufferInstance(context.ropeFreqsSin)) {
        fail('QSA requires an indexer cache and explicit f32 rotary tables on the GPU.');
      }
      context.signal?.throwIfAborted();
      const { queryHeads, keyHeads, headDim } = cache.contract;
      const projection = await project('qsa_index_qk', input, weights.projection,
        (queryHeads + keyHeads) * headDim, this.hyper.streamWidth);
      let query = null;
      try {
        query = await cache.append(context.recorder ?? null, layerIdx, { projection,
          qNormWeight: weights.qNorm, kNormWeight: weights.kNorm,
          rope: { cos: context.ropeFreqsCos, sin: context.ropeFreqsSin },
          startPos: context.currentSeqLen, numTokens: input.shape[0] });
        const call = { startPos: context.currentSeqLen, numTokens: input.shape[0] };
        const selection = await cache.select(context.recorder ?? null, layerIdx, query, call);
        return { ...selection, layerIdx, numTokens: input.shape[0], visibleRows: call.startPos + call.numTokens };
      } finally {
        releaseOrTrack(context.recorder, projection.buffer);
        if (query) releaseOrTrack(context.recorder, query.buffer);
      }
    });
  }

  /** @type {import('./session.js').HyperConnectionSession['releaseQsa']} */
  releaseQsa(selection, recorder) {
    releaseOrTrack(recorder ?? undefined, selection.selection);
    releaseOrTrack(recorder ?? undefined, selection.meta);
    if (selection.scores) releaseOrTrack(recorder ?? undefined, selection.scores.buffer);
  }

  reset() {
    if (this.#closed || this.#pending.size) fail('cannot reset a closed or active session.');
    const failures = [];
    try { this.qsa?.reset(); } catch (error) { failures.push(error); }
    for (const layer of this.layers.values()) {
      try { layer.history.reset(); } catch (error) { failures.push(error); }
      try { layer.conv.reset(); } catch (error) { failures.push(error); }
    }
    this.#epoch += 1;
    this.#invalid = failures.length > 0;
    if (failures.length) throw new AggregateError(failures, 'Composite reset failed; reset successfully or reload before reuse.');
  }

  /** @param {number} length */
  assertLength(length) {
    for (const index of this.qsa?.contract.layers ?? []) {
      if (this.qsa?.length(index) !== length) fail(`QSA layer ${index} is not at committed position ${length}.`);
    }
    for (const [index, layer] of this.layers) {
      if (layer.history.length !== length) fail(`PLE layer ${index} is not at committed position ${length}.`);
    }
  }

  /** @type {import('./session.js').HyperConnectionSession['checkpoint']} */
  async checkpoint() {
    if (this.#closed || this.#pending.size) fail('cannot checkpoint a closed or active session.');
    return this.#operation(async () => {
      const epoch = this.#epoch;
      const device = getDevice();
      if (!device) fail('GPU device is unavailable.');
      /** @type {Map<number, import('./session.js').PleCheckpoint>} */
      const layers = new Map();
      try {
        const encoder = device.createCommandEncoder();
        for (const [index, layer] of this.layers) {
          const conv = acquireBuffer(layer.conv.buffer.size, undefined, 'ple_checkpoint');
          layers.set(index, { history: layer.history.checkpoint(), conv });
          encoder.copyBufferToBuffer(layer.conv.buffer, 0, conv, 0, layer.conv.buffer.size);
        }
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        if (this.#closed || epoch !== this.#epoch) fail('checkpoint was invalidated during preparation.');
        return { epoch, qsa: this.qsa?.checkpoint() ?? null, layers };
      } catch (error) {
        for (const layer of layers.values()) releaseBuffer(layer.conv);
        throw error;
      }
    });
  }

  /** @type {import('./session.js').HyperConnectionSession['restore']} */
  async restore(snapshot) {
    if (this.#closed || this.#pending.size || snapshot.epoch !== this.#epoch) fail('stale or active checkpoint restore.');
    return this.#operation(async () => {
      const device = getDevice();
      if (!device) fail('GPU device is unavailable.');
      const encoder = device.createCommandEncoder();
      for (const [index, saved] of snapshot.layers) {
        const layer = this.layers.get(index);
        if (!layer) fail(`checkpoint layer ${index} is missing.`);
        encoder.copyBufferToBuffer(saved.conv, 0, layer.conv.buffer, 0, layer.conv.buffer.size);
        layer.history.restore(saved.history);
      }
      if (snapshot.qsa) this.qsa?.restore(snapshot.qsa);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      this.#epoch += 1;
    });
  }

  /** @type {import('./session.js').HyperConnectionSession['releaseCheckpoint']} */
  releaseCheckpoint(snapshot) {
    const failures = [];
    for (const layer of snapshot.layers.values()) {
      try { releaseBuffer(layer.conv); } catch (error) { failures.push(error); }
    }
    snapshot.layers.clear();
    if (failures.length) throw new AggregateError(failures, 'PLE checkpoint cleanup failed.');
  }

  destroy() {
    if (!this.#destroyPromise) {
      this.#closed = true;
      this.#abort.abort(new DOMException('Hyper-connection session closed.', 'AbortError'));
      this.#destroyPromise = this.#dispose();
    }
    return this.#destroyPromise;
  }

  async #dispose() {
    await Promise.allSettled([...this.#pending]);
    const results = await Promise.allSettled([this.qsa?.destroy(), ...[...this.layers.values()].map(layer => layer.conv.destroy())]);
    for (const layer of this.layers.values()) layer.table.clear();
    this.layers.clear();
    const failures = results.filter(result => result.status === 'rejected').map(result => result.reason);
    if (failures.length) throw new AggregateError(failures, 'Hyper-connection state cleanup failed.');
  }
}

/** @type {typeof import('./session.js').createHyperConnectionSession} */
export async function createHyperConnectionSession(config, loader, capacity) {
  if (!config.hyperConnections) return null;
  const hyper = config.hyperConnections;
  const indexed = config.indexedAttention;
  const ngram = config.ngramEmbedding;
  const bindings = ngram && config.ngramEmbeddingSession ? bindNgramTables(loader, ngram, config.ngramEmbeddingSession) : new Map();
  /** @type {import('./session.js').HyperConnectionSessionInput['layers']} */
  const layers = new Map();
  const qsa = indexed ? new QsaIndexerCache({ contract: indexed, capacity: Math.ceil(capacity / indexed.compressRatio) * indexed.compressRatio }) : null;
  const session = new HyperConnectionSession({ hyper, ngram, qsa, layers });
  try {
    if (ngram) for (const layer of ngram.layers) {
      const options = bindings.get(layer.decoderLayerIndex);
      if (!options) fail(`missing n-gram table binding for layer ${layer.decoderLayerIndex}.`);
      const table = new NgramTableReader(options);
      const history = new NgramTokenHistory(ngram, layer);
      const conv = new PleConvState({ channels: hyper.streamCount * hyper.streamWidth, stateLength: ngram.convStateLength });
      layers.set(layer.decoderLayerIndex, { table, history, conv });
    }
    return session;
  } catch (error) {
    try { await session.destroy(); } catch (cleanup) { throw new AggregateError([error, cleanup], 'Hyper-connection initialization failed.', { cause: error }); }
    throw error;
  }
}

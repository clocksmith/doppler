import { isWeightBuffer, isSplitWeightBuffer, isGpuBufferInstance, getWeightDtype } from '../../../gpu/weight-buffer.js';
import { readBufferSlice } from '../../../memory/buffer-pool.js';
import { dequantizeQ4KMRowWise } from '../../../formats/q4k-dequant.js';
import { QK_K, QK4_K_BLOCK_SIZE } from '../../../config/schema/quantization.schema.js';
import { decodeSnapshotData } from '../../../debug/tensor.js';

/** @type {typeof import('./activity-weights.js').captureActivityWeights} */
export async function captureActivityWeights(weights, limits, signal, tiedReference = false) {
  const { weightBlockValues, weightReadValues, weightRowGroupSize, weightRawRows, weightMaxBytes } = limits;
  if (![weightBlockValues, weightReadValues, weightRowGroupSize, weightRawRows, weightMaxBytes].every(n => Number.isSafeInteger(n) && n > 0)) {
    throw new Error('Weight observation needs explicit tile and memory budgets.');
  }
  /** @type {import('../../../debug/activity-trace.js').ActivityWeight[]} */
  const result = [];
  const seen = new Set();
  let retained = 0, haveContribution = false;
  /** @param {string} id @param {unknown} weight @param {number | null} layer @param {string} role */
  async function visit(id, weight, layer, role) {
    if (!weight || seen.has(weight)) return;
    seen.add(weight);
    if (!isWeightBuffer(weight) && !isSplitWeightBuffer(weight) && !isGpuBufferInstance(weight)) {
      if (typeof weight === 'object' && !ArrayBuffer.isView(weight)) {
        for (const [name, child] of Object.entries(weight)) await visit(`${id}/${name}`, child, layer, name);
      }
      return;
    }
    signal?.throwIfAborted();
    const shaped = isWeightBuffer(weight) || isSplitWeightBuffer(weight);
    const shape = shaped ? [...weight.shape] : [];
    const dtype = getWeightDtype(weight) ?? 'unknown';
    const byteLength = isWeightBuffer(weight) ? weight.buffer.size : isGpuBufferInstance(weight) ? weight.size
      : weight.sections.reduce((n, section) => n + section.buffer.size, 0);
    /** @type {import('../../../debug/activity-trace.js').ActivityWeight} */
    const entry = { id, layer, role, shape, dtype, layout: shaped ? weight.layout : 'unknown',
      byteLength, resident: true, status: 'unavailable', reason: null,
      blockSize: weightBlockValues, rms: [], raw: [], rawOffset: 0, columnEnergy: [], rowGroupSize: weightRowGroupSize };
    result.push(entry);
    if (!isWeightBuffer(weight) || weight.layout !== 'row' || shape.length !== 2 || !['q4k', 'f16', 'f32'].includes(dtype)) {
      entry.reason = 'This loaded weight has no supported row-major observation descriptor.'; return;
    }
    const [rows, cols] = shape;
    const total = rows * cols;
    const referenceWindow = tiedReference && (id === 'embed' || id === 'lm_head');
    const contribute = !haveContribution && (role === 'down' || role === 'ffnDown');
    const stored = Math.ceil(total / weightBlockValues) + (contribute ? Math.ceil(rows / weightRowGroupSize) * cols : 0) + ((contribute || referenceWindow) ? Math.min(rows, weightRawRows) * cols : 0);
    if (retained + stored * 8 > weightMaxBytes) { entry.reason = 'Weight recording budget reached.'; return; }
    const rowBytes = dtype === 'q4k' ? Math.ceil(cols / QK_K) * QK4_K_BLOCK_SIZE : cols * (dtype === 'f16' ? 2 : 4);
    if (rowBytes * rows > weight.buffer.size) { entry.reason = 'Weight descriptor exceeds storage.'; return; }
    entry.rms = new Array(Math.ceil(total / weightBlockValues)).fill(0);
    if (contribute) entry.columnEnergy = new Array(Math.ceil(rows / weightRowGroupSize) * cols).fill(0);
    const rowsPerRead = Math.max(1, Math.floor(weightReadValues / cols));
    try {
      for (let row = 0; row < rows; row += rowsPerRead) {
        signal?.throwIfAborted();
        const count = Math.min(rowsPerRead, rows - row);
        const start = row * rowBytes, size = count * rowBytes;
        const aligned = Math.floor(start / 4) * 4, prefix = start - aligned;
        const raw = await readBufferSlice(weight.buffer, aligned, Math.ceil((prefix + size) / 4) * 4);
        const data = raw.slice(prefix, prefix + size);
        const values = dtype === 'q4k' ? dequantizeQ4KMRowWise(new Uint8Array(data), [count, cols]) : decodeSnapshotData(data, dtype);
        for (let i = 0; i < values.length; i++) {
          const absolute = row * cols + i, v = values[i];
          entry.rms[Math.floor(absolute / weightBlockValues)] += v * v;
          if ((contribute || referenceWindow) && absolute < weightRawRows * cols) entry.raw.push(v);
          if (contribute) {
            entry.columnEnergy[Math.floor(Math.floor(absolute / cols) / weightRowGroupSize) * cols + absolute % cols] += v * v;
          }
        }
      }
      entry.rms = entry.rms.map((sum, i) => Math.sqrt(sum / Math.min(weightBlockValues, total - i * weightBlockValues)));
      entry.status = 'captured'; retained += stored * 8;
      if (contribute) haveContribution = true;
    } catch (error) {
      signal?.throwIfAborted();
      entry.status = 'failed'; entry.reason = String(error); entry.rms = []; entry.raw = []; entry.columnEnergy = [];
    }
  }
  let layer = -1;
  for (const [name, weight] of weights) {
    // The loaded map's layer container order is declared by initialization;
    // the explicit layer key supplies identity, never tensor geometry.
    const match = /^layer_(\d+)$/.exec(name);
    layer = match ? Number(match[1]) : -1;
    await visit(name, weight, layer < 0 ? null : layer, name);
  }
  const head = result.find(w => w.id === 'lm_head'), reference = result.find(w => w.id === 'embed');
  if (tiedReference && head?.raw.length && reference?.raw.length === head.raw.length
    && JSON.stringify(head.shape) === JSON.stringify(reference.shape)) {
    head.referenceId = reference.id;
    head.referenceDtype = reference.dtype;
    head.error = head.raw.map((v, i) => v - reference.raw[i]);
  }
  return result;
}

/** @type {typeof import('./activity-weights.js').prepareActivityRecording} */
export async function prepareActivityRecording(state, opts) {
  if (opts.__activationTrace?.activity) {
    const activity = opts.__activationTrace.activity;
    const config = state.modelConfig;
    if (!config) throw new Error('Activity capture requires a loaded model configuration.');
    activity.setReader(readBufferSlice);
    activity.configure({ numLayers: config.numLayers, hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize, numHeads: config.numHeads, numKVHeads: config.numKVHeads,
      headDim: config.headDim, layerTypes: config.layerTypes, vocabSize: config.vocabSize, useTiedEmbeddings: config.useTiedEmbeddings });
    activity.setWeights(await captureActivityWeights(state.weights, /** @type {import('../../../debug/activity-trace.js').ActivityLimits} */ (opts.__activationTrace.snapshot('recording').capturePolicy), opts.signal, config.useTiedEmbeddings === true));
    activity.unavailable('quantization-error', 'No pinned source-reference tensors were supplied with this recording.');
    activity.unavailable('expert-routing', 'No expert-routing boundary was captured for this execution.');
  }
}

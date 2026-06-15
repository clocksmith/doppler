import assert from 'node:assert/strict';

import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { createTensor } from '../../src/gpu/tensor.js';
import { createSplitWeightBuffer, createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';
import { runSoftEmbeddingLogitsF16, runSoftEmbeddingSplitF16 } from '../../src/gpu/kernels/soft-embedding.js';
import { f32ToF16Array, f16ToF32Bits } from '../../src/inference/kv-cache/types.js';

const gpu = await probeNodeGPU();
if (!gpu.ready) {
  console.log(`soft-embedding-split-f16-parity.test: skipped (${gpu.reason})`);
  process.exit(0);
}

const device = getDevice();

function makeStorageBuffer(data, label) {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buffer = device.createBuffer({
    label,
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function computeSoftEmbeddingReference(logits, embeddingRows, rows, cols, hidden, temperature) {
  const expected = new Float32Array(rows * hidden);
  for (let token = 0; token < rows; token += 1) {
    let max = -Infinity;
    for (let vocab = 0; vocab < cols; vocab += 1) {
      max = Math.max(max, logits[(token * cols) + vocab] / temperature);
    }
    let sum = 0;
    const probs = new Float32Array(cols);
    for (let vocab = 0; vocab < cols; vocab += 1) {
      const probability = Math.exp((logits[(token * cols) + vocab] / temperature) - max);
      probs[vocab] = probability;
      sum += probability;
    }
    for (let vocab = 0; vocab < cols; vocab += 1) {
      const probability = probs[vocab] / sum;
      for (let h = 0; h < hidden; h += 1) {
        expected[(token * hidden) + h] += probability * embeddingRows[(vocab * hidden) + h];
      }
    }
  }
  return expected;
}

const numTokens = 2;
const vocabSize = 6;
const hiddenSize = 4;

const probabilities = new Float32Array([
  0.10, 0.20, 0.05, 0.15, 0.30, 0.20,
  0.25, 0.05, 0.20, 0.10, 0.15, 0.25,
]);
const embeddingF32 = new Float32Array([
  0.50, -0.25, 0.75, 0.10,
  -0.40, 0.30, 0.20, -0.60,
  0.90, -0.10, -0.20, 0.40,
  -0.15, 0.80, 0.35, -0.45,
  0.25, -0.55, 0.65, 0.05,
  -0.70, 0.15, -0.30, 0.95,
]);
const embeddingF16 = f32ToF16Array(embeddingF32);
const embeddingRounded = new Float32Array(embeddingF16.length);
for (let i = 0; i < embeddingF16.length; i += 1) {
  embeddingRounded[i] = f16ToF32Bits(embeddingF16[i]);
}

const expected = new Float32Array(numTokens * hiddenSize);
for (let token = 0; token < numTokens; token += 1) {
  for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
    let sum = 0;
    for (let vocab = 0; vocab < vocabSize; vocab += 1) {
      sum += probabilities[token * vocabSize + vocab] * embeddingRounded[vocab * hiddenSize + hidden];
    }
    expected[token * hiddenSize + hidden] = sum;
  }
}

const softmaxBuffer = makeStorageBuffer(probabilities, 'soft_embedding_softmax');
const section0Data = embeddingF16.subarray(0, 3 * hiddenSize);
const section1Data = embeddingF16.subarray(3 * hiddenSize);
const section0Buffer = makeStorageBuffer(section0Data, 'soft_embedding_section0');
const section1Buffer = makeStorageBuffer(section1Data, 'soft_embedding_section1');
const splitEmbedding = createSplitWeightBuffer(
  [
    { buffer: section0Buffer, rowStart: 0, rowCount: 3 },
    { buffer: section1Buffer, rowStart: 3, rowCount: 3 },
  ],
  'f16',
  'row',
  [vocabSize, hiddenSize],
  'embed_tokens'
);

let output = null;
let logitsOutput = null;
try {
  const softmaxTensor = createTensor(
    softmaxBuffer,
    'f32',
    [numTokens, vocabSize],
    'soft_embedding_softmax'
  );
  output = await runSoftEmbeddingSplitF16(
    softmaxTensor,
    splitEmbedding,
    numTokens,
    hiddenSize,
    vocabSize
  );
  const actual = new Float32Array(await readBuffer(output.buffer, expected.byteLength));

  let maxAbsDiff = 0;
  for (let i = 0; i < expected.length; i += 1) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual[i] - expected[i]));
  }
  console.log(`soft-embedding-split-f16-parity: max_abs_diff=${maxAbsDiff.toExponential(3)}`);
  assert.ok(maxAbsDiff < 1e-5, `split soft embedding diverged: max_abs_diff=${maxAbsDiff}`);

  const logitsVocabSize = 96;
  const logitsHiddenSize = 4;
  const logitsNumTokens = 2;
  const temperature = 0.7;
  const logits = new Float32Array(logitsNumTokens * logitsVocabSize);
  for (let i = 0; i < logits.length; i += 1) {
    logits[i] = Math.sin(i * 0.173) * 4.0 + Math.cos(i * 0.037) * 1.5;
  }
  const logitsEmbeddingF32 = new Float32Array(logitsVocabSize * logitsHiddenSize);
  for (let i = 0; i < logitsEmbeddingF32.length; i += 1) {
    logitsEmbeddingF32[i] = Math.sin(i * 0.071) * 0.75 + Math.cos(i * 0.113) * 0.25;
  }
  const logitsEmbeddingF16 = f32ToF16Array(logitsEmbeddingF32);
  const logitsEmbeddingRounded = new Float32Array(logitsEmbeddingF16.length);
  for (let i = 0; i < logitsEmbeddingF16.length; i += 1) {
    logitsEmbeddingRounded[i] = f16ToF32Bits(logitsEmbeddingF16[i]);
  }
  const logitsExpected = computeSoftEmbeddingReference(
    logits,
    logitsEmbeddingRounded,
    logitsNumTokens,
    logitsVocabSize,
    logitsHiddenSize,
    temperature
  );
  const logitsBuffer = makeStorageBuffer(logits, 'soft_embedding_logits');
  const logitsEmbeddingBuffer = makeStorageBuffer(logitsEmbeddingF16, 'soft_embedding_logits_embedding');
  try {
    logitsOutput = await runSoftEmbeddingLogitsF16(
      createTensor(logitsBuffer, 'f32', [logitsNumTokens, logitsVocabSize], 'soft_embedding_logits'),
      createWeightBuffer(
        logitsEmbeddingBuffer,
        'f16',
        'row',
        [logitsVocabSize, logitsHiddenSize],
        'embed_tokens'
      ),
      logitsNumTokens,
      logitsHiddenSize,
      logitsVocabSize,
      { temperature, chunkRows: 32 }
    );
    const logitsActual = new Float32Array(await readBuffer(logitsOutput.buffer, logitsExpected.byteLength));
    let logitsMaxAbsDiff = 0;
    for (let i = 0; i < logitsExpected.length; i += 1) {
      logitsMaxAbsDiff = Math.max(logitsMaxAbsDiff, Math.abs(logitsActual[i] - logitsExpected[i]));
    }
    console.log(`soft-embedding-logits-f16-parity: max_abs_diff=${logitsMaxAbsDiff.toExponential(3)}`);
    assert.ok(logitsMaxAbsDiff < 1e-4, `logits soft embedding diverged: max_abs_diff=${logitsMaxAbsDiff}`);
  } finally {
    if (logitsOutput?.buffer) releaseBuffer(logitsOutput.buffer);
    logitsBuffer.destroy();
    logitsEmbeddingBuffer.destroy();
  }
} finally {
  if (output?.buffer) releaseBuffer(output.buffer);
  softmaxBuffer.destroy();
  section0Buffer.destroy();
  section1Buffer.destroy();
  destroyDevice();
}

console.log('soft-embedding-split-f16-parity.test: ok');

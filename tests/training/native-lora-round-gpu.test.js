import assert from 'node:assert/strict';
import { bootstrapNativeTrainingHost, releaseNativeTrainingHost } from '../../src/training.js';
import { initDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { createUploadedTensor } from '../../src/experimental/training/tensor-factory.js';
import { releaseBuffer } from '../../src/memory/buffer-pool.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { quantizeToQ4KMRowWise } from '../../src/converter/quantizer.js';
import { createNativeQwenLoRATrainer } from '../../src/experimental/training/native-qwen-lora.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Executed WebGPU suffix oracle. Adapter metadata distinguishes hardware from
// software providers; this fixture does not establish full-model training quality.
await bootstrapNativeTrainingHost();
await initDevice();
const hiddenSize = 4, intermediateSize = 6, vocabSize = 8;
const norm = createUploadedTensor(new Float32Array(hiddenSize).fill(1), 'f32', [hiddenSize], 'round_norm');
const lmValues = Float32Array.from({ length: vocabSize * hiddenSize }, (_, i) => Math.sin(i * 0.17) * 0.35);
const packed = quantizeToQ4KMRowWise(lmValues, [vocabSize, hiddenSize]).quantized;
const lm = createUploadedTensor(packed, 'f32', [packed.byteLength / 4], 'round_lm');
const pipeline = {
  manifest: { modelId: 'native-lora-round-suffix-fixture', fixtureVersion: 1 },
  modelConfig: { numLayers: 1, layerTypes: ['full_attention'], postFeedforwardNorm: false,
    hiddenSize, intermediateSize, vocabSize, rmsNormEps: 1e-6, rmsNormWeightOffset: false, useTiedEmbeddings: false },
  weights: new Map([
    ['final_norm', createWeightBuffer(norm.buffer, 'f32', 'row', [hiddenSize], 'norm')],
    ['lm_head', createWeightBuffer(lm.buffer, 'q4k', 'row', [vocabSize, hiddenSize], 'lm')],
  ]),
  async prefillForLoRATraining(tokens) {
    const activation = createUploadedTensor(Float32Array.from({ length: tokens.length * intermediateSize },
      (_, i) => Math.sin(tokens[Math.floor(i / intermediateSize)] * 0.31 + i % intermediateSize) * 0.6),
    'f32', [tokens.length, intermediateSize], 'round_activation');
    const baseHidden = createUploadedTensor(Float32Array.from({ length: tokens.length * hiddenSize },
      (_, i) => Math.cos(tokens[Math.floor(i / hiddenSize)] * 0.23 + i % hiddenSize) * 0.4),
    'f32', [tokens.length, hiddenSize], 'round_hidden');
    return { activation, baseHidden, dispose() { releaseBuffer(activation.buffer); releaseBuffer(baseHidden.buffer); } };
  },
};
const trainer = createNativeQwenLoRATrainer({ pipeline, baseModelId: pipeline.manifest.modelId,
  layerIdx: 0, module: 'down_proj', rank: 2, alpha: 4,
  optimizer: { type: 'adamw', lr: 0.01, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01 },
  gradient: { maxNorm: 0.1 }, precision: { activations: 'f32', gradients: 'f32', loraParams: 'f32' } });
const samples = [
  { inputIds: [1, 3], targetIds: [2, 4], supervisedTokenCount: 2 },
  { inputIds: [4], targetIds: [2], supervisedTokenCount: 1 },
];
function values(snapshot) {
  const bytes = Buffer.from(snapshot.bytes, 'base64');
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}
function compare(a, b) {
  let maxAbsoluteError = 0;
  for (const key of ['A', 'B']) {
    const left = values(a.adapter[key]), right = values(b.adapter[key]);
    for (let i = 0; i < left.length; i++) maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(left[i] - right[i]));
  }
  for (let j = 0; j < 2; j++) for (const key of ['m', 'v']) {
    const left = values(a.optimizer[j][key]), right = values(b.optimizer[j][key]);
    for (let i = 0; i < left.length; i++) maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(left[i] - right[i]));
  }
  assert.ok(maxAbsoluteError < 1e-5, `collective update diverged: ${maxAbsoluteError}`);
  return maxAbsoluteError;
}
try {
  const round = await trainer.createTrainingRound({ jobId: 'gpu-suffix-oracle', roundId: '0',
    assignments: samples.map((sample, i) => ({ participantId: String(i), sampleHash: computeCanonicalSha256(sample), supervisedTokenCount: sample.supervisedTokenCount })) });
  await trainer.trainStep([1, 3, 4], [2, 4, 2], 3);
  const reference = await trainer.createCheckpoint();
  const contributions = [];
  for (let i = 0; i < samples.length; i++) contributions.push(await trainer.computeGradientContribution(round, String(i), samples[i]));
  assert.equal((await trainer.createCheckpoint()).stepCount, 0);
  await assert.rejects(() => trainer.applyGradientContributions(round, contributions.slice(0, 1)), /incomplete/);
  assert.equal(computeCanonicalSha256(await trainer.createCheckpoint()), round.checkpointHash);
  const corrupt = structuredClone(contributions);
  corrupt[1].gradients.B.bytes = Buffer.from(new Float32Array(values(corrupt[1].gradients.B).length).fill(NaN).buffer).toString('base64');
  await assert.rejects(() => trainer.applyGradientContributions(round, corrupt), /non-finite/);
  assert.equal(computeCanonicalSha256(await trainer.createCheckpoint()), round.checkpointHash);
  const receipt = await trainer.applyGradientContributions(round, contributions.toReversed());
  const actual = await trainer.createCheckpoint();
  assert.equal(receipt.stepCount, 1);
  assert.equal(receipt.supervisedTokenCount, 3);
  const maxAbsoluteError = compare(reference, actual);
  await assert.rejects(() => trainer.applyGradientContributions(round, contributions), /stale coordinator/);
  const malformed = structuredClone(round.checkpoint);
  malformed.adapter.B.bytes = 'AA==';
  assert.throws(() => trainer.restoreCheckpoint(malformed), /tensor_size_mismatch/);
  assert.equal(computeCanonicalSha256(await trainer.createCheckpoint()), computeCanonicalSha256(actual));
  const adapter = await trainer.exportAdapter({ id: 'round-oracle', name: 'Round oracle', weightsPath: 'adapter.safetensors' });
  assert.ok(adapter.weights.byteLength > 0 && adapter.weightsSha256);
  console.log(JSON.stringify({ passed: true, scope: 'executed WebGPU suffix oracle; one adapter; no full-model quality claim',
    adapter: getKernelCapabilities().adapterInfo,
    maxAbsoluteError, stepCount: receipt.stepCount, supervisedTokenCount: receipt.supervisedTokenCount,
    weightsSha256: adapter.weightsSha256 }, null, 2));
} finally {
  trainer.dispose(); releaseBuffer(norm.buffer); releaseBuffer(lm.buffer);
  await releaseNativeTrainingHost();
}

import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { f16ToF32 } from '../../src/loader/dtype-utils.js';

const [modelDirectory, baselinePath, candidatePath, destination] = process.argv.slice(2);
assert(destination, 'Supply model directory, baseline capture, candidate capture and destination');
const manifestBytes = await readFile(resolve(modelDirectory, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const baseline = JSON.parse(await readFile(baselinePath));
const candidate = JSON.parse(await readFile(candidatePath));
const modelIdentity = `sha256:${digest(manifestBytes)}`;
assert.equal(baseline.modelIdentity, modelIdentity);
assert.equal(candidate.modelIdentity, modelIdentity);
assert.deepEqual(baseline.generation, candidate.generation);
assert.deepEqual(baseline.prompt, candidate.prompt);
const observations = report => report.diagnostics.timeline;
const embedding = report => observations(report).find(row => row.opId === 'embed.out');
const normalization = report => observations(report).find(row => row.opId === 'layer.0.attn.post_input_norm');
const input = embedding(baseline).capture.data;
assert.deepEqual(input, embedding(candidate).capture.data, 'Compare the same observed operands');
const hiddenSize = manifest.architecture.hiddenSize;
const policy = manifest.inference.normalization;
assert(Number.isSafeInteger(hiddenSize) && hiddenSize > 0);
assert(input.length % hiddenSize === 0);
assert(Number.isFinite(policy.rmsNormEps) && policy.rmsNormEps > 0);
assert.equal(typeof policy.rmsNormWeightOffset, 'boolean');
const tensor = manifest.tensors['model.language_model.layers.0.input_layernorm.weight'];
assert.deepEqual(tensor.shape, [hiddenSize]);
assert.equal(tensor.dtype, 'F16');
assert.equal(tensor.size, hiddenSize * 2);
const shard = await readFile(resolve(modelDirectory, manifest.shards[tensor.shard].filename));
const weight = new Float64Array(hiddenSize);
for (let i = 0; i < hiddenSize; i++) {
  weight[i] = f16ToF32(shard.readUInt16LE(tensor.offset + i * 2)) + Number(policy.rmsNormWeightOffset);
}
const reference = new Float64Array(input.length);
for (let offset = 0; offset < input.length; offset += hiddenSize) {
  let sumSquares = 0;
  for (let i = 0; i < hiddenSize; i++) sumSquares += input[offset + i] ** 2;
  const scale = 1 / Math.sqrt(sumSquares / hiddenSize + policy.rmsNormEps);
  for (let i = 0; i < hiddenSize; i++) reference[offset + i] = input[offset + i] * scale * weight[i];
}
const errors = values => {
  assert.equal(values.length, reference.length);
  let maximumAbsoluteError = 0, squaredError = 0;
  for (let i = 0; i < reference.length; i++) {
    assert(Number.isFinite(values[i]));
    const error = values[i] - reference[i];
    maximumAbsoluteError = Math.max(maximumAbsoluteError, Math.abs(error));
    squaredError += error * error;
  }
  return { maximumAbsoluteError, rmsError: Math.sqrt(squaredError / reference.length) };
};
const report = {
  schema: 'doppler.input-normalization-package-reference/v1',
  scope: 'Float64 normalization on identical captured embeddings and exact stored norm weights; not full-model qualification',
  modelIdentity, hiddenSize, tokens: input.length / hiddenSize, normalization: policy,
  weightSha256: digest(shard.subarray(tensor.offset, tensor.offset + tensor.size)),
  referenceSha256: digest(Buffer.from(reference.buffer)),
  baseline: { packageVersion: baseline.packageVersion, captureSha256: digest(await readFile(baselinePath)),
    ...errors(normalization(baseline).capture.data) },
  candidate: { packageVersion: candidate.packageVersion, captureSha256: digest(await readFile(candidatePath)),
    ...errors(normalization(candidate).capture.data) },
};
await writeFile(destination, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));

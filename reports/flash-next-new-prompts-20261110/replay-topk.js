// Replay production routing on retained logits; no model execution or input substitution.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const rawRun = await fs.readFile(process.argv[2]);
const run = JSON.parse(rawRun);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const moduleAt = file => import(pathToFileURL(path.join(run.runtimeRoot, file)).href);
const { probeNodeGPU } = await moduleAt('tests/helpers/gpu-probe.js');
const { getDevice, getKernelCapabilities, destroyDevice } = await moduleAt('src/gpu/device.js');
const { acquireBuffer, releaseBuffer, readBuffer, uploadData } = await moduleAt('src/memory/buffer-pool.js');
const { runSoftmaxTopK } = await moduleAt('src/gpu/kernels/softmax.js');
const { getKernelConfig } = await moduleAt('src/gpu/kernels/kernel-configs.js');
const errors = [];
const probe = await probeNodeGPU({ installFileFetchShim: true });
assert(probe.ready, probe.reason);
const device = getDevice();
device.addEventListener('uncapturederror', event => errors.push(event.error.message));
const config = getKernelConfig('topk', run.variant);
assert.equal(config.shaderFile, run.expectedShader, 'Replay must bind the compiled model routing shader');
assert.ok(config.variantMetadata?.maxExperts >= run.experts && config.variantMetadata?.maxTopK >= run.topK,
  'Replay geometry must fit the declared kernel limits');
const shader = await fs.readFile(path.join(run.runtimeRoot, 'src/gpu/kernels', config.shaderFile));
const result = { schema: 'doppler.flash-next-routing-physical-replay/v1',
  scope: 'Production top-k shader replay on retained source/GPU logits; not a live capture of model routing or model qualification',
  runSha256: hash(rawRun), producerSha256: hash(await fs.readFile(new URL(import.meta.url))),
  shader: { file: config.shaderFile, sourceSha256: hash(shader), variant: run.variant },
  adapter: getKernelCapabilities().adapterInfo, rows: [], errors,
  acceptanceChanged: false, normalModelQualified: false };
assert.doesNotMatch(JSON.stringify(result.adapter), /swiftshader|llvmpipe|software/i);
await fs.writeFile(run.outputPath, JSON.stringify(result, null, 2), { flag: 'wx' });
async function verified(descriptor) {
  const bytes = await fs.readFile(descriptor.file);
  assert.equal(hash(bytes), descriptor.sha256, 'Retained replay input hash mismatch');
  return bytes;
}
try {
  for (const layer of run.layers) {
    const expectedRaw = await verified(layer.sourceExperts);
    const expected = new Uint32Array(expectedRaw.buffer.slice(expectedRaw.byteOffset, expectedRaw.byteOffset + expectedRaw.byteLength));
    const scoresRaw = await verified(layer.sourceWeights);
    const scores = new Float32Array(scoresRaw.buffer.slice(scoresRaw.byteOffset, scoresRaw.byteOffset + scoresRaw.byteLength));
    assert.equal(expected.length, run.tokens * run.topK);
    assert.equal(scores.length, expected.length);
    assert(scores.every(Number.isFinite));
    for (const lane of ['source', 'gpu']) {
      const bytes = await verified(layer[lane]);
      assert.equal(bytes.byteLength, run.tokens * run.experts * 4);
      const input = acquireBuffer(bytes.byteLength);
      let output;
      try {
        uploadData(input, bytes);
        output = await runSoftmaxTopK(input, run.tokens, run.experts, run.topK,
          { normalize: run.normalize, inputDtype: 'f32', weightsDtype: 'f32', variant: run.variant, modelType: run.modelType });
        const actual = new Uint32Array(await readBuffer(output.indices, expectedRaw.byteLength));
        const weights = new Float32Array(await readBuffer(output.weights, expectedRaw.byteLength));
        assert(weights.every(Number.isFinite));
        const setDifferences = [], orderDifferences = [];
        let maxAbsoluteWeightError = 0, maxRelativeWeightError = 0, weightSquaredError = 0, comparableWeights = 0;
        for (let token = 0; token < run.tokens; token++) {
          const a = Array.from(actual.subarray(token * run.topK, (token + 1) * run.topK));
          const b = Array.from(expected.subarray(token * run.topK, (token + 1) * run.topK));
          if (a.some((value, i) => value !== b[i])) orderDifferences.push(token);
          for (let slot = 0; slot < run.topK; slot++) {
            const sourceSlot = b.indexOf(a[slot]);
            if (sourceSlot < 0) continue;
            const reference = scores[token * run.topK + sourceSlot];
            const error = Math.abs(weights[token * run.topK + slot] - reference);
            maxAbsoluteWeightError = Math.max(maxAbsoluteWeightError, error);
            if (reference !== 0) maxRelativeWeightError = Math.max(maxRelativeWeightError, error / Math.abs(reference));
            weightSquaredError += error * error;
            comparableWeights++;
          }
          a.sort((x, y) => x - y); b.sort((x, y) => x - y);
          if (a.some((value, i) => value !== b[i])) setDifferences.push(token);
        }
        result.rows.push({ layer: layer.layer, lane, inputSha256: layer[lane].sha256,
          outputIndicesSha256: hash(actual), setDifferences, orderDifferences,
          sourceWeightsSha256: layer.sourceWeights.sha256, outputWeightsSha256: hash(weights), comparableWeights,
          maxAbsoluteWeightError, maxRelativeWeightError, weightErrorRms: Math.sqrt(weightSquaredError / comparableWeights),
          differingChoices: setDifferences.map(token => ({ token,
            actual: Array.from(actual.subarray(token * run.topK, (token + 1) * run.topK)),
            source: Array.from(expected.subarray(token * run.topK, (token + 1) * run.topK)) })) });
      } finally {
        await device.queue.onSubmittedWorkDone();
        if (output) { releaseBuffer(output.indices); releaseBuffer(output.weights); }
        releaseBuffer(input);
      }
    }
  }
  result.completed = true;
} finally {
  destroyDevice();
  await fs.writeFile(run.outputPath, JSON.stringify(result, null, 2));
}
const summary = row => row ? { layer: row.layer, lane: row.lane, differences: row.setDifferences.length,
  firstChoices: row.differingChoices.slice(0, 2) } : null;
console.log(JSON.stringify({ completed: result.completed, rows: result.rows.length,
  firstSourceDifference: summary(result.rows.find(r => r.lane === 'source' && r.setDifferences.length)),
  firstGpuDifference: summary(result.rows.find(r => r.lane === 'gpu' && r.setDifferences.length)), errors }));
assert.deepEqual(errors, []);

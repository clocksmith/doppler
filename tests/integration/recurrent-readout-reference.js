/** Separate inherited recurrent-state error from readout arithmetic error.
 * Inputs must be probes that preserve the uninstrumented physical execution. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pairLinearCaptures, decodeCapturedTensor } from '../kernels/recurrent-reference.js';

const [capturePath, normalizedPath, statePath, outputPath, gatePath, destination] = process.argv.slice(2);
assert(destination, 'Supply capture, normalized-Q, state, raw-output, gate observations and destination');
const bytes = await readFile(capturePath), capture = JSON.parse(bytes);
const pairs = pairLinearCaptures(capture); assert.equal(pairs.length, 1);
const { input, coordinate } = pairs[0], p = input.params;
const hash = b => createHash('sha256').update(b).digest('hex');
const results = [{ platform: 'mac', stages: {} }, { platform: 'linux', stages: {} }];
const sources = [];
let baselineSha256 = null;
for (const [path, stages] of [[normalizedPath, ['normalizedQ']], [statePath, ['updatedState']],
  [outputPath, ['rawOutput', 'invRms']], [gatePath, ['gate']]]) {
  const bytes = await readFile(path), probe = JSON.parse(bytes);
  assert.equal(probe.observationValidation?.preservesUninstrumentedExecution, true, 'Reject perturbing probes');
  assert.equal(probe.captureSha256, hash(await readFile(capturePath)));
  assert.deepEqual(probe.coordinate, coordinate);
  baselineSha256 ??= probe.observationValidation.baselineSha256;
  assert.equal(probe.observationValidation.baselineSha256, baselineSha256);
  sources.push({ stages, sha256: hash(bytes), shaderSha256: probe.shaderSha256 });
  for (const result of results) {
    const records = probe.results.filter(r => r.platform === result.platform); assert.equal(records.length, 1);
    const bytes = Buffer.from(records[0].activationData, 'base64');
    const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
    for (const name of stages) {
      assert(probe.observedStages.includes(name));
      const f = probe.traceLayout.fields[name]; result.stages[name] = data.slice(f.offset, f.offset + f.length);
    }
    if (!result.output) {
      const bytes = Buffer.from(records[0].data, 'base64');
      result.output = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4).slice();
    }
  }
}
const compare = (actual, reference) => {
  assert.equal(actual.length, reference.length);
  let maxDifference = 0, square = 0;
  for (let i = 0; i < actual.length; i++) {
    assert(Number.isFinite(actual[i]) && Number.isFinite(reference[i]));
    const diff = Math.abs(actual[i] - reference[i]); maxDifference = Math.max(diff, maxDifference); square += diff ** 2;
  }
  return { elements: actual.length, maxDifference, rmsError: Math.sqrt(square / actual.length) };
};
const normWeight = decodeCapturedTensor(input, 'normWeight');
for (const result of results) {
  const s = result.stages, raw = new Float64Array(result.output.length);
  const rms = new Float64Array(p.numTokens * p.numVHeads), gated = new Float64Array(raw.length);
  for (let row = 0; row < rms.length; row++) {
    let squared = 0;
    for (let v = 0; v < p.headVDim; v++) {
      const out = row * p.headVDim + v;
      for (let k = 0; k < p.headKDim; k++) {
        raw[out] += s.updatedState[(row * p.headKDim + k) * p.headVDim + v] * s.normalizedQ[row * p.headKDim + k];
      }
      squared += s.rawOutput[out] ** 2;
      const weight = normWeight[p.normMode === 'per_head' ? (row % p.numVHeads) * p.headVDim + v : v];
      gated[out] = s.rawOutput[out] * s.invRms[row] * weight * s.gate[out];
    }
    rms[row] = 1 / Math.sqrt(squared / p.headVDim + Math.fround(p.rmsNormEps));
  }
  result.comparisons = { rawDotGivenObservedStateAndQ: compare(s.rawOutput, raw),
    rmsGivenObservedRawOutput: compare(s.invRms, rms), gatedProductGivenObservedOperands: compare(result.output, gated) };
  delete result.stages; delete result.output;
}
const receipt = { scope: 'Float64 conditional references on validated GPU operands; no model acceptance or arithmetic substitution',
  captureSha256: hash(bytes), coordinate, baselineSha256, sources, results };
await writeFile(destination, JSON.stringify(receipt, null, 2));
console.log(JSON.stringify(receipt.results));

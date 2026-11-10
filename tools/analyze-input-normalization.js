#!/usr/bin/env node
import fs from 'node:fs';
import crypto from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { halfValue } from './numerical-discontinuity-core.js';
import { fma, inverseRootCandidates } from './dyadic-diagnostic.js';

const [protocolPath, inputPath, evidencePath, outputPath] = process.argv.slice(2);
if (!outputPath) throw new Error('Usage: analyze-input-normalization.js protocol inputs.gz norm-evidence.json output');
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const protocolBytes = fs.readFileSync(protocolPath);
const p = JSON.parse(protocolBytes);
if (p.workgroupSize !== 256 || p.shape?.[1] !== 1024 || !Number.isInteger(p.shape[0]) || p.shape[0] < 1 ||
    p.weightOffset !== true || p.allowedInverseSqrtErrorUlp !== 2 || p.epsilon !== 1e-6) {
  throw new Error('Unsupported frozen normalization contract');
}
const inputBytes = fs.readFileSync(inputPath);
if (sha(inputBytes) !== p.inputBundleSha256) throw new Error('Capture identity mismatch');
const input = JSON.parse(gunzipSync(inputBytes));
const evidenceBytes = fs.readFileSync(evidencePath);
const evidence = JSON.parse(evidenceBytes);
for (const key of ['manifestSha256', 'runtimeArchiveSha256', 'shaderSha256', 'weightsSha256']) {
  if (evidence[key] !== p[key]) throw new Error(`Frozen identity mismatch: ${key}`);
}
const weightBytes = Buffer.from(evidence.weightsF16, 'base64');
if (sha(weightBytes) !== p.weightsSha256 || sha(Buffer.from(evidence.shader)) !== p.shaderSha256) throw new Error('Evidence content mismatch');
const x = input.first.find((r) => r.op === 'embed.out');
const y = input.first.find((r) => r.op === 'layer.0.attn.post_input_norm');
const [tokens, width] = p.shape;
if (x.values.length !== tokens * width || y.values.length !== tokens * width ||
    x.referenceValues.length !== tokens * width || y.referenceValues.length !== tokens * width ||
    weightBytes.length !== width * 2) throw new Error('Frozen shape mismatch');
for (const values of [x.values, x.referenceValues, y.values]) {
  if (values.some((v) => !Number.isFinite(v) || !Object.is(v, Math.fround(v)))) throw new Error('Expected finite F32 data');
  // JSON number arrays cannot retain the sign bit of zero. Reject the ambiguous
  // case rather than calling numerical equality a bitwise witness.
  if (values.some((v) => v === 0)) throw new Error('Zero needs a bit-preserving input encoding');
}
if (y.referenceValues.some((v) => !Number.isFinite(v))) throw new Error('Nonfinite reference');
if (x.values.some((v, i) => !Object.is(v, x.referenceValues[i]))) throw new Error('Norm inputs differ');
const weights = Array.from({ length: width }, (_, i) => halfValue(weightBytes.readUInt16LE(2 * i)));
const r = Math.fround;
const models = [];
for (const explicitFma of p.models.explicitFma) for (const residualSubtraction of p.models.residualSubtraction) for (const outputAssociation of p.models.outputAssociation) {
  if (!['fused', 'separate'].includes(explicitFma) || !['fused', 'separate'].includes(residualSubtraction) ||
      !['left', 'right'].includes(outputAssociation)) throw new Error('Unknown arithmetic model');
  models.push({ explicitFma, residualSubtraction, outputAssociation, exactRows: 0, exactElementsInBestRows: 0, corruptedRowsRejected: 0, wrongWeightRowsRejected: 0, rows: [] });
}
let proxyMaxError = 0;
let observedMaxError = 0;
for (let token = 0; token < tokens; token++) {
  const row = x.values.slice(token * width, (token + 1) * width);
  const observed = y.values.slice(token * width, (token + 1) * width);
  const sums = new Float32Array(p.workgroupSize);
  const perThread = width / p.workgroupSize;
  if (!Number.isInteger(perThread)) throw new Error('Unsupported reduction geometry');
  let squareSum = 0;
  for (let i = 0; i < width; i++) {
    // F16 embedding products are exact in F64; this sum is only the independent
    // high-precision diagnostic, never the witness acceptance criterion.
    squareSum += row[i] * row[i];
    const lane = Math.floor(i / perThread);
    sums[lane] = r(sums[lane] + r(row[i] * row[i]));
  }
  for (let stride = p.workgroupSize / 2; stride >= 1; stride /= 2) for (let lane = 0; lane < stride; lane++) sums[lane] = r(sums[lane] + sums[lane + stride]);
  const epsilonSum = r(r(sums[0] / width) + r(p.epsilon));
  const estimates = inverseRootCandidates(epsilonSum);
  const idealInverse = 1 / Math.sqrt(squareSum / width + p.epsilon);
  for (let i = 0; i < width; i++) {
    const ideal = row[i] * idealInverse * (1 + weights[i]);
    proxyMaxError = Math.max(proxyMaxError, Math.abs(ideal - y.referenceValues[token * width + i]));
    observedMaxError = Math.max(observedMaxError, Math.abs(ideal - observed[i]));
  }
  for (const model of models) {
    const madd = model.explicitFma === 'fused' ? fma : (a, b, c) => r(r(a * b) + c);
    let best = 0; const matchingEstimates = [];
    let corruptedMatches = false; let wrongWeightMatches = false;
    for (const estimate of estimates) {
      const squared = r(estimate * estimate);
      const error = madd(estimate, estimate, -squared);
      const first = madd(-epsilonSum, squared, 1);
      const residual = model.residualSubtraction === 'fused' ? fma(-epsilonSum, error, first) : r(first - r(epsilonSum * error));
      const inv = madd(r(0.5 * estimate), residual, estimate);
      let matches = 0; let corrupted = 0; let wrong = 0;
      for (let i = 0; i < width; i++) {
        const gain = r(1 + weights[i]);
        const predicted = model.outputAssociation === 'left' ? r(r(row[i] * inv) * gain) : r(row[i] * r(inv * gain));
        const badGain = model.outputAssociation === 'left' ? r(r(row[i] * inv) * weights[i]) : r(row[i] * r(inv * weights[i]));
        matches += Object.is(predicted, observed[i]);
        corrupted += Object.is(predicted, i === 0 ? r(observed[i] + 0.01) : observed[i]);
        wrong += Object.is(badGain, observed[i]);
      }
      best = Math.max(best, matches);
      if (matches === width) matchingEstimates.push(estimate);
      corruptedMatches ||= corrupted === width;
      wrongWeightMatches ||= wrong === width;
    }
    model.exactRows += matchingEstimates.length > 0;
    model.exactElementsInBestRows += best;
    model.corruptedRowsRejected += !corruptedMatches;
    model.wrongWeightRowsRejected += !wrongWeightMatches;
    model.rows.push({ token, bestMatchingElements: best, matchingEstimates });
  }
}
const result = { schema: 'doppler.input-normalization-result/v1', qualified: false, modelId: evidence.modelId,
  protocolSha256: sha(protocolBytes), inputsSha256: sha(inputBytes), evidenceSha256: sha(evidenceBytes),
  analyzerSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  arithmeticSha256: sha(fs.readFileSync(new URL('./dyadic-diagnostic.js', import.meta.url))),
  tokens, width, values: tokens * width, proxyMaxError, observedMaxError, models,
  scope: 'Exact operator witnesses on the identified inputs; no claim about subsequent layers or complete-model qualification' };
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ ...result, models: models.map(({ rows, ...m }) => m) }));

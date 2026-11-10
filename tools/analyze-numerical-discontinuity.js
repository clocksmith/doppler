#!/usr/bin/env node
import fs from 'node:fs';
import crypto from 'node:crypto';
import { gunzipSync } from 'node:zlib';
import { dotEnvelope, halfValue, nearestHalf, cacheDecision } from './numerical-discontinuity-core.js';

const [protocolPath, bundlePath, bundleSha256, output] = process.argv.slice(2);
if (!output) throw new Error('Usage: analyze-numerical-discontinuity.js protocol inputs.gz expected-sha256 output');
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const protocolBytes = fs.readFileSync(protocolPath);
const protocol = JSON.parse(protocolBytes);
const bundleBytes = fs.readFileSync(bundlePath);
if (sha(bundleBytes) !== bundleSha256) throw new Error('Input artifact digest mismatch');
const bundle = JSON.parse(gunzipSync(bundleBytes));
if (bundle.protocolSha256 !== sha(protocolBytes)) throw new Error('Frozen protocol digest mismatch');
const top = (values, count = 10) => values.map((v, id) => ({ v, id })).sort((a, b) => b.v - a.v || a.id - b.id).slice(0, count).map((r) => r.id);
const result = { schema: 'doppler.discontinuity-result/v1', qualified: false,
  scope: 'Offline arithmetic on historical physical captures; conditional local bounds, not a new correctness contract',
  protocolSha256: sha(protocolBytes), inputSha256: bundleSha256,
  analyzerSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  coreSha256: sha(fs.readFileSync(new URL('./numerical-discontinuity-core.js', import.meta.url))),
    upstreamBound: null, routers: [], kv: [], firstCapturedDifferences: [],
  negativeControls: { outsideEnvelopeRejected: 0, insideEnvelopeAlterationAcceptedLocally: 0, changedCacheBitRejected: 0, corruptedInputCorrectlyRoundedButUnresolved: 0 },
};
for (const router of bundle.routers) {
  const bytes = Buffer.from(router.weightsF16, 'base64');
  if (sha(bytes) !== router.weightsSha256) throw new Error('Weight slice identity mismatch');
  const weights = Array.from({ length: 512 }, (_, expert) => Array.from({ length: 2560 }, (_, i) => halfValue(bytes.readUInt16LE((expert * 2560 + i) * 2))));
  for (const row of router.rows) {
    const candidates = new Set([...top(row.logits, 12), ...top(row.sourceLogits, 12)]);
    const summary = { layer: router.layer, position: row.position,
      reservedValidationSlice: !protocol.flashNext.knownDiagnosticPositions.includes(row.position),
      outputs: 512, outsideLocalEnvelope: 0, sourceOutsideLocalEnvelope: 0,
      forwardReplayBitMatches: 0, reverseReplayBitMatches: 0, maxLocalError: 0, maxErrorBoundRatio: 0,
      minBound: Infinity, maxBound: 0, inputMaxDifference: Math.max(...row.input.map((x, i) => Math.abs(x - row.sourceInput[i]))),
      gpuLogitTop10: top(row.logits), sourceLogitTop10: top(row.sourceLogits),
      selectionScope: 'Raw logit ordering; does not substitute for captured GPU softmax/top-k tie semantics',
      candidates: [], negativeControls: [] };
    const forwardLogits = [];
    const reverseLogits = [];
    for (let expert = 0; expert < 512; expert++) {
      const a = dotEnvelope(row.input, weights[expert], row.logits[expert]);
      const b = dotEnvelope(row.sourceInput, weights[expert], row.sourceLogits[expert]);
      if (!a.forwardWithin || !a.reverseWithin || !b.forwardWithin || !b.reverseWithin) throw new Error('Constructive F32 replay violated the analytic model');
      summary.outsideLocalEnvelope += !a.within;
      summary.sourceOutsideLocalEnvelope += !b.within;
      summary.forwardReplayBitMatches += a.forward === row.logits[expert];
      summary.reverseReplayBitMatches += a.reverse === row.logits[expert];
      summary.maxLocalError = Math.max(summary.maxLocalError, a.error);
      summary.maxErrorBoundRatio = Math.max(summary.maxErrorBoundRatio, a.error / a.bound);
      summary.minBound = Math.min(summary.minBound, a.bound);
      summary.maxBound = Math.max(summary.maxBound, a.bound);
      forwardLogits.push(a.forward); reverseLogits.push(a.reverse);
      if (candidates.has(expert)) summary.candidates.push({ expert, gpu: row.logits[expert], source: row.sourceLogits[expert],
        sameGpuInputExactDot: a.exact, sameSourceInputExactDot: b.exact, localBound: a.bound, sourceLocalBound: b.bound,
        forwardReplay: a.forward, reverseReplay: a.reverse,
        inheritedInputEffect: a.exact - b.exact,
        inheritedInputEffectStatus: 'measured-difference-not-an-allowed-error-budget' });
      if (expert === 0) {
        const outside = Math.fround(a.exact + 2 * a.bound);
        const inside = Math.fround(a.exact + a.bound / 2);
        result.negativeControls.outsideEnvelopeRejected += !a.contains(outside);
        result.negativeControls.insideEnvelopeAlterationAcceptedLocally += inside !== row.logits[expert] && a.contains(inside);
        summary.negativeControls.push({ expert, actual: row.logits[expert], insideAlteration: inside, outsideAlteration: outside,
          insideAcceptedLocally: a.contains(inside), outsideRejected: !a.contains(outside) });
      }
    }
    summary.forwardLogitTop10 = top(forwardLogits);
    summary.reverseLogitTop10 = top(reverseLogits);
    const lost = summary.sourceLogitTop10.filter((id) => !summary.gpuLogitTop10.includes(id));
    const gained = summary.gpuLogitTop10.filter((id) => !summary.sourceLogitTop10.includes(id));
    summary.changedLogitBoundary = lost.flatMap((a) => gained.map((b) => {
      const x = summary.candidates.find((c) => c.expert === a);
      const y = summary.candidates.find((c) => c.expert === b);
      const gap = x.sameGpuInputExactDot - y.sameGpuInputExactDot;
      const uncertainty = x.localBound + y.localBound;
      return { sourceExpert: a, gpuExpert: b, sameGpuInputExactGap: gap, conditionalGapEnvelope: [gap - uncertainty, gap + uncertainty],
        orderingCertified: Math.abs(gap) > uncertainty, upstreamVerdict: 'unresolved' };
    }));
    summary.crossRunVerdict = 'unresolved-no-independent-upstream-bound';
    result.routers.push(summary);
    console.log(JSON.stringify({ layer: router.layer, position: row.position, outside: summary.outsideLocalEnvelope, ratio: summary.maxErrorBoundRatio }));
  }
}
for (const row of bundle.kv) {
  const summary = { layer: row.layer, kind: row.kind, values: row.values.length,
    incorrectStores: 0, incorrectReferenceStores: 0, changedStores: 0, adjacentChanges: 0,
    crossedMidpoints: 0, maxPreStoreDifference: 0, examples: [], upstreamVerdict: 'unresolved-no-independent-input-bound' };
  for (let i = 0; i < row.values.length; i++) {
    const d = cacheDecision(row.values[i], row.stored[i], row.referenceValues[i], row.referenceStored[i]);
    summary.incorrectStores += !d.ownCorrect;
    summary.incorrectReferenceStores += !d.referenceCorrect;
    summary.changedStores += d.different;
    summary.adjacentChanges += d.adjacent;
    summary.crossedMidpoints += d.crossedMidpoint === true;
    summary.maxPreStoreDifference = Math.max(summary.maxPreStoreDifference, Math.abs(row.values[i] - row.referenceValues[i]));
    if (d.different && summary.examples.length < 8) summary.examples.push({ position: Math.floor(i / 512), dimension: i % 512,
      value: row.values[i], referenceValue: row.referenceValues[i], stored: row.stored[i], referenceStored: row.referenceStored[i], midpoint: d.midpoint });
  }
  const i = 0;
  result.negativeControls.changedCacheBitRejected += !cacheDecision(row.values[i], row.stored[i] ^ 1, row.referenceValues[i], row.referenceStored[i]).ownCorrect;
  const corrupted = row.values[i] + 0.01;
  const diagnostic = cacheDecision(corrupted, nearestHalf(corrupted), row.referenceValues[i], row.referenceStored[i]);
  result.negativeControls.corruptedInputCorrectlyRoundedButUnresolved += diagnostic.ownCorrect && diagnostic.upstreamStatus.startsWith('unresolved');
  result.kv.push(summary);
}
for (const row of bundle.first) {
  let maximum = 0; let firstIndex = null;
  for (let i = 0; i < row.values.length; i++) {
    const delta = Math.abs(row.values[i] - row.referenceValues[i]);
    maximum = Math.max(maximum, delta);
    if (delta && firstIndex === null) firstIndex = i;
  }
  result.firstCapturedDifferences.push({ op: row.op, firstIndex, maximum, comparedValues: row.values.length });
}
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ output, negativeControls: result.negativeControls, kv: result.kv.map(({ examples, ...r }) => r), first: result.firstCapturedDifferences }));

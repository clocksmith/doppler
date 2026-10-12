// Synthetic report-conversion controls; no model, GPU, signing or eligibility.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { producePhysicalReports, sha256 } from './produce-physical-reports.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
const bytes = value => Buffer.from(JSON.stringify(value));
const vector = (values, width) => { const b = Buffer.alloc(values.length * width);
  values.forEach((value, i) => width === 8 ? b.writeDoubleLE(value, i * width) : b.writeFloatLE(value, i * width)); return b.toString('base64'); };
function fixture() {
  const graph = { kernels: { synthetic: true } }, plan = { modelId: 'synthetic', splitLayer: 1 };
  const identity = { executionGraphHash: computeCanonicalSha256(graph), synthetic: true };
  const manifest = { modelId: 'synthetic', architecture: { vocabSize: 2 }, inference: { execution: graph } };
  const generation = { maxTokens: 51, temperature: 0 };
  const evaluatorBytes = Buffer.from('synthetic independent evaluator'), archiveBytes = Buffer.from('synthetic archive');
  const reference = { planId: 'legacy-plan', generation, numericalReference: { kind: 'independent-f64',
    evaluatorSha256: sha256(evaluatorBytes).slice(7), proposalSha256: 'proposal', outputIndexSha256: 'index', tolerance: 0.001, steps: 55 },
    expected: [2, 2, 51].map(count => ({ text: 'synthetic answer', steps: Array.from({ length: count }, (_, step) => ({
      tokenId: 3, stopReason: step + 1 === count ? 'eos-token' : null,
      logitsDtype: 'f64', logits: vector([0.125, -0.5], 8) })) })) };
  const requestContractBytes = reference.expected.map((row, requestIndex) => bytes({ requestIndex,
    steps: row.steps.map((_, index) => index), prefixLengths: row.steps.map((_, index) => 2 + index), inputIds: [1, 2, ...row.steps.map(() => 3)] }));
  const expected = { schema: 'doppler.physical-resident-producer-input/v1', archiveHash: sha256(archiveBytes),
    referenceHash: sha256(bytes(reference)), manifestHash: sha256(bytes(manifest)), graphHash: sha256(bytes(graph)),
    evaluatorHash: sha256(evaluatorBytes), proposalSha256: 'proposal', outputIndexSha256: 'index',
    applicationBundleHash: 'synthetic-application', requestFixtureSha256: 'requests',
    requestContractHashes: requestContractBytes.map(sha256), factorySha256: 'factory', identityHelperSha256: 'helper',
    legacyPlanId: 'legacy-plan', partitionPlan: plan,
    placements: [{ partitionIndex: 0, platform: 'first' }, { partitionIndex: 1, platform: 'second' }],
    providers: { first: { adapter: { vendor: 'synthetic-first' } }, second: { adapter: { vendor: 'synthetic-second' } } } };
  const capture = { phase: 'reference', requestFixtureSha256: 'requests',
    pairedBinding: { archiveSha256: expected.archiveHash.slice(7), manifestIdentity: expected.manifestHash, graphSha256: expected.graphHash.slice(7) },
    runs: reference.expected.map((row, attempt) => ({ attempt, generation, completed: true, stopReason: 'eos-token', text: row.text,
      steps: row.steps.length, inputTokens: 2, numerical: row.steps.map((entry, step) => ({ step, tokenId: entry.tokenId,
        stopReason: entry.stopReason, inputTokenIds: step === 0 ? [1, 2] : [3], logits: vector([0.125, -0.5], 4), maxDifference: 0, matches: true })) })),
    descriptors: ['first', 'second'].map((platform, index) => ({ platform, observedProvider: expected.providers[platform], browser: 'synthetic-browser',
      descriptor: { index }, plan, legacyPlanId: 'legacy-plan', planId: computeCanonicalSha256(plan),
      openingObservation: { partitionIndex: index, modelIdentity: expected.manifestHash, planId: computeCanonicalSha256(plan),
        sources: { factory: { sha256: 'factory' }, identity: { sha256: 'helper' } },
        initialExecutionIdentity: identity, resolvedRuntimeSession: { runtime: { session: { intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32' } } } } })),
    residentSettlement: [0, 1].map(() => ({ snapshot: { pool: { resources: { retainedModel: { count: 0 }, active: { count: 0 }, deferredCleanup: { count: 0 } } },
      device: { categories: { weights: 0 }, maxBytes: 1420000000, peakBytes: 100, lost: false }, gpuValidationErrors: [] } })) };
  const run = { ok: true, package: { integrity: 'sha512-' + createHash('sha512').update(archiveBytes).digest('base64') },
    browserIdentity: 'synthetic-application', referenceSha256: expected.referenceHash.slice(7), maxGpuBufferBytes: 1420000000,
    referenceBinding: { manifestIdentity: expected.manifestHash, archiveSha256: expected.archiveHash.slice(7) } };
  return { capture, reference, manifest, graph, run, input: { expected, captureBytes: bytes(capture), referenceBytes: bytes(reference),
    manifestBytes: bytes(manifest), graphBytes: bytes(graph), runBytes: bytes(run), archiveBytes, evaluatorBytes, requestContractBytes,
    computeCanonicalSha256, observeInitialExecutionIdentity: () => identity } };
}
test('emits existing report schema only after all55 independently recomputed comparisons', () => {
  const f = fixture(), result = producePhysicalReports(f.input);
  assert.equal(result.contract.comparedSteps, 55);
  assert.deepEqual(result.reports.map(row => row.partitionIndex), [0, 1]);
  assert(result.reports.every(row => row.comparison.steps.length === 55 && row.comparison.steps.every(step => step.maxAbsoluteError === 0)));
  assert.equal(result.reports[0].comparison.captureHash, sha256(f.input.captureBytes));
});
const mutations = {
  'failed actual logits despite passing self-report': f => { f.capture.runs[0].numerical[0].logits = vector([0.13, -0.5], 4); },
  'nonfinite output': f => { f.capture.runs[0].numerical[0].logits = vector([NaN, -0.5], 4); },
  'truncated output': f => { f.capture.runs[0].numerical[0].logits = vector([0.125], 4); },
  'missing step': f => { f.capture.runs[2].numerical.pop(); },
  'missing request': f => { f.capture.runs.pop(); },
  'changed prompt': f => { f.capture.runs[0].numerical[0].inputTokenIds[0] = 9; },
  'changed continuation': f => { f.capture.runs[2].numerical[5].inputTokenIds = [9]; },
  'changed sampled token': f => { f.capture.runs[0].numerical[0].tokenId = 9; },
  'changed sampling': f => { f.capture.runs[0].generation = { temperature: 1 }; },
  'failed cleanup': f => { f.capture.residentSettlement[0].snapshot.pool.resources.active.count = 1; },
  'changed allocation ceiling': f => { f.capture.residentSettlement[0].snapshot.device.maxBytes++; },
  'lost device': f => { f.capture.residentSettlement[1].snapshot.device.lost = true; },
  'missing actual identity observation': f => { delete f.capture.descriptors[0].openingObservation; },
  'forged identity': f => { f.capture.descriptors[0].openingObservation.initialExecutionIdentity = {}; },
  'scalar session': f => { f.capture.descriptors[0].openingObservation.resolvedRuntimeSession.runtime.session.intermediatePrecision = 'f32'; },
  'changed factory bytes': f => { f.capture.descriptors[0].openingObservation.sources.factory.sha256 = 'other'; },
  'same physical host for both partitions': f => { f.capture.descriptors[1].platform = 'first'; f.capture.descriptors[1].observedProvider = f.input.expected.providers.first; },
  'reversed physical placement': f => { f.capture.descriptors.reverse(); f.capture.descriptors[0].descriptor.index = 0; },
  'changed provider': f => { f.capture.descriptors[0].observedProvider = {}; },
  'changed traversal': f => { f.capture.descriptors[0].plan = { splitLayer: 2 }; },
  'failed outer controller': f => { f.run.ok = false; },
  'changed package': f => { f.run.package.integrity = 'other'; },
  'changed application': f => { f.run.browserIdentity = 'other'; },
};
for (const [name, mutate] of Object.entries(mutations)) test(`rejects ${name}`, () => {
  const f = fixture(); mutate(f); f.input.captureBytes = bytes(f.capture); f.input.runBytes = bytes(f.run);
  assert.throws(() => producePhysicalReports(f.input));
});
test('rejects a replacement reference or evaluator before interpreting captures', () => {
  for (const key of ['referenceBytes', 'evaluatorBytes', 'manifestBytes', 'graphBytes', 'archiveBytes']) {
    const f = fixture(); f.input[key] = Buffer.from('changed'); assert.throws(() => producePhysicalReports(f.input));
  }
});

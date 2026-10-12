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
    browserVersions: { first: 'synthetic-browser', second: 'synthetic-browser' },
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

function directFixture() {
  const f = fixture();
  const delivery = { scope: 'Immutable numerical fixture delivery; not deployed application or signed admission',
    archiveSha256: f.input.expected.archiveHash.slice(7), files: [{ path: 'synthetic-only' }] };
  f.input.deliveryBytes = bytes(delivery); delete f.input.runBytes;
  Object.assign(f.input.expected, { deliveryMode: 'direct-fixture', installedFiles: 1, testDeliveryHash: sha256(f.input.deliveryBytes) });
  f.capture.testDelivery = { schema: 'reploid.numerical-test-delivery/v1', testDeliveryHash: f.input.expected.testDeliveryHash,
    archiveSha256: delivery.archiveSha256, snapshotFiles: 1, installedFiles: 1, scope: delivery.scope, applicationQualification: false };
  f.input.captureBytes = bytes(f.capture); return f;
}
test('direct fixture preserves test-delivery scope without inventing run.ok or application identity', () => {
  const f = directFixture(), result = producePhysicalReports(f.input);
  assert.equal(result.reports[0].provenance.testDeliveryHash, f.input.expected.testDeliveryHash);
  assert.equal(result.reports[0].provenance.applicationBundleHash, undefined);
});
for (const field of ['testDeliveryHash', 'archiveSha256', 'snapshotFiles', 'installedFiles', 'applicationQualification']) {
  test(`direct fixture rejects changed ${field}`, () => {
    const f = directFixture(); f.capture.testDelivery[field] = field.endsWith('Files') ? 0 : 'changed';
    f.input.captureBytes = bytes(f.capture); assert.throws(() => producePhysicalReports(f.input));
  });
}

test('direct fixture rejects an extraneous application envelope', () => {
  const f = directFixture(); f.input.runBytes = bytes(f.run); assert.throws(() => producePhysicalReports(f.input));
});

function nativeFixture() {
  const f = directFixture();
  const provider = { module: 'webgpu', version: 'synthetic', entrySha256: 'a'.repeat(64), packageSha256: 'b'.repeat(64),
    observation: 'Actual process.report sharedObjects after resident preparation',
    libraries: [{ path: '/synthetic/node_modules/webgpu/linux.node', bytes: 42, sha256: 'c'.repeat(64) }] };
  f.input.expected.surfaces = { first: 'node-webgpu', second: 'browser-webgpu' };
  f.input.expected.nativeHosts = { first: { node: 'v22.synthetic', nativeDeliverySha256: 'd'.repeat(64), testDeliveryHash: 'sha256:' + 'e'.repeat(64), provider } };
  const d = f.capture.descriptors[0]; delete d.browser;
  Object.assign(d, { surface: 'node-webgpu', nativePid: 123, node: 'v22.synthetic', nativeProvider: structuredClone(provider),
    nativeHost: { schema: 'reploid.native-partition-fixture/v1', nativeDeliverySha256: 'd'.repeat(64),
      testDelivery: { archiveSha256: f.input.expected.archiveHash.slice(7), installedFiles: 1, testDeliveryHash: 'sha256:' + 'e'.repeat(64) } } });
  f.capture.nativeHostSettlement = { pid: 123, closed: true, providerOpened: true, ownedProviderReleased: true };
  f.input.captureBytes = bytes(f.capture); return f;
}
test('native surface requires its independently declared installed host and mapped library observations', () => {
  const f = nativeFixture(), { reports } = producePhysicalReports(f.input);
  assert.deepEqual(reports.map(report => report.surface), ['node-webgpu', 'browser-webgpu']);
  assert.equal(reports[0].provenance.nodeVersion, 'v22.synthetic');
  assert.equal(reports[0].provenance.browserVersion, undefined);
});
for (const [name, mutate] of Object.entries({
  'undeclared native surface': f => { delete f.input.expected.surfaces; },
  'missing native binary': f => { f.capture.descriptors[0].nativeProvider.libraries = []; },
  'changed native binary': f => { f.capture.descriptors[0].nativeProvider.libraries[0].sha256 = 'f'.repeat(64); },
  'different native installation': f => { f.capture.descriptors[0].nativeHost.nativeDeliverySha256 = 'f'.repeat(64); },
  'wrong installed archive': f => { f.capture.descriptors[0].nativeHost.testDelivery.archiveSha256 = 'f'.repeat(64); },
  'failed native cleanup': f => { f.capture.nativeHostSettlement.ownedProviderReleased = false; },
  'missing native cleanup': f => { delete f.capture.nativeHostSettlement; },
  'different native fixture delivery': f => { f.capture.descriptors[0].nativeHost.testDelivery.testDeliveryHash = 'sha256:' + 'f'.repeat(64); },
  'other native process cleanup': f => { f.capture.nativeHostSettlement.pid = 456; },
  'missing observed native PID': f => { delete f.capture.descriptors[0].nativePid; },
  'second native participant': f => { f.capture.descriptors[1].surface = 'node-webgpu'; },
  'different Node': f => { f.capture.descriptors[0].node = 'other'; },
})) test(`rejects ${name}`, () => {
  const f = nativeFixture(); mutate(f); f.input.captureBytes = bytes(f.capture);
  assert.throws(() => producePhysicalReports(f.input));
});

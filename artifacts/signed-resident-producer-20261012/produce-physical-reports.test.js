// Synthetic report-conversion controls; no model, GPU, signing or eligibility.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { producePhysicalReports, sha256, validateProducerBinding } from './produce-physical-reports.js';
import { stableSortObject } from '../../src/formats/stable-sort-object.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
const bytes = value => Buffer.from(JSON.stringify(value));
const vector = (values, width) => { const b = Buffer.alloc(values.length * width);
  values.forEach((value, i) => width === 8 ? b.writeDoubleLE(value, i * width) : b.writeFloatLE(value, i * width)); return b.toString('base64'); };
function fixture() {
  const graph = { kernels: { synthetic: true, Z: true, a: true } }, plan = { modelId: 'synthetic', splitLayer: 1 };
  const identity = { executionGraphHash: sha256(bytes(stableSortObject(graph))), synthetic: true };
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
        initialExecutionIdentity: identity, resolvedRuntimeSession: { manifestInference: { execution: graph }, runtime: { session: { intermediatePrecision: 'two-f32', linearRecurrentStatePrecision: 'two-f32' } } } } })),
    residentSettlement: [0, 1].map(() => ({ barriers: 1, snapshot: { pool: { resources: { retainedModel: { count: 0 }, active: { count: 0 }, deferredCleanup: { count: 0 } } },
      device: { categories: { weights: 0 }, maxBytes: 1420000000, peakBytes: 100 }, gpuValidationErrors: [], deviceEvents: [], observationErrors: [] } })) };
  const run = { ok: true, package: { integrity: 'sha512-' + createHash('sha512').update(archiveBytes).digest('base64') },
    browserIdentity: 'synthetic-application', referenceSha256: expected.referenceHash.slice(7), maxGpuBufferBytes: 1420000000,
    referenceBinding: { manifestIdentity: expected.manifestHash, archiveSha256: expected.archiveHash.slice(7) } };
  return { capture, reference, manifest, graph, run, input: { expected, captureBytes: bytes(capture), referenceBytes: bytes(reference),
    manifestBytes: bytes(manifest), graphBytes: bytes(graph), runBytes: bytes(run), archiveBytes, evaluatorBytes, requestContractBytes,
    computeCanonicalSha256, observeInitialExecutionIdentity: () => identity } };
}
test('emits existing report schema only after all55 independently recomputed comparisons', () => {
  const f = fixture(), result = producePhysicalReports(f.input);
  assert.notEqual(f.input.observeInitialExecutionIdentity().executionGraphHash, computeCanonicalSha256(f.graph));
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
  'lost device': f => { f.capture.residentSettlement[1].snapshot.deviceEvents.push({ type: 'device-lost', reason: 'unknown' }); },
  'missing device observation': f => { delete f.capture.residentSettlement[0].snapshot.deviceEvents; },
  'failed memory observation': f => { f.capture.residentSettlement[0].snapshot.observationErrors.push({ operation: 'device' }); },
  'unconfirmed queue settlement': f => { f.capture.residentSettlement[0].barriers = 0; },
  'missing actual identity observation': f => { delete f.capture.descriptors[0].openingObservation; },
  'changed observed graph with copied identity': f => { f.capture.descriptors[0].openingObservation.resolvedRuntimeSession.manifestInference.execution = { kernels: { other: true } }; },
  'missing observed graph': f => { delete f.capture.descriptors[0].openingObservation.resolvedRuntimeSession.manifestInference; },
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


test('post-run correction binding retains exact pre-run and actual capture/reference authority', () => {
  const expected = { producerHash: 'old', referenceHash: 'reference' };
  const correction = { schema: 'doppler.resident-report-validator-correction/v1',
    correction: 'execution-graph-hash-domain-and-observed-device-schema', reviewStatus: 'approved',
    originalExpectationsHash: 'expectations', originalProducerHash: 'old', correctedProducerHash: 'new',
    captureHash: 'capture', referenceHash: 'reference' };
  assert.doesNotThrow(() => validateProducerBinding(expected, 'expectations', 'new', 'capture', correction));
  assert.throws(() => validateProducerBinding(expected, 'expectations', 'new', 'capture'));
  for (const key of Object.keys(correction)) {
    assert.throws(() => validateProducerBinding(expected, 'expectations', 'new', 'capture', { ...correction, [key]: 'changed' }));
  }
  assert.doesNotThrow(() => validateProducerBinding(expected, 'expectations', 'old', 'capture'));
  assert.throws(() => validateProducerBinding(expected, 'expectations', 'old', 'capture', correction));
});

import { createOrderedLayerPartitionPlan, hashLayerPartitionPlan, validateOrderedLayerPartitionPlan } from '../../src/config/layer-partition-plan.js';
import { assertOrderedResidentContract } from '../../src/config/ordered-resident-qualification.js';
import { hashPartitionGeneration } from '../../src/partitions.js';
function orderedFixture(count = 3, hostCount = count) {
  const f = directFixture(), expected = f.input.expected;
  Object.assign(f.manifest.architecture, { numLayers: 24, hiddenSize: 4 });
  f.input.manifestBytes = bytes(f.manifest); expected.manifestHash = sha256(f.input.manifestBytes);
  const plan = createOrderedLayerPartitionPlan({ modelId: f.manifest.modelId, numLayers: 24, hiddenSize: 4, vocabSize: 2,
    splitLayers: Array.from({ length: count - 1 }, (_, i) => Math.floor(24 * (i + 1) / count)), activationPrecision: 'two-f32' });
  const generation = { maxTokens: 51, maxSeqLen: 64, temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, repetitionPenaltyWindow: 0, useChatTemplate: false };
  f.reference.generation = generation; f.input.referenceBytes = bytes(f.reference); expected.referenceHash = sha256(f.input.referenceBytes);
  Object.assign(expected, { schema: 'doppler.physical-resident-producer-input/v2', partitionPlan: plan,
    logicalStageCount: count, physicalHostCount: hostCount, maxGpuBufferBytes: 1420000000,
    deadlines: { stageMs: 300000, cleanupMs: 10000, observationMs: 2000 }, requesterId: 'synthetic-requester', placementGeneration: 1,
    placements: [], providers: {}, surfaces: {}, browserVersions: {}, executables: {} });
  const prototype = f.capture.descriptors[0], identityHash = computeCanonicalSha256(prototype.openingObservation.initialExecutionIdentity);
  f.capture.descriptors = []; f.capture.residentSettlement = []; f.capture.executorSettlements = [];
  const settlement = directFixture().capture.residentSettlement[0];
  for (const owner of ['retainedModel', 'active', 'deferredCleanup']) settlement.snapshot.pool.resources[owner].bytes = 0;
  for (let index = 0; index < count; index++) {
    const executorId = `executor-${index}`, platform = `host-${index % hostCount}`;
    expected.placements.push({ partitionIndex: index, executorId, platform, participantId: executorId,
      transportId: `controller:${executorId}`, residentId: `resident-${index}`, executionId: identityHash });
    expected.providers[executorId] = { adapter: { vendor: `synthetic-${platform}` } };
    expected.surfaces[executorId] = 'browser-webgpu'; expected.browserVersions[executorId] = 'synthetic-browser';
    expected.executables[executorId] = { sha256: 'b'.repeat(64), sizeBytes: 100 };
    const descriptor = structuredClone(prototype);
    Object.assign(descriptor, { executorId, platform, observedProvider: expected.providers[executorId], plan, planId: hashLayerPartitionPlan(plan),
      executableIdentity: expected.executables[executorId], executorIdentity: { pid: 1000 + index, contextId: `context-${index}` },
      descriptor: { index, schema: 'doppler.resident-partition/v2', stageCount: count, ready: true, modelId: f.manifest.modelId,
        modelIdentity: expected.manifestHash, planId: hashLayerPartitionPlan(plan), generationDigest: hashPartitionGeneration(generation), layerRange: plan.partitions[index].layerRange } });
    Object.assign(descriptor.openingObservation, { partitionIndex: index, modelIdentity: expected.manifestHash, planId: descriptor.planId });
    f.capture.descriptors.push(descriptor);
    f.capture.residentSettlement.push({ ...structuredClone(settlement), partitionIndex: index, executorId,
      initialExecutionIdentityHash: identityHash, residentClosed: true, attemptsClosed: 3, elapsedMs: 1 });
    f.capture.executorSettlements.push({ partitionIndex: index, executorId, executorIdentity: descriptor.executorIdentity, contextClosed: true });
  }
  expected.orderedContract = { schema: 'doppler.resident-qualification-contract/v2', manifestIdentity: expected.manifestHash,
    executionGraphHash: prototype.openingObservation.initialExecutionIdentity.executionGraphHash,
    partitionPlan: plan, partitionPlanHash: hashLayerPartitionPlan(plan), referenceHash: expected.referenceHash,
    evaluatorHash: expected.evaluatorHash, tolerance: 0.001, comparedSteps: 55,
    stages: expected.placements.map(({ executorId }, index) => ({ index, surface: expected.surfaces[executorId], providerIdentityHash: computeCanonicalSha256(expected.providers[executorId]) })) };
  f.capture.deadlines = expected.deadlines; f.capture.logicalStageCount = count; f.capture.physicalHostCount = hostCount;
  f.capture.pairedBinding.manifestIdentity = expected.manifestHash;
  for (const run of f.capture.runs) {
    run.generation = generation;
    run.identity = { schema: 'doppler.resident-partition-attempt/v2', modelId: f.manifest.modelId, modelIdentity: expected.manifestHash,
      planId: hashLayerPartitionPlan(plan), requesterId: expected.requesterId, placementGeneration: expected.placementGeneration,
      threadId: `thread-${run.attempt}`, attemptId: `attempt-${run.attempt}`,
      stages: expected.placements.map(({ partitionIndex: index, participantId, transportId, residentId, executionId }) => ({ index, participantId, transportId, residentId, executionId })) };
    let position = 0;
    for (const observed of run.numerical) {
      let prior = null;
      observed.stageExecutions = expected.placements.map((placement, index) => {
        const raw = Buffer.alloc(observed.inputTokenIds.length * plan.hiddenSize * 8);
        const output = index + 1 < count ? { shape: [1, observed.inputTokenIds.length, plan.hiddenSize], dtype: 'f32', precision: 'two-f32',
          components: 2, elementStrideBytes: 8, step: observed.step, seqOffset: position, data: raw.toString('base64'), byteLength: raw.length, storageHash: sha256(raw) } : null;
        const result = { partitionIndex: index, executorId: placement.executorId, method: 'executeStage', completed: true, elapsedMs: 1,
          identity: run.identity, step: observed.step, tokenPosition: position, inputTokenCount: observed.inputTokenIds.length,
          inputTokenIds: observed.inputTokenIds, generationDigest: hashPartitionGeneration(generation), initialExecutionIdentityHash: identityHash,
          activationInput: prior, activationOutput: output, logitsHash: output ? null : sha256(Buffer.from(observed.logits, 'base64')) };
        prior = output; return result;
      });
      position += observed.inputTokenIds.length;
    }
  }
  Object.assign(f.input, { captureBytes: bytes(f.capture), assertOrderedResidentContract, validateOrderedLayerPartitionPlan, hashPartitionGeneration });
  return f;
}
for (const [count, hosts] of [[2, 2], [3, 3], [3, 2], [4, 3], [7, 3]]) test(`ordered synthetic ${count} stages/${hosts} hosts bind all prefixes and exact stage lifecycles`, () => {
  const f = orderedFixture(count, hosts), result = producePhysicalReports(f.input);
  assert.equal(result.reports.length, count);
  assert.equal(result.contract.schema, 'doppler.resident-qualification-contract/v2');
  assert(result.reports.every(r => r.schema === 'doppler.resident-partition-qualification/v2' && r.comparison.steps.length === 55));
  assert.equal(result.reports[0].provenance.physicalHostCount, hosts);
  assert.equal(result.reports[0].provenance.logicalStageCount, count);
});
for (const [name, mutate] of Object.entries({
  'exceeded stage deadline': f => f.capture.runs[0].numerical[0].stageExecutions[0].elapsedMs = 300001,
  'exceeded cleanup deadline': f => f.capture.residentSettlement[0].elapsedMs = 10001,
  'absent stage execution': f => f.capture.runs[0].numerical[0].stageExecutions.pop(),
  'reordered stage execution': f => f.capture.runs[0].numerical[0].stageExecutions.reverse(),
  'copied wrong prefix': f => f.capture.runs[0].numerical[0].stageExecutions[1].step++,
  'wrong generation': f => f.capture.runs[0].numerical[0].stageExecutions[1].generationDigest = 'wrong',
  'wrong input token': f => f.capture.runs[0].numerical[0].stageExecutions[1].inputTokenIds = [9],
  'wrong input activation': f => f.capture.runs[0].numerical[0].stageExecutions[1].activationInput.data = 'AA==',
  'scalar activation reinterpretation': f => f.capture.runs[0].numerical[0].stageExecutions[0].activationOutput.components = 1,
  'extra nonfinal logits': f => f.capture.runs[0].numerical[0].stageExecutions[0].logitsHash = sha256('wrong'),
  'wrong final output binding': f => f.capture.runs[0].numerical[0].stageExecutions[2].logitsHash = sha256('wrong'),
  'unexecuted middle stage': f => f.capture.runs[0].numerical[0].stageExecutions[1].completed = false,
  'wrong actual opening identity': f => f.capture.runs[0].numerical[0].stageExecutions[1].initialExecutionIdentityHash = sha256('wrong'),
  'wrong process cleanup': f => f.capture.executorSettlements[0].executorIdentity = { pid: 999, contextId: 'wrong' },
  'wrong descriptor generation': f => f.capture.descriptors[1].descriptor.generationDigest = 'wrong',
  'duplicate actual executor context': f => { f.capture.descriptors[1].platform = f.capture.descriptors[0].platform; f.capture.descriptors[1].executorIdentity = f.capture.descriptors[0].executorIdentity; },
  'unclosed context': f => f.capture.executorSettlements[1].contextClosed = false,
  'wrong resident cleanup': f => f.capture.residentSettlement[1].executorId = 'other',
  'unclosed attempt': f => f.capture.residentSettlement[1].attemptsClosed = 2,
  'changed browser binary': f => f.capture.descriptors[0].executableIdentity = { sha256: 'c'.repeat(64), sizeBytes: 100 },
  'mislabelled physical count': f => f.capture.physicalHostCount = 2,
  'missing stage descriptor': f => f.capture.descriptors.pop(),
  'changed middle layer range': f => f.capture.descriptors[1].descriptor.layerRange = [9, 15],
  'v1 capture without ordered traces': f => { for (const run of f.capture.runs) for (const step of run.numerical) delete step.stageExecutions; },
})) test(`ordered rejects ${name}`, () => {
  const f = orderedFixture(); mutate(f); f.input.captureBytes = bytes(f.capture);
  assert.throws(() => producePhysicalReports(f.input));
});
function orderedNativeFixture() {
  const f = orderedFixture(), template = nativeFixture();
  f.input.expected.nativeHosts = {};
  f.capture.nativeHostSettlements = [];
  for (const index of [0, 1]) {
    const d = f.capture.descriptors[index], key = d.executorId, frozen = structuredClone(template.input.expected.nativeHosts.first);
    const original = template.capture.descriptors[0];
    Object.assign(d, { surface: 'node-webgpu', nativePid: d.executorIdentity.pid,
      node: original.node, nativeProvider: structuredClone(original.nativeProvider), nativeHost: structuredClone(original.nativeHost) });
    d.nativeHost.testDelivery.archiveSha256 = f.input.expected.archiveHash.slice(7);
    delete d.browser;
    f.input.expected.surfaces[key] = 'node-webgpu'; f.input.expected.nativeHosts[key] = frozen;
    f.input.expected.orderedContract.stages[index].surface = 'node-webgpu';
    f.capture.nativeHostSettlements.push({ executorId: key, pid: d.nativePid, closed: true, providerOpened: true, ownedProviderReleased: true });
  }
  f.input.captureBytes = bytes(f.capture); return f;
}
test('ordered mixed route binds two native process/provider settlements independently', () => {
  const f = orderedNativeFixture();
  assert.deepEqual(producePhysicalReports(f.input).reports.map(r => r.surface), ['node-webgpu', 'node-webgpu', 'browser-webgpu']);
});
for (const [name, mutate] of Object.entries({
  'missing second native release': f => f.capture.nativeHostSettlements.pop(),
  'borrowed process release': f => f.capture.nativeHostSettlements[1].pid = f.capture.nativeHostSettlements[0].pid,
  'native/context pid disagreement': f => f.capture.descriptors[0].nativePid++,
  'duplicate native settlement': f => f.capture.nativeHostSettlements[1] = f.capture.nativeHostSettlements[0],
})) test(`ordered rejects ${name}`, () => {
  const f = orderedNativeFixture(); mutate(f); f.input.captureBytes = bytes(f.capture);
  assert.throws(() => producePhysicalReports(f.input));
});

test('ordered cleanup preserves reusable allocation categories while requiring empty live ownership', () => {
  const f = orderedFixture();
  f.capture.residentSettlement[0].snapshot.device.categories.weights = 256;
  f.capture.residentSettlement[0].snapshot.pool.resources.reusable = { count: 1, bytes: 256 };
  f.input.captureBytes = bytes(f.capture); assert.equal(producePhysicalReports(f.input).reports.length, 3);
  f.capture.residentSettlement[0].snapshot.pool.resources.active.bytes = 4;
  f.input.captureBytes = bytes(f.capture); assert.throws(() => producePhysicalReports(f.input));
});

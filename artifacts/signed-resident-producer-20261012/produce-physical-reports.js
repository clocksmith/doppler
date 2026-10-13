// Offline evidence conversion only. Never executes a model or changes a reference.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

export const sha256 = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const parse = bytes => JSON.parse(bytes);
const same = (actual, expected, message) => assert.deepEqual(actual, expected, message);
function vector(text, bytesPerElement, length) {
  assert.equal(typeof text, 'string', 'Raw logits required');
  const bytes = Buffer.from(text, 'base64');
  assert.equal(bytes.toString('base64'), text, 'Noncanonical or corrupt logits encoding');
  assert.equal(bytes.length, length * bytesPerElement, 'Vocabulary byte count differs');
  const values = bytesPerElement === 8 ? new Float64Array(length) : new Float32Array(length);
  for (let i = 0; i < length; i++) values[i] = bytesPerElement === 8 ? bytes.readDoubleLE(i * 8) : bytes.readFloatLE(i * 4);
  assert(values.every(Number.isFinite), 'Nonfinite logits');
  return values;
}

/** Ordered capture checks bind actual stage execution, not descriptive stage inventory. */
function orderedStageExecutions({ expected, capture, manifest, reference, computeCanonicalSha256, hashPartitionGeneration }) {
  const plan = expected.partitionPlan, count = plan.stageCount;
  same(expected.logicalStageCount, count);
  same(expected.maxGpuBufferBytes, 1420000000);
  same(expected.deadlines, { stageMs: 300000, cleanupMs: 10000, observationMs: 2000 });
  same(capture.deadlines, expected.deadlines);
  same(expected.placements.map(row => row.partitionIndex), Array.from({ length: count }, (_, index) => index));
  same(new Set(expected.placements.map(row => row.executorId)).size, count, 'Distinct declared executor contexts required');
  for (const placement of expected.placements) {
    for (const key of ['executorId', 'platform', 'participantId', 'transportId', 'residentId', 'executionId']) {
      assert.equal(typeof placement[key], 'string'); assert(placement[key].length, `Missing stage ${key}`);
    }
  }
  same(expected.physicalHostCount, new Set(expected.placements.map(row => row.platform)).size,
    'Physical host count must be separate from logical stage count');
  assert(Number.isSafeInteger(expected.physicalHostCount) && expected.physicalHostCount > 0);
  same(new Set(capture.descriptors.map(row => `${row.platform}:${row.executorIdentity?.pid}:${row.executorIdentity?.contextId}`)).size, count, 'Stage contexts must be distinct');
  same(capture.logicalStageCount, count); same(capture.physicalHostCount, expected.physicalHostCount);
  const generationDigest = hashPartitionGeneration(reference.generation), attempts = new Set();
  for (const run of capture.runs) {
    const identity = run.identity;
    same(identity.schema, 'doppler.resident-partition-attempt/v2');
    same(identity.modelId, manifest.modelId); same(identity.modelIdentity, expected.manifestHash);
    same(identity.planId, expected.orderedContract.partitionPlanHash);
    same(identity.requesterId, expected.requesterId); same(identity.placementGeneration, expected.placementGeneration);
    assert(typeof identity.threadId === 'string' && identity.threadId.length);
    assert(typeof identity.attemptId === 'string' && identity.attemptId.length && !attempts.has(identity.attemptId));
    attempts.add(identity.attemptId);
    same(identity.stages, expected.placements.map(({ partitionIndex: index, participantId, transportId, residentId, executionId }) =>
      ({ index, participantId, transportId, residentId, executionId })));
    let position = 0;
    for (const observed of run.numerical) {
      same(observed.stageExecutions.length, count, 'Every stage execution must be observed for every prefix');
      let prior = null;
      for (const [index, execution] of observed.stageExecutions.entries()) {
        const placement = expected.placements[index], descriptor = capture.descriptors[index];
        same(execution.partitionIndex, index); same(execution.executorId, placement.executorId);
        same(execution.method, 'executeStage'); same(execution.completed, true);
        assert(Number.isFinite(execution.elapsedMs) && execution.elapsedMs >= 0 && execution.elapsedMs <= expected.deadlines.stageMs, 'Stage deadline not met');
        same(execution.identity, identity); same(execution.step, observed.step);
        same(execution.tokenPosition, position); same(execution.inputTokenCount, observed.inputTokenIds.length);
        same(execution.inputTokenIds, observed.inputTokenIds); same(execution.generationDigest, generationDigest);
        same(execution.initialExecutionIdentityHash, computeCanonicalSha256(descriptor.openingObservation.initialExecutionIdentity));
        same(execution.initialExecutionIdentityHash, placement.executionId, 'Stage execution does not use the declared prepared identity');
        same(execution.activationInput, prior, 'Activation boundary changed between ordered stages');
        if (index + 1 < count) {
          const activation = execution.activationOutput;
          same(activation.shape, [1, observed.inputTokenIds.length, plan.hiddenSize]);
          same(activation.dtype, plan.activationDtype); same(activation.precision, plan.activationPrecision);
          same(activation.components, plan.activationComponents); same(activation.elementStrideBytes, plan.activationElementStrideBytes);
          same(activation.step, observed.step); same(activation.seqOffset, position);
          const raw = Buffer.from(activation.data, 'base64'); same(raw.toString('base64'), activation.data);
          same(raw.length, observed.inputTokenIds.length * plan.hiddenSize * plan.activationElementStrideBytes);
          same(activation.byteLength, raw.length); same(activation.storageHash, sha256(raw));
          same(execution.logitsHash, null, 'Nonfinal stages cannot claim sampling output');
          prior = activation;
        } else {
          same(execution.activationOutput, null);
          same(execution.logitsHash, sha256(Buffer.from(observed.logits, 'base64')));
        }
      }
      position += observed.inputTokenIds.length;
    }
  }
}

/** Frozen expectations and independently assembled F64 bytes precede captures.
 * Identity derivation is supplied by the exact installed archive's existing helper.
 * Raw report flags/error summaries are not trusted as numerical comparison.
 */
export function producePhysicalReports({ expected, captureBytes, referenceBytes, manifestBytes, graphBytes,
  runBytes, deliveryBytes, archiveBytes, evaluatorBytes, requestContractBytes, observeInitialExecutionIdentity, computeCanonicalSha256, assertOrderedResidentContract,
  validateOrderedLayerPartitionPlan, hashPartitionGeneration }) {
  const ordered = expected.schema === 'doppler.physical-resident-producer-input/v2';
  assert(ordered || expected.schema === 'doppler.physical-resident-producer-input/v1', 'Unsupported producer input');
  for (const [name, bytes] of Object.entries({ archive: archiveBytes, reference: referenceBytes,
    manifest: manifestBytes, graph: graphBytes, evaluator: evaluatorBytes })) {
    same(sha256(bytes), expected[`${name}Hash`], `${name} differs from frozen input`);
  }
  const manifest = parse(manifestBytes), graph = parse(graphBytes), reference = parse(referenceBytes);
  const capture = parse(captureBytes), run = runBytes ? parse(runBytes) : null;
  same(manifest.inference.execution, graph, 'Candidate graph differs');
  same(reference.numericalReference.kind, 'independent-f64');
  same(reference.numericalReference.evaluatorSha256, expected.evaluatorHash.slice(7));
  same(reference.numericalReference.proposalSha256, expected.proposalSha256);
  same(reference.numericalReference.outputIndexSha256, expected.outputIndexSha256);
  same(reference.numericalReference.tolerance, 0.001);
  same(reference.numericalReference.steps, 55);
  if (expected.deliveryMode === 'direct-fixture') {
    assert.equal(runBytes, undefined, 'Direct fixture cannot claim an application-controller envelope');
    same(sha256(deliveryBytes), expected.testDeliveryHash, 'Frozen test delivery changed');
    const delivery = parse(deliveryBytes);
    same(delivery.archiveSha256, expected.archiveHash.slice(7));
    same(capture.testDelivery.schema, 'reploid.numerical-test-delivery/v1');
    same(capture.testDelivery.testDeliveryHash, expected.testDeliveryHash);
    same(capture.testDelivery.archiveSha256, delivery.archiveSha256);
    same(capture.testDelivery.snapshotFiles, delivery.files.length);
    assert(Number.isSafeInteger(expected.installedFiles) && expected.installedFiles > 0);
    same(capture.testDelivery.installedFiles, expected.installedFiles, 'Installed archive closure was not fully observed');
    same(capture.testDelivery.applicationQualification, false);
    same(delivery.scope, 'Immutable numerical fixture delivery; not deployed application or signed admission');
    same(capture.testDelivery.scope, delivery.scope);
  } else {
    same(expected.deliveryMode ?? 'application-controller', 'application-controller');
    same(run.package.integrity, 'sha512-' + createHash('sha512').update(archiveBytes).digest('base64'));
    same(run.browserIdentity, expected.applicationBundleHash, 'Application differs');
    same(run.referenceSha256, expected.referenceHash.slice(7));
    same(run.referenceBinding.manifestIdentity, expected.manifestHash);
    same(run.referenceBinding.archiveSha256, expected.archiveHash.slice(7));
    same(run.maxGpuBufferBytes, 1420000000);
    assert.equal(run.ok, true, 'Physical controller did not finish successfully');
    assert(!run.failure, 'Failed physical controller cannot qualify');
  }
  assert(!capture.failure && !capture.cleanupFailure, 'Failed physical execution cannot qualify');
  same(capture.phase, 'reference');
  same(capture.requestFixtureSha256, expected.requestFixtureSha256);
  same(capture.pairedBinding.archiveSha256, expected.archiveHash.slice(7));
  same(capture.pairedBinding.manifestIdentity, expected.manifestHash);
  same(capture.pairedBinding.graphSha256, expected.graphHash.slice(7));
  same(reference.planId, expected.legacyPlanId, 'Frozen legacy traversal differs');
  if (ordered) {
    assert.equal(typeof assertOrderedResidentContract, 'function', 'Exact archive ordered validator required');
    assertOrderedResidentContract(expected.orderedContract);
    validateOrderedLayerPartitionPlan(manifest, expected.partitionPlan);
    same(expected.orderedContract.partitionPlan, expected.partitionPlan);
    same(expected.orderedContract.manifestIdentity, expected.manifestHash);
    same(expected.orderedContract.referenceHash, expected.referenceHash);
    same(expected.orderedContract.evaluatorHash, expected.evaluatorHash);
    same(expected.orderedContract.comparedSteps, 55); same(expected.orderedContract.tolerance, 0.001);
    orderedStageExecutions({ expected, capture, manifest, reference, computeCanonicalSha256, hashPartitionGeneration });
  }
  same(capture.runs.length, reference.expected.length, 'Missing or extra request');
  same(requestContractBytes.length, reference.expected.length);
  same(expected.requestContractHashes.length, reference.expected.length);
  const steps = [];
  for (let request = 0; request < reference.expected.length; request++) {
    same(sha256(requestContractBytes[request]), expected.requestContractHashes[request], 'Frozen tokens differ');
    const tokens = parse(requestContractBytes[request]), actual = capture.runs[request], target = reference.expected[request];
    same(tokens.requestIndex, request); same(actual.attempt, request);
    same(actual.generation, reference.generation, 'Sampling/generation changed');
    assert(actual.completed && !actual.error && !actual.cancelled, 'Incomplete request');
    same(actual.stopReason, 'eos-token'); same(actual.text, target.text);
    same(actual.steps, target.steps.length); same(actual.numerical.length, target.steps.length);
    same(actual.inputTokens, tokens.prefixLengths[0]);
    for (let step = 0; step < target.steps.length; step++) {
      const observed = actual.numerical[step], independent = target.steps[step];
      same(observed.step, step); same(tokens.steps[step], step);
      same(independent.logitsDtype, 'f64', 'Historical F32 outputs are not accuracy targets');
      same(observed.inputTokenIds, step === 0 ? tokens.inputIds.slice(0, tokens.prefixLengths[0])
        : [target.steps[step - 1].tokenId], 'Actual input history changed');
      same(observed.tokenId, independent.tokenId); same(observed.stopReason, independent.stopReason);
      const actualValues = vector(observed.logits, 4, manifest.architecture.vocabSize);
      const targetValues = vector(independent.logits, 8, manifest.architecture.vocabSize);
      let maxAbsoluteError = 0, maxIndex = 0, squaredError = 0;
      for (let i = 0; i < actualValues.length; i++) {
        const error = Math.abs(actualValues[i] - targetValues[i]); squaredError += error * error;
        if (error > maxAbsoluteError) { maxAbsoluteError = error; maxIndex = i; }
      }
      assert(maxAbsoluteError <= 0.001, `Independent comparison failed at request ${request}, step ${step}: ${maxAbsoluteError}`);
      steps.push({ index: steps.length, request, step, maxAbsoluteError, maxIndex,
        rmsError: Math.sqrt(squaredError / actualValues.length) });
    }
  }
  same(steps.length, 55, 'All 55 ordered prefixes required');
  const stageCount = ordered ? expected.partitionPlan.stageCount : 2;
  same(capture.descriptors.length, stageCount); same(capture.residentSettlement.length, stageCount);
  if (!ordered) {
    same(expected.placements.map(row => row.partitionIndex).sort(), [0, 1]);
    same(new Set(expected.placements.map(row => row.platform)).size, 2, 'Two distinct predeclared physical hosts required');
  }
  const contract = ordered ? expected.orderedContract : { schema: 'doppler.resident-qualification-contract/v1', referenceHash: expected.referenceHash,
    evaluatorHash: expected.evaluatorHash, tolerance: 0.001, comparedSteps: 55 };
  const nativeDescriptors = capture.descriptors.filter(row => row.surface === 'node-webgpu');
  if (!ordered) assert(nativeDescriptors.length <= 1, 'This fixture supports one native and one browser participant');
  const reports = capture.descriptors.map((descriptor, slot) => {
    const observed = descriptor.openingObservation, partitionIndex = descriptor.descriptor.index;
    assert(Number.isSafeInteger(partitionIndex) && partitionIndex >= 0 && partitionIndex < stageCount);
    if (ordered) {
      same(partitionIndex, slot); same(descriptor.executorId, expected.placements[slot].executorId);
      same(descriptor.descriptor.schema, 'doppler.resident-partition/v2');
      same(descriptor.descriptor.stageCount, stageCount); same(descriptor.descriptor.ready, true);
      same(descriptor.descriptor.modelId, manifest.modelId); same(descriptor.descriptor.modelIdentity, expected.manifestHash);
      same(descriptor.descriptor.planId, expected.orderedContract.partitionPlanHash);
      same(descriptor.descriptor.generationDigest, hashPartitionGeneration(reference.generation));
      same(descriptor.descriptor.layerRange, expected.partitionPlan.partitions[slot].layerRange);
      assert(Number.isSafeInteger(descriptor.executorIdentity?.pid) && descriptor.executorIdentity.pid > 0);
      assert(typeof descriptor.executorIdentity.contextId === 'string' && descriptor.executorIdentity.contextId.length);
      same(capture.executorSettlements.length, stageCount);
      const closed = capture.executorSettlements[slot];
      same(closed.partitionIndex, partitionIndex); same(closed.executorId, descriptor.executorId);
      same(closed.executorIdentity, descriptor.executorIdentity); same(closed.contextClosed, true);
      assert(!closed.error, 'Executor context did not close');
    }
    const providerKey = ordered ? descriptor.executorId : descriptor.platform;
    same(descriptor.platform, expected.placements.find(row => row.partitionIndex === partitionIndex).platform,
      'Physical placement differs from the pre-run declaration');
    same(descriptor.plan, expected.partitionPlan, 'Declared paired traversal differs');
    same(descriptor.legacyPlanId, expected.legacyPlanId);
    same(observed.partitionIndex, partitionIndex); same(observed.modelIdentity, expected.manifestHash);
    same(observed.planId, descriptor.planId); same(descriptor.planId, computeCanonicalSha256(descriptor.plan));
    same(observed.sources.factory.sha256, expected.factorySha256);
    same(observed.sources.identity.sha256, expected.identityHelperSha256);
    assert(expected.providers[providerKey], 'Provider was not declared before observation');
    same(descriptor.observedProvider, expected.providers[providerKey], 'Provider differs');
    if (ordered) assert(expected.surfaces?.[providerKey], 'Ordered executor surface must be explicit');
    const surface = expected.surfaces?.[providerKey] ?? 'browser-webgpu';
    assert(['browser-webgpu', 'node-webgpu'].includes(surface), 'Unknown declared provider surface');
    // Old browser captures predate the explicit surface field. No native fallback.
    same(descriptor.surface ?? 'browser-webgpu', surface, 'Provider surface differs');
    if (ordered) {
      same(contract.stages[partitionIndex].surface, surface);
      same(contract.stages[partitionIndex].providerIdentityHash, computeCanonicalSha256(descriptor.observedProvider));
      same(descriptor.executableIdentity, expected.executables[providerKey], 'Executor executable changed');
      assert.match(descriptor.executableIdentity.sha256, /^[a-f0-9]{64}$/);
      assert(Number.isSafeInteger(descriptor.executableIdentity.sizeBytes) && descriptor.executableIdentity.sizeBytes > 0);
    }
    if (surface === 'node-webgpu') {
      const native = expected.nativeHosts?.[providerKey];
      assert(native, 'Native host must be frozen before execution');
      same(descriptor.node, native.node, 'Node version differs');
      same(descriptor.nativeHost?.schema, 'reploid.native-partition-fixture/v1');
      same(descriptor.nativeHost?.nativeDeliverySha256, native.nativeDeliverySha256, 'Native application differs');
      same(descriptor.nativeHost?.testDelivery?.archiveSha256, expected.archiveHash.slice(7));
      same(descriptor.nativeHost?.testDelivery?.installedFiles, expected.installedFiles);
      same(descriptor.nativeHost?.testDelivery?.testDeliveryHash, native.testDeliveryHash, 'Native fixture delivery differs');
      assert.match(native.testDeliveryHash, /^sha256:[a-f0-9]{64}$/);
      assert.equal(typeof native.node, 'string'); assert(native.node.length);
      assert.match(native.nativeDeliverySha256, /^[a-f0-9]{64}$/);
      same(native.provider?.module, 'webgpu');
      for (const key of ['entrySha256', 'packageSha256']) assert.match(native.provider[key], /^[a-f0-9]{64}$/);
      same(descriptor.nativeProvider, native.provider, 'Actual mapped native provider differs');
      assert(native.provider?.libraries?.some(library => library.path.endsWith('.node') && library.path.includes('webgpu')),
        'Native WebGPU binary is missing');
      for (const library of native.provider.libraries) {
        assert.match(library.sha256, /^[a-f0-9]{64}$/);
        assert(Number.isSafeInteger(library.bytes) && library.bytes > 0);
      }
      if (ordered) same(descriptor.nativePid, descriptor.executorIdentity.pid, 'Native executor process identity differs');
      assert(Number.isSafeInteger(descriptor.nativePid) && descriptor.nativePid > 0, 'Observed native process identity is missing');
      const nativeSettlement = ordered ? capture.nativeHostSettlements?.find(row => row.executorId === descriptor.executorId) : capture.nativeHostSettlement;
      if (ordered) {
        same(capture.nativeHostSettlements.length, nativeDescriptors.length);
        same(new Set(capture.nativeHostSettlements.map(row => row.executorId)).size, nativeDescriptors.length);
      }
      same(nativeSettlement?.pid, descriptor.nativePid, 'Cleanup belongs to another native process');
      same(nativeSettlement.closed, true);
      same(nativeSettlement.providerOpened, true);
      same(nativeSettlement.ownedProviderReleased, true);
      assert(!nativeSettlement.error, 'Native provider cleanup failed');
    } else {
      assert.equal(typeof descriptor.browser, 'string'); assert(descriptor.browser.length);
      same(descriptor.browser, expected.browserVersions[providerKey], 'Browser version differs from pre-run observation');
    }
    const session = observed.resolvedRuntimeSession.runtime.session;
    same(session.intermediatePrecision, 'two-f32'); same(session.linearRecurrentStatePrecision, 'two-f32');
    const identity = observeInitialExecutionIdentity(observed.resolvedRuntimeSession);
    same(identity, observed.initialExecutionIdentity, 'Recorded execution identity differs from actual resolved session');
    same(observed.resolvedRuntimeSession.manifestInference?.execution, graph,
      'Observed execution graph differs from frozen candidate graph');
    if (ordered) same(identity.executionGraphHash, contract.executionGraphHash);
    const settlement = capture.residentSettlement[slot]; assert(!settlement.error);
    if (ordered) {
      same(settlement.partitionIndex, partitionIndex); same(settlement.executorId, descriptor.executorId);
      same(settlement.initialExecutionIdentityHash, computeCanonicalSha256(identity));
      same(settlement.residentClosed, true); same(settlement.attemptsClosed, capture.runs.length);
      assert(Number.isFinite(settlement.elapsedMs) && settlement.elapsedMs >= 0 && settlement.elapsedMs <= expected.deadlines.cleanupMs, 'Cleanup deadline not met');
    }
    const { snapshot } = settlement;
    for (const owner of ['retainedModel', 'active', 'deferredCleanup']) same(snapshot.pool.resources[owner].count, 0, 'Ownership did not settle');
    if (ordered) {
      for (const owner of ['retainedModel', 'active', 'deferredCleanup']) same(snapshot.pool.resources[owner].bytes, 0, 'Retained ownership bytes did not settle');
    } else same(snapshot.device.categories.weights, 0);
    same(snapshot.device.maxBytes, 1420000000);
    assert(Number.isSafeInteger(snapshot.device.peakBytes) && snapshot.device.peakBytes >= 0
      && snapshot.device.peakBytes <= 1420000000);
    assert(Number.isSafeInteger(settlement.barriers) && settlement.barriers > 0, 'Queue settlement was not confirmed');
    same(snapshot.deviceEvents, [], 'Device event was observed');
    same(snapshot.observationErrors, [], 'Memory observation failed');
    same(snapshot.gpuValidationErrors, []);
    return { schema: ordered ? 'doppler.resident-partition-qualification/v2' : 'doppler.resident-partition-qualification/v1', operation: 'residentPartition', status: 'passed',
      modelId: manifest.modelId, manifestIdentity: expected.manifestHash, surface,
      executionGraphHash: identity.executionGraphHash, initialExecutionIdentity: identity,
      ...(ordered ? { providerIdentity: descriptor.observedProvider } : {}),
      partitionPlan: descriptor.plan, partitionPlanHash: descriptor.planId, partitionIndex,
      comparison: { contractHash: computeCanonicalSha256(contract), referenceHash: expected.referenceHash,
        captureHash: sha256(captureBytes), evaluatorHash: expected.evaluatorHash, tolerance: 0.001, complete: true, steps },
      provenance: { ...(runBytes ? { controllerHash: sha256(runBytes), applicationBundleHash: expected.applicationBundleHash }
          : { testDeliveryHash: expected.testDeliveryHash, applicationQualification: false }), archiveHash: expected.archiveHash, provider: descriptor.observedProvider,
        ...(surface === 'node-webgpu' ? { nodeVersion: descriptor.node, nativeDeliverySha256: descriptor.nativeHost.nativeDeliverySha256, nativeProvider: descriptor.nativeProvider }
          : { browserVersion: descriptor.browser }), pairedProviders: capture.descriptors.map(x => x.observedProvider),
        ...(ordered ? { logicalStageCount: stageCount, physicalHostCount: expected.physicalHostCount, executorId: descriptor.executorId, executableIdentity: descriptor.executableIdentity } : {}),
        scope: ordered ? 'This resident participates in the captured ordered route. Logical stages and physical hosts are distinct; signed opening, authenticated transport, standalone accuracy and deployment are not established.' : 'This resident participates in the captured cooperative pair. Neither isolated-partition accuracy, signed opening, source answer quality nor deployment is established.' } };
  });
  same(reports.map(row => row.partitionIndex).sort((a, b) => a - b), Array.from({ length: stageCount }, (_, index) => index));
  for (const report of reports) same(report.partitionPlan, reports[0].partitionPlan, 'Route used different plans');
  return { contract, reports };
}

/** Explicit post-run validator correction; never edits the pre-run expectations. */
export function validateProducerBinding(expected, expectationsHash, producerHash, captureHash, correction = null) {
  if (expected.producerHash === producerHash) {
    same(correction, null, 'Unexpected validator correction for unchanged producer');
    return;
  }
  assert(correction, 'Producer differs from pre-run freeze; reviewed correction binding required');
  same(correction.schema, 'doppler.resident-report-validator-correction/v1');
  same(correction.correction, 'execution-graph-hash-domain-and-observed-device-schema');
  same(correction.reviewStatus, 'approved');
  same(correction.originalExpectationsHash, expectationsHash);
  same(correction.originalProducerHash, expected.producerHash);
  same(correction.correctedProducerHash, producerHash);
  same(correction.captureHash, captureHash);
  same(correction.referenceHash, expected.referenceHash);
}

async function main() {
  const [configPath, outputDirectory] = process.argv.slice(2);
  assert(configPath && outputDirectory, 'Usage: node produce-physical-reports.js CONFIG.json NEW_OUTPUT_DIRECTORY');
  const config = parse(await readFile(configPath));
  const readBound = async binding => {
    assert(binding?.path && /^sha256:[a-f0-9]{64}$/.test(binding.hash), 'Exact retained file binding required');
    const bytes = await readFile(binding.path); same(sha256(bytes), binding.hash, 'Retained file changed'); return bytes;
  };
  const expected = parse(await readBound(config.expectations));
  const producerHash = sha256(await readFile(new URL(import.meta.url)));
  const correctionBytes = config.producerReevaluation ? await readBound(config.producerReevaluation) : null;
  validateProducerBinding(expected, config.expectations.hash, producerHash, config.capture.hash,
    correctionBytes ? parse(correctionBytes) : null);
  const fields = Object.fromEntries(await Promise.all(['capture', 'reference', 'manifest', 'graph', 'archive', 'evaluator',
    expected.deliveryMode === 'direct-fixture' ? 'delivery' : 'run']
    .map(async name => [name + 'Bytes', await readBound(config[name])])));
  // Import the exact archive closure, never the live checkout or copied helper dependencies.
  const extracted = await mkdtemp(resolve(tmpdir(), 'doppler-resident-producer-'));
  try {
    const archivePath = resolve(extracted, 'bound-package.tgz');
    await writeFile(archivePath, fields.archiveBytes, { flag: 'wx' });
    const execute = promisify(execFile);
    const { stdout: listing } = await execute('tar', ['-tzf', archivePath]);
    assert(listing.trim().split('\n').every(path => path.startsWith('package/')
      && !path.split('/').includes('..')), 'Archive contains an unexpected member path');
    await execute('tar', ['-xzf', archivePath, '-C', extracted]);
    const helperPath = resolve(extracted, 'package/src/config/initial-execution-identity.js');
    same(sha256(await readFile(helperPath)).slice(7), expected.identityHelperSha256);
    same(sha256(await readFile(resolve(extracted, 'package/src/client/resident-partitions.js'))).slice(7),
      expected.factorySha256);
    const { observeInitialExecutionIdentity } = await import(pathToFileURL(helperPath));
    const { computeCanonicalSha256 } = await import(pathToFileURL(
      resolve(extracted, 'package/src/formats/canonical-hash.js')));
    const orderedPorts = {};
    if (expected.schema === 'doppler.physical-resident-producer-input/v2') {
      Object.assign(orderedPorts, await import(pathToFileURL(resolve(extracted, 'package/src/config/ordered-resident-qualification.js'))));
      Object.assign(orderedPorts, await import(pathToFileURL(resolve(extracted, 'package/src/config/layer-partition-plan.js'))));
      Object.assign(orderedPorts, await import(pathToFileURL(resolve(extracted, 'package/src/inference/pipelines/text/resident-partition-contract.js'))));
    }
    const result = producePhysicalReports({ ...fields, expected, ...orderedPorts,
      requestContractBytes: await Promise.all(config.requestContracts.map(readBound)),
      observeInitialExecutionIdentity, computeCanonicalSha256 });
    const { assertResidentQualification } = await import(pathToFileURL(
      resolve(extracted, 'package/src/converter/resident-qualification.js')));
    for (const report of result.reports) assertResidentQualification(report, {
      manifest: parse(fields.manifestBytes), manifestIdentity: expected.manifestHash,
      executionGraphHash: report.executionGraphHash, initialExecutionIdentity: report.initialExecutionIdentity,
      contract: result.contract,
    });
    if (orderedPorts.assertOrderedResidentQualificationSet) orderedPorts.assertOrderedResidentQualificationSet(result.reports.map(report => ({
      ...report, orderedContract: result.contract, providerIdentityHash: computeCanonicalSha256(report.providerIdentity), comparedSteps: report.comparison.steps.length,
    })));
    await mkdir(outputDirectory); // Existing evidence is never overwritten.
    await writeFile(resolve(outputDirectory, 'contract.json'), JSON.stringify(result.contract, null, 2) + '\n', { flag: 'wx' });
    for (const report of result.reports) {
      report.provenance.producerHash = producerHash;
      if (correctionBytes) Object.assign(report.provenance, {
        preRunProducerHash: expected.producerHash, preRunExpectationsHash: config.expectations.hash,
        validatorCorrectionHash: sha256(correctionBytes), qualificationReview: 'post-run-validator-correction',
      });
      await writeFile(resolve(outputDirectory, `partition-${report.partitionIndex}.json`), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    }
  } finally {
    await rm(extracted, { recursive: true, force: true });
  }
}
if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await main();

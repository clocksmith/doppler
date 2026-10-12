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

/** Frozen expectations and independently assembled F64 bytes precede captures.
 * Identity derivation is supplied by the exact installed archive's existing helper.
 * Raw report flags/error summaries are not trusted as numerical comparison.
 */
export function producePhysicalReports({ expected, captureBytes, referenceBytes, manifestBytes, graphBytes,
  runBytes, archiveBytes, evaluatorBytes, requestContractBytes, observeInitialExecutionIdentity, computeCanonicalSha256 }) {
  same(expected.schema, 'doppler.physical-resident-producer-input/v1');
  for (const [name, bytes] of Object.entries({ archive: archiveBytes, reference: referenceBytes,
    manifest: manifestBytes, graph: graphBytes, evaluator: evaluatorBytes })) {
    same(sha256(bytes), expected[`${name}Hash`], `${name} differs from frozen input`);
  }
  const manifest = parse(manifestBytes), graph = parse(graphBytes), reference = parse(referenceBytes);
  const capture = parse(captureBytes), run = parse(runBytes);
  same(manifest.inference.execution, graph, 'Candidate graph differs');
  same(reference.numericalReference.kind, 'independent-f64');
  same(reference.numericalReference.evaluatorSha256, expected.evaluatorHash.slice(7));
  same(reference.numericalReference.proposalSha256, expected.proposalSha256);
  same(reference.numericalReference.outputIndexSha256, expected.outputIndexSha256);
  same(reference.numericalReference.tolerance, 0.001);
  same(reference.numericalReference.steps, 55);
  same(run.package.integrity, 'sha512-' + createHash('sha512').update(archiveBytes).digest('base64'));
  same(run.browserIdentity, expected.applicationBundleHash, 'Application differs');
  same(run.referenceSha256, expected.referenceHash.slice(7));
  same(run.referenceBinding.manifestIdentity, expected.manifestHash);
  same(run.referenceBinding.archiveSha256, expected.archiveHash.slice(7));
  same(run.maxGpuBufferBytes, 1420000000);
  assert.equal(run.ok, true, 'Physical controller did not finish successfully');
  assert(!run.failure && !capture.failure && !capture.cleanupFailure, 'Failed physical execution cannot qualify');
  same(capture.phase, 'reference');
  same(capture.requestFixtureSha256, expected.requestFixtureSha256);
  same(capture.pairedBinding.archiveSha256, expected.archiveHash.slice(7));
  same(capture.pairedBinding.manifestIdentity, expected.manifestHash);
  same(capture.pairedBinding.graphSha256, expected.graphHash.slice(7));
  same(reference.planId, expected.legacyPlanId, 'Frozen legacy traversal differs');
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
  same(capture.descriptors.length, 2); same(capture.residentSettlement.length, 2);
  same(expected.placements.map(row => row.partitionIndex).sort(), [0, 1]);
  same(new Set(expected.placements.map(row => row.platform)).size, 2, 'Two distinct predeclared physical hosts required');
  const contract = { schema: 'doppler.resident-qualification-contract/v1', referenceHash: expected.referenceHash,
    evaluatorHash: expected.evaluatorHash, tolerance: 0.001, comparedSteps: 55 };
  const reports = capture.descriptors.map((descriptor, slot) => {
    const observed = descriptor.openingObservation, partitionIndex = descriptor.descriptor.index;
    assert([0, 1].includes(partitionIndex));
    same(descriptor.platform, expected.placements.find(row => row.partitionIndex === partitionIndex).platform,
      'Physical placement differs from the pre-run declaration');
    same(descriptor.plan, expected.partitionPlan, 'Declared paired traversal differs');
    same(descriptor.legacyPlanId, expected.legacyPlanId);
    same(observed.partitionIndex, partitionIndex); same(observed.modelIdentity, expected.manifestHash);
    same(observed.planId, descriptor.planId); same(descriptor.planId, computeCanonicalSha256(descriptor.plan));
    same(observed.sources.factory.sha256, expected.factorySha256);
    same(observed.sources.identity.sha256, expected.identityHelperSha256);
    assert(expected.providers[descriptor.platform], 'Provider was not declared before observation');
    same(descriptor.observedProvider, expected.providers[descriptor.platform], 'Provider differs');
    assert.equal(typeof descriptor.browser, 'string'); assert(descriptor.browser.length);
    const session = observed.resolvedRuntimeSession.runtime.session;
    same(session.intermediatePrecision, 'two-f32'); same(session.linearRecurrentStatePrecision, 'two-f32');
    const identity = observeInitialExecutionIdentity(observed.resolvedRuntimeSession);
    same(identity, observed.initialExecutionIdentity, 'Recorded execution identity differs from actual resolved session');
    same(identity.executionGraphHash, computeCanonicalSha256(graph));
    const settlement = capture.residentSettlement[slot]; assert(!settlement.error);
    const { snapshot } = settlement;
    for (const owner of ['retainedModel', 'active', 'deferredCleanup']) same(snapshot.pool.resources[owner].count, 0, 'Ownership did not settle');
    same(snapshot.device.categories.weights, 0); same(snapshot.device.maxBytes, 1420000000);
    assert(snapshot.device.peakBytes <= 1420000000 && snapshot.device.lost === false);
    same(snapshot.gpuValidationErrors, []);
    return { schema: 'doppler.resident-partition-qualification/v1', operation: 'residentPartition', status: 'passed',
      modelId: manifest.modelId, manifestIdentity: expected.manifestHash, surface: 'browser-webgpu',
      executionGraphHash: identity.executionGraphHash, initialExecutionIdentity: identity,
      partitionPlan: descriptor.plan, partitionPlanHash: descriptor.planId, partitionIndex,
      comparison: { contractHash: computeCanonicalSha256(contract), referenceHash: expected.referenceHash,
        captureHash: sha256(captureBytes), evaluatorHash: expected.evaluatorHash, tolerance: 0.001, complete: true, steps },
      provenance: { controllerHash: sha256(runBytes), archiveHash: expected.archiveHash,
        applicationBundleHash: expected.applicationBundleHash, provider: descriptor.observedProvider,
        browserVersion: descriptor.browser, pairedProviders: capture.descriptors.map(x => x.observedProvider),
        scope: 'This resident participates in the captured cooperative pair. Neither isolated-partition accuracy, signed opening, source answer quality nor deployment is established.' } };
  });
  same(reports.map(row => row.partitionIndex).sort(), [0, 1]);
  same(reports[0].partitionPlan, reports[1].partitionPlan, 'Pair used different plans');
  return { contract, reports };
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
  same(expected.producerHash, sha256(await readFile(new URL(import.meta.url))), 'Producer differs from pre-run freeze');
  const fields = Object.fromEntries(await Promise.all(['capture', 'reference', 'manifest', 'graph', 'run', 'archive', 'evaluator']
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
    const result = producePhysicalReports({ ...fields, expected,
      requestContractBytes: await Promise.all(config.requestContracts.map(readBound)),
      observeInitialExecutionIdentity, computeCanonicalSha256 });
    const { assertResidentQualification } = await import(pathToFileURL(
      resolve(extracted, 'package/src/converter/resident-qualification.js')));
    for (const report of result.reports) assertResidentQualification(report, {
      manifest: parse(fields.manifestBytes), manifestIdentity: expected.manifestHash,
      executionGraphHash: report.executionGraphHash, initialExecutionIdentity: report.initialExecutionIdentity,
      contract: result.contract,
    });
    await mkdir(outputDirectory); // Existing evidence is never overwritten.
    await writeFile(resolve(outputDirectory, 'contract.json'), JSON.stringify(result.contract, null, 2) + '\n', { flag: 'wx' });
    for (const report of result.reports) {
      report.provenance.producerHash = sha256(await readFile(new URL(import.meta.url)));
      await writeFile(resolve(outputDirectory, `partition-${report.partitionIndex}.json`), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    }
  } finally {
    await rm(extracted, { recursive: true, force: true });
  }
}
if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await main();

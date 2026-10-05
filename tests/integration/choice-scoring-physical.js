// Installed-package physical qualification. Run in an isolated Node process.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { create, globals } from 'webgpu';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { assertChoiceScoringReferenceTranscript } from '../../src/config/choice-scoring-reference.js';
import { hashStableJson } from '../../src/tooling/program-bundle/materialize.js';

const [packageRoot, modelPath, contractPath, referencePath, outputPath, archivePath] = process.argv.slice(2);
assert(archivePath, 'Expected installed package, model, contract, CPU reference, output, archive paths');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const read = async path => JSON.parse(await readFile(path, 'utf8'));
const contract = await read(contractPath);
const reference = await read(referencePath);
assert.equal(reference.contractSha256, sha(await readFile(contractPath)), 'CPU reference contract changed');
const manifestBytes = await readFile(resolve(modelPath, 'manifest.json'));
const manifest = JSON.parse(manifestBytes);
assert.equal(sha(manifestBytes), contract.manifestSha256, 'Model manifest changed');
assert.equal(reference.modelManifestSha256, contract.manifestSha256);
const metadata = await read(resolve(packageRoot, 'package.json'));
assert.equal(metadata.name, 'doppler-gpu');
const report = { schema: 'doppler.choice-scoring-physical/v1', passed: false,
  archiveSha256: sha(await readFile(archivePath)), runtimeVersion: metadata.version,
  contractSha256: reference.contractSha256, referenceSha256: sha(await readFile(referencePath)),
  manifestSha256: contract.manifestSha256, modelId: contract.modelId,
  provider: { name: 'webgpu', version: '0.4.0', backend: 'vulkan' }, cases: [], checks: [] };
const devices = [];
const destroyed = new WeakSet();
let submittedControl = null, model;
Object.assign(globalThis, globals);
const gpu = create(['backend=vulkan']);
Object.defineProperty(globalThis, 'navigator', { value: { gpu }, configurable: true });
const requestAdapter = gpu.requestAdapter.bind(gpu);
gpu.requestAdapter = async options => {
  const adapter = await requestAdapter(options);
  const requestDevice = adapter.requestDevice.bind(adapter);
  adapter.requestDevice = async options => {
    const device = await requestDevice(options);
    const destroy = device.destroy.bind(device);
    device.destroy = () => { destroyed.add(device); return destroy(); };
    const submit = device.queue.submit.bind(device.queue);
    device.queue.submit = commands => {
      const result = submit(commands);
      const abort = submittedControl;
      submittedControl = null;
      abort?.abort(new Error('Cancelled after real GPU submission'));
      return result;
    };
    devices.push(device);
    return device;
  };
  return adapter;
};
try {
  const api = await import(pathToFileURL(resolve(packageRoot, metadata.exports['./compat'].import)));
  const validation = await import(pathToFileURL(resolve(packageRoot, metadata.exports['./choice-scoring-contract'].import)));
  model = await api.load({ url: pathToFileURL(resolve(modelPath) + '/').href }, { cache: false, isolatedLoader: true });
  report.deviceInfo = model.deviceInfo;
  assert(!/swiftshader|llvmpipe|software/i.test(JSON.stringify(report.deviceInfo)), 'Physical GPU required');
  const request = expected => ({ prompt: expected.prompt, choices: contract.choices, maxSeqLen: contract.maxSeqLen });
  for (const expected of reference.cases) {
    const input = request(expected);
    const result = validation.validateChoiceScoringResult(input, await model.scoreChoices(input));
    const tokenIds = result.choices.map(choice => choice.tokenId);
    const errors = result.choices.map((choice, index) => Math.abs(choice.logit - expected.logits[index]));
    const tokensMatch = JSON.stringify(tokenIds) === JSON.stringify(expected.tokenIds)
      && result.promptTokenCount === expected.promptTokenIds.length;
    const maximumAbsoluteLogitError = Math.max(...errors);
    const passed = tokensMatch && maximumAbsoluteLogitError <= contract.maximumAbsoluteLogitError
      && (!contract.requireReferenceSelection || result.selectedId === expected.selectedId);
    const promptTokenIds = model.advanced.tokenizePrompt(input.prompt, { useChatTemplate: false });
    report.cases.push({ id: expected.id, input, output: result, promptTokenIds, expectedId: expected.expectedId,
      maximumAbsoluteLogitError, tokensMatch, passed });
    console.log(JSON.stringify({ id: expected.id, selectedId: result.selectedId, maximumAbsoluteLogitError, passed }));
  }
  const before = new AbortController();
  before.abort(new Error('Cancelled before scoring'));
  await assert.rejects(model.scoreChoices(request(reference.cases[0]), { signal: before.signal }), /Cancelled before/);
  report.checks.push({ id: 'cancel-before-submission', passed: true });
  const during = new AbortController();
  submittedControl = during;
  await assert.rejects(model.scoreChoices(request(reference.cases[0]), { signal: during.signal }), /Cancelled after real/);
  assert(during.signal.aborted, 'Cancellation must occur after real submission');
  for (const device of devices) if (!destroyed.has(device)) await device.queue.onSubmittedWorkDone();
  report.checks.push({ id: 'cancel-after-submission-settled', passed: true });
  const reused = await model.scoreChoices(request(reference.cases[0]));
  assert.equal(reused.selectedId, report.cases[0].output.selectedId);
  assert(reused.choices.every((choice, index) => Math.abs(choice.logit - report.cases[0].output.choices[index].logit)
    <= contract.maximumAbsoluteLogitError));
  report.checks.push({ id: 'resident-reuse-after-cancellation', passed: true });
  report.correctChoices = report.cases.filter(row => row.output.selectedId === row.expectedId).length;
  report.passed = report.cases.every(row => row.passed) && report.correctChoices >= contract.minimumCorrectChoices
    && reference.correctChoices >= contract.minimumCorrectChoices;
} catch (error) {
  report.failure = String(error.stack || error);
  report.passed = false;
} finally {
  try {
    for (const device of devices) if (!destroyed.has(device)) await device.queue.onSubmittedWorkDone();
    await model?.unload();
    for (const device of devices) if (!destroyed.has(device)) device.destroy();
    report.checks.push({ id: 'weights-unloaded-and-owned-devices-destroyed', passed: true });
  } catch (error) { report.cleanupFailure = String(error.stack || error); report.passed = false; }
  await mkdir(dirname(resolve(outputPath)), { recursive: true });
  await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
  if (report.passed) {
    const sourceReference = { schema: 'doppler.choice-scoring-source-reference/v1',
      engine: reference.implementation, engineVersions: reference.versions,
      contractHash: `sha256:${reference.contractSha256}`, manifestHash: `sha256:${reference.modelManifestSha256}`,
      maximumAbsoluteLogitError: contract.maximumAbsoluteLogitError, minimumCorrectChoices: contract.minimumCorrectChoices,
      cases: reference.cases.map(row => ({ id: row.id, input: { prompt: row.prompt,
        choices: contract.choices, maxSeqLen: contract.maxSeqLen }, expectedId: row.expectedId, promptTokenIds: row.promptTokenIds,
      output: { schema: 'doppler.choice-scores/v1', interpretation: 'next-token-logits', calibration: null,
        choices: contract.choices.map((choice, index) => ({ ...choice, tokenId: row.tokenIds[index], logit: row.logits[index] })),
        selectedId: row.selectedId, promptTokenCount: row.promptTokenIds.length } })) };
    const qualification = { schema: 'doppler.choiceScoringModelQualification.v1', passed: true,
      model: { modelId: contract.modelId, manifestHash: `sha256:${contract.manifestSha256}` },
      runtime: { surface: 'node-webgpu', executionGraphHash: hashStableJson(manifest.inference.execution),
        adapterInfo: report.deviceInfo, archiveSha256: report.archiveSha256 },
      reference: sourceReference, referenceDigest: computeCanonicalSha256(sourceReference),
      observation: { cases: report.cases.map(({ id, input, output, promptTokenIds, expectedId }) =>
        ({ id, input, output, promptTokenIds, expectedId })) } };
    assertChoiceScoringReferenceTranscript({ schema: 'doppler.choice-scoring-reference-transcript/v1', operation: 'scoreChoices',
      modelId: qualification.model.modelId, manifestHash: qualification.model.manifestHash,
      surface: qualification.runtime.surface, executionGraphHash: qualification.runtime.executionGraphHash,
      source: { kind: 'physical-test', path: outputPath, hash: `sha256:${sha(await readFile(outputPath))}` },
      reference: qualification.reference, referenceDigest: qualification.referenceDigest, observation: qualification.observation });
    await writeFile(outputPath + '.qualification.json', JSON.stringify(qualification, null, 2) + '\n');
  }
  console.log(JSON.stringify({ outputPath, passed: report.passed, correctChoices: report.correctChoices,
    failure: report.failure ?? null }));
}
// This harness owns the provider process. Exit only after submitted work,
// model cleanup, device destruction and durable evidence have all settled.
process.exit(report.passed ? 0 : 1);

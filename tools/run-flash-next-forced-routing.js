// Test-process-only Flash-Next forced-routing control; never production execution
// or model qualification. Runs the provisioned learned-forward diagnostic with the
// MoE top-k selection observed ('observe') or forced to the CPU source fixture
// ('indices', 'indices-weights'); see tests/helpers/reference-routing-selection-control.js.
//
// Usage: node tools/run-flash-next-forced-routing.js PROTOCOL.json
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const [protocolPath] = process.argv.slice(2);
assert.ok(protocolPath, 'Usage: node tools/run-flash-next-forced-routing.js PROTOCOL.json');
const protocolBytes = await fs.readFile(protocolPath);
const protocol = JSON.parse(protocolBytes);
assert.equal(protocol.schema, 'doppler.flash-next-forced-routing-protocol/v1');
assert.equal(protocol.qualificationAuthority, false, 'a forced-routing control never qualifies the model');
for (const key of ['mode', 'runtimeRoot', 'baseRunPath', 'baseRunSha256', 'fixtureDirectory', 'sequenceId',
  'referencePath', 'inputsPath', 'outputDirectory']) {
  assert.ok(protocol[key] !== undefined && protocol[key] !== null, `protocol.${key} is required`);
}
const root = path.resolve(protocol.runtimeRoot);
const fromRoot = file => pathToFileURL(path.join(root, file)).href;
const output = path.resolve(protocol.outputDirectory);

const baseBytes = await fs.readFile(protocol.baseRunPath);
assert.equal(sha256(baseBytes), protocol.baseRunSha256, 'base run identity');
const run = JSON.parse(baseBytes);
const manifestSha256 = sha256(await fs.readFile(path.join(run.modelDir, 'manifest.json')));
assert.equal(manifestSha256, run.expectedManifestSha256);
const inputsSha256 = sha256(await fs.readFile(protocol.inputsPath));
const referenceSha256 = sha256(await fs.readFile(protocol.referencePath));

const helper = await import(fromRoot('tests/helpers/reference-routing-selection-control.js'));
const fixture = await helper.loadRoutingSelectionFixture(protocol.fixtureDirectory,
  { manifestSha256, inputsSha256, referenceSha256 });
assert.equal(fixture.receipt.sequenceId, protocol.sequenceId);

Object.assign(run, {
  referencePath: path.resolve(protocol.referencePath), inputsPath: path.resolve(protocol.inputsPath),
  outputPath: path.join(output, 'observation.json'), runtimeRevision: protocol.runtimeRevision ?? null,
  sequences: [{ id: protocol.sequenceId, maxTokens: 1 }], captureLayers: false, captureLogitRows: false,
  captureRouterLayers: undefined, captureQsaLayers: undefined, lifecycle: false,
});
assert.equal(run.referenceRouterControl, undefined, 'the logit-replacement control is a different experiment');
await fs.mkdir(output);
await fs.writeFile(path.join(output, 'run.json'), JSON.stringify(run, null, 2) + '\n', { flag: 'wx' });

// The top-k observation hook must be registered before any module loads moe-gpu.js.
const { installMoeRoutingObservation } = await import(fromRoot('tests/helpers/moe-routing-observation-loader.js'));
const routeUrl = fromRoot('src/inference/pipelines/text/moe-gpu.js');
let selection = null;
const routeHook = installMoeRoutingObservation(routeUrl, sha256(await fs.readFile(new URL(routeUrl))), async record => {
  await selection.observeTopK(record);
  await writeControl();
});
const { MoERouter } = await import(fromRoot('src/inference/moe-router.js'));
const memory = await import(fromRoot('src/memory/buffer-pool.js'));
const { getDevice } = await import(fromRoot('src/gpu/device.js'));
// protocol.recordSelections: true writes each layer's executed GPU expert indices to
// OUTPUT/selections for tools/evaluate-routing-aware-acceptance.js.
assert.ok([undefined, true, false].includes(protocol.recordSelections), 'protocol.recordSelections must be boolean');
const selectionDumpDirectory = protocol.recordSelections ? path.join(output, 'selections') : null;
if (selectionDumpDirectory) await fs.mkdir(selectionDumpDirectory);
selection = helper.installReferenceRoutingSelectionControl(fixture, protocol.mode, { MoERouter, memory, getDevice },
  { selectionDumpDirectory });
const control = { protocolSha256: sha256(protocolBytes), baseRunSha256: protocol.baseRunSha256,
  driverSha256: sha256(await fs.readFile(new URL(import.meta.url))),
  helperSha256: sha256(await fs.readFile(new URL(fromRoot('tests/helpers/reference-routing-selection-control.js')))) };
const writeControl = () => fs.writeFile(path.join(output, 'control.json'),
  JSON.stringify({ ...selection.control, ...control }, null, 2) + '\n');
process.env.DOPPLER_FLASH_NEXT_PROBE = path.join(output, 'run.json');
try {
  await import(fromRoot('tests/integration/flash-next-learned-physical.test.js'));
  const layers = fixture.receipt.layers;
  assert.deepEqual(selection.control.topKLayers.map(row => row.layer), Array.from({ length: layers }, (_, i) => i));
  assert.deepEqual(selection.control.routerLayers.map(row => row.layer), Array.from({ length: layers }, (_, i) => i));
  control.completed = true;
} catch (error) {
  control.failure = { name: error.name, message: error.message };
  throw error;
} finally {
  routeHook.restore();
  selection.restore();
  control.routingObservation = routeHook.receipt;
  await writeControl();
  const observed = JSON.parse(await fs.readFile(run.outputPath));
  observed.scope = protocol.mode === 'observe'
    ? 'Test-only routing readback; no arithmetic replacement'
    : 'Counterfactual forced CPU source routing; NOT normal learned-model qualification';
  observed.counterfactual = { ...selection.control, ...control };
  observed.normalModelQualified = false;
  await fs.writeFile(run.outputPath, JSON.stringify(observed, null, 2) + '\n');
}

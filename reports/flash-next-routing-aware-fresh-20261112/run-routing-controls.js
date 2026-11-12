// Unforced-observe and forced-indices routing runs for many pinned cases in one process.
//
// Usage (from the runtime snapshot root):
//   node reports/flash-next-routing-aware-fresh-20261112/run-routing-controls.js BATCH.json
// BATCH: { root, plainRunPath, plainRunSha256, referencePath, inputsPath, runtimeRevision,
//          cases: [caseId, ...], modes: ['observe', 'indices'] }
//
// Same mechanism as tools/run-flash-next-forced-routing.js and
// tests/helpers/reference-routing-selection-control.js ('observe' reads the executed GPU
// top-k back and records it; 'indices' masks router logits outside the CPU fixture set),
// but the model loads once: every Flash-Next GPU process first verifies the n-gram table
// pieces it touches, which costs minutes per process on this host. Each case/mode writes
// ROOT/gpu/<mode>/<case>/{control.json, observation.json, prefill.f32[, selections/]} in the
// shape compare-forced-routing.py and build-evidence.py read. Test-only; never qualification.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const snapshot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fromRoot = file => pathToFileURL(path.join(snapshot, file)).href;
const [batchPath] = process.argv.slice(2);
assert.ok(batchPath, 'Usage: run-routing-controls.js BATCH.json');
const batchBytes = await fs.readFile(batchPath);
const batch = JSON.parse(batchBytes);
const plainBytes = await fs.readFile(batch.plainRunPath);
assert.equal(sha256(plainBytes), batch.plainRunSha256, 'plain run identity');
const run = JSON.parse(plainBytes);
const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
assert.equal(sha256(manifestBytes), run.expectedManifestSha256);
const manifest = JSON.parse(manifestBytes);
const inputsBytes = await fs.readFile(batch.inputsPath);
const inputs = JSON.parse(inputsBytes);
const referenceBytes = await fs.readFile(batch.referencePath);
const reference = JSON.parse(referenceBytes);
const modes = batch.modes ?? ['observe', 'indices'];
assert.ok(modes.every(mode => ['observe', 'indices'].includes(mode)));

const helper = await import(fromRoot('tests/helpers/reference-routing-selection-control.js'));
const { installMoeRoutingObservation } = await import(fromRoot('tests/helpers/moe-routing-observation-loader.js'));
const routeUrl = fromRoot('src/inference/pipelines/text/moe-gpu.js');
let selection = null;
const routeHook = installMoeRoutingObservation(routeUrl, sha256(await fs.readFile(new URL(routeUrl))), async record => {
  if (selection) await selection.observeTopK(record);
});
const [{ MoERouter }, memory, device, gpu, runtime, storage, provider] = await Promise.all([
  import(fromRoot('src/inference/moe-router.js')), import(fromRoot('src/memory/buffer-pool.js')),
  import(fromRoot('src/gpu/device.js')), import(fromRoot('tests/helpers/gpu-probe.js')),
  import(fromRoot('src/inference/pipelines/text.js')), import(fromRoot('src/storage/artifact-storage-context.js')),
  import(fromRoot('src/tooling/node-webgpu.js')),
]);
const batchReport = { schema: 'doppler.flash-next-routing-controls-batch/v1', startedAtUtc: new Date().toISOString(),
  batchSha256: sha256(batchBytes), driverSha256: sha256(await fs.readFile(new URL(import.meta.url))),
  helperSha256: sha256(await fs.readFile(new URL(fromRoot('tests/helpers/reference-routing-selection-control.js')))),
  routingObservation: routeHook.receipt, gpuErrors: [], runs: [] };
const batchOut = path.join(batch.root, 'gpu', `routing-controls-${Date.now()}.json`);
let pipeline, primaryError;
try {
  const probe = await gpu.probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason);
  batchReport.provider = probe.providerReceipt;
  batchReport.adapterInfo = device.getKernelCapabilities()?.adapterInfo ?? null;
  device.getDevice().addEventListener('uncapturederror', event => batchReport.gpuErrors.push(event.error.message));
  pipeline = await runtime.createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
    storage: storage.createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
  for (const id of batch.cases) {
    const input = inputs.sequences.find(row => row.id === id);
    const expected = reference.sequences[id];
    assert.ok(input && expected && input.tokens.length === expected.tokens, `${id}: input and reference required`);
    const fixtureDirectory = path.join(batch.root, 'fixture', id);
    const fixture = await helper.loadRoutingSelectionFixture(fixtureDirectory,
      { manifestSha256: sha256(manifestBytes), inputsSha256: sha256(inputsBytes), referenceSha256: sha256(referenceBytes) });
    assert.equal(fixture.receipt.sequenceId, id);
    for (const mode of modes) {
      const output = path.join(batch.root, 'gpu', mode, id);
      await fs.mkdir(output);
      const selectionDumpDirectory = mode === 'observe' ? path.join(output, 'selections') : null;
      if (selectionDumpDirectory) await fs.mkdir(selectionDumpDirectory);
      const startedAtUtc = new Date().toISOString();
      const errorsBefore = batchReport.gpuErrors.length;
      selection = helper.installReferenceRoutingSelectionControl(fixture, mode,
        { MoERouter, memory, getDevice: device.getDevice }, { selectionDumpDirectory });
      const control = { batchSha256: batchReport.batchSha256, driverSha256: batchReport.driverSha256,
        helperSha256: batchReport.helperSha256, adapterInfo: batchReport.adapterInfo, startedAtUtc, completed: false };
      let logits = null, tokenIds = null;
      try {
        pipeline.resetGenerationState();
        const result = await pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
          maxTokens: 1, temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1,
          onLogits: row => { logits ??= Float32Array.from(row); } });
        tokenIds = Array.from(result.tokenIds);
        const layers = fixture.receipt.layers;
        assert.deepEqual(selection.control.topKLayers.map(row => row.layer), Array.from({ length: layers }, (_, i) => i));
        assert.deepEqual(selection.control.routerLayers.map(row => row.layer), Array.from({ length: layers }, (_, i) => i));
        await device.getDevice().queue.onSubmittedWorkDone();
        control.completed = batchReport.gpuErrors.length === errorsBefore;
      } catch (error) {
        control.failure = { name: error.name, message: error.message };
      } finally {
        selection.restore();
        await fs.writeFile(path.join(output, 'control.json'), JSON.stringify({ ...selection.control, ...control }, null, 2) + '\n');
        selection = null;
      }
      const observation = { schema: 'doppler.flash-next-learned-observation/v1',
        scope: mode === 'observe' ? 'Test-only routing readback; no arithmetic replacement'
          : 'Counterfactual forced CPU source routing; NOT normal learned-model qualification',
        normalModelQualified: false, manifestSha256: sha256(manifestBytes), referenceSha256: sha256(referenceBytes),
        inputsSha256: sha256(inputsBytes), runtimeRevision: batch.runtimeRevision ?? null, completed: control.completed,
        gpuErrors: batchReport.gpuErrors.slice(errorsBefore), cleanupFailures: [], rows: [] };
      if (control.completed && logits) {
        const bytes = Buffer.from(logits.buffer);
        const file = path.join(output, 'prefill.f32');
        await fs.writeFile(file, bytes, { flag: 'wx' });
        observation.rows.push({ id, inputTokenCount: input.tokens.length, tokenId: tokenIds[0], tokenIds,
          prefillLogitsFile: file, prefillLogitsSha256: sha256(bytes), kvStorageDtype: pipeline.kvCache?.kvDtype ?? null });
      }
      await fs.writeFile(path.join(output, 'observation.json'), JSON.stringify(observation, null, 2) + '\n');
      batchReport.runs.push({ id, mode, completed: control.completed, startedAtUtc, failure: control.failure ?? null });
      await fs.writeFile(batchOut, JSON.stringify(batchReport, null, 2) + '\n');
      console.log(JSON.stringify({ id, mode, completed: control.completed, tokenId: tokenIds?.[0] ?? null,
        failure: control.failure?.message ?? null }));
    }
  }
  batchReport.completed = true;
} catch (error) {
  primaryError = error;
  batchReport.failure = { name: error.name, message: error.message, stack: error.stack };
} finally {
  routeHook.restore();
  const failures = [];
  try { await pipeline?.unload(); } catch (error) { failures.push(error); }
  try { device.destroyDevice(); } catch (error) { failures.push(error); }
  try { await provider.releaseNodeWebGPU(); } catch (error) { failures.push(error); }
  batchReport.cleanupFailures = failures.map(String);
  await fs.writeFile(batchOut, JSON.stringify(batchReport, null, 2) + '\n');
  if (primaryError) throw primaryError;
  if (failures.length) throw new AggregateError(failures, 'Routing-controls cleanup failed');
}

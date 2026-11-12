import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { installPrefillLogitRowsObservation } from '../helpers/prefill-logit-rows-loader.js';

// Provisioned GPU task-quality capture for the routing-aware acceptance gate
// (docs/routing-aware-acceptance.md, step 7). For each pinned case it records:
// - teacher-forced prefill logits at the prompt set's sampled positions, from the
//   unchanged prefill logits path evaluated on a prefix of the prefill hidden rows;
// - the greedy continuation (temperature 0, top-k 1) of at most maxNewTokens tokens.
// The full last-position row from the extra path must equal the production prefill
// logits byte for byte. It never reads the CPU reference and never qualifies a model.
const runPath = process.env.DOPPLER_FLASH_NEXT_TASK_QUALITY;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_TASK_QUALITY to the explicit run JSON.');
  console.log('flash-next task-quality capture: skipped (no provisioned run)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath);
  const run = JSON.parse(runBytes);
  assert.equal(run.schema, 'doppler.flash-next-task-quality-run/v1');
  for (const key of ['modelDir', 'expectedManifestSha256', 'runtimeConfig', 'inputsPath', 'promptSetPath', 'outputDirectory',
    'sequences', 'policySha256', 'promptSetSha256']) assert.ok(run[key] !== undefined && run[key] !== null, `run.${key} is required`);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.expectedManifestSha256, 'the run must name the exact prepared manifest');
  const manifest = JSON.parse(manifestBytes);
  const inputsBytes = await fs.readFile(run.inputsPath);
  const inputs = JSON.parse(inputsBytes);
  const promptSet = JSON.parse(await fs.readFile(run.promptSetPath));
  const { samplePositions } = await import('../../tools/lib/routing-aware-acceptance.js');
  const { computeCanonicalJsonSha256 } = await import('../../tools/lib/canonical-json.js');
  assert.equal(computeCanonicalJsonSha256(promptSet), run.promptSetSha256, 'pinned prompt set identity');
  const maxNewTokens = promptSet.generation.maxNewTokens;
  assert.ok(Number.isSafeInteger(maxNewTokens) && maxNewTokens > 0);
  const output = path.resolve(run.outputDirectory);
  await fs.mkdir(output);
  const sourceUrl = new URL('../../src/inference/pipelines/text/generator/prefill-runtime.js', import.meta.url);
  let pending = null;
  const observer = installPrefillLogitRowsObservation(sourceUrl.href, hash(await fs.readFile(sourceUrl)), async record => {
    if (!pending || pending.done || record.startPos !== 0) return;
    assert.equal(record.numTokens, pending.tokens, 'prefill must cover the whole prompt in one call');
    const rows = [];
    for (const position of pending.positions) {
      const logits = await record.compute(position + 1);
      assert.equal(logits.length, record.vocabSize, 'one complete logit row per sampled position');
      assert.ok(logits.every(Number.isFinite), 'sampled logits must be finite');
      rows.push(Float32Array.from(logits));
    }
    pending.rows = rows;
    pending.done = true;
  });
  const report = { schema: 'doppler.flash-next-task-quality-observation/v1',
    scope: 'GPU teacher-forced sampled-position logits and greedy continuations for the routing-aware gate; not model qualification',
    normalModelQualified: false, runStartedAtUtc: new Date().toISOString(),
    process: { pid: process.pid, node: process.versions.node },
    harnessSha256: hash(await fs.readFile(new URL(import.meta.url))), runSha256: hash(runBytes),
    manifestSha256: hash(manifestBytes), inputsSha256: hash(inputsBytes), policySha256: run.policySha256,
    promptSetSha256: run.promptSetSha256, runtimeConfig: run.runtimeConfig, instrumentation: observer.receipt,
    cases: [], gpuErrors: [] };
  let pipeline, destroyDevice, releaseNodeWebGPU, primaryError;
  try {
    const [gpu, runtime, storage, device, provider] = await Promise.all([
      import('../helpers/gpu-probe.js'), import('../../src/inference/pipelines/text.js'),
      import('../../src/storage/artifact-storage-context.js'), import('../../src/gpu/device.js'),
      import('../../src/tooling/node-webgpu.js'),
    ]);
    destroyDevice = device.destroyDevice;
    releaseNodeWebGPU = provider.releaseNodeWebGPU;
    const probe = await gpu.probeNodeGPU({ installFileFetchShim: true });
    assert(probe.ready, probe.reason);
    report.provider = probe.providerReceipt;
    report.adapterInfo = device.getKernelCapabilities()?.adapterInfo ?? null;
    device.getDevice().addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
    pipeline = await runtime.createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
      storage: storage.createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
    report.stopTokenIds = Array.from((pipeline.modelConfig ?? pipeline._state?.modelConfig)?.stopTokenIds ?? []);
    for (const id of run.sequences) {
      const input = inputs.sequences.find(row => row.id === id);
      assert.ok(input && promptSet.cases.some(entry => entry.id === id), `${id}: pinned input required`);
      const tokens = input.tokens.length;
      const positions = samplePositions(promptSet.samplePositions, tokens);
      pending = { id, tokens, positions, rows: null, done: false };
      pipeline.resetGenerationState();
      const frames = [];
      const started = Date.now();
      const result = await pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
        maxTokens: maxNewTokens, temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1,
        onLogits: logits => { frames.push(Float32Array.from(logits)); } });
      assert.ok(pending.done, `${id}: the sampled-position logit rows were not observed`);
      const last = pending.rows[pending.rows.length - 1];
      assert.equal(positions.at(-1), tokens - 1);
      const production = Buffer.from(frames[0].buffer);
      const lastBytes = Buffer.from(last.buffer);
      const rowsBytes = Buffer.concat(pending.rows.map(row => Buffer.from(row.buffer)));
      const rowsFile = path.join(output, `${id}.sampled-logits.f32`);
      await fs.writeFile(rowsFile, rowsBytes, { flag: 'wx' });
      const prefillFile = path.join(output, `${id}.prefill.f32`);
      await fs.writeFile(prefillFile, production, { flag: 'wx' });
      const row = { caseId: id, inputTokenCount: tokens, positions,
        sampledLogits: { file: path.basename(rowsFile), shape: [positions.length, last.length], sha256: hash(rowsBytes) },
        prefillLogits: { file: path.basename(prefillFile), sha256: hash(production) },
        lastSampledRowMatchesProductionPrefill: hash(lastBytes) === hash(production),
        gpuTokens: Array.from(result.tokenIds), gpuText: pipeline.tokenizer.decode(result.tokenIds),
        stopReason: pipeline.stats?.stopReason ?? null, seconds: (Date.now() - started) / 1000 };
      assert.ok(row.lastSampledRowMatchesProductionPrefill, `${id}: extra logits path differs from production prefill logits`);
      report.cases.push(row);
      pending = null;
      await fs.writeFile(path.join(output, 'observation.json'), JSON.stringify(report, null, 2));
      console.log(JSON.stringify({ id, tokens, positions: positions.length, gpuTokens: row.gpuTokens, text: row.gpuText, seconds: row.seconds }));
    }
    await device.getDevice().queue.onSubmittedWorkDone();
    assert.deepEqual(report.gpuErrors, []);
    report.completed = true;
  } catch (error) {
    primaryError = error;
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    observer.restore();
    const failures = [];
    try { await pipeline?.unload(); } catch (error) { failures.push(error); }
    try { destroyDevice?.(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU?.(); } catch (error) { failures.push(error); }
    report.cleanupFailures = failures.map(String);
    await fs.mkdir(output, { recursive: true });
    await fs.writeFile(path.join(output, 'observation.json'), JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(primaryError ? [primaryError, ...failures] : failures, 'Task-quality cleanup failed');
  }
}

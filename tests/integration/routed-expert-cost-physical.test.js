import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { getBufferPool } from '../../src/memory/buffer-pool.js';
import { checkLearnedLifecycle } from '../helpers/flash-next-lifecycle.js';
import { planRoutedExpertBank } from '../../src/loader/experts/routed-expert-plan.js';

// Explicit provisioned comparison, not a quickstart/model-promotion benchmark.
// Both modes observe complete logits with the same inference configuration.
const runPath = process.env.DOPPLER_ROUTED_EXPERT_COST;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_ROUTED_EXPERT_COST to a run JSON.');
  console.log('routed expert cost: skipped (no provisioned run)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath), run = JSON.parse(runBytes);
  assert.equal(run.schema, 'doppler.routed-expert-cost-run/v1');
  assert.ok(['bank', 'routed'].includes(run.mode));
  assert.ok(Number.isSafeInteger(run.repetitions) && run.repetitions > 0);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.manifestSha256);
  const manifest = JSON.parse(manifestBytes);
  const provider = () => execFileSync('systemctl', ['--user', 'show', run.providerUnit,
    '--property=ActiveState,MainPID', '--no-pager'], { encoding: 'utf8' }).trim();
  const providerBefore = provider();
  assert.ok(providerBefore.includes('ActiveState=active'), 'the existing provider must remain active');
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert.ok(probe.ready, probe.reason);
  const device = getDevice(), errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const sourceSha256 = {};
  for (const file of run.sourceFiles) sourceSha256[file] = hash(await fs.readFile(file));
  const report = { schema: 'doppler.routed-expert-cost-observation/v1',
    scope: 'Single-host diagnostic comparison with full-logit observation; no throughput or model-qualification claim',
    coldWarmDefinition: 'First request has an empty expert cache; repeats reset generation state but retain the expert cache. OS caches are neither cleared nor controlled.',
    mode: run.mode, runSha256: hash(runBytes), sourceSha256, manifestSha256: hash(manifestBytes),
    providerBefore, provider: probe.providerReceipt, node: process.versions.node,
    runtimeConfig: run.runtimeConfig, prompt: run.prompt, rows: [], gpuErrors: errors };
  const save = () => fs.writeFile(run.outputPath, JSON.stringify(report, null, 2));
  const disk = async () => Object.fromEntries((await fs.readFile('/proc/self/io', 'utf8')).trim()
    .split('\n').map(line => { const [key, value] = line.split(':'); return [key, Number(value.trim())]; }));
  let pipeline, uploaded = 0, expertTensorBytes = 0, expertTensorLoads = 0;
  const originalWrite = device.queue.writeBuffer.bind(device.queue);
  device.queue.writeBuffer = (buffer, offset, data, dataOffset = 0, size) => {
    const unit = ArrayBuffer.isView(data) ? (data.BYTES_PER_ELEMENT ?? 1) : 1;
    uploaded += size === undefined ? data.byteLength - dataOffset * unit : size * unit;
    return originalWrite(buffer, offset, data, dataOffset, size);
  };
  const snap = async () => ({ atMs: performance.now(), disk: await disk(), uploadBytes: uploaded,
    expertTensorBytes, expertTensorLoads, source: pipeline.dopplerLoader.shardCache.customReadStats,
    routed: pipeline.dopplerLoader.getRoutedExpertCacheStats(), pool: getBufferPool().getStats(),
    memory: process.memoryUsage(), maxRssKiB: process.resourceUsage().maxRSS });
  try {
    const loadedAt = performance.now();
    pipeline = await createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
      storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
    report.loadMs = performance.now() - loadedAt;
    const loader = pipeline.dopplerLoader, originalLoad = loader._loadTensor.bind(loader);
    const expertTensorNames = new Set(Array.from({ length: manifest.architecture.numLayers }, (_, layer) =>
      planRoutedExpertBank(manifest, loader.tensorLocations, layer).planes.map(plane => plane.name)).flat());
    loader._loadTensor = async (name, ...args) => {
      const value = await originalLoad(name, ...args);
      if (expertTensorNames.has(name)) {
        expertTensorLoads++; expertTensorBytes += loader.tensorLocations.get(name).size;
      }
      return value;
    };
    for (let repetition = 0; repetition < run.repetitions; repetition++) {
      pipeline.resetGenerationState();
      const row = { repetition, cache: repetition ? 'retained' : 'empty', before: await snap(), phases: [] };
      report.rows.push(row); await save();
      const logits = [];
      const output = await pipeline.generateTokenIds(run.prompt, { useChatTemplate: false,
        maxTokens: run.maxTokens, temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, seed: 0,
        onLogits(values, metadata) {
          assert.ok(values.every(Number.isFinite));
          logits.push(Buffer.from(Buffer.from(values.buffer, values.byteOffset, values.byteLength)));
          row.phases.push({ atMs: performance.now(), tokenId: metadata.tokenId,
            inputTokenCount: metadata.inputTokenCount, uploadBytes: uploaded,
            expertTensorBytes, expertTensorLoads, source: loader.shardCache.customReadStats,
            routed: loader.getRoutedExpertCacheStats(), pool: getBufferPool().getStats() });
          console.log(JSON.stringify({ mode: run.mode, repetition, phase: row.phases.length,
            tokenId: metadata.tokenId, elapsedMs: performance.now() - row.before.atMs }));
        } });
      await device.queue.onSubmittedWorkDone();
      if (run.mode === 'bank' && repetition === 0) {
        assert.ok(expertTensorLoads > 0, 'the bank payload observer must bind to actual artifact tensor names');
      }
      row.after = await snap(); row.stats = structuredClone(pipeline.stats);
      row.tokenIds = Array.from(output.tokenIds); row.text = output.text ?? pipeline.tokenizer.decode(output.tokenIds);
      row.logits = [];
      for (let i = 0; i < logits.length; i++) {
        const file = `${run.outputPath}.repeat${repetition}.row${i}.f32`;
        await fs.writeFile(file, logits[i]); row.logits.push({ file, sha256: hash(logits[i]), count: logits[i].byteLength / 4 });
      }
      await save();
    }
    if (run.lifecycle) {
      report.lifecycle = [];
      const input = { tokens: await pipeline.tokenizer.encode(run.prompt) };
      await checkLearnedLifecycle(pipeline, input, report.rows[0].tokenIds, async result => {
        report.lifecycle.push(result); await save(); console.log(JSON.stringify({ lifecycle: result.case }));
      });
    }
    assert.deepEqual(errors, []);
    report.completed = true;
  } catch (error) {
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    const cleanup = [];
    try { await pipeline?.unload(); } catch (error) { cleanup.push(String(error)); }
    device.queue.writeBuffer = originalWrite;
    try { destroyDevice(); } catch (error) { cleanup.push(String(error)); }
    try { await releaseNodeWebGPU(); } catch (error) { cleanup.push(String(error)); }
    report.cleanupFailures = cleanup; report.providerAfter = provider();
    await save();
    assert.equal(report.providerAfter, providerBefore, 'the provider service and PID must remain unchanged');
    assert.deepEqual(cleanup, []);
  }
}

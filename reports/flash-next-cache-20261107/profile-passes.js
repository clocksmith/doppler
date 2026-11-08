// Diagnostic-only pass timestamps. Primary cost runs never install this observer.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { probeNodeGPU } from '../../tests/helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { observeComputePasses } from '../../tools/experiments/compute-pass-observer.js';

const hash = value => createHash('sha256').update(value).digest('hex');
const bytes = await fs.readFile(process.argv[2]);
const run = JSON.parse(bytes);
assert.equal(run.schema, 'doppler.flash-next-pass-profile/v1');
const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
assert.equal(hash(manifestBytes), run.manifestSha256);
const report = { schema: 'doppler.flash-next-pass-profile-observation/v1',
  diagnosticOnly: true, performanceRepresentative: false, configSha256: hash(bytes),
  scriptSha256: hash(await fs.readFile(new URL(import.meta.url))), phases: [], gpuErrors: [] };
let pipeline, observer;
try {
  const probe = await probeNodeGPU({ installFileFetchShim: true });
  assert(probe.ready, probe.reason);
  report.provider = probe.providerReceipt;
  const device = getDevice();
  device.addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
  pipeline = await createPipeline(JSON.parse(manifestBytes), { runtimeConfig: run.runtimeConfig,
    storage: createNodeFileArtifactStorageContext(pathToFileURL(run.modelDir).href, JSON.parse(manifestBytes)) });
  const options = { useChatTemplate: false, temperature: 0, topK: 1, topP: 1,
    repetitionPenalty: 1, seed: 0 };
  await pipeline.generateTokenIds(run.prompt, { ...options, maxTokens: run.warmupTokens });
  pipeline.resetGenerationState();
  const cache = pipeline.dopplerLoader.getRoutedExpertCacheStats();
  observer = observeComputePasses(device, run.observer);
  const start = performance.now();
  const result = await pipeline.generateTokenIds(run.prompt, { ...options, maxTokens: run.profileTokens,
    onLogits(values) { report.phases.push({ atMs: performance.now() - start,
      count: values.length, sha256: hash(Buffer.from(values.buffer, values.byteOffset, values.byteLength)),
      routed: pipeline.dopplerLoader.getRoutedExpertCacheStats() }); } });
  await device.queue.onSubmittedWorkDone();
  report.generationWallMs = performance.now() - start;
  report.tokenIds = Array.from(result.tokenIds);
  report.cacheBefore = cache;
  report.cacheAfter = pipeline.dopplerLoader.getRoutedExpertCacheStats();
  report.computePasses = await observer.collect();
  assert.deepEqual(report.gpuErrors, []);
  report.completed = true;
} catch (error) {
  report.failure = { message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  const cleanup = [];
  try { await getDevice()?.queue.onSubmittedWorkDone(); observer?.dispose(); } catch (error) { cleanup.push(String(error)); }
  try { await pipeline?.unload(); } catch (error) { cleanup.push(String(error)); }
  try { destroyDevice(); await releaseNodeWebGPU(); } catch (error) { cleanup.push(String(error)); }
  report.cleanupFailures = cleanup;
  await fs.writeFile(run.outputPath, JSON.stringify(report, null, 2) + '\n');
  if (cleanup.length) process.exitCode = 1;
}

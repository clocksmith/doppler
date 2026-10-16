#!/usr/bin/env node
// Exercise the installed consumer boundary. Does not advertise or change a catalog.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const [configPath, suitePath, outputPath] = process.argv.slice(2);
assert(!await fs.stat(outputPath).catch(() => null), 'Preserve prior observations');
const configBytes = await fs.readFile(configPath), suiteBytes = await fs.readFile(suitePath);
const config = JSON.parse(configBytes), suite = JSON.parse(suiteBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const from = (root, file) => import(pathToFileURL(path.join(root, file)).href);
const consumer = config.reploidDirectory;
const { createNativeWholeModel } = await from(consumer, 'server/native-whole-model.js');
const { createNativeGpuHost } = await from(consumer, 'server/native-participant.js');
const { createReploidDopplerRuntimeService } = await from(consumer, 'self/infrastructure/doppler-runtime-service.js');
const api = await from(config.installedPackage, 'src/index.js');
const { getBufferPool } = await from(config.installedPackage, 'src/memory/buffer-pool.js');
const { getDevice } = await from(config.installedPackage, 'src/gpu/device.js');
const requireAt = createRequire(path.join(consumer, 'package.json'));
const gpu = createNativeGpuHost(() => import(pathToFileURL(requireAt.resolve('webgpu')).href));
const service = createReploidDopplerRuntimeService({ loadModule: async () => api, expectedVersion: api.DOPPLER_VERSION });
const manifestBytes = await fs.readFile(path.join(config.modelDirectory, 'manifest.json'));
assert.equal(hash(manifestBytes), config.manifestSha256);
const manifest = JSON.parse(manifestBytes);
const model = { id: manifest.modelId, identity: 'sha256:' + hash(manifestBytes), provider: 'doppler',
  contextLength: suite.contextTokens, generation: { ...suite.generation, maxSeqLen: suite.contextTokens },
  adapters: [], loadOptions: { runtimeConfig: config.runtimeConfig } };
assert(typeof model.id === 'string');
const report = { schema: 'doppler.installed-reploid-provider-observation/v1', qualified: false,
  scope: 'Native resident provider boundary; no WebRTC, browser persistence, public deployment or simultaneous GPU execution claim',
  configSha256: hash(configBytes), suiteSha256: hash(suiteBytes), model, packageVersion: api.DOPPLER_VERSION,
  toolSha256: hash(await fs.readFile(new URL(import.meta.url))), sourceHashes: {}, turns: [], loads: [], cleanup: [] };
for (const file of ['server/native-whole-model.js', 'server/native-participant.js',
  'self/infrastructure/doppler-runtime-service.js', 'self/providers/work-resident-provider.js',
  'self/providers/work-provider.js', 'self/providers/work-device.js']) {
  report.sourceHashes[file] = hash(await fs.readFile(path.join(consumer, file)));
}
let provider;
async function prepare() {
  const available = Number((await fs.readFile('/proc/meminfo', 'utf8')).match(/^MemAvailable:\s+(\d+)/m)[1]) * 1024;
  const base = '/sys/class/drm/card1/device/';
  const gttTotal = Number(await fs.readFile(base + 'mem_info_gtt_total', 'utf8'));
  const gttUsed = Number(await fs.readFile(base + 'mem_info_gtt_used', 'utf8'));
  const budget = config.runtimeConfig.shared.bufferPool.budget.maxTotalBytes;
  assert(available > budget + 8 * 1024 ** 3 && gttTotal - gttUsed > budget + 1024 ** 3);
  provider = createNativeWholeModel({ model, modelDirectory: config.modelDirectory,
    service, nativeGpu: () => gpu.open() });
  const start = performance.now();
  const ready = await provider.contribute(true);
  assert(ready.ready);
  report.loads.push({ elapsedMs: performance.now() - start, available, gttUsed, gttTotal,
    state: ready, pool: getBufferPool().getStats(), allocations: getBufferPool().getLabelStats() });
}
async function turn(task) {
  const start = performance.now(), chunks = [];
  const result = await provider.generate(task.messages, delta => chunks.push(delta),
    { signal: new AbortController().signal, maxOutputTokens: suite.generation.maxTokens });
  assert.equal(chunks.join(''), result.content);
  assert.equal(result.content.trim(), task.expected);
  assert.equal(result.modelIdentity, model.identity);
  const row = { id: task.id, elapsedMs: performance.now() - start, chunks, result };
  report.turns.push(row);
  return row;
}
async function retire() {
  await provider?.close();
  if (getDevice()) await getDevice().queue.onSubmittedWorkDone();
  const pool = getBufferPool().getStats();
  report.cleanup.push(pool);
  assert.equal(pool.resources.retainedModel.bytes, 0);
  assert.equal(pool.resources.active.bytes, 0);
  assert.equal(pool.resources.deferredCleanup.bytes, 0);
}
try {
  await prepare();
  await turn(suite.tasks[0]);
  await turn(suite.tasks[1]);
  const before = getBufferPool().getStats().resources.retainedModel;
  // The consumer serializes these against one resident session. No duplicate weights.
  await Promise.all([turn(suite.tasks[2]), turn(suite.tasks[3])]);
  const after = getBufferPool().getStats().resources.retainedModel;
  assert.deepEqual(after, before);
  report.concurrentRequests = { count: 2, activeExecutions: 1, before, after, passed: true };
  const capacity = await turn(suite.capacityTask);
  const promptTokens = capacity.result.completion.promptTokenCount;
  assert(promptTokens >= suite.capacityTask.minPromptTokens && promptTokens <= suite.capacityTask.maxPromptTokens);
  report.capacity = { promptTokens, configuredContextTokens: suite.contextTokens,
    outputLimit: suite.generation.maxTokens, pool: getBufferPool().getStats() };
  const controller = new AbortController();
  let chunks = 0, rejection = null;
  try {
    await provider.generate([{ role: 'user', content: suite.cancellationPrompt }], () => {
      chunks++; controller.abort();
    }, { signal: controller.signal, maxOutputTokens: suite.generation.maxTokens });
  } catch (error) { rejection = { name: error.name, message: error.message }; }
  assert(chunks > 0 && controller.signal.aborted && rejection);
  report.cancellation = { chunks, rejection, passed: true };
  await turn(suite.tasks[0]);
  report.poolBeforeRetirement = getBufferPool().getStats();
  await retire();
  await prepare();
  await turn(suite.tasks[0]);
  report.reopened = true;
  report.passed = true;
} catch (error) {
  report.error = { name: error.name, message: error.message, stack: error.stack };
  report.passed = false; process.exitCode = 1;
} finally {
  try { await retire(); }
  catch (error) { report.cleanupError = error.message; report.passed = false; process.exitCode = 1; }
  try { await gpu.close(); }
  catch (error) { report.deviceCleanupError = error.message; report.passed = false; process.exitCode = 1; }
  await fs.writeFile(outputPath, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}

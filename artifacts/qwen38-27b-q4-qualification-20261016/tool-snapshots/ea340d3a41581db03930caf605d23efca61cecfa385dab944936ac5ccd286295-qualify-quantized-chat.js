#!/usr/bin/env node
// Installed runtime lifecycle and frozen task observations; not an accuracy oracle.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

const [configPath, suitePath, outputPath] = process.argv.slice(2);
const config = JSON.parse(await fs.readFile(configPath));
const suiteBytes = await fs.readFile(suitePath);
const suite = JSON.parse(suiteBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const moduleAt = relative => import(pathToFileURL(path.join(config.installedPackage, relative)).href);
const { load, DOPPLER_VERSION } = await moduleAt('src/index.js');
const { getDevice, destroyDevice } = await moduleAt('src/gpu/device.js');
const { releaseNodeWebGPU } = await moduleAt('src/tooling/node-webgpu.js');
const { getBufferPool } = await moduleAt('src/memory/buffer-pool.js');
const manifest = await fs.readFile(path.join(config.modelDirectory, 'manifest.json'));
assert.equal(hash(manifest), config.manifestSha256);
const report = { schema: 'doppler.quantized-chat-observation/v1', qualified: false,
  packageVersion: DOPPLER_VERSION, modelIdentity: hash(manifest), suiteSha256: hash(suiteBytes),
  toolSha256: hash(await fs.readFile(new URL(import.meta.url))), tasks: [], perplexity: [], lifecycle: {} };
let session;
const observe = async () => fs.writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
async function generate(task) {
  session.resetGenerationState();
  const inputIds = session.advanced.tokenizePrompt(task.messages, suite.generation);
  const start = performance.now();
  const result = await session.generateWithEvidence(task.messages, suite.generation);
  const stats = structuredClone(session.advanced.getStats());
  const naturalStop = ['eos-token', 'stop-token'].includes(stats.stopReason);
  return { id: task.id, inputIds, result, stats, elapsedMs: performance.now() - start,
    passed: naturalStop && result.outputText.trim() === task.expected };
}
function nll(logits, target) {
  assert(logits instanceof Float32Array && logits.every(Number.isFinite));
  let max = -Infinity;
  for (const value of logits) max = Math.max(max, value);
  let total = 0;
  for (const value of logits) total += Math.exp(value - max);
  return max + Math.log(total) - logits[target];
}
try {
  assert(!await fs.stat(outputPath).catch(() => null), 'Preserve prior observations');
  const memory = await fs.readFile('/proc/meminfo', 'utf8');
  const available = Number(memory.match(/^MemAvailable:\s+(\d+)/m)[1]) * 1024;
  const gtt = '/sys/class/drm/card1/device/';
  const gttTotal = Number(await fs.readFile(gtt + 'mem_info_gtt_total', 'utf8'));
  const gttUsed = Number(await fs.readFile(gtt + 'mem_info_gtt_used', 'utf8'));
  const budget = config.runtimeConfig.shared.bufferPool.budget.maxTotalBytes;
  assert(available > budget + 8 * 1024 ** 3 && gttTotal - gttUsed > budget + 1024 ** 3,
    'Insufficient system or GPU memory before load');
  report.memoryBeforeLoad = { available, gttTotal, gttUsed, budget };
  const start = performance.now();
  session = await load({ url: pathToFileURL(config.modelDirectory + path.sep).href }, { runtimeConfig: config.runtimeConfig });
  report.loadMs = performance.now() - start;
  report.resolvedSession = session.advanced.getResolvedRuntimeSession();
  for (const task of suite.tasks) {
    const row = await generate(task);
    report.tasks.push(row);
    await observe();
    console.log(JSON.stringify({ stage: 'task', id: row.id, passed: row.passed, text: row.result.outputText }));
  }
  for (const document of suite.documents) {
    session.resetGenerationState();
    const options = { ...suite.generation, useChatTemplate: false };
    const ids = session.advanced.tokenizePrompt(document.text, options);
    const initial = session.advanced.tokenizePrompt(document.firstText, options);
    assert.equal(initial.length, 1);
    assert.equal(initial[0], ids[0]);
    const first = await session.advanced.prefillWithLogits(document.firstText, options);
    const losses = [nll(first.logits, ids[1])];
    first.cache.destroy(); first.linearAttention = null;
    for (let index = 1; index + 1 < ids.length; index++) {
      const step = await session.advanced.decodeStepLogits([ids[index]], options);
      losses.push(nll(step.logits, ids[index + 1]));
    }
    const meanNll = losses.reduce((sum, value) => sum + value, 0) / losses.length;
    report.perplexity.push({ id: document.id, inputIds: ids, losses, meanNll, perplexity: Math.exp(meanNll) });
    await observe();
  }
  const repeated = await generate(suite.tasks[0]);
  assert.deepEqual(repeated.result.tokenIds, report.tasks[0].result.tokenIds);
  report.lifecycle.repeated = repeated;
  const controller = new AbortController();
  session.resetGenerationState();
  let callbacks = 0, abortRejection = null;
  try {
    for await (const chunk of session.generate(suite.cancellationPrompt, { ...suite.generation,
      signal: controller.signal, onToken() { callbacks++; controller.abort(); } })) { void chunk; }
  } catch (error) {
    if (!controller.signal.aborted || error.name !== 'AbortError') throw error;
    abortRejection = { name: error.name, message: error.message };
  }
  report.lifecycle.cancellation = { callbacks, abortRejection, stats: structuredClone(session.advanced.getStats()) };
  assert(callbacks > 0 && controller.signal.aborted
    && (abortRejection || report.lifecycle.cancellation.stats.stopReason === 'aborted'));
  report.lifecycle.afterCancellation = await generate(suite.tasks[0]);
  assert.deepEqual(report.lifecycle.afterCancellation.result.tokenIds, report.tasks[0].result.tokenIds);
  await getDevice().queue.onSubmittedWorkDone();
  report.memory = getBufferPool().getStats();
  report.lifecycle.passed = true;
  report.quality = { passed: report.tasks.every(row => row.passed), correct: report.tasks.filter(row => row.passed).length,
    total: report.tasks.length, scope: 'Frozen exact-answer tasks only; no general quality or preference claim' };
} catch (error) { report.error = { name: error.name, message: error.message, stack: error.stack }; process.exitCode = 1; }
finally {
  try {
    await session?.unload();
    if (getDevice()) await getDevice().queue.onSubmittedWorkDone();
    report.afterUnload = getBufferPool().getStats();
    assert.equal(report.afterUnload.resources.active.bytes, 0);
    assert.equal(report.afterUnload.resources.deferredCleanup.bytes, 0);
  } catch (error) { report.cleanupError = error.message; process.exitCode = 1; }
  try { destroyDevice(); } finally { await releaseNodeWebGPU(); }
  await observe();
}

#!/usr/bin/env node
// Installed Doppler diagnostic: fixed input history, separate from quality/timing claims.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

const [configPath, outputPath] = process.argv.slice(2);
const configBytes = await fs.readFile(configPath);
const config = JSON.parse(configBytes);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const root = path.resolve(config.installedPackage);
const moduleAt = relative => import(pathToFileURL(path.join(root, relative)).href);
const { load, DOPPLER_VERSION } = await moduleAt('src/index.js');
const { getDevice, destroyDevice } = await moduleAt('src/gpu/device.js');
const { releaseNodeWebGPU } = await moduleAt('src/tooling/node-webgpu.js');
const { getBufferPool } = await moduleAt('src/memory/buffer-pool.js');
const inventory = JSON.parse(await fs.readFile(config.inventoryPath));
const manifestBytes = await fs.readFile(path.join(config.modelDirectory, 'manifest.json'));
assert.equal(hash(manifestBytes), inventory.manifestSha256);
const contractBytes = await fs.readFile(config.referenceContract);
const contract = JSON.parse(contractBytes);
for (const entry of Object.values(contract.inputs)) {
  assert.equal(hash(await fs.readFile(entry.path)), entry.sha256, entry.path);
}
const transcript = JSON.parse(await fs.readFile(contract.inputs.transcript.path));
const output = path.resolve(outputPath);
await fs.mkdir(output);
const result = { schema: 'doppler.quantized-generation-observation/v1', qualified: false,
  scope: 'Installed runtime forced-token diagnostic; no cross-precision token acceptance gate',
  configSha256: hash(configBytes), toolSha256: hash(await fs.readFile(new URL(import.meta.url))),
  packageVersion: DOPPLER_VERSION, manifestSha256: inventory.manifestSha256,
  contractSha256: hash(contractBytes), inputCases: contract.cases, rows: [], states: [], stages: [] };
let session;
async function record(position, logits, caseId) {
  assert(logits instanceof Float32Array && logits.length > 0);
  assert(logits.every(Number.isFinite));
  const bytes = Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength);
  const file = `${caseId}-logits-${position}.f32`;
  await fs.writeFile(path.join(output, file), bytes, { flag: 'wx' });
  let top = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[top]) top = i;
  result.rows.push({ caseId, position, file, sha256: hash(bytes), count: logits.length,
    greedyToken: top, greedyText: session.advanced.decodeTokenIds([top]),
    stats: structuredClone(session.advanced.getStats()) });
}
try {
  const meminfo = await fs.readFile('/proc/meminfo', 'utf8');
  const available = Number(meminfo.match(/^MemAvailable:\s+(\d+)/m)[1]) * 1024;
  const gpu = '/sys/class/drm/card1/device/';
  const gttTotal = Number(await fs.readFile(gpu + 'mem_info_gtt_total', 'utf8'));
  const gttUsed = Number(await fs.readFile(gpu + 'mem_info_gtt_used', 'utf8'));
  const budget = config.runtimeConfig.shared.bufferPool.budget.maxTotalBytes;
  assert(available > budget + 8 * 1024 ** 3 && gttTotal - gttUsed > budget + 1024 ** 3);
  result.memoryBeforeLoad = { available, gttTotal, gttUsed, budget };
  const start = performance.now();
  session = await load({ url: pathToFileURL(config.modelDirectory + path.sep).href },
    { runtimeConfig: config.runtimeConfig });
  result.stages.push({ stage: 'loaded', elapsedMs: performance.now() - start,
    pool: getBufferPool().getStats(), allocations: getBufferPool().getLabelStats(),
    session: session.advanced.getResolvedRuntimeSession() });
  console.log(JSON.stringify({ stage: 'loaded', elapsedMs: performance.now() - start }));
  for (const item of contract.cases) {
  assert(/^[a-z0-9-]+$/.test(item.id), 'Case ID must be a safe filename');
  const prompt = item.prefillText ?? transcript.prompt;
  const inputIds = session.advanced.tokenizePrompt(prompt, config.generation);
  if (!item.prefillText) assert.deepEqual(inputIds, transcript.promptTokenIds);
  assert.deepEqual(item.inputIds.slice(0, inputIds.length), inputIds);
  session.resetGenerationState();
  const first = await session.advanced.prefillWithLogits(prompt, config.generation);
  try {
    assert.deepEqual(first.tokens, inputIds);
    if (item.positions.includes(inputIds.length - 1)) await record(inputIds.length - 1, first.logits, item.id);
    for (const layer of contract.stateLayers ?? []) {
      const state = first.linearAttention.layers.get(layer);
      assert(state && state.recurrentStatePrecision === 'f32' && state.convStatePrecision === 'f32');
      for (const field of ['recurrentState', 'convState']) {
        const values = state[field];
        assert(values.every(Number.isFinite));
        const bytes = Buffer.from(values.buffer, values.byteOffset, values.byteLength);
        const file = `${item.id}-layer-${layer}-${field}.f32`;
        await fs.writeFile(path.join(output, file), bytes, { flag: 'wx' });
        result.states.push({ caseId: item.id, layer, field, position: inputIds.length - 1,
          file, sha256: hash(bytes), count: values.length, numVHeads: state.numVHeads,
          headKDim: state.headKDim, headVDim: state.headVDim,
          convDim: state.convDim, convKernelSize: state.convKernelSize });
      }
    }
  } finally { first.cache?.destroy(); first.linearAttention = null; }
  for (let position = inputIds.length; position < item.inputIds.length; position++) {
    const step = await session.advanced.decodeStepLogits([item.inputIds[position]], config.generation);
    if (item.positions.includes(position)) await record(position, step.logits, item.id);
    if (position % 16 === 0) console.log(JSON.stringify({ stage: 'forced-token', position }));
  }
  }
  result.stages.push({ stage: 'forced-trajectory-complete', pool: getBufferPool().getStats(),
    allocations: getBufferPool().getLabelStats() });
  result.executionCompleted = true;
} catch (error) {
  result.error = { name: error.name, message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  try {
    await session?.unload();
    if (getDevice()) await getDevice().queue.onSubmittedWorkDone();
    result.afterUnload = getBufferPool().getStats();
  } catch (error) { result.cleanupError = error.message; process.exitCode = 1; }
  try { destroyDevice(); } finally { await releaseNodeWebGPU(); }
  await fs.writeFile(path.join(output, 'observation.json'), JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ completed: result.executionCompleted ?? false, error: result.error ?? null }));
}

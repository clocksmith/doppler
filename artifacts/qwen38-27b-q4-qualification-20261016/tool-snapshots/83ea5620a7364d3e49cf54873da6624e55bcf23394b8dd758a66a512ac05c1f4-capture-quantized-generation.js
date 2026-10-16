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
const contract = JSON.parse(await fs.readFile(config.referenceContract));
const transcript = JSON.parse(await fs.readFile(contract.inputs.transcript.path));
const output = path.resolve(outputPath);
await fs.mkdir(output);
const result = { schema: 'doppler.quantized-generation-observation/v1', qualified: false,
  scope: 'Installed runtime forced-token diagnostic; no cross-precision token acceptance gate',
  configSha256: hash(configBytes), toolSha256: hash(await fs.readFile(new URL(import.meta.url))),
  packageVersion: DOPPLER_VERSION, manifestSha256: inventory.manifestSha256, rows: [], stages: [] };
let session;
async function record(position, logits) {
  assert(logits instanceof Float32Array && logits.length > 0);
  assert(logits.every(Number.isFinite));
  const bytes = Buffer.from(logits.buffer, logits.byteOffset, logits.byteLength);
  const file = `logits-${position}.f32`;
  await fs.writeFile(path.join(output, file), bytes, { flag: 'wx' });
  let top = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[top]) top = i;
  result.rows.push({ position, file, sha256: hash(bytes), count: logits.length,
    greedyToken: top, greedyText: session.advanced.decodeTokenIds([top]),
    stats: structuredClone(session.advanced.getStats()) });
}
try {
  const start = performance.now();
  session = await load({ url: pathToFileURL(config.modelDirectory + path.sep).href },
    { runtimeConfig: config.runtimeConfig });
  result.stages.push({ stage: 'loaded', elapsedMs: performance.now() - start,
    pool: getBufferPool().getStats(), session: session.advanced.getResolvedRuntimeSession() });
  console.log(JSON.stringify({ stage: 'loaded', elapsedMs: performance.now() - start }));
  const item = contract.cases[0];
  const inputIds = session.advanced.tokenizePrompt(transcript.prompt, config.generation);
  assert.deepEqual(inputIds, transcript.promptTokenIds);
  assert.deepEqual(item.inputIds.slice(0, inputIds.length), inputIds);
  session.resetGenerationState();
  const first = await session.advanced.prefillWithLogits(transcript.prompt, config.generation);
  try {
    assert.deepEqual(first.tokens, inputIds);
    if (item.positions.includes(inputIds.length - 1)) await record(inputIds.length - 1, first.logits);
  } finally { first.cache?.destroy(); first.linearAttention = null; }
  for (let position = inputIds.length; position < item.inputIds.length; position++) {
    const step = await session.advanced.decodeStepLogits([item.inputIds[position]], config.generation);
    if (item.positions.includes(position)) await record(position, step.logits);
    if (position % 16 === 0) console.log(JSON.stringify({ stage: 'forced-token', position }));
  }
  result.stages.push({ stage: 'forced-trajectory-complete', pool: getBufferPool().getStats() });
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

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { installHyperConnectionObservation } from '../helpers/hyper-connection-observation-loader.js';
import { installMoeRoutingObservation } from '../helpers/moe-routing-observation-loader.js';
import { installMoeSharedSplitObservation } from '../helpers/moe-shared-split-observation-loader.js';

// Provisioned GPU stage capture for same-input operator checks over every layer, the
// root mixer (Flash-Next's final norm) and the LM head (docs/routing-aware-acceptance.md,
// step 6). It extends flash-next-upstream-trace-physical.test.js from one stop layer to a
// complete unforced prefill: hyper-connection boundaries at the declared positions, each
// sequence mixer's complete input, executed top-k indices and weights, the routed-expert
// output before the shared-expert add, the root mixer and the last-position logits.
// Observation only: awaited readbacks, no replacement. The CPU replay is
// reports/flash-next-routing-aware-fresh-20261112/audit-same-input-all-layers.py.
const runPath = process.env.DOPPLER_FLASH_NEXT_SAME_INPUT_CAPTURE;
if (!runPath) {
  assert(!process.argv.includes('--require-model'), 'Set DOPPLER_FLASH_NEXT_SAME_INPUT_CAPTURE.');
  console.log('Flash-Next same-input capture: skipped (no provisioned run)');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const runBytes = await fs.readFile(runPath);
  const run = JSON.parse(runBytes);
  assert.equal(run.schema, 'doppler.flash-next-same-input-capture-run/v1');
  for (const key of ['modelDir', 'expectedManifestSha256', 'runtimeConfig', 'inputsPath', 'sequenceId', 'positions',
    'outputDirectory']) assert.ok(run[key] !== undefined && run[key] !== null, `run.${key} is required`);
  const manifestBytes = await fs.readFile(path.join(run.modelDir, 'manifest.json'));
  assert.equal(hash(manifestBytes), run.expectedManifestSha256);
  const manifest = JSON.parse(manifestBytes);
  const numLayers = manifest.architecture.numLayers;
  const inputsBytes = await fs.readFile(run.inputsPath);
  const input = JSON.parse(inputsBytes).sequences.find(row => row.id === run.sequenceId);
  assert.ok(input, 'pinned sequence required');
  const tokens = input.tokens.length;
  const positions = run.positions;
  assert.ok(positions.length > 0 && positions.every((p, i) => Number.isSafeInteger(p) && p >= 0 && p < tokens
    && (i === 0 || p > positions[i - 1])), 'positions must be increasing token indices');
  assert.equal(positions.at(-1), tokens - 1, 'the last position feeds the LM head check');
  const output = path.resolve(run.outputDirectory);
  await fs.mkdir(output);
  const report = { schema: 'doppler.flash-next-same-input-capture/v1',
    scope: 'Unforced GPU prefill stage capture for same-input operator checks; observation only, not model qualification',
    normalModelQualified: false, runStartedAtUtc: new Date().toISOString(),
    process: { pid: process.pid, node: process.versions.node },
    harnessSha256: hash(await fs.readFile(new URL(import.meta.url))), runSha256: hash(runBytes),
    manifestSha256: hash(manifestBytes), inputsSha256: hash(inputsBytes), sequenceId: run.sequenceId, tokens, positions,
    policySha256: run.policySha256 ?? null, runtimeConfig: run.runtimeConfig, tensors: [], moeRouting: [], routedOutputs: [],
    gpuErrors: [] };
  let memory = null;
  const write = async (name, bytes) => {
    const file = path.join(output, name);
    await fs.writeFile(file, bytes, { flag: 'wx' });
    return { file: name, sha256: hash(bytes) };
  };
  const readRows = async (tensor, width) => {
    const rows = [];
    for (const position of positions) rows.push(Buffer.from(await memory.readBufferSlice(tensor.buffer, position * width * 4, width * 4)));
    return Buffer.concat(rows);
  };
  // Hyper-connection stages kept for the CPU replay; the low-rank internals are covered by
  // the norm -> branchInput and norm -> injected checks.
  const keepStages = new Set(['input', 'norm', 'branchInput', 'branchOutput', 'injected']);
  let layer = -1;
  const hcUrl = new URL('../../src/inference/pipelines/text/hyper-connection/decoder-layer.js', import.meta.url);
  const hc = installHyperConnectionObservation(hcUrl.href, hash(await fs.readFile(hcUrl)), async record => {
    if (record.numTokens !== tokens) return;
    if (record.role === 'attn_hc' && record.stage === 'input') layer++;
    if (!keepStages.has(record.stage)) return;
    assert.equal(record.tensor.dtype, 'f32');
    const [rows, width] = record.tensor.shape;
    assert.equal(rows, tokens);
    const label = record.role === 'hc_mixer' ? 'final' : layer;
    const entry = { layer: label, role: record.role, stage: record.stage, shape: [positions.length, width],
      positions: 'run.positions', ...(await write(`L${label}.${record.role}.${record.stage}.f32`, await readRows(record.tensor, width))) };
    report.tensors.push(entry);
    if (record.role === 'attn_hc' && record.stage === 'branchInput') {
      const full = Buffer.from(await memory.readBuffer(record.tensor.buffer, tokens * width * 4));
      report.tensors.push({ layer: label, role: record.role, stage: 'branchInputAllPositions', shape: [tokens, width],
        positions: 'all', ...(await write(`L${label}.attn_hc.branchInputAllPositions.f32`, full)) });
    }
  });
  const moeUrl = new URL('../../src/inference/pipelines/text/moe-gpu.js', import.meta.url);
  const routing = installMoeRoutingObservation(moeUrl.href, hash(await fs.readFile(moeUrl)), async record => {
    if (record.numTokens !== tokens) return;
    assert.equal(record.activationDtype, 'f32');
    const row = { layer: record.layer, shape: [tokens, record.topK] };
    for (const [name, buffer, suffix] of [['indices', record.indicesBuffer, 'u32'], ['weights', record.weightsBuffer, 'f32']]) {
      const bytes = Buffer.from(await memory.readBuffer(buffer, tokens * record.topK * 4));
      row[name] = await write(`L${record.layer}.moe-${name}.${suffix}`, bytes);
    }
    report.moeRouting.push(row);
  });
  const ffnUrl = new URL('../../src/inference/pipelines/text/ffn/moe.js', import.meta.url);
  const split = installMoeSharedSplitObservation(ffnUrl.href, hash(await fs.readFile(ffnUrl)), async record => {
    if (record.numTokens !== tokens) return;
    assert.equal(record.sharedSlot, false, 'prefill keeps the shared expert separate from the routed output');
    assert.equal(record.routed.dtype, 'f32');
    const width = record.routed.shape.at(-1);
    report.routedOutputs.push({ layer: record.layer, shape: [positions.length, width], positions: 'run.positions',
      ...(await write(`L${record.layer}.moe-routed.f32`, await readRows(record.routed, width))) });
  });
  let pipeline, destroyDevice, releaseNodeWebGPU, primaryError;
  try {
    const [gpu, runtime, storage, pool, device, provider] = await Promise.all([
      import('../helpers/gpu-probe.js'), import('../../src/inference/pipelines/text.js'),
      import('../../src/storage/artifact-storage-context.js'), import('../../src/memory/buffer-pool.js'),
      import('../../src/gpu/device.js'), import('../../src/tooling/node-webgpu.js'),
    ]);
    memory = pool;
    destroyDevice = device.destroyDevice;
    releaseNodeWebGPU = provider.releaseNodeWebGPU;
    const probe = await gpu.probeNodeGPU({ installFileFetchShim: true });
    assert(probe.ready, probe.reason);
    report.provider = probe.providerReceipt;
    report.adapterInfo = device.getKernelCapabilities()?.adapterInfo ?? null;
    device.getDevice().addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
    pipeline = await runtime.createPipeline(manifest, { runtimeConfig: run.runtimeConfig,
      storage: storage.createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(run.modelDir)).href, manifest) });
    let logits = null;
    const result = await pipeline.generateTokenIds('', { inputIds: input.tokens, useChatTemplate: false,
      maxTokens: 1, temperature: 0, topK: 1, topP: 1, seed: 0, repetitionPenalty: 1,
      onLogits: row => { logits ??= Float32Array.from(row); } });
    assert.ok(logits?.length === manifest.architecture.vocabSize, 'complete last-position logits required');
    report.prefillLogits = await write('prefill-logits.f32', Buffer.from(logits.buffer));
    report.tokenId = result.tokenIds[0];
    const covered = role => new Set(report.tensors.filter(row => row.role === role).map(row => row.layer));
    for (const role of ['attn_hc', 'mlp_hc']) assert.equal(covered(role).size, numLayers, `${role} must cover every layer`);
    assert.equal(report.tensors.filter(row => row.role === 'hc_mixer').length, 3, 'root mixer input, norm and output');
    assert.deepEqual(report.moeRouting.map(row => row.layer), Array.from({ length: numLayers }, (_, i) => i));
    assert.deepEqual(report.routedOutputs.map(row => row.layer), Array.from({ length: numLayers }, (_, i) => i));
    await device.getDevice().queue.onSubmittedWorkDone();
    assert.deepEqual(report.gpuErrors, []);
    report.diagnosticCompleted = true;
  } catch (error) {
    primaryError = error;
    report.failure = { name: error.name, message: error.message, stack: error.stack };
    throw error;
  } finally {
    const failures = [];
    try { await pipeline?.unload(); } catch (error) { failures.push(error); }
    try { destroyDevice?.(); } catch (error) { failures.push(error); }
    try { await releaseNodeWebGPU?.(); } catch (error) { failures.push(error); }
    hc.restore();
    routing.restore();
    split.restore();
    report.instrumentation = { hyperConnections: hc.receipt, routing: routing.receipt, sharedSplit: split.receipt };
    report.cleanupFailures = failures.map(String);
    await fs.writeFile(path.join(output, 'observation.json'), JSON.stringify(report, null, 2));
    if (failures.length) throw new AggregateError(primaryError ? [primaryError, ...failures] : failures, 'Same-input capture cleanup failed');
  }
}

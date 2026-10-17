import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { runPipelineOperation } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createLayerPartitionPlan } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { createPartitionAttempt, clonePartitionAttempt } from '../../src/inference/pipelines/text/partition-attempt.js';
import { executePartitionLayers } from '../../src/inference/pipelines/text/partition-execution.js';
import { cloneLinearAttentionRuntime, resetLinearAttentionRuntime } from '../../src/inference/pipelines/text/linear-attention.js';
import { cloneLinearAttentionRuntimeOnGPU } from '../../src/inference/pipelines/text/linear-attention/state-clone.js';
import { ensureLayerRuntimeGpuBuffers } from '../../src/inference/pipelines/text/linear-attention/plan.js';
import { readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { getDevice, destroyDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

// Run in an immutable copied source tree, with an externally provisioned model.
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
assert.equal(config.schema, 'doppler.recurrent-clone-profile/v1');
assert.equal(config.warmupPairs, 2); assert.equal(config.measuredPairs, 12);
assert.equal(config.continuationSteps, 128);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const report = { config, startedAt: new Date().toISOString(), passed: false, pairs: [], continuation: [], artifacts: [] };
let pipeline, parent, child, reference;
const signal = new AbortController().signal;
const fields = ['convWeight', 'dtBias', 'aLog', 'normWeight', 'convState', 'recurrentState'];
try {
  const manifestBytes = await fs.readFile(path.join(config.modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  report.modelId = manifest.modelId; report.manifestHash = hash(manifestBytes);
  assert.equal(report.manifestHash, config.manifestHash);
  for (const file of [manifest.tokenizer.file, ...manifest.shards.map(shard => shard.filename)]) {
    const bytes = await fs.readFile(path.join(config.modelDirectory, file));
    report.artifacts.push({ file, bytes: bytes.length, sha256: hash(bytes) });
  }
  report.gpu = await probeNodeGPU({ installFileFetchShim: true }); assert(report.gpu.ready, report.gpu.reason);
  report.adapter = getKernelCapabilities().adapterInfo;
  const device = getDevice();
  const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    activationDtype: manifest.inference.session.compute.defaults.activationDtype });
  const runtimeConfig = { inference: { session: { kvcache: { maxSeqLen: config.maxSeqLen } } } };
  const loadedAt = performance.now();
  pipeline = await createPipeline(manifest, { runtimeConfig, partition: { plan, index: 0 },
    storage: createNodeFileArtifactStorageContext(pathToFileURL(config.modelDirectory).href, manifest) });
  report.loadMs = performance.now() - loadedAt;
  report.plan = plan; report.runtimeConfig = runtimeConfig;
  report.kernelPath = pipeline.resolvedKernelPath; report.kernelPathSource = pipeline.kernelPathSource;
  report.session = pipeline.resolvedRuntimeSession;
  parent = createPartitionAttempt(pipeline, config.maxSeqLen);
  const input = Array.from(pipeline.tokenizer.encode(config.prompt)); report.inputTokenIds = input;
  const execute = (attempt, ids) => runPipelineOperation(pipeline, async () => {
    const result = await executePartitionLayers(attempt.state, { numTokens: ids.length, tokenIds: ids }, signal);
    attempt.state.currentSeqLen += ids.length;
    return result.activationBytes;
  });
  const prefillAt = performance.now(); await execute(parent, input); report.prefillMs = performance.now() - prefillAt;
  const runtime = parent.state.linearAttentionRuntime;
  report.recurrentLayers = runtime.layers.size;
  report.recurrentBytes = [...runtime.layers.values()].reduce((sum, layer) => sum + layer.convState.byteLength + layer.recurrentState.byteLength, 0);
  const copy = async mode => {
    if (mode === 'gpu') return cloneLinearAttentionRuntimeOnGPU(runtime);
    const result = await cloneLinearAttentionRuntime(runtime);
    try {
      for (const layer of result.layers.values()) ensureLayerRuntimeGpuBuffers(layer);
      await device.queue.onSubmittedWorkDone(); return result;
    } catch (error) { resetLinearAttentionRuntime(result); throw error; }
  };
  for (let pair = -config.warmupPairs; pair < config.measuredPairs; pair++) {
    const row = { pair, warmup: pair < 0, samples: [] };
    for (const mode of pair % 2 === 0 ? ['cpu-roundtrip', 'gpu'] : ['gpu', 'cpu-roundtrip']) {
      const start = performance.now(); const cloned = await copy(mode); const elapsedMs = performance.now() - start;
      try {
        // Observation is outside timing and checks every state/static buffer.
        for (const [index, source] of runtime.layers) for (const field of fields) {
          const [expected, actual] = await Promise.all([
            readBuffer(source[field + 'GPU'], source[field].byteLength),
            readBuffer(cloned.layers.get(index)[field + 'GPU'], source[field].byteLength),
          ]);
          assert.deepEqual(new Uint8Array(actual), new Uint8Array(expected));
        }
      } finally { resetLinearAttentionRuntime(cloned); }
      row.samples.push({ mode, elapsedMs });
    }
    report.pairs.push(row); console.log(JSON.stringify(row));
  }
  // Exercise the production composite owner, including KV, then close the parent.
  child = await clonePartitionAttempt(parent, signal);
  reference = createPartitionAttempt(pipeline, config.maxSeqLen);
  await reference.state.kvCache.copyFrom(parent.state.kvCache);
  reference.state.linearAttentionRuntime = await cloneLinearAttentionRuntime(runtime);
  reference.state.currentSeqLen = parent.state.currentSeqLen;
  parent.close(); parent = null;
  for (let step = 0; step < config.continuationSteps; step++) {
    const token = input[step % input.length];
    const expected = await execute(reference, [token]); const actual = await execute(child, [token]);
    assert.deepEqual(new Uint8Array(actual), new Uint8Array(expected), `Continuation byte mismatch at ${step}`);
    report.continuation.push({ step, token, sha256: hash(new Uint8Array(actual)) });
  }
  report.passed = true;
} catch (error) { report.error = error.stack; }
finally {
  for (const attempt of [parent, child, reference]) attempt?.close();
  await pipeline?.unload();
  if (getDevice()) { await getDevice().queue.onSubmittedWorkDone(); report.finalPool = getBufferPool().getStats(); }
  destroyDevice(); releaseNodeWebGPU();
  report.completedAt = new Date().toISOString();
  await fs.writeFile(config.output, JSON.stringify(report, null, 2) + '\n');
}
assert(report.passed, report.error);
console.log(JSON.stringify({ passed: report.passed, recurrentBytes: report.recurrentBytes, output: config.output }));

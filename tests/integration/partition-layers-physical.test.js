import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { runPipelineOperation } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createLayerPartitionPlan, comparePartitionExecution } from '../../src/inference/pipelines/text/layer-partition-contract.js';
import { assertPartitionExecutionSupported, executePartitionLayers } from '../../src/inference/pipelines/text/partition-execution.js';
import { readBuffer, releaseBuffer, isBufferActive } from '../../src/memory/buffer-pool.js';
import { destroyDevice, getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const modelDirectory = process.env.DOPPLER_PARTITION_MODEL_DIR;
if (!modelDirectory) {
  console.log('partition-layers-physical: skipped; set DOPPLER_PARTITION_MODEL_DIR to a local RDRR generation model');
} else {
  const manifestBytes = await fs.readFile(path.join(modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    activationDtype: manifest.inference.session.compute.defaults.activationDtype });
  const signal = new AbortController().signal;
  const runtimeConfig = { inference: { session: { kvcache: { maxSeqLen: 128 } } } };
  const report = { kind: 'local-partition-numerical-diagnostic', modelId: manifest.modelId,
    manifestHash: 'sha256:' + createHash('sha256').update(manifestBytes).digest('hex'),
    plan, runtimeConfig, steps: [] };
  report.artifacts = [];
  for (const filename of ['manifest.json', manifest.tokenizer.file, ...manifest.shards.map(shard => shard.filename)]) {
    const bytes = await fs.readFile(path.join(modelDirectory, filename));
    report.artifacts.push({ filename, bytes: bytes.byteLength,
      sha256: createHash('sha256').update(bytes).digest('hex') });
  }
  const pipelines = [];
  let snapshot;
  try {
    const gpu = await probeNodeGPU({ installFileFetchShim: true });
    assert.ok(gpu.ready, gpu.reason);
    report.providerReceipt = gpu.providerReceipt;
    report.adapter = getKernelCapabilities().adapterInfo;
    for (const index of [null, 0, 1]) {
      const storage = createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(modelDirectory)).href, manifest);
      const pipeline = await createPipeline(manifest, { runtimeConfig, storage,
        ...(index === null ? {} : { partition: { plan, index } }) });
      pipelines.push(pipeline);
    }
    const [reference, a, b] = pipelines;
    const prompt = 'The color of the sky is';
    let inputIds = Array.from(reference.tokenizer.encode(prompt));
    report.prompt = prompt;
    report.promptTokenIds = [...inputIds];
    report.cacheAllocations = pipelines.map(p => p.kvCache.getMemoryStats());
    report.gpuWeightAllocationBytes = pipelines.map(p => [...p.dopplerLoader.gpuBuffers]
      .reduce((bytes, buffer) => bytes + buffer.size, 0));
    report.weightLayers = pipelines.map(p => [...p.weights.keys()].filter(k => /^layer_\d+$/.test(String(k))));
    assert.equal(a.kvCache.numLayers, plan.splitLayer);
    assert.equal(b.kvCache.numLayers, plan.totalLayers - plan.splitLayer);
    for (let step = 0; step < 3; step++) {
      const expected = step === 0
        ? await reference.prefillWithLogits(prompt, { inputIds, useChatTemplate: false })
        : await reference.decodeStepLogits(inputIds, { useChatTemplate: false });
      if (step === 0) snapshot = expected.cache;
      const boundary = await runPipelineOperation(a, async state => {
        assertPartitionExecutionSupported(state, plan);
        const result = await executePartitionLayers(state, { numTokens: inputIds.length, tokenIds: inputIds }, signal);
        state.currentSeqLen += inputIds.length;
        return result.activationBytes;
      });
      const observed = await runPipelineOperation(b, async state => {
        assertPartitionExecutionSupported(state, plan);
        const result = await executePartitionLayers(state,
          { numTokens: inputIds.length, activationBytes: boundary }, signal);
        try {
          const logits = new Float32Array(await readBuffer(result.logits.logitsBuffer, result.logits.vocabSize * 4));
          state.currentSeqLen += inputIds.length;
          return logits;
        } finally { releaseBuffer(result.logits.logitsBuffer); }
      });
      const comparison = comparePartitionExecution({ splitOutput: observed, referenceOutput: expected.logits, tolerance: 1e-4 });
      const argmax = logits => logits.reduce((best, value, index) => value > logits[best] ? index : best, 0);
      const tokenId = argmax(expected.logits);
      report.steps.push({ step, tokenId, splitTokenId: argmax(observed), activationBytes: boundary.byteLength, comparison });
      console.log(JSON.stringify(report.steps.at(-1)));
      assert.ok(comparison.matches, `partition logits differ at step ${step}: ${JSON.stringify(comparison)}`);
      assert.equal(argmax(observed), tokenId);
      inputIds = [tokenId];
    }
    const activeBuffers = () => a.getBufferPool().getStats().activeBuffers;
    const beforeFailures = activeBuffers();
    await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
      { numTokens: 1, tokenIds: [-1] }, signal)), /vocabulary/);
    await assert.rejects(runPipelineOperation(b, state => executePartitionLayers(state,
      { numTokens: 1, activationBytes: new ArrayBuffer(4) }, signal)), /byte length/);
    const alreadyAborted = new AbortController();
    alreadyAborted.abort(new Error('cancel-before-dispatch'));
    await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
      { numTokens: 1, tokenIds: inputIds }, alreadyAborted.signal)), /cancel-before-dispatch/);
    assert.equal(activeBuffers(), beforeFailures, 'Validation and pre-dispatch cancellation leak no active pooled buffers');

    const device = getDevice();
    const createEncoder = device.createCommandEncoder;
    device.createCommandEncoder = function (descriptor) {
      const encoder = createEncoder.call(this, descriptor);
      if (descriptor?.label === 'resident_partition_layers') {
        encoder.finish = () => { throw new Error('injected-recording-failure'); };
      }
      return encoder;
    };
    try {
      await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
        { numTokens: 1, tokenIds: inputIds }, signal)), /injected-recording-failure/);
    } finally { device.createCommandEncoder = createEncoder; }
    assert.equal(activeBuffers(), beforeFailures, 'Recording failure releases all temporary pooled buffers');

    const pool = a.getBufferPool();
    const acquire = pool.acquire;
    const release = pool.release;
    pool.release = function (buffer) {
      assert.ok(isBufferActive(buffer), 'Output allocation failure must not release a buffer twice');
      return release.call(this, buffer);
    };
    pool.acquire = function (size, usage, label) {
      if (label === 'resident_partition_output') throw new Error('injected-output-allocation-failure');
      return acquire.call(this, size, usage, label);
    };
    try {
      await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
        { numTokens: 1, tokenIds: inputIds }, signal)), /injected-output-allocation-failure/);
    } finally { pool.acquire = acquire; pool.release = release; }
    assert.equal(activeBuffers(), beforeFailures, 'Allocation failure releases recorded intermediates');

    const duringReadback = new AbortController();
    const mapAsync = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = async function (...args) {
      const result = await mapAsync.apply(this, args);
      duringReadback.abort(new Error('cancel-during-readback'));
      return result;
    };
    try {
      await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
        { numTokens: 1, tokenIds: inputIds }, duringReadback.signal)), /cancel-during-readback/);
    } finally { GPUBuffer.prototype.mapAsync = mapAsync; }
    assert.equal(activeBuffers(), beforeFailures, 'Readback cancellation releases mapped staging and output');

    const afterSubmit = new AbortController();
    const submit = device.queue.submit;
    let submitted = 0;
    device.queue.submit = function (...args) {
      const result = submit.apply(this, args);
      submitted++;
      afterSubmit.abort(new Error('cancel-after-submit'));
      return result;
    };
    try {
      await assert.rejects(runPipelineOperation(a, state => executePartitionLayers(state,
        { numTokens: 1, tokenIds: inputIds }, afterSubmit.signal)), /cancel-after-submit/);
    } finally { device.queue.submit = submit; }
    assert.equal(submitted, 1);
    assert.equal(activeBuffers(), beforeFailures, 'Submitted cancellation settles and releases temporary pooled buffers');
    report.failureChecks = ['invalid-token', 'activation-length', 'cancel-before-dispatch', 'recording-failure',
      'output-allocation-failure', 'cancel-during-readback', 'cancel-after-submit'];
    console.log(JSON.stringify(report));
  } finally {
    await getDevice()?.queue.onSubmittedWorkDone();
    snapshot?.destroy();
    for (const pipeline of pipelines.reverse()) await pipeline.unload();
    destroyDevice();
    await releaseNodeWebGPU();
  }
}

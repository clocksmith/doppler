// One JSON-bound model and ordered split against the same unpartitioned GPU
// program. This is same-device runtime evidence, not network or source accuracy.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { createManifestResidentPartitionFactory } from '../../src/client/resident-partitions.js';
import { createOrderedLayerPartitionPlan, hashLayerPartitionPlan, deserializeActivationFrame,
  serializeActivationFrame, comparePartitionExecution, hashPartitionGeneration } from '../../src/partitions.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { destroyDevice, getDevice, getKernelCapabilities } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const configurationPath = process.env.DOPPLER_ORDERED_SPLIT_CONFIG;
if (!configurationPath) {
  assert(!process.argv.includes('--require-model'), 'DOPPLER_ORDERED_SPLIT_CONFIG is required');
  console.log('ordered-split-physical: skipped; configure DOPPLER_ORDERED_SPLIT_CONFIG');
} else {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const configurationBytes = await fs.readFile(configurationPath);
  const config = JSON.parse(configurationBytes);
  assert.equal(config.schema, 'doppler.ordered-split-physical/v1');
  assert(path.isAbsolute(config.modelDirectory) && path.isAbsolute(config.outputPath));
  assert.equal(typeof config.prompt, 'string');
  assert(Number.isSafeInteger(config.steps) && config.steps >= 2 && config.steps <= 4);
  const manifestBytes = await fs.readFile(path.join(config.modelDirectory, 'manifest.json'));
  const manifestIdentity = 'sha256:' + hash(manifestBytes);
  assert.equal(config.modelIdentity, manifestIdentity, 'Exact model identity before GPU allocation');
  const manifest = JSON.parse(manifestBytes);
  const generation = resolveGenerationOptions({ ...config.generation, maxTokens: config.steps });
  assert.equal(generation.temperature, 0, 'This control compares deterministic greedy prefixes');
  const hyper = manifest.inference.hyperConnections;
  const plan = createOrderedLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    ...config.partition, ...(hyper ? { residualStreams: { streamCount: hyper.streamCount, streamWidth: hyper.streamWidth } } : {}) });
  const planId = hashLayerPartitionPlan(plan);
  const storageOptions = config.runtimeConfig.loading?.shardCache ?? {};
  const storage = () => createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(config.modelDirectory)).href, manifest,
    { verifyHashes: storageOptions.verifyHashes, verificationBufferBytes: storageOptions.verificationBufferBytes });
  const report = { schema: 'doppler.ordered-split-physical-result/v1',
    scope: 'One same-device ordered resident route versus the same whole-model GPU program; no independent accuracy or transport qualification',
    configSha256: hash(configurationBytes), modelIdentity: manifestIdentity, modelId: manifest.modelId,
    plan, planId, runtimeConfig: config.runtimeConfig, generation,
    referenceSource: 'unsplit-doppler', tolerance: 1e-4, rows: [], gpuErrors: [], cleanupFailures: [], passed: false };
  report.artifactVerification = config.artifactVerification ?? null;
  const residents = [];
  let reference = null;
  const controller = new AbortController();
  const signal = controller.signal;
  const stop = () => controller.abort(new Error('Ordered split control interrupted'));
  process.once('SIGTERM', stop);
  process.once('SIGINT', stop);
  try {
    const gpu = await probeNodeGPU({ installFileFetchShim: true });
    assert(gpu.ready, gpu.reason);
    report.provider = gpu.providerReceipt;
    report.adapter = getKernelCapabilities()?.adapterInfo;
    assert(!/swiftshader|llvmpipe|lavapipe/.test(JSON.stringify(report.adapter).toLowerCase()));
    getDevice().addEventListener('uncapturederror', event => report.gpuErrors.push(event.error.message));
    // Close the complete control before loading stages, avoiding two resident
    // copies of a large model merely to run this comparison.
    reference = await createPipeline(manifest, { runtimeConfig: config.runtimeConfig, storage: storage() });
    const inputIds = [...reference.tokenizer.encode(config.prompt)];
    report.inputTokenIds = inputIds;
    const frames = [], selected = [];
    for await (const token of reference.generateTokens('', { ...generation, inputIds, useChatTemplate: false, signal,
      onLogits(logits) {
        assert.equal(logits.length, manifest.architecture.vocabSize);
        assert(logits.every(Number.isFinite));
        frames.push(Float32Array.from(logits));
        console.log(JSON.stringify({ phase: 'whole-model-control', logitRows: frames.length }));
      } })) selected.push(token);
    assert(frames.length >= 2 && frames.length === selected.length, 'The control must exercise prefill and decode');
    report.controlLogits = frames.map(row => hash(Buffer.from(row.buffer)));
    report.controlTokenIds = selected;
    await reference.unload(); reference = null;
    const factory = createManifestResidentPartitionFactory({ manifest, manifestIdentity,
      runtimeConfig: config.runtimeConfig, createStorage: async () => storage() });
    const stages = plan.partitions.map(({ index }) => ({ index, participantId: `stage-${index}`,
      transportId: `local-stage-${index}`, residentId: `resident-${index}`, executionId: `execution-${index}` }));
    for (const stage of stages) residents.push(await factory.openResidentPartition({
      model: { id: manifest.modelId, identity: manifestIdentity, generation }, plan, planId,
      index: stage.index, participantId: stage.participantId, limits: config.limits, signal,
    }));
    report.descriptors = residents.map(resident => resident.getDescriptor());
    const identity = { schema: 'doppler.resident-partition-attempt/v2', modelId: manifest.modelId,
      modelIdentity: manifestIdentity, planId, threadId: 'control-thread', attemptId: 'control-attempt',
      requesterId: 'local-control', placementGeneration: 0, stages };
    const continuation = stages.map(() => null);
    const generationDigest = hashPartitionGeneration(generation);
    let ids = inputIds, position = 0;
    for (let step = 0; step < frames.length; step++) {
      let activation = null, final = null;
      for (const stage of stages) {
        const result = await residents[stage.index].executeStage({ identity, partitionIndex: stage.index,
          step, tokenPosition: position, inputTokenCount: ids.length, inputTokenIds: ids,
          generation, generationDigest, maxTokens: generation.maxTokens,
          continuation: continuation[stage.index], signal,
          ...(activation ? { activation } : {}) });
        continuation[stage.index] = result.continuation;
        if (result.activationTensor) {
          const tensor = result.activationTensor;
          activation = deserializeActivationFrame(serializeActivationFrame({ ...tensor, data: tensor.data }));
          assert.equal(tensor.data.byteLength, ids.length * plan.partitions[stage.index].outputContract.hiddenSize * 4);
        } else final = result;
      }
      assert(final?.logits, 'Only the final stage must produce logits');
      const comparison = comparePartitionExecution({ splitOutput: final.logits, referenceOutput: frames[step], tolerance: 1e-4 });
      const row = { step, tokenPosition: position, inputTokenCount: ids.length,
        tokenId: final.tokenId, referenceTokenId: selected[step], comparison,
        logitsSha256: hash(Buffer.from(final.logits.buffer)) };
      report.rows.push(row); console.log(JSON.stringify(row));
      assert(comparison.matches, 'Stage logits differ: ' + JSON.stringify(row));
      assert.equal(final.tokenId, selected[step]);
      position += ids.length; ids = [selected[step]];
    }
    for (const resident of residents) await resident.closeAttempt({ identity });
    report.passed = report.gpuErrors.length === 0;
    assert(report.passed, 'Uncaptured GPU errors');
  } catch (error) {
    report.failure = String(error); throw error;
  } finally {
    for (const resource of [...residents].reverse()) {
      try { await resource.close(); } catch (error) { report.cleanupFailures.push(String(error)); }
    }
    try { await reference?.unload(); await destroyDevice(); await releaseNodeWebGPU(); }
    catch (error) { report.cleanupFailures.push(String(error)); }
    if (report.cleanupFailures.length) report.passed = false;
    process.removeListener('SIGTERM', stop);
    process.removeListener('SIGINT', stop);
    await fs.writeFile(config.outputPath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ passed: report.passed, rows: report.rows.length,
      gpuErrors: report.gpuErrors, cleanupFailures: report.cleanupFailures }));
  }
  assert(report.passed);
}

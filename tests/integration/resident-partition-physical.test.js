import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { deserializeActivationFrame, serializeActivationFrame,
  comparePartitionExecution } from '../../src/partitions.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';
import { createResidentPartitionSession } from '../../src/inference/pipelines/text/resident-partition.js';
import { createModelHandle } from '../../src/client/model-host/model-session.js';
import { createCapsuleProgramAdapter } from '../../src/client/runtime/capsule-program-adapter.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { sampleCapsuleLogits, stoppingReason } from '../../src/inference/generation-step.js';
import { samplingSeedForToken } from '../../src/inference/token-sampling.js';
import { createPartitionAttempt } from '../../src/inference/pipelines/text/partition-attempt.js';
import { acquireBuffer, isBufferActive } from '../../src/memory/buffer-pool.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';
import { resolvePhysicalPartitionConfig, bindPhysicalIndependentReference, physicalInputDigest,
  INDEPENDENT_TOLERANCE } from '../helpers/resident-partition-physical-config.js';

const modelDirectory = process.env.DOPPLER_PARTITION_MODEL_DIR;
if (!modelDirectory) {
  assert(!process.env.DOPPLER_PARTITION_CONFIG, 'Explicit physical config requires DOPPLER_PARTITION_MODEL_DIR');
  console.log('resident-partition-physical: skipped; set DOPPLER_PARTITION_MODEL_DIR');
} else {
  const manifestBytes = await fs.readFile(path.join(modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const modelIdentity = 'sha256:' + createHash('sha256').update(manifestBytes).digest('hex');
  const configPath = process.env.DOPPLER_PARTITION_CONFIG;
  const configBytes = configPath ? await fs.readFile(configPath) : null;
  const configuration = resolvePhysicalPartitionConfig(manifest, modelIdentity,
    configBytes ? JSON.parse(configBytes) : null, { chat: process.env.DOPPLER_PARTITION_CHAT === '1',
      prompts: process.env.DOPPLER_PARTITION_PROMPTS
        ? JSON.parse(await fs.readFile(process.env.DOPPLER_PARTITION_PROMPTS, 'utf8')) : undefined });
  const { plan, planId, generation, limits, prompts, chat } = configuration;
  const independent = configuration.reference ? bindPhysicalIndependentReference(
    await fs.readFile(path.resolve(path.dirname(configPath), configuration.reference.path)),
    configuration.reference, configuration) : null;
  const runtimeConfig = { inference: { session: { kvcache: { maxSeqLen: generation.maxSeqLen } } } };
  const signal = new AbortController().signal;
  const pipelines = [], residents = [], handles = [], references = [];
  const report = { kind: 'same-device-resident-partition-diagnostic', modelIdentity, planId, generation,
    actualModelInference: true, publicCapsuleAcquisition: false, capacityQualification: false,
    configDigest: configBytes ? physicalInputDigest(configBytes) : null, limits,
    parity: { source: 'unsplit-doppler', tolerance: 1e-4 },
    independentAccuracy: independent ? { sourceDigest: configuration.reference.sha256,
      provenance: independent.provenance, tolerance: INDEPENDENT_TOLERANCE,
      scope: 'Unsplit forced-prefix logits at independently produced contexts; not resident sampling or complete-model qualification', steps: [] } : null,
    steps: [] };
  try {
    const gpu = await probeNodeGPU({ installFileFetchShim: true });
    assert.ok(gpu.ready, gpu.reason); report.provider = gpu.providerReceipt;
    for (const index of [null, 0, 1]) {
      const pipeline = await createPipeline(manifest, { runtimeConfig,
        storage: createNodeFileArtifactStorageContext(pathToFileURL(path.resolve(modelDirectory)).href, manifest),
        ...(index === null ? {} : { partition: { plan, index } }) });
      pipelines.push(pipeline);
      const handle = createModelHandle(pipeline, { modelId: manifest.modelId, manifestHash: modelIdentity.slice(7) });
      handles.push(handle);
      const tokenPorts = createCapsuleProgramAdapter(handle,
        { modelId: manifest.modelId, program: { executionGraphHash: 'diagnostic' } },
        { phases: { prefill: [], decode: [] } });
      if (index === null) { references.push(tokenPorts); continue; }
      const allocation = resolveResidentPartitionAllocation(manifest, modelIdentity,
        { model: { id: manifest.modelId, identity: modelIdentity }, plan, planId, index,
          participantId: index === 0 ? 'a' : 'b', limits, generation });
      residents.push(await createResidentPartitionSession(pipeline, allocation, tokenPorts, () => handle.unload()));
    }
    const [reference] = pipelines, [a, b] = residents, [referenceTokens] = references;
    report.descriptors = residents.map(resident => resident.getDescriptor());
    const identity = id => ({ modelId: manifest.modelId, modelIdentity, planId,
      threadId: id, attemptId: id, participantA: 'a', participantB: 'b' });
    async function unsplit(prompt, options) {
      reference.reset();
      const input = referenceTokens.tokenize(prompt, options), context = [...input], decoder = referenceTokens.createIncrementalDecoder();
      const steps = []; let text = '', snapshot;
      try {
        for (let step = 0; step < options.maxTokens; step++) {
          const output = step === 0
            ? await reference.prefillWithLogits(prompt, { inputIds: input, useChatTemplate: false })
            : await reference.decodeStepLogits(context, { useChatTemplate: false });
          if (step === 0) snapshot = output.cache;
          const tokenId = sampleCapsuleLogits(output.logits, context,
            { ...options, seed: samplingSeedForToken(options.seed, step) }, referenceTokens.getTokenContract());
          context.push(tokenId); text += decoder.push(tokenId);
          const stopReason = stoppingReason(tokenId, step + 1, options, referenceTokens.getTokenContract(), () => text + decoder.pendingText());
          steps.push({ tokenId, logits: output.logits, stopReason });
          if (stopReason) break;
        }
        text += decoder.finish({ stopReason: steps.at(-1)?.stopReason }); return { steps, text, input };
      } finally { snapshot?.destroy(); }
    }
    const cacheLength = pipelines[1].runtimeConfig.inference.session.kvcache.maxSeqLen;
    const probes = [createPartitionAttempt(pipelines[1], cacheLength), createPartitionAttempt(pipelines[1], cacheLength)];
    const buffers = probes.map(() => acquireBuffer(64, undefined, 'partition_recurrent_cleanup'));
    probes.forEach((probe, index) => probe.state.linearAttentionRuntime.layers.set(0, { recurrentStateGPU: buffers[index] }));
    probes[0].close(); probes[0].close();
    assert.equal(isBufferActive(buffers[0]), false);
    assert.equal(isBufferActive(buffers[1]), true, 'Closing one attempt must preserve the other recurrent state');
    probes[1].close(); assert.equal(isBufferActive(buffers[1]), false);
    const expected = [];
    for (const prompt of prompts) expected.push(await unsplit(prompt, generation));
    if (independent) for (const [thread, sample] of independent.expected.entries()) {
      assert.deepEqual(sample.input, expected[thread].input, 'Independent reference tokenization changed');
      reference.reset();
      const last = sample.steps.at(-1);
      let snapshot;
      try {
        for (let index = 0; index <= last.step; index++) {
          const output = index === 0
            ? await reference.prefillWithLogits(prompts[thread], { inputIds: sample.input, useChatTemplate: false })
            : await reference.decodeStepLogits([last.context[sample.input.length + index - 1]], { useChatTemplate: false });
          if (index === 0) snapshot = output.cache;
          const selected = sample.steps.find(step => step.step === index);
          if (!selected) continue;
          const comparison = comparePartitionExecution({ splitOutput: output.logits,
            referenceOutput: selected.logits, tolerance: INDEPENDENT_TOLERANCE, minCosineSimilarity: -1 });
          report.independentAccuracy.steps.push({ thread, step: index, comparison });
          assert.ok(comparison.matches, `Independent numerical accuracy: ${JSON.stringify(comparison)}`);
        }
      } finally { snapshot?.destroy(); }
    }
    if (process.env.DOPPLER_PARTITION_REFERENCE_OUT) {
      await fs.writeFile(process.env.DOPPLER_PARTITION_REFERENCE_OUT, JSON.stringify({ kind: 'doppler-parity-capture',
        scope: 'Doppler-generated regression fixture; not independent numerical accuracy', modelIdentity, planId, generation, prompts,
        expected: expected.map(result => ({ input: result.input, text: result.text, steps: result.steps.map(step => ({ tokenId: step.tokenId,
          stopReason: step.stopReason, logits: Buffer.from(step.logits.buffer, step.logits.byteOffset, step.logits.byteLength).toString('base64') })) })) }));
    }
    const requests = await Promise.all(prompts.map((messages, index) => a.tokenize({ messages, identity: identity(String(index)), signal })));
    const states = requests.map((request, index) => ({ identity: identity(String(index)), ids: request.tokenIds,
      position: 0, continuationA: null, continuationB: null, text: '', done: false }));
    async function step(state, index, options = generation) {
      const common = { identity: state.identity, step: index, tokenPosition: state.position,
        inputTokenCount: state.ids.length, maxTokens: options.maxTokens, generation: options, signal };
      const left = await a.executeGroup0({ ...common, tokenIds: state.ids, continuation: state.continuationA });
      const activation = deserializeActivationFrame(serializeActivationFrame(left.activationTensor));
      const right = await b.executeGroup1({ ...common, activation, inputTokenIds: state.ids, continuation: state.continuationB });
      state.position += state.ids.length; state.ids = [right.tokenId]; state.continuationA = left.continuation;
      state.continuationB = right.continuation; state.text += right.delta; state.done = right.done;
      return right;
    }
    for (let start = 0; start < states.length; start += limits.maxConcurrentAttempts) {
      for (let index = 0; index < generation.maxTokens; index++) {
        for (let thread = start; thread < Math.min(states.length, start + limits.maxConcurrentAttempts); thread++) {
          if (states[thread].done) continue;
          const result = await step(states[thread], index);
          const expectedStep = expected[thread].steps[index];
          const comparison = comparePartitionExecution({ splitOutput: result.logits, referenceOutput: expectedStep.logits, tolerance: 1e-4 });
          assert.ok(comparison.matches, JSON.stringify(comparison));
          assert.equal(result.tokenId, expectedStep.tokenId);
          assert.equal(result.stopReason, expectedStep.stopReason);
          report.steps.push({ thread, step: index, tokenId: result.tokenId, stopReason: result.stopReason, comparison });
        }
      }
      for (const state of states.slice(start, start + limits.maxConcurrentAttempts)) {
        for (const resident of residents) await resident.closeAttempt({ identity: state.identity });
      }
    }
    assert.deepEqual(states.map(state => state.text), expected.map(result => result.text));
    report.texts = states.map(state => state.text);
    if (chat) assert.ok(states.every(state => state.done), 'Each chat must complete with model stopping');
    for (const state of states) for (const resident of residents) await resident.closeAttempt({ identity: state.identity });
    for (const resident of residents) assert.equal(resident.getDescriptor().ready, true);
    const short = { identity: identity('short'), ids: requests[0].tokenIds, position: 0, continuationA: null, continuationB: null, text: '' };
    const result = await step(short, 0, resolveGenerationOptions({ ...generation, maxTokens: 1 }));
    assert.equal(result.done, true); assert.equal(result.stopReason, 'max-tokens');
    for (const resident of residents) await resident.closeAttempt({ identity: short.identity });
    await assert.rejects(a.executeGroup0({ identity: short.identity, tokenIds: requests[0].tokenIds, step: 0,
      tokenPosition: 0, inputTokenCount: requests[0].tokenIds.length, maxTokens: generation.maxTokens, generation, continuation: null, signal }), /retired/);
    const victim = { identity: identity('cancel'), ids: requests[0].tokenIds, position: 0, continuationA: null, continuationB: null, text: '' };
    const cancelled = new AbortController(); cancelled.abort(new Error('cancel-before-submit'));
    await assert.rejects(a.executeGroup0({ identity: victim.identity, tokenIds: victim.ids, step: 0,
      tokenPosition: 0, inputTokenCount: victim.ids.length, maxTokens: generation.maxTokens, generation, continuation: null,
      signal: cancelled.signal }), /cancel-before-submit/);
    await Promise.all(residents.map(resident => resident.closeAttempt({ identity: victim.identity })));
    // Exercise the actual GPU output path with a denied character allocation.
    const outputTokens = createCapsuleProgramAdapter(handles[2],
      { modelId: manifest.modelId, program: { executionGraphHash: 'diagnostic' } },
      { phases: { prefill: [], decode: [] } });
    const deniedAllocation = resolveResidentPartitionAllocation(manifest, modelIdentity,
      { model: { id: manifest.modelId, identity: modelIdentity }, plan, planId, index: 1,
        participantId: 'b', limits: { ...limits, maxOutputCharacters: 1 }, generation });
    const denied = await createResidentPartitionSession(pipelines[2], deniedAllocation, outputTokens, async () => {});
    residents.push(denied);
    const deniedIdentity = identity('output-denied');
    const deniedState = { ids: requests[0].tokenIds, position: 0, continuationA: null, continuationB: null };
    assert.ok(expected[0].text.length > 1, 'The physical fixture must exceed one output character');
    let outputError;
    try {
      for (let index = 0; index < generation.maxTokens; index++) {
        const common = { identity: deniedIdentity, step: index, tokenPosition: deniedState.position,
          inputTokenCount: deniedState.ids.length, maxTokens: generation.maxTokens, generation, signal };
        const left = await a.executeGroup0({ ...common, tokenIds: deniedState.ids, continuation: deniedState.continuationA });
        const activation = deserializeActivationFrame(serializeActivationFrame(left.activationTensor));
        try {
          const right = await denied.executeGroup1({ ...common, activation, inputTokenIds: deniedState.ids,
            continuation: deniedState.continuationB });
          deniedState.position += deniedState.ids.length; deniedState.ids = [right.tokenId];
          deniedState.continuationA = left.continuation; deniedState.continuationB = right.continuation;
        } catch (error) { outputError = error; break; }
      }
      assert.equal(outputError?.code, 'DOPPLER_RESIDENT_OUTPUT_LIMIT');
    } finally {
      await Promise.all([a.closeAttempt({ identity: deniedIdentity }), denied.closeAttempt({ identity: deniedIdentity })]);
      await denied.close();
    }
    report.outputAllocationFailure = { code: outputError.code, settled: true };
    report.checks = ['isolated-recurrent-buffer-cleanup', 'interleaved-attempts', 'full-logit-parity', 'sampling-penalties', 'text-parity', 'length-finalization', 'closed-replay', 'cancel-before-submit'];
    console.log(JSON.stringify(report));
  } catch (error) {
    console.error(JSON.stringify({ ...report, failure: { name: error.name, message: error.message } }));
    throw error;
  } finally {
    await Promise.allSettled(residents.map(resident => resident.close()));
    for (const pipeline of pipelines.reverse()) await pipeline.unload();
    destroyDevice(); await releaseNodeWebGPU();
  }
}

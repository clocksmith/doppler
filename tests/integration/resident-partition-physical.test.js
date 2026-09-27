import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { createPipeline } from '../../src/inference/pipelines/text.js';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { createLayerPartitionPlan, deserializeActivationFrame, serializeActivationFrame,
  comparePartitionExecution } from '../../src/partitions.js';
import { resolveResidentPartitionAllocation } from '../../src/inference/pipelines/text/resident-partition-contract.js';
import { createResidentPartitionSession } from '../../src/inference/pipelines/text/resident-partition.js';
import { createModelHandle } from '../../src/client/model-host/model-session.js';
import { createCapsuleProgramAdapter } from '../../src/client/runtime/capsule-program-adapter.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { resolveGenerationOptions } from '../../src/config/generation-contract.js';
import { sampleCapsuleLogits, stoppingReason } from '../../src/inference/generation-step.js';
import { destroyDevice, getDevice } from '../../src/gpu/device.js';
import { releaseNodeWebGPU } from '../../src/tooling/node-webgpu.js';

const modelDirectory = process.env.DOPPLER_PARTITION_MODEL_DIR;
if (!modelDirectory) {
  console.log('resident-partition-physical: skipped; set DOPPLER_PARTITION_MODEL_DIR');
} else {
  const manifestBytes = await fs.readFile(path.join(modelDirectory, 'manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const modelIdentity = 'sha256:' + createHash('sha256').update(manifestBytes).digest('hex');
  const plan = createLayerPartitionPlan({ modelId: manifest.modelId, ...manifest.architecture,
    activationDtype: manifest.inference.session.compute.defaults.activationDtype });
  const planId = computeCanonicalSha256(plan);
  const generation = resolveGenerationOptions({ maxTokens: 3, maxSeqLen: 128, temperature: 0,
    topK: 0, topP: 1, repetitionPenalty: 1.1, repetitionPenaltyWindow: 0,
    presencePenalty: 0.1, useChatTemplate: false });
  const limits = { maxTokens: 3, maxPromptTokens: 64, maxActivationBytes: 1024 * 1024,
    maxOutputCharacters: 1024, maxAttempts: 16, maxConcurrentAttempts: 2 };
  const runtimeConfig = { inference: { session: { kvcache: { maxSeqLen: generation.maxSeqLen } } } };
  const signal = new AbortController().signal;
  const pipelines = [], residents = [], handles = [], references = [];
  const report = { kind: 'same-device-resident-partition-diagnostic', modelIdentity, planId, generation,
    actualModelInference: true, publicCapsuleAcquisition: false, steps: [] };
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
          const tokenId = sampleCapsuleLogits(output.logits, context, options, referenceTokens.getTokenContract());
          context.push(tokenId); text += decoder.push(tokenId);
          const stopReason = stoppingReason(tokenId, step + 1, options, referenceTokens.getTokenContract(), () => text + decoder.pendingText());
          steps.push({ tokenId, logits: output.logits, stopReason });
          if (stopReason) break;
        }
        text += decoder.finish(); return { steps, text, input };
      } finally { snapshot?.destroy(); }
    }
    const prompts = ['The color of the sky is', 'The capital of France is'];
    const expected = [];
    for (const prompt of prompts) expected.push(await unsplit(prompt, generation));
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
    for (let index = 0; index < generation.maxTokens; index++) {
      for (let thread = 0; thread < states.length; thread++) {
        if (states[thread].done) continue;
        const result = await step(states[thread], index);
        const expectedStep = expected[thread].steps[index];
        const comparison = comparePartitionExecution({ splitOutput: result.logits, referenceOutput: expectedStep.logits, tolerance: 1e-4 });
        assert.ok(comparison.matches, JSON.stringify(comparison));
        assert.equal(result.tokenId, expectedStep.tokenId);
        assert.equal(result.stopReason, expectedStep.stopReason);
        report.steps.push({ thread, step: index, tokenId: result.tokenId, comparison });
      }
    }
    assert.deepEqual(states.map(state => state.text), expected.map(result => result.text));
    report.texts = states.map(state => state.text);
    for (const state of states) for (const resident of residents) await resident.closeAttempt({ identity: state.identity });
    for (const resident of residents) assert.equal(resident.getDescriptor().ready, true);
    const short = { identity: identity('short'), ids: requests[0].tokenIds, position: 0, continuationA: null, continuationB: null, text: '' };
    const result = await step(short, 0, resolveGenerationOptions({ ...generation, maxTokens: 1 }));
    assert.equal(result.done, true); assert.equal(result.stopReason, 'max-tokens');
    for (const resident of residents) await resident.closeAttempt({ identity: short.identity });
    await assert.rejects(a.executeGroup0({ identity: short.identity, tokenIds: requests[0].tokenIds, step: 0,
      tokenPosition: 0, inputTokenCount: requests[0].tokenIds.length, maxTokens: 3, generation, continuation: null, signal }), /retired/);
    const victim = { identity: identity('cancel'), ids: requests[0].tokenIds, position: 0, continuationA: null, continuationB: null, text: '' };
    const cancelled = new AbortController(); cancelled.abort(new Error('cancel-before-submit'));
    await assert.rejects(a.executeGroup0({ identity: victim.identity, tokenIds: victim.ids, step: 0,
      tokenPosition: 0, inputTokenCount: victim.ids.length, maxTokens: 3, generation, continuation: null,
      signal: cancelled.signal }), /cancel-before-submit/);
    await Promise.all(residents.map(resident => resident.closeAttempt({ identity: victim.identity })));
    report.checks = ['interleaved-attempts', 'full-logit-parity', 'sampling-penalties', 'text-parity', 'length-finalization', 'closed-replay', 'cancel-before-submit'];
    console.log(JSON.stringify(report));
  } finally {
    await Promise.allSettled(residents.map(resident => resident.close()));
    for (const pipeline of pipelines.reverse()) await pipeline.unload();
    destroyDevice(); await releaseNodeWebGPU();
  }
}

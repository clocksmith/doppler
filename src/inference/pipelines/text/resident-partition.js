import { runPipelineOperation } from '../shader-scoped-pipeline.js';
import { isDeviceLost } from '../../../gpu/device-state.js';
import { readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { computeCanonicalSha256 } from '../../../formats/canonical-hash.js';
import { resolveGenerationOptions } from '../../../config/generation-contract.js';
import { sampleCapsuleLogits, stoppingReason } from '../../generation-step.js';
import { assertPartitionExecutionSupported, executePartitionLayers } from './partition-execution.js';
import { createPartitionAttempt } from './partition-attempt.js';
import { assertResidentPartitionIdentity } from './resident-partition-contract.js';
import { hashLayerPartitionPlan } from './layer-partition-contract.js';
import { markDeviceWeights, getDeviceMemorySnapshot } from '../../../memory/device-budget.js';

/** @type {import('./resident-partition.js').createResidentPartitionSession} */
export async function createResidentPartitionSession(pipeline, allocation, tokens, closeProgram) {
  const descriptor = await runPipelineOperation(pipeline, owner => {
    if (owner.gpuContext?.device) markDeviceWeights(owner.gpuContext.device, owner.dopplerLoader?.gpuBuffers ?? []);
    assertPartitionExecutionSupported(owner, allocation.plan);
    if (owner.modelPartition?.index !== allocation.index
      || hashLayerPartitionPlan(owner.modelPartition.plan) !== allocation.planId) {
      throw new Error('Loaded pipeline partition differs from its accepted allocation.');
    }
    // Resident execution creates one cache per attempt. The opening pipeline's
    // empty cache is never used by those attempts and must not reserve a second
    // copy of the participant's entire context allocation.
    owner.kvCache?.destroy();
    owner.kvCache = null;
    return { schema: /** @type {const} */ ('doppler.resident-partition/v1'), ready: true,
      modelId: allocation.model.id, modelIdentity: allocation.model.identity, planId: allocation.planId,
      index: allocation.index, layerRange: [...allocation.plan.partitions[allocation.index].layerRange],
      residentWeightBytes: [...(owner.dopplerLoader?.gpuBuffers ?? [])].reduce((bytes, buffer) => bytes + buffer.size, 0),
      generationDigest: computeCanonicalSha256(allocation.generation) };
  });
  /** @type {Map<string, import('./resident-partition.js').ResidentAttempt>} */
  const attempts = new Map();
  let closed = false;
  /** @type {Promise<void> | null} */
  let closing = null;

  function assertOpen() {
    if (closed || isDeviceLost(pipeline.gpuContext?.device)) throw new Error('Resident partition is closed or its device was lost.');
  }
  /** @param {import('./resident-partition-contract.js').ResidentPartitionIdentity} identity */
  function attemptFor(identity) {
    const binding = assertResidentPartitionIdentity(identity, allocation);
    let attempt = attempts.get(identity.attemptId);
    if (attempt && attempt.binding !== binding) throw new Error('Resident attempt identity collision.');
    if (!attempt) {
      if (attempts.size >= allocation.limits.maxAttempts) throw new Error('Resident attempt budget exhausted.');
      attempt = { binding, identity: structuredClone(identity), nonce: crypto.randomUUID(), step: 0, position: 0,
        retired: false, done: false, controller: new AbortController(), pending: null, settlement: null,
        execution: null, generationDigest: null, contextTokens: [], text: '', decoder: null };
      attempts.set(identity.attemptId, attempt);
    }
    return attempt;
  }
  /** @param {number[]} tokenIds @param {number} count */
  function checkTokens(tokenIds, count) {
    if (!Array.isArray(tokenIds) || tokenIds.length !== count || tokenIds.some(id => !Number.isSafeInteger(id)
      || id < 0 || id >= allocation.plan.vocabSize)) throw new Error('Resident input token context is invalid.');
  }
  /** @param {import('./resident-partition-contract.js').ResidentPartitionARequest | import('./resident-partition-contract.js').ResidentPartitionBRequest} input */
  async function execute(input) {
    assertOpen(); input.signal.throwIfAborted();
    const { signal, ...data } = input;
    if (!Number.isSafeInteger(data.inputTokenCount) || data.inputTokenCount < 1
      || data.inputTokenCount > (data.step === 0 ? allocation.limits.maxPromptTokens : 1)) {
      throw new Error('Resident input exceeds its prompt allocation.');
    }
    const inputIds = 'tokenIds' in data ? data.tokenIds : data.inputTokenIds;
    checkTokens(inputIds, data.inputTokenCount);
    if ('activation' in data && data.activation?.tensorData?.byteLength > allocation.limits.maxActivationBytes) {
      throw new Error('Resident activation exceeds its byte allocation.');
    }
    const request = structuredClone(data);
    const generation = resolveGenerationOptions(request.generation);
    if (generation.maxTokens !== request.maxTokens || request.maxTokens > allocation.generation.maxTokens
      || computeCanonicalSha256({ ...generation, maxTokens: allocation.generation.maxTokens }) !== descriptor.generationDigest) {
      throw new Error('Resident generation settings differ from the accepted program.');
    }
    const attempt = attemptFor(request.identity);
    if (attempt.retired || attempt.done || attempt.pending || request.step !== attempt.step
      || request.tokenPosition !== attempt.position || request.step >= request.maxTokens) {
      throw new Error('Resident attempt is retired, busy or out of order.');
    }
    const continuation = request.step === 0 ? null : { nonce: attempt.nonce, step: attempt.step, position: attempt.position };
    if (computeCanonicalSha256(request.continuation) !== computeCanonicalSha256(continuation)) {
      throw new Error('Resident continuation mismatch.');
    }
    const generationDigest = computeCanonicalSha256(generation);
    if (attempt.generationDigest && attempt.generationDigest !== generationDigest) throw new Error('Resident attempt generation settings changed.');
    attempt.generationDigest = generationDigest;
    if (request.step === 0 && request.inputTokenCount + request.maxTokens > generation.maxSeqLen) {
      throw new Error('Resident maxSeqLen must cover prompt and generated tokens.');
    }
    if (!attempt.execution && [...attempts.values()].filter(item => item.execution || item.pending).length >= allocation.limits.maxConcurrentAttempts) {
      throw new Error('Resident concurrency allocation exhausted.');
    }
    const combined = AbortSignal.any([signal, attempt.controller.signal]);
    const operation = runPipelineOperation(pipeline, async owner => {
      try {
        assertOpen(); combined.throwIfAborted();
        attempt.execution ??= createPartitionAttempt(owner, request.inputTokenCount + request.maxTokens);
        const state = attempt.execution.state;
        const ids = 'tokenIds' in request ? request.tokenIds : request.inputTokenIds;
        if ('activation' in request) {
          const activation = request.activation;
          if (activation?.dtype !== allocation.plan.activationDtype || activation.step !== request.step
            || activation.seqOffset !== request.tokenPosition
            || JSON.stringify(activation.shape) !== JSON.stringify([1, request.inputTokenCount, allocation.plan.hiddenSize])) {
            throw new Error('Resident activation identity or shape mismatch.');
          }
        }
        const executionStarted = performance.now();
        const result = await executePartitionLayers(state, { numTokens: request.inputTokenCount,
          ...('tokenIds' in request ? { tokenIds: ids } : { activationBytes: request.activation.tensorData }) }, combined);
        state.currentSeqLen += request.inputTokenCount;
        state.decodeStepCount++;
        attempt.step++; attempt.position = state.currentSeqLen;
        const next = { nonce: attempt.nonce, step: attempt.step, position: attempt.position };
        const memory = getDeviceMemorySnapshot(owner.gpuContext?.device ?? null);
        const metrics = { ...result.timing, executionMs: performance.now() - executionStarted,
          logitsReadbackMs: 0, samplingMs: 0,
          memory: memory && { maxBytes: memory.maxBytes, liveBytes: memory.liveBytes,
            peakBytes: memory.peakBytes, categories: memory.categories } };
        if ('activationBytes' in result && result.activationBytes) {
          if (result.activationBytes.byteLength > allocation.limits.maxActivationBytes) throw new Error('Resident activation exceeds its byte allocation.');
          return { activationTensor: { shape: [1, request.inputTokenCount, allocation.plan.hiddenSize],
            dtype: allocation.plan.activationDtype, data: result.activationBytes, step: request.step,
            seqOffset: request.tokenPosition }, continuation: next, metrics };
        }
        if (!result.logits) throw new Error('Resident B did not produce logits.');
        let logits;
        const readbackStarted = performance.now();
        try { logits = new Float32Array(await readBuffer(result.logits.logitsBuffer, result.logits.vocabSize * Float32Array.BYTES_PER_ELEMENT)); }
        finally { releaseBuffer(result.logits.logitsBuffer); }
        metrics.logitsReadbackMs = performance.now() - readbackStarted;
        combined.throwIfAborted();
        if (request.step === 0) attempt.contextTokens.push(...ids);
        else if (ids.length !== 1 || ids[0] !== attempt.contextTokens.at(-1)) throw new Error('Resident continuation token differs from the selected token.');
        const samplingStarted = performance.now();
        const tokenContract = tokens.getTokenContract();
        const tokenId = sampleCapsuleLogits(logits, attempt.contextTokens, generation, tokenContract);
        attempt.contextTokens.push(tokenId);
        attempt.decoder ??= tokens.createIncrementalDecoder();
        const decoder = attempt.decoder;
        let delta = decoder.push(tokenId);
        attempt.text += delta;
        const stopReason = stoppingReason(tokenId, attempt.step, generation, tokenContract,
          () => attempt.text + decoder.pendingText());
        if (stopReason) { const final = decoder.finish(); delta += final; attempt.text += final; attempt.done = true; }
        if (attempt.text.length > allocation.limits.maxOutputCharacters) throw new Error('Resident output exceeds its character allocation.');
        metrics.samplingMs = performance.now() - samplingStarted;
        return { identity: request.identity, step: request.step, tokenPosition: request.tokenPosition,
          tokenId, delta, done: attempt.done, stopReason, continuation: next, logits, metrics };
      } catch (error) {
        attempt.retired = true;
        throw error;
      } finally {
        // Attempt caches cannot be destroyed until submitted work has settled,
        // including cancellation or a failure after recording started.
        await owner.gpuContext?.device?.queue.onSubmittedWorkDone();
      }
    });
    attempt.pending = operation;
    try { const result = await operation; combined.throwIfAborted(); return result; }
    catch (error) { attempt.retired = true; throw error; }
    finally { attempt.pending = null; }
  }
  /** @type {import('./resident-partition-contract.js').ResidentPartitionSession['closeAttempt']} */
  function closeAttempt({ identity }) {
    const binding = assertResidentPartitionIdentity(identity, allocation);
    const existing = attempts.get(identity.attemptId);
    if (existing && existing.binding !== binding) throw new Error('Resident attempt identity collision.');
    // A closed session cannot accept delayed submissions, so it needs no new tombstone.
    const attempt = existing ?? (closed ? null : attemptFor(identity));
    if (!attempt) return Promise.resolve();
    if (attempt.settlement) return attempt.settlement;
    attempt.retired = true;
    attempt.controller.abort(new Error('Resident attempt closed.'));
    attempt.settlement = (async () => {
      await attempt.pending?.catch(() => {});
      attempt.execution?.close(); attempt.execution = null;
      attempt.decoder = null; attempt.contextTokens = []; attempt.text = '';
    })();
    return attempt.settlement;
  }
  return {
    getDescriptor: () => ({ ...structuredClone(descriptor), ready: !closed && !isDeviceLost(pipeline.gpuContext?.device) }),
    async tokenize(request) {
      assertOpen(); request.signal.throwIfAborted();
      if (allocation.index !== 0) throw new Error('Only partition A tokenizes input.');
      const messages = structuredClone(request.messages);
      if (typeof messages !== 'string' && (!Array.isArray(messages) || messages.some(message =>
        !message || !['system', 'user', 'assistant'].includes(message.role) || typeof message.content !== 'string'
        || Object.keys(message).some(key => !['role', 'content'].includes(key))))) {
        throw new Error('Resident partitions accept text-only messages; multimodal input requires a different partition contract.');
      }
      const attempt = attemptFor(request.identity);
      if (attempt.retired || attempt.done || attempt.pending || attempt.step !== 0) {
        throw new Error('Resident attempt is retired, busy or out of order.');
      }
      const combined = AbortSignal.any([request.signal, attempt.controller.signal]);
      const operation = runPipelineOperation(pipeline, async () => {
        assertOpen(); combined.throwIfAborted();
        const tokenIds = await tokens.tokenize(messages, allocation.generation);
        combined.throwIfAborted();
        return tokenIds;
      });
      attempt.pending = operation;
      try {
        const tokenIds = await operation;
        combined.throwIfAborted();
        checkTokens(tokenIds, tokenIds.length);
        if (!tokenIds.length || tokenIds.length > allocation.limits.maxPromptTokens) throw new Error('Resident prompt exceeds its token allocation.');
        return { modelIdentity: allocation.model.identity, tokenIds, generation: resolveGenerationOptions(allocation.generation) };
      } catch (error) { attempt.retired = true; throw error; }
      finally { attempt.pending = null; }
    },
    async executeGroup0(request) {
      if (allocation.index !== 0) throw new Error('Partition B cannot execute group A.');
      return /** @type {import('./resident-partition-contract.js').ResidentPartitionAResult} */ (await execute(request));
    },
    async executeGroup1(request) {
      if (allocation.index !== 1) throw new Error('Partition A cannot execute group B.');
      return /** @type {import('./resident-partition-contract.js').ResidentPartitionBResult} */ (await execute(request));
    },
    closeAttempt,
    close() {
      if (closing) return closing;
      closed = true;
      closing = (async () => {
        const results = await Promise.allSettled([...attempts.values()].map(attempt => closeAttempt({ identity: attempt.identity })));
        try { await closeProgram(); }
        finally { attempts.clear(); }
        const failures = results.filter(result => result.status === 'rejected').map(result => result.reason);
        if (failures.length) throw new AggregateError(failures, 'Resident attempt cleanup failed.');
      })();
      return closing;
    },
  };
}

import { ERROR_CODES, createDopplerError } from '../../../errors/doppler-error.js';
import { runPipelineOperation } from '../shader-scoped-pipeline.js';
import { isDeviceLost } from '../../../gpu/device-state.js';
import { readBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { computeCanonicalSha256 } from '../../../formats/canonical-hash.js';
import { resolveGenerationOptions } from '../../../config/generation-contract.js';
import { sampleCapsuleLogits, stoppingReason } from '../../generation-step.js';
import { samplingSeedForToken } from '../../token-sampling.js';
import { assertPartitionExecutionSupported, executePartitionLayers } from './partition-execution.js';
import { createPartitionAttempt } from './partition-attempt.js';
import { hashPartitionGeneration, assertResidentPartitionIdentity, resolveResidentCanonicalGeneration } from './resident-partition-contract.js';
import { admitResidentPrompt } from '../../resident-state-contract.js';
import { createResidentTargetState, createResidentCursorMirror } from '../../resident-target-state.js';
import { createResidentOutputState } from '../../resident-output-state.js';
import { ORDERED_LAYER_PARTITION_SCHEMA, hashLayerPartitionPlan, resolveActivationTensorLayout, validateActivationTensorShape } from './layer-partition-contract.js';
import { markDeviceWeights, getDeviceMemorySnapshot } from '../../../memory/device-budget.js';
import { preparePairedLinearAttentionPipelines } from '../../../gpu/kernels/paired-linear-attention.js';

/** @type {import('./resident-partition.js').createResidentPartitionSession} */
export async function createResidentPartitionSession(pipeline, allocation, tokens, closeProgram, signal) {
  const ordered = allocation.plan.schema === ORDERED_LAYER_PARTITION_SCHEMA;
  const partition = allocation.plan.partitions[allocation.index];
  const canonical = resolveResidentCanonicalGeneration(allocation);
  if (canonical && allocation.index === 1 && typeof tokens.createRawIncrementalDecoder !== 'function') {
    throw new Error('Canonical resident requires a raw incremental decoder.');
  }
  const descriptor = await runPipelineOperation(pipeline, async owner => {
    signal?.throwIfAborted();
    if (owner.gpuContext?.device) markDeviceWeights(owner.gpuContext.device, owner.dopplerLoader?.gpuBuffers ?? []);
    assertPartitionExecutionSupported(owner, allocation.plan);
    if (owner.modelPartition?.index !== allocation.index
      || hashLayerPartitionPlan(owner.modelPartition.plan) !== allocation.planId) {
      throw new Error('Loaded pipeline partition differs from its accepted allocation.');
    }
    const [firstLayer, lastLayer] = allocation.plan.partitions[allocation.index].layerRange;
    if (allocation.plan.activationPrecision === 'two-f32'
      && owner.modelConfig?.layerTypes?.slice(firstLayer, lastLayer + 1).includes('linear_attention')) {
      await preparePairedLinearAttentionPipelines(owner.gpuContext?.device, signal,
        owner.modelConfig.linearNormGateActivation);
    }
    signal?.throwIfAborted();
    // Resident execution creates one cache per attempt. The opening pipeline's
    // empty cache is never used by those attempts and must not reserve a second
    // copy of the participant's entire context allocation.
    owner.kvCache?.destroy();
    owner.kvCache = null;
    return { schema: ordered ? /** @type {const} */ ('doppler.resident-partition/v2')
      : /** @type {const} */ ('doppler.resident-partition/v1'), ready: true,
      ...(ordered ? { stageCount: allocation.plan.partitions.length } : {}),
      modelId: allocation.model.id, modelIdentity: allocation.model.identity, planId: allocation.planId,
      index: allocation.index, layerRange: [...allocation.plan.partitions[allocation.index].layerRange],
      residentWeightBytes: [...(owner.dopplerLoader?.gpuBuffers ?? [])].reduce((bytes, buffer) => bytes + buffer.size, 0),
      generationDigest: computeCanonicalSha256(allocation.generation),
      ...(canonical ? { canonicalGenerationId: canonical.generationId } : {}) };
  });
  /** @type {Map<string, import('./resident-partition.js').ResidentAttempt>} */
  const attempts = new Map();
  let closed = false;
  /** @type {Promise<void> | null} */
  let closing = null;

  function assertOpen() {
    if (closed || isDeviceLost(pipeline.gpuContext?.device)) throw new Error('Resident partition is closed or its device was lost.');
  }
  /** @param {import('./resident-partition-contract.js').ResidentExecutionIdentity} identity */
  function attemptFor(identity) {
    const binding = assertResidentPartitionIdentity(identity, allocation);
    let attempt = attempts.get(identity.attemptId);
    if (attempt && attempt.binding !== binding) throw new Error('Resident attempt identity collision.');
    if (!attempt) {
      if (attempts.size >= allocation.limits.maxAttempts) throw new Error('Resident attempt budget exhausted.');
      attempt = { binding, identity: structuredClone(identity), nonce: crypto.randomUUID(), step: 0, position: 0,
        retired: false, done: false, controller: new AbortController(), pending: null, settlement: null,
        execution: null, generationDigest: null, contextTokens: [], text: '', decoder: null,
        canonicalState: null, canonicalMirror: null, canonicalPrompt: null, canonicalAwaiting: false };
      attempts.set(identity.attemptId, attempt);
    }
    return attempt;
  }
  /** @param {number[]} tokenIds @param {number} count */
  function checkTokens(tokenIds, count) {
    if (!Array.isArray(tokenIds) || tokenIds.length !== count || tokenIds.some(id => !Number.isSafeInteger(id)
      || id < 0 || id >= allocation.plan.vocabSize)) throw new Error('Resident input token context is invalid.');
  }
  /** @param {import('./resident-partition-contract.js').ResidentPartitionARequest | import('./resident-partition-contract.js').ResidentPartitionBRequest} input
   * @param {boolean} canonicalMode */
  async function execute(input, canonicalMode = false) {
    assertOpen(); input.signal.throwIfAborted();
    if (Boolean(canonical) !== canonicalMode) throw new Error('Resident canonical execution mode differs from its allocation.');
    const { signal, ...data } = input;
    if (!Number.isSafeInteger(data.inputTokenCount) || data.inputTokenCount < 1
      || data.inputTokenCount > (data.step === 0 ? allocation.limits.maxPromptTokens : 1)) {
      throw new Error('Resident input exceeds its prompt allocation.');
    }
    if (allocation.plan.activationPrecision === 'two-f32') {
      const { expectedBytes } = validateActivationTensorShape({ shape: [1, data.inputTokenCount, allocation.plan.hiddenSize],
        dtype: allocation.plan.activationDtype, precision: allocation.plan.activationPrecision,
        components: allocation.plan.activationComponents, elementStrideBytes: allocation.plan.activationElementStrideBytes });
      if (expectedBytes > allocation.limits.maxActivationBytes) throw new Error('Resident activation exceeds its byte allocation.');
    }
    const inputIds = 'tokenIds' in data ? data.tokenIds : data.inputTokenIds;
    checkTokens(inputIds, data.inputTokenCount);
    if ('activation' in data && data.activation?.tensorData?.byteLength > allocation.limits.maxActivationBytes) {
      throw new Error('Resident activation exceeds its byte allocation.');
    }
    const request = structuredClone(data);
    const generation = resolveGenerationOptions(request.generation);
    if (!canonicalMode && (generation.maxTokens !== request.maxTokens || request.maxTokens > allocation.generation.maxTokens
      || computeCanonicalSha256({ ...generation, maxTokens: allocation.generation.maxTokens }) !== descriptor.generationDigest)) {
      throw new Error('Resident generation settings differ from the accepted program.');
    }
    const attempt = attemptFor(request.identity);
    if (attempt.retired || attempt.done || attempt.pending || request.step !== attempt.step
      || request.tokenPosition !== attempt.position || request.step >= request.maxTokens) {
      throw new Error('Resident attempt is retired, busy or out of order.');
    }
    if (canonicalMode) {
      const parent = (attempt.canonicalState ?? attempt.canonicalMirror)?.getCursor();
      const expectedIds = parent ? [parent.pendingTokenId] : attempt.canonicalPrompt;
      if (!canonical || !(attempt.canonicalState || attempt.canonicalMirror) || attempt.canonicalAwaiting
        || request.generationId !== canonical.generationId || request.baseCursorId !== (parent?.cursorId ?? null)
        || JSON.stringify(inputIds) !== JSON.stringify(expectedIds)) {
        throw new Error('Canonical step has a stale cursor, changed generation, input or pending commit.');
      }
    }
    const continuation = request.step === 0 ? null : { nonce: attempt.nonce, step: attempt.step, position: attempt.position };
    if (computeCanonicalSha256(request.continuation) !== computeCanonicalSha256(continuation)) {
      throw new Error('Resident continuation mismatch.');
    }
    const generationDigest = computeCanonicalSha256(generation);
    if (attempt.generationDigest && attempt.generationDigest !== generationDigest) throw new Error('Resident attempt generation settings changed.');
    attempt.generationDigest = generationDigest;
    if (!canonicalMode && request.step === 0 && request.inputTokenCount + request.maxTokens > generation.maxSeqLen) {
      throw new Error('Resident maxSeqLen must cover prompt and generated tokens.');
    }
    if (!attempt.execution && [...attempts.values()].filter(item => item.execution || item.pending).length >= allocation.limits.maxConcurrentAttempts) {
      throw new Error('Resident concurrency allocation exhausted.');
    }
    const combined = AbortSignal.any([signal, attempt.controller.signal]);
    const operation = runPipelineOperation(pipeline, async owner => {
      try {
        assertOpen(); combined.throwIfAborted();
        const ids = 'tokenIds' in request ? request.tokenIds : request.inputTokenIds;
        if ('activation' in request) {
          const activation = request.activation;
          const format = resolveActivationTensorLayout({ dtype: allocation.plan.activationDtype,
            precision: allocation.plan.activationPrecision, components: allocation.plan.activationComponents,
            elementStrideBytes: allocation.plan.activationElementStrideBytes });
          if (format.precision === 'two-f32' ? activation.precision !== format.precision
            || activation.components !== format.components || activation.elementStrideBytes !== format.elementStrideBytes
            : activation.precision !== undefined || activation.components !== undefined || activation.elementStrideBytes !== undefined) {
            throw new Error('Resident activation precision or component layout mismatch.');
          }
          validateActivationTensorShape({ ...activation, dtype: allocation.plan.activationDtype,
            byteLength: activation.tensorData?.byteLength, hiddenSize: allocation.plan.hiddenSize });
          if (activation?.dtype !== allocation.plan.activationDtype || activation.step !== request.step
            || activation.seqOffset !== request.tokenPosition
            || JSON.stringify(activation.shape) !== JSON.stringify([1, request.inputTokenCount, allocation.plan.hiddenSize])) {
            throw new Error('Resident activation identity or shape mismatch.');
          }
        }
        attempt.execution ??= createPartitionAttempt(owner, canonicalMode && canonical ? canonical.policy.materializedInputCapacity
          : request.inputTokenCount + request.maxTokens);
        const state = attempt.execution.state;
        const executionStarted = performance.now();
        const result = await executePartitionLayers(state, { numTokens: request.inputTokenCount,
          ...('tokenIds' in request ? { tokenIds: ids } : { activationBytes: request.activation.tensorData }) }, combined);
        state.currentSeqLen += request.inputTokenCount;
        state.decodeStepCount++;
        attempt.step++; attempt.position = state.currentSeqLen;
        const next = { nonce: attempt.nonce, step: attempt.step, position: attempt.position };
        const memory = getDeviceMemorySnapshot(owner.gpuContext?.device ?? null);
        /** @type {import('./resident-partition-contract.js').ResidentPartitionMetrics} */
        const metrics = { ...result.timing, executionMs: performance.now() - executionStarted,
          logitsReadbackMs: 0, samplingMs: 0,
          memory: memory && { maxBytes: memory.maxBytes, liveBytes: memory.liveBytes,
            peakBytes: memory.peakBytes, categories: memory.categories } };
        if ('activationBytes' in result && result.activationBytes) {
          if (ordered && partition.hasLmHead) throw new Error('Final ordered stage must produce logits.');
          if (canonicalMode) attempt.canonicalAwaiting = true;
          if (result.activationBytes.byteLength > allocation.limits.maxActivationBytes) throw new Error('Resident activation exceeds its byte allocation.');
          return { activationTensor: { shape: [1, request.inputTokenCount, allocation.plan.hiddenSize],
            dtype: allocation.plan.activationDtype,
            ...(allocation.plan.activationPrecision === 'two-f32' ? { precision: allocation.plan.activationPrecision,
              components: allocation.plan.activationComponents, elementStrideBytes: allocation.plan.activationElementStrideBytes } : {}), data: result.activationBytes, step: request.step,
            seqOffset: request.tokenPosition }, continuation: next, metrics };
        }
        if (!partition.hasLmHead && ordered) throw new Error('Only the final ordered stage may produce logits.');
        if (!result.logits) throw new Error('Resident final stage did not produce logits.');
        let logits;
        const readbackStarted = performance.now();
        try { logits = new Float32Array(await readBuffer(result.logits.logitsBuffer, result.logits.vocabSize * Float32Array.BYTES_PER_ELEMENT)); }
        finally { releaseBuffer(result.logits.logitsBuffer); }
        metrics.logitsReadbackMs = performance.now() - readbackStarted;
        combined.throwIfAborted();
        if (canonicalMode) {
          const target = attempt.canonicalState;
          if (!target) throw new Error('Canonical target state is unavailable.');
          const parent = target.getCursor();
          const samplingStarted = performance.now();
          const transition = parent ? target.advance(parent.cursorId, logits) : target.bootstrap(logits);
          attempt.done = transition.successor.terminal;
          metrics.samplingMs = performance.now() - samplingStarted;
          return { identity: request.identity, step: request.step, tokenPosition: request.tokenPosition,
            tokenId: transition.acceptedTokenIds[0], delta: transition.delta, done: attempt.done,
            stopReason: transition.successor.stopReason, continuation: next, logits, metrics, transition };
        }
        if (request.step === 0) attempt.contextTokens.push(...ids);
        else if (ids.length !== 1 || ids[0] !== attempt.contextTokens.at(-1)) throw new Error('Resident continuation token differs from the selected token.');
        const samplingStarted = performance.now();
        const tokenContract = tokens.getTokenContract();
        const tokenId = sampleCapsuleLogits(logits, attempt.contextTokens,
          { ...generation, seed: samplingSeedForToken(generation.seed, request.step) }, tokenContract);
        attempt.contextTokens.push(tokenId);
        attempt.decoder ??= tokens.createIncrementalDecoder();
        const decoder = attempt.decoder;
        let delta = decoder.push(tokenId);
        const assertOutputAllowance = () => {
          const usage = decoder.getUsage?.();
          if (usage) metrics.outputUsage = usage;
          if ((usage?.rawCharacters ?? attempt.text.length) > allocation.limits.maxOutputCharacters) {
            throw createDopplerError(ERROR_CODES.RESIDENT_OUTPUT_LIMIT, 'Resident output exceeds its character allocation.');
          }
        };
        attempt.text += delta;
        assertOutputAllowance();
        const stopReason = stoppingReason(tokenId, attempt.step, generation, tokenContract,
          () => attempt.text + decoder.pendingText());
        if (stopReason) { const final = decoder.finish({ stopReason }); delta += final; attempt.text += final; attempt.done = true; }
        assertOutputAllowance();
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
      attempt.canonicalState = null; attempt.canonicalMirror = null; attempt.canonicalPrompt = null; attempt.canonicalAwaiting = false;
    })();
    return attempt.settlement;
  }
  return {
    getDescriptor: () => ({ ...structuredClone(descriptor), ready: !closed && !isDeviceLost(pipeline.gpuContext?.device) }),
    getRecoveryCapabilities: () => Object.freeze({ schema: /** @type {const} */ ('doppler.resident-recovery/v1'),
      inputReplay: false, checkpointExport: false, checkpointImport: false }),
    getCanonicalGeneration: () => canonical,
    async admitCanonicalPrompt(request) {
      assertOpen(); request.signal.throwIfAborted();
      if (!canonical || request.generationId !== canonical.generationId) throw new Error('Canonical generation allocation mismatch.');
      checkTokens(request.tokenIds, request.tokenIds.length);
      if (request.tokenIds.length > allocation.limits.maxPromptTokens) throw new Error('Resident prompt exceeds its token allocation.');
      const admitted = admitResidentPrompt(request.tokenIds, canonical.policy, allocation.plan.vocabSize, canonical.generationId);
      const attempt = attemptFor(request.identity);
      if (attempt.retired || attempt.done || attempt.pending || attempt.step !== 0 || attempt.canonicalPrompt) {
        throw new Error('Canonical admission is already used, busy or retired.');
      }
      if (!('kind' in admitted)) { attempt.done = true; return admitted; }
      attempt.canonicalPrompt = [...admitted.tokenIds];
      try {
        if (allocation.index === 0) {
          attempt.canonicalMirror = createResidentCursorMirror(admitted.tokenIds, canonical.policy,
            allocation.plan.vocabSize, canonical.generationId);
        } else {
          if (!tokens.createRawIncrementalDecoder) throw new Error('Canonical raw decoder is unavailable.');
          attempt.canonicalState = createResidentTargetState(admitted.tokenIds, canonical.policy, allocation.plan.vocabSize,
            canonical.generationId, createResidentOutputState(tokens.createRawIncrementalDecoder(),
              tokens.createIncrementalDecoder(), canonical.policy, allocation.plan.vocabSize));
        }
      } catch (error) { attempt.retired = true; throw error; }
      return admitted;
    },
    async executeCanonicalGroup0(request) {
      if (!canonical || allocation.index !== 0) throw new Error('Canonical partition A execution is unavailable.');
      return /** @type {import('./resident-partition-contract.js').ResidentPartitionAResult} */ (await execute({ ...request,
        generation: { ...allocation.generation, maxTokens: canonical.policy.maxGeneratedTokens,
          maxSeqLen: canonical.policy.materializedInputCapacity }, maxTokens: canonical.policy.maxGeneratedTokens }, true));
    },
    async executeCanonicalGroup1(request) {
      if (!canonical || allocation.index !== 1) throw new Error('Canonical partition B execution is unavailable.');
      return /** @type {import('./resident-partition-contract.js').ResidentCanonicalBResult} */ (await execute({ ...request,
        generation: { ...allocation.generation, maxTokens: canonical.policy.maxGeneratedTokens,
          maxSeqLen: canonical.policy.materializedInputCapacity }, maxTokens: canonical.policy.maxGeneratedTokens }, true));
    },
    async commitCanonicalTransition(request) {
      assertOpen(); request.signal.throwIfAborted();
      if (!canonical || allocation.index !== 0) throw new Error('Only canonical partition A accepts target transitions.');
      const attempt = attemptFor(request.identity);
      if (attempt.retired || attempt.done || attempt.pending || !attempt.canonicalAwaiting || !attempt.canonicalMirror) {
        throw new Error('Canonical input stage has no settled numerical step awaiting commit.');
      }
      try {
        const transition = attempt.canonicalMirror.acceptTransition(request.baseCursorId, request.transition);
        attempt.canonicalAwaiting = false; attempt.done = transition.successor.terminal;
        return transition;
      } catch (error) { attempt.retired = true; throw error; }
    },
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
    async executeStage(request) {
      if (!ordered) throw new Error('Ordered stage execution requires a v3 layer plan.');
      if (request.partitionIndex !== allocation.index
        || (partition.hasEmbedding ? request.activation !== undefined : request.activation === undefined)) {
        throw new Error('Ordered stage index or input role mismatch.');
      }
      const { signal: stageSignal, ...input } = request;
      stageSignal.throwIfAborted();
      const snapshot = structuredClone(input);
      if (hashPartitionGeneration(snapshot.generation) !== snapshot.generationDigest) {
        throw new Error('Ordered stage generation digest mismatch.');
      }
      const layout = resolveActivationTensorLayout({ dtype: allocation.plan.activationDtype,
        precision: allocation.plan.activationPrecision, components: allocation.plan.activationComponents,
        elementStrideBytes: allocation.plan.activationElementStrideBytes });
      const { expectedBytes } = validateActivationTensorShape({ ...layout, dtype: allocation.plan.activationDtype,
        shape: [1, snapshot.inputTokenCount, allocation.plan.hiddenSize] });
      if (expectedBytes > allocation.limits.maxActivationBytes) throw new Error('Resident activation exceeds its byte allocation.');
      const stepInput = { identity: snapshot.identity, step: snapshot.step, tokenPosition: snapshot.tokenPosition,
        inputTokenCount: snapshot.inputTokenCount, maxTokens: snapshot.maxTokens,
        generation: snapshot.generation, continuation: snapshot.continuation, signal: stageSignal };
      const result = partition.hasEmbedding
        ? await execute({ ...stepInput, tokenIds: snapshot.inputTokenIds })
        : await execute({ ...stepInput, inputTokenIds: snapshot.inputTokenIds,
          activation: /** @type {import('./resident-partition-contract.js').ResidentPartitionBRequest['activation']} */ (snapshot.activation) });
      return { ...result, identity: snapshot.identity, partitionIndex: allocation.index,
        step: snapshot.step, tokenPosition: snapshot.tokenPosition };
    },
    async executeGroup0(request) {
      if (ordered || allocation.index !== 0) throw new Error('Legacy group A execution requires a two-stage input partition.');
      return /** @type {import('./resident-partition-contract.js').ResidentPartitionAResult} */ (await execute(request));
    },
    async executeGroup1(request) {
      if (ordered || allocation.index !== 1) throw new Error('Legacy group B execution requires a two-stage output partition.');
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

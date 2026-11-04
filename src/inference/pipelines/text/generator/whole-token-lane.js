import { getDevice } from '../../../../gpu/device.js';
import { isGreedySampling } from '../../../../rules/greedy-sampling.js';
import { FINITENESS_RESET_WORDS, FinitenessError, parseFinitenessStatusWords } from '../finiteness-guard-status.js';
import { advanceDecodeStepCount } from '../tsir-fixture-writer.js';
import { queueLinearAttentionCheckpoint } from '../linear-attention/state-checkpoint.js';
import { resolveActiveExecutionPlan } from '../execution-plan.js';
import { WholeTokenProgram } from './whole-token-program.js';

// Whole-token program lane of single-token decode (decodeLoop.wholeTokenProgram):
// without a built program, step 1 of a generation runs the ordinary recorded step and
// later steps are observed until the program is built, then replay it. A step the
// built program cannot run (outside its declared positions, or with a changed plan or
// resource) invalidates it and is observed again. A built program is kept for the next
// generation, whose step 1 is observed and must match it (WholeTokenProgram.verify)
// before step 2 replays. While the prefill's last commands run on the GPU, the kept
// program can instead be verified on a step 1 that is recorded and never submitted
// (verifyWholeTokenProgramAhead); a match replays step 1 itself. See
// whole-token-program.js.

/** @typedef {import('../state.js').PipelineState} PipelineState */
/** @typedef {{ seqLen?: unknown, keysGPU?: GPUBuffer | null, valuesGPU?: GPUBuffer | null }} KVLayerView */
/** @typedef {{ seqLen?: unknown, convStateGPU?: GPUBuffer | null, recurrentStateGPU?: GPUBuffer | null }} LinearLayerView */
/**
 * The KV cache fields the lane reads: its layers (a layer-range cache maps local to
 * model layer indices), capacity and bookkeeping entry point.
 * @typedef {{ layers?: (KVLayerView | null)[], layerIndices?: number[], maxSeqLen?: number,
 *   markAppended(layerIdx: number, startPos: number, numTokens: number): void }} KVCacheView
 */
/**
 * @typedef {{ kind: 'kv', owner: { seqLen: number }, layerIdx: number }
 *   | { kind: 'linear', owner: { seqLen: number } }} SequenceAdvancer
 */

/** @param {PipelineState} state */
function kvCacheOf(state) {
  return /** @type {KVCacheView | null} */ (/** @type {unknown} */ (state.kvCache));
}

/** @param {PipelineState} state */
function linearLayersOf(state) {
  const layers = /** @type {{ layers?: unknown } | null} */ (state.linearAttentionRuntime)?.layers;
  return layers instanceof Map ? /** @type {Map<number, LinearLayerView | null>} */ (layers) : null;
}

/**
 * Sequence bookkeeping a decode step advances on the CPU: KV layers (through the
 * cache's own markAppended) and linear-attention layer states.
 * @param {PipelineState} state
 * @returns {SequenceAdvancer[]}
 */
function sequenceAdvancers(state) {
  /** @type {SequenceAdvancer[]} */
  const fields = [];
  const kv = kvCacheOf(state);
  kv?.layers?.forEach((layer, local) => {
    if (layer && typeof layer.seqLen === 'number') {
      fields.push({ kind: 'kv', owner: /** @type {{ seqLen: number }} */ (layer), layerIdx: kv.layerIndices?.[local] ?? local });
    }
  });
  for (const layer of linearLayersOf(state)?.values() ?? []) {
    if (layer && typeof layer.seqLen === 'number') fields.push({ kind: 'linear', owner: /** @type {{ seqLen: number }} */ (layer) });
  }
  return fields;
}

/**
 * @param {PipelineState} state
 * @param {SequenceAdvancer[]} advancers
 * @param {number} pos
 */
function applySequenceAdvance(state, advancers, pos) {
  for (const advancer of advancers) {
    if (advancer.kind === 'kv') /** @type {KVCacheView} */ (kvCacheOf(state)).markAppended(advancer.layerIdx, pos, 1);
    else advancer.owner.seqLen = Math.max(advancer.owner.seqLen, pos + 1);
  }
}

/**
 * The plan and resources a whole-token program binds by identity, read after a step:
 * a plan switch (finiteness fallback), a recreated KV cache, reallocated decode buffers
 * or replaced linear-attention state makes the program describe a different step.
 * @param {PipelineState} state
 * @returns {unknown[]}
 */
function wholeTokenIdentity(state) {
  const parts = [
    state.executionPlanState ?? null,
    state.executionPlanState?.activePlanId ?? null,
    state.kvCache ?? null,
    state.linearAttentionRuntime ?? null,
    /** @type {{ buffers?: unknown } | null} */ (state.decodeBuffers)?.buffers ?? null,
    state.decodeRing ?? null,
    state.finitenessBuffer ?? null,
    state.weights ?? null,
  ];
  for (const layer of kvCacheOf(state)?.layers ?? []) parts.push(layer?.keysGPU ?? null, layer?.valuesGPU ?? null);
  for (const layer of linearLayersOf(state)?.values() ?? []) {
    parts.push(layer, layer?.convStateGPU ?? null, layer?.recurrentStateGPU ?? null);
  }
  return parts;
}

/** Thrown by a decode step recorded for verification ahead, in place of its submit. */
export const WHOLE_TOKEN_DRY_STEP_STOP = Object.freeze({ reason: 'whole-token step verified ahead' });

/** @type {typeof import('./whole-token-lane.js').verifyWholeTokenProgramAhead} */
export async function verifyWholeTokenProgramAhead(state, position, token, runRecordedDry) {
  const lane = state.wholeTokenProgram;
  if (!lane?.program.ready || lane.verifiedAhead != null) return false;
  const { program } = lane;
  program.beginGeneration();
  // The dry step advances CPU sequence bookkeeping as an executed step would; all of it is
  // restored, and the fields it advanced become the replay's advancers.
  const fields = sequenceAdvancers(state);
  const before = fields.map(({ owner }) => owner.seqLen);
  const kvCache = /** @type {{ currentSeqLen?: number, truncate?(length: number): void } | null} */ (/** @type {unknown} */ (state.kvCache));
  const saved = { seqLen: state.currentSeqLen, steps: state.decodeStepCount, kvSeqLen: kvCache?.currentSeqLen };
  state.currentSeqLen = position;
  let verified = false;
  try {
    verified = await program.verifyAhead(position, token, runRecordedDry, () => wholeTokenIdentity(state),
      kvCapacityLimits(state), WHOLE_TOKEN_DRY_STEP_STOP);
    if (verified) {
      const advancers = fields.filter(({ owner }, i) => owner.seqLen !== before[i]);
      if (advancers.some(({ owner }) => owner.seqLen !== position + 1)) {
        throw new Error('[WholeTokenProgram] A decode step advanced sequence state by more than one position.');
      }
      lane.advancers = advancers;
      lane.verifiedAhead = position;
    }
  } finally {
    state.currentSeqLen = saved.seqLen;
    state.decodeStepCount = saved.steps;
    fields.forEach(({ owner }, i) => { owner.seqLen = before[i]; });
    // The cache owns its sequence length (a resident cache refuses outside writes).
    if (typeof saved.kvSeqLen === 'number') /** @type {{ truncate(length: number): void }} */ (kvCache).truncate(saved.kvSeqLen);
  }
  state.stats.wholeTokenProgram = { ...program.stats, dispatches: program.ready ? program.dispatchCount : null };
  return verified;
}

/** @type {typeof import('./whole-token-lane.js').wholeTokenProgramRequested} */
export function wholeTokenProgramRequested(state, opts) {
  const session = /** @type {{ decodeLoop?: { wholeTokenProgram?: unknown, batchSize?: number, stopCheckMode?: string, readbackInterval?: number, readbackMode?: string } }} */ (
    state.resolvedRuntimeSession?.runtime?.session ?? state.runtimeConfig.inference.session);
  const requested = session.decodeLoop?.wholeTokenProgram ?? false;
  if (requested !== true && requested !== false) {
    throw new Error('decodeLoop.wholeTokenProgram must be a boolean when set.');
  }
  if (!requested) return false;
  const blockers = [];
  const batchSize = session.decodeLoop?.batchSize;
  if (typeof batchSize === 'number' && batchSize > 1 && (state.kvCache?.layout !== 'contiguous' || state.convLayerStates.size || ('modelPartition' in state && state.modelPartition != null))) {
    blockers.push('batch replay without contiguous whole-model checkpointable state');
  }
  if (typeof batchSize === 'number' && batchSize > 1 && session.decodeLoop?.stopCheckMode !== 'batch') blockers.push('batch replay without stopCheckMode=batch');
  if (typeof batchSize === 'number' && batchSize > 1
    && (session.decodeLoop?.readbackInterval !== 1 || session.decodeLoop?.readbackMode !== 'sequential')) {
    blockers.push('batch replay without sequential per-token readback');
  }
  if (typeof batchSize === 'number' && batchSize > 1 && state.executionPlanState
    && resolveActiveExecutionPlan(state.executionPlanState).finitenessGuardEnabled) blockers.push('batch replay with active finiteness fallback');
  if (opts.debug) blockers.push('debug');
  if (opts.profile) blockers.push('profile');
  if (opts.executionObserver) blockers.push('executionObserver');
  if (typeof opts.onLogits === 'function') blockers.push('onLogits');
  if (typeof opts.logitMaskFn === 'function' || opts.suppressTokenIds.length > 0) blockers.push('logit masking');
  // Inputs the program has no declared field for: the token history and a per-token random draw.
  if (!(opts.repetitionPenalty === 1 && opts.presencePenalty === 0)) blockers.push('repetition or presence penalty');
  const greedyThreshold = /** @type {number} */ (state.runtimeConfig.inference.sampling.greedyThreshold);
  if (!isGreedySampling(opts.temperature, opts.topK, greedyThreshold)) {
    blockers.push('stochastic sampling');
  }
  if (/** @type {{ pleCache?: unknown }} */ (/** @type {unknown} */ (state)).pleCache) blockers.push('per-layer inputs');
  if (blockers.length > 0) {
    throw new Error(`decodeLoop.wholeTokenProgram cannot run with ${blockers.join(', ')}; disable one of them.`);
  }
  return true;
}

/**
 * Rows a replayed token appends must fit the KV cache the lane advances.
 * @param {PipelineState} state
 */
function kvCapacityLimits(state) {
  const capacity = kvCacheOf(state)?.maxSeqLen;
  return typeof capacity === 'number' && Number.isSafeInteger(capacity)
    ? [{ owner: 'kv-cache capacity', min: 0, max: capacity - 1 }] : [];
}

/**
 * One decode step of the lane. `runRecorded` runs the ordinary recorded step; given
 * an `attach` function, it hands that the step's recorder first.
 * @param {import('../state.js').PipelineState} state
 * @param {number[]} currentIds
 * @param {(attach: ((recorder: import('../../../../gpu/command-recorder.js').CommandRecorder) => void) | null) => Promise<number>} runRecorded
 * @returns {Promise<number>}
 */
export async function decodeWholeTokenStep(state, currentIds, runRecorded) {
  const step = state.decodeStepCount + 1;
  const pos = state.currentSeqLen;
  const lastToken = currentIds[currentIds.length - 1];
  if (step === 1 && state.wholeTokenProgram?.verifiedAhead != null) {
    // Verified ahead at this position: step 1 replays (a different position was rejected below).
    if (state.wholeTokenProgram.verifiedAhead !== pos) state.wholeTokenProgram.program.invalidate(
      `verified ahead at position ${state.wholeTokenProgram.verifiedAhead}, not ${pos}`);
    state.wholeTokenProgram.verifiedAhead = null;
  } else if (step === 1) {
    const kept = state.wholeTokenProgram?.program;
    if (kept?.ready) {
      // Bind group descriptors from earlier generations are still known, so the kept
      // program is verified on this step and replays from step 2.
      kept.beginGeneration();
    } else {
      kept?.destroy();
      const device = getDevice();
      if (!device) throw new Error('[WholeTokenProgram] Requires an active GPU device.');
      state.wholeTokenProgram = { program: new WholeTokenProgram(device), advancers: null };
      return runRecorded(null);
    }
  }
  const lane = state.wholeTokenProgram;
  if (!lane) throw new Error('[WholeTokenProgram] Generation reached decode step > 1 without a program.');
  const { program } = lane;
  if (program.ready && !program.awaitingVerification) {
    const reason = program.rejects(pos, wholeTokenIdentity(state));
    if (reason !== null) program.invalidate(reason);
  }
  if (!program.ready || program.awaitingVerification) {
    const fields = sequenceAdvancers(state);
    const before = fields.map(({ owner }) => owner.seqLen);
    const step = program.awaitingVerification ? program.verify.bind(program) : program.observe.bind(program);
    const token = await step(pos, lastToken, (attach) => runRecorded(attach),
      () => wholeTokenIdentity(state), kvCapacityLimits(state));
    const advancers = fields.filter(({ owner }, i) => owner.seqLen !== before[i]);
    lane.advancers = advancers;
    if (advancers.some(({ owner }) => owner.seqLen !== pos + 1)) {
      throw new Error('[WholeTokenProgram] A decode step advanced sequence state by more than one position.');
    }
    state.stats.wholeTokenProgram = { ...program.stats, dispatches: program.ready ? program.dispatchCount : null };
    return token;
  }
  const wallStart = performance.now();
  advanceDecodeStepCount(state);
  if (state.finitenessBuffer) getDevice()?.queue.writeBuffer(state.finitenessBuffer, 0, FINITENESS_RESET_WORDS);
  const words = await program.run(pos, lastToken);
  const next = validateWholeTokenResult(state, words);
  applySequenceAdvance(state, lane.advancers ?? [], pos);
  state.currentSeqLen = pos + 1;
  state.stats.wholeTokenProgram = { ...program.stats, dispatches: program.dispatchCount };
  state.stats.wholeTokenProgramWallMs = (state.stats.wholeTokenProgramWallMs ?? 0) + (performance.now() - wallStart);
  return next;
}

/** @param {PipelineState} state @param {Uint32Array} words */
function validateWholeTokenResult(state, words) {
  if (state.finitenessBuffer) {
    const status = parseFinitenessStatusWords(words, 1);
    if (status.triggered) throw new FinitenessError(`F16 bounds exceeded during generation${status.metadata}`);
  }
  const token = words[0];
  if (token >= (state.modelConfig?.vocabSize ?? 0)) throw new Error(`[WholeTokenProgram] Sampled token ${token} is outside the vocabulary.`);
  return token;
}

/**
 * Own tentative sequential GPU steps until the consumer publishes a prefix.
 * Returning early (EOS, stop sequence, abort, generator.return or callback error)
 * restores KV/recurrent state and replays exactly that prefix with the same program.
 * @type {typeof import('./whole-token-lane.js').prepareWholeTokenBatch}
 */
export async function prepareWholeTokenBatch(state, currentIds, requestedCount) {
  const lane = state.wholeTokenProgram, pos = state.currentSeqLen;
  if (!lane?.program.ready || lane.program.awaitingVerification
    || (state.decodeStepCount === 0 && lane.verifiedAhead !== pos)) return null;
  const { program } = lane;
  const reason = program.rejects(pos, wholeTokenIdentity(state));
  if (reason !== null) { program.invalidate(reason); return null; }
  // The configured size is a ceiling. Start with ordinary token delivery, then
  // grow the readback window no faster than the tokens already decoded.
  const count = Math.min(requestedCount, state.decodeStepCount, program.positionRange.max - pos + 1);
  if (count < 2) return null;
  const cache = state.kvCache;
  if (!cache || cache.layout !== 'contiguous' || state.convLayerStates.size || ('modelPartition' in state && state.modelPartition != null)) {
    throw new Error('[WholeTokenProgram] Batch requires contiguous whole-model checkpointable state.');
  }
  const device = getDevice();
  if (!device) throw new Error('[WholeTokenProgram] Batch requires a live GPU device.');
  const wallStart = performance.now(), initialToken = currentIds[currentIds.length - 1];
  const advancers = lane.advancers ?? [], before = advancers.map(({ owner }) => owner.seqLen);
  const initialSteps = state.decodeStepCount;
  const initialSubmissions = program.stats.replaySubmissions;
  // Capture precedes decoding on the same ordered queue. Settlement or recovery
  // completes that work before the saved buffers are released.
  const snapshot = state.linearAttentionRuntime
    ? await queueLinearAttentionCheckpoint(state.linearAttentionRuntime) : null;
  let published = 0, closed = false, checkpointSubmissions = snapshot ? 1 : 0;
  /** @type {number[]} */
  let tokens = [];
  const restore = async () => {
    if (snapshot) { await snapshot.restore(); checkpointSubmissions++; }
    cache.truncate(pos);
    for (let i = 0; i < advancers.length; i++) advancers[i].owner.seqLen = before[i];
    state.currentSeqLen = pos; state.decodeStepCount = initialSteps; published = 0;
    program.resumeAfterRestore(pos);
    if (state.finitenessBuffer) device.queue.writeBuffer(state.finitenessBuffer, 0, FINITENESS_RESET_WORDS);
  };
  const advance = () => {
    advanceDecodeStepCount(state);
    applySequenceAdvance(state, advancers, pos + published);
    state.currentSeqLen = pos + ++published;
  };
  const close = () => {
    if (closed) return;
    closed = true; snapshot?.close();
    state.batchingStats.totalBatchedTimeMs += performance.now() - wallStart;
    state.batchingStats.gpuSubmissions += program.stats.replaySubmissions - initialSubmissions + checkpointSubmissions;
    state.batchingStats.resolvedBatchTokens = (state.batchingStats.resolvedBatchTokens ?? 0) + published;
    state.stats.wholeTokenProgram = { ...program.stats, dispatches: program.dispatchCount };
    state.stats.wholeTokenProgramWallMs = (state.stats.wholeTokenProgramWallMs ?? 0) + performance.now() - wallStart;
  };
  try {
    const stream = await program.startBatch(pos, initialToken, count);
    const tokensFromGPU = (async function* () {
      for await (const words of stream.rows) {
        const token = validateWholeTokenResult(state, words);
        tokens.push(token); yield token;
      }
    })();
    lane.verifiedAhead = null;
    state.batchingStats.batchedForwardCalls += 1;
    state.stats.decodeMode = 'batched_gpu'; state.stats.batchGuardReason = null;
    state.batchingStats.requestedBatchTokens = Math.max(state.batchingStats.requestedBatchTokens ?? 0, requestedCount);
    state.batchingStats.effectiveBatchTokens = Math.max(state.batchingStats.effectiveBatchTokens ?? 0, count);
    return {
      tokens: tokensFromGPU,
      publish() {
        if (closed || published >= tokens.length) throw new Error('[WholeTokenProgram] Invalid batch publication.');
        advance();
      },
      async settle() {
        if (closed) return;
        try {
          await stream.settle();
          state.batchingStats.executedBatchTokens = (state.batchingStats.executedBatchTokens ?? 0) + count;
          if (published !== count) {
            const accepted = published;
            await restore(); published = 0;
            for (let index = 0; index < accepted; index++) {
              const words = await program.run(pos + index, index ? tokens[index - 1] : initialToken);
              if (validateWholeTokenResult(state, words) !== tokens[index]) throw new Error('[WholeTokenProgram] Restored prefix changed its sampled token.');
              advance();
            }
          }
        } catch (error) {
          try { await restore(); }
          catch (settlement) { throw new AggregateError([error, settlement], 'Whole-token prefix replay and state recovery failed.'); }
          throw error;
        } finally { close(); }
      },
    };
  } catch (error) {
    try { await restore(); }
    catch (settlement) { throw new AggregateError([error, settlement], 'Whole-token batch and state recovery failed.'); }
    finally { close(); }
    throw error;
  }
}
// Recorded generation mechanisms share this lane owner; each caller supplies its
// declared execution program and retains exclusive model/cache ownership.
export { createCompiledGreedyWindow } from './compiled-greedy-window.js';

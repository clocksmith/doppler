import { getDevice } from '../../../../gpu/device.js';
import { isGreedySampling } from '../../../../rules/greedy-sampling.js';
import { FINITENESS_RESET_WORDS, FinitenessError, parseFinitenessStatusWords } from '../finiteness-guard-status.js';
import { advanceDecodeStepCount } from '../tsir-fixture-writer.js';
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
  const session = /** @type {{ decodeLoop?: { wholeTokenProgram?: unknown } }} */ (
    state.resolvedRuntimeSession?.runtime?.session ?? state.runtimeConfig.inference.session);
  const requested = session.decodeLoop?.wholeTokenProgram ?? false;
  if (requested !== true && requested !== false) {
    throw new Error('decodeLoop.wholeTokenProgram must be a boolean when set.');
  }
  if (!requested) return false;
  const blockers = [];
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
  const next = words[0];
  if (state.finitenessBuffer) {
    const finitenessStatus = parseFinitenessStatusWords(words, 1);
    if (finitenessStatus.triggered) {
      throw new FinitenessError(`F16 bounds exceeded during generation${finitenessStatus.metadata}`);
    }
  }
  applySequenceAdvance(state, lane.advancers ?? [], pos);
  state.currentSeqLen = pos + 1;
  state.stats.wholeTokenProgram = { ...program.stats, dispatches: program.dispatchCount };
  state.stats.wholeTokenProgramWallMs = (state.stats.wholeTokenProgramWallMs ?? 0) + (performance.now() - wallStart);
  if (next >= (state.modelConfig?.vocabSize ?? 0)) {
    throw new Error(`[WholeTokenProgram] Sampled token ${next} is outside the vocabulary.`);
  }
  return next;
}

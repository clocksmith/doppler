import { log } from "../../../../debug/index.js";
import { releaseBuffer } from "../../../../memory/buffer-pool.js";
import { createPartitionAttempt } from './partition-attempt.js';
import { executePartitionLayers } from './partition-execution.js';
import { resolveLayerPartition, getPartitionActivationContract, validateActivationTensorShape } from "../execution/layer-partition-contract.js";

// runtime.shared.kernelWarmup.warmPartition: one prompt pass (and optionally a few single-token
// steps) through this stage's own layers while the stage opens. It compiles the pipelines the
// stage selects on first use and, for routed-expert stages, loads the experts those tokens
// select, so the first request finds them resident. It runs in a private attempt that is closed
// before the stage reports ready: no sequence state survives it. The time counts as opening time.
// A stage that starts from token ids runs the configured ids; a later stage has no real boundary
// activation at opening and runs zeros of the declared shape, which only compiles its pipelines.

const FIELD = 'runtime.shared.kernelWarmup.warmPartition';
const MAX_DECODE_STEPS = 8;

/** @type {import('./resident-partition-warmup.js').runResidentPartitionWarmup} */
export async function runResidentPartitionWarmup(owner, allocation, signal) {
  const warm = owner.runtimeConfig?.shared?.kernelWarmup?.warmPartition;
  if (warm == null) return null;
  const { tokenIds, decodeSteps } = warm;
  const vocabSize = allocation.plan.vocabSize;
  if (!Array.isArray(tokenIds) || tokenIds.length < 1 || tokenIds.length > allocation.limits.maxPromptTokens
    || tokenIds.some(id => !Number.isSafeInteger(id) || id < 0 || id >= vocabSize)) {
    throw new Error(`${FIELD}.tokenIds must hold 1 to limits.maxPromptTokens token ids inside the vocabulary.`);
  }
  if (!Number.isSafeInteger(decodeSteps) || decodeSteps < 0 || decodeSteps > MAX_DECODE_STEPS) {
    throw new Error(`${FIELD}.decodeSteps must be an integer from 0 to ${MAX_DECODE_STEPS}.`);
  }
  if (!owner.manifest || !owner.modelPartition) throw new Error('Partition warm-up requires a loaded, assigned model.');
  const partition = resolveLayerPartition(owner.manifest, owner.modelPartition);
  if (!partition) throw new Error('Partition warm-up allocation is missing.');
  const plan = allocation.plan;
  const hiddenSize = getPartitionActivationContract(plan).hiddenSize;
  /** @param {number} count */
  const zeros = count => new Uint8Array(validateActivationTensorShape({ shape: [1, count, hiddenSize],
    dtype: plan.activationDtype, precision: plan.activationPrecision, components: plan.activationComponents,
    elementStrideBytes: plan.activationElementStrideBytes }).expectedBytes);
  const started = performance.now();
  signal?.throwIfAborted();
  const active = signal ?? new AbortController().signal;
  const attempt = createPartitionAttempt(owner, tokenIds.length + decodeSteps);
  const state = attempt.state;
  /** @param {number[]} ids */
  const step = async ids => {
    const result = await executePartitionLayers(state, { numTokens: ids.length, tokenIds: ids,
      ...(partition.hasEmbedding ? {} : { activationBytes: zeros(ids.length) }) }, active);
    if ('logits' in result && result.logits) releaseBuffer(result.logits.logitsBuffer);
    state.currentSeqLen += ids.length;
    state.decodeStepCount++;
  };
  try {
    await step(tokenIds);
    // Single-token steps select their own pipelines; the repeated id is a placeholder, not an answer.
    for (let index = 0; index < decodeSteps; index++) await step([tokenIds[tokenIds.length - 1]]);
  } finally {
    // The private attempt's caches may not be destroyed until its submitted work has settled.
    await owner.gpuContext?.device?.queue.onSubmittedWorkDone();
    await attempt.close();
  }
  const receipt = { promptTokens: tokenIds.length, decodeSteps, tokenInput: partition.hasEmbedding,
    durationMs: performance.now() - started };
  log.info('Pipeline', `Partition warm-up: ${receipt.promptTokens} prompt tokens, ${decodeSteps} single-token steps, `
    + `${receipt.tokenInput ? 'token ids' : 'zero activations'} (${receipt.durationMs.toFixed(0)}ms)`);
  return receipt;
}

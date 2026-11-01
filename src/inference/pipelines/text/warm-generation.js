import { log } from '../../../debug/index.js';

// runtime.shared.kernelWarmup.warmGeneration: one greedy generation at the end of a text
// pipeline load. It compiles the pipelines this model's plan binds, and with
// decodeLoop.wholeTokenProgram it builds the program that the next generation verifies
// on its first decode step. Sequence state is reset afterwards; compiled pipelines and
// the kept program stay. The time counts as load time (phase "warmGeneration").

const FIELD = 'runtime.shared.kernelWarmup.warmGeneration';

/** @type {typeof import('./warm-generation.js').runWarmGeneration} */
export async function runWarmGeneration(pipeline, warmGeneration) {
  const { prompt, maxTokens } = warmGeneration;
  if (typeof prompt !== 'string' || prompt.length === 0) throw new Error(`${FIELD}.prompt must be a non-empty string.`);
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 2) throw new Error(`${FIELD}.maxTokens must be an integer of at least 2.`);
  let generatedTokens = 0;
  /** @type {string | null} */
  let stopReason = null;
  try {
    const tokens = pipeline.generator.generateTokens(prompt, {
      maxTokens, useChatTemplate: false,
      temperature: 0, topK: 1, topP: 1, repetitionPenalty: 1, presencePenalty: 0, suppressTokenIds: [],
    });
    for await (const _token of tokens) generatedTokens++;
    stopReason = pipeline.stats.stopReason ?? null;
  } finally {
    pipeline.reset();
  }
  const wholeTokenProgramReady = pipeline.wholeTokenProgram?.program.ready === true;
  if (generatedTokens < maxTokens) {
    log.warn('Pipeline', `Warm generation stopped after ${generatedTokens} of ${maxTokens} tokens (${stopReason}).`);
  }
  return { generatedTokens, maxTokens, stopReason, wholeTokenProgramReady };
}

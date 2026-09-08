import { GENERATION_CONTRACT, GenerationError, resolveGenerationOptions, validateGenerationInput } from '../../config/generation-contract.js';
import { applyRepetitionPenalty, applyPresencePenalty, sample } from '../../inference/token-sampling.js';

export const requireGenerationOptions = resolveGenerationOptions;

export function sampleCapsuleLogits(sourceLogits, contextTokens, options, tokenContract = {}) {
  const logits = Float32Array.from(sourceLogits || []);
  if (logits.length === 0) throw new Error('Capsule execution returned empty logits.');
  applyRepetitionPenalty(logits, contextTokens, options.repetitionPenalty, options.repetitionPenaltyWindow);
  applyPresencePenalty(logits, contextTokens, options.presencePenalty, options.repetitionPenaltyWindow);
  return sample(logits, { ...options, padTokenId: tokenContract.padTokenId });
}

function stoppingReason(tokenId, generatedTokens, options, tokenContract, program) {
  if (tokenId === tokenContract.eosTokenId) return 'eos-token';
  if (tokenContract.stopTokenIds?.includes(tokenId)) return 'stop-token';
  if (options.stopSequences.length > 0) {
    const text = program.decodeTokens(generatedTokens);
    if (options.stopSequences.some(sequence => text.endsWith(sequence))) return 'stop-sequence';
  }
  return generatedTokens.length >= options.maxTokens ? 'max-tokens' : null;
}

export function createSessionController(commandExecutor, resourceBinder, program) {
  if (!commandExecutor || !resourceBinder || !program) {
    throw new Error('createSessionController requires commandExecutor, resourceBinder, and program.');
  }
  let closed = false;

  return {
    async *generateTokens(targetPlan, request = {}) {
      if (closed) throw new Error('Capsule runtime session is closed.');
      const { prompt, promptTokens: inputTokens, signal, modules, ...requestedSampling } = request;
      const input = Object.fromEntries(Object.entries({ prompt, promptTokens: inputTokens }).filter(([, value]) => value !== undefined));
      validateGenerationInput(input);
      const sampling = resolveGenerationOptions(requestedSampling);
      const options = { ...input, ...sampling, signal, modules };
      if (signal?.aborted) throw new GenerationError('aborted', 'Generation aborted before prefill.', { cause: signal.reason });
      const promptTokens = inputTokens ? [...inputTokens] : program.tokenize(prompt, { useChatTemplate: sampling.useChatTemplate });
      if (promptTokens.length === 0) throw new GenerationError('invalidRequest', 'Capsule generation prompt must produce at least one token.');
      const dimensions = { seqLen: promptTokens.length, maxSeqLen: sampling.maxSeqLen, batchSize: 1 };
      if (dimensions.maxSeqLen < promptTokens.length + sampling.maxTokens) {
        throw new GenerationError('invalidRequest', 'Capsule generation requires maxSeqLen large enough for prompt and generated tokens.');
      }
      const contextTokens = [...promptTokens];
      const generatedTokens = [];
      const tokenContract = program.getTokenContract();
      let stepResult = null;
      try {
        program.reset();
        resourceBinder.bindSlots(targetPlan.memoryLayout, dimensions);
        resourceBinder.writeSlot('input_tokens', Uint32Array.from(promptTokens));
        const prefill = await commandExecutor.executePhase('prefill', targetPlan.phases.prefill, {
          signal, modules, context: { prompt: prompt ?? '', promptTokens, generationOptions: options },
        });
        stepResult = prefill.results.at(-1);
        for (let step = 0; step < sampling.maxTokens; step += 1) {
          if (signal?.aborted) throw new GenerationError('aborted', 'Generation aborted during decode.', { cause: signal.reason });
          const tokenId = sampleCapsuleLogits(stepResult?.logits, contextTokens, sampling, tokenContract);
          program.releaseStepResult(stepResult);
          stepResult = null;
          generatedTokens.push(tokenId);
          contextTokens.push(tokenId);
          const stopReason = stoppingReason(tokenId, generatedTokens, sampling, tokenContract, program);
          yield tokenId;
          if (signal?.aborted) throw new GenerationError('aborted', 'Generation aborted after token delivery.', { cause: signal.reason });
          if (stopReason) return {
            sampling,
            completion: {
              schema: GENERATION_CONTRACT.completionSchema, stopReason,
              promptTokenCount: promptTokens.length, generatedTokenCount: generatedTokens.length,
            },
          };
          const decode = await commandExecutor.executePhase('decode', targetPlan.phases.decode, {
            signal, modules, context: { contextTokens, generationOptions: options },
          });
          stepResult = decode.results.at(-1);
        }
      } finally {
        program.releaseStepResult(stepResult);
        resourceBinder.releaseTransient();
      }
    },

    async close() {
      if (closed) return;
      closed = true;
      resourceBinder.releaseAll();
      await program.close();
    },
  };
}

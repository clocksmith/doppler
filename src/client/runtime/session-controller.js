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

function stoppingReason(tokenId, generatedCount, options, tokenContract, getText) {
  if (tokenId === tokenContract.eosTokenId) return 'eos-token';
  if (tokenContract.stopTokenIds?.includes(tokenId)) return 'stop-token';
  if (options.stopSequences.length > 0) {
    const text = getText();
    if (options.stopSequences.some(sequence => text.endsWith(sequence))) return 'stop-sequence';
  }
  return generatedCount >= options.maxTokens ? 'max-tokens' : null;
}

export function createSessionController(commandExecutor, resourceBinder, program) {
  if (!commandExecutor || !resourceBinder || !program) {
    throw new Error('createSessionController requires commandExecutor, resourceBinder, and program.');
  }
  let closed = false;

  return {
    async *generateTokens(targetPlan, request = {}, { incremental = false } = {}) {
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
      const generatedTokens = !incremental && sampling.stopSequences.length ? [] : null;
      const stopDecoder = incremental && sampling.stopSequences.length ? program.createIncrementalDecoder() : null;
      const stopLength = sampling.stopSequences.reduce((length, sequence) => Math.max(length, sequence.length), 0);
      let stopTail = '';
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
          resourceBinder.assertDeviceAvailable();
          const tokenId = targetPlan.tokenSelection === undefined
            ? sampleCapsuleLogits(stepResult?.logits, contextTokens, sampling, tokenContract)
            : stepResult?.tokenId;
          if (targetPlan.tokenSelection !== undefined && (!Number.isInteger(tokenId) || tokenId < 0
            || !Number.isInteger(stepResult?.vocabSize) || tokenId >= stepResult.vocabSize)) {
            throw new Error('Declared GPU token selection returned an invalid token result.');
          }
          program.releaseStepResult(stepResult);
          stepResult = null;
          generatedTokens?.push(tokenId);
          contextTokens.push(tokenId);
          if (stopDecoder) stopTail = (stopTail + stopDecoder.push(tokenId)).slice(-stopLength);
          const stopText = () => stopDecoder ? stopTail + stopDecoder.pendingText()
            : generatedTokens ? program.decodeTokens(generatedTokens) : '';
          const stopReason = stoppingReason(tokenId, step + 1, sampling, tokenContract, stopText);
          yield tokenId;
          if (signal?.aborted) throw new GenerationError('aborted', 'Generation aborted after token delivery.', { cause: signal.reason });
          if (stopReason) return {
            sampling,
            completion: {
              schema: GENERATION_CONTRACT.completionSchema, stopReason,
              promptTokenCount: promptTokens.length, generatedTokenCount: step + 1,
            },
          };
          const decode = await commandExecutor.executePhase('decode', targetPlan.phases.decode, {
            signal, modules, context: { contextTokens, generationOptions: options },
          });
          stepResult = decode.results.at(-1);
        }
      } finally {
        try { program.releaseStepResult(stepResult); }
        finally { resourceBinder.releaseTransient(); }
      }
    },

    async close() {
      if (closed) return;
      closed = true;
      const errors = [];
      try { resourceBinder.releaseAll(); } catch (error) { errors.push(error); }
      try { await program.close(); } catch (error) { errors.push(error); }
      if (errors.length === 1) throw errors[0];
      if (errors.length) throw new AggregateError(errors, 'Capsule session cleanup failed.', { cause: errors[0] });
    },
  };
}

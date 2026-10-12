import { GENERATION_CONTRACT, GenerationError, resolveGenerationOptions, validateGenerationInput } from '../../config/generation-contract.js';
import { sampleCapsuleLogits, stoppingReason } from '../../inference/generation-step.js';
import { samplingSeedForToken } from '../../inference/token-sampling.js';
import { canonicalGenerationIdentity } from '../../config/canonical-generation-policy.js';
import { contractAssert as assert, exactRecord } from '../../config/canonical-contract-codec.js';
import { admitResidentPrompt } from '../../inference/resident-state-contract.js';
import { createResidentOutputState } from '../../inference/resident-output-state.js';
import { createResidentTargetState } from '../../inference/resident-target-state.js';
export { sampleCapsuleLogits } from '../../inference/generation-step.js';

export const requireGenerationOptions = resolveGenerationOptions;

export function createSessionController(commandExecutor, resourceBinder, program) {
  if (!commandExecutor || !resourceBinder || !program) {
    throw new Error('createSessionController requires commandExecutor, resourceBinder, and program.');
  }
  let closed = false;
  let active = false;

  return {
    // This is the ordinary numerical path. It never creates speculative state
    // or publishes a transition; advancing the iterator is the caller's barrier.
    async *generateCanonicalTransitions(targetPlan, request) {
      if (closed) throw new Error('Capsule runtime session is closed.');
      if (active) throw new Error('Capsule generation is already active.');
      const { signal, modules, ...wire } = request;
      exactRecord(wire, ['schema', 'promptTokens', 'policy', 'executionIdentity']);
      assert(wire.schema === 'doppler.canonical-generation-request/v1', 'Unsupported canonical generation request.');
      signal?.throwIfAborted();
      assert(targetPlan.tokenSelection === undefined, 'Canonical CPU sampling cannot use a GPU token-selection plan.');
      const { vocabSize, maxSeqLen } = program.getGenerationGeometry();
      const { policy, generationId } = canonicalGenerationIdentity(wire.policy, wire.executionIdentity, vocabSize);
      assert(policy.materializedInputCapacity <= maxSeqLen, 'Canonical capacity exceeds model geometry.');
      const admitted = admitResidentPrompt(wire.promptTokens, policy, vocabSize, generationId);
      if (!('kind' in admitted)) { yield admitted; return; }
      const promptTokens = [...admitted.tokenIds];
      const options = { maxSeqLen: policy.materializedInputCapacity, maxTokens: policy.maxGeneratedTokens,
        temperature: policy.temperature, topK: policy.topK, topP: policy.topP,
        repetitionPenalty: policy.repetitionPenalty, repetitionPenaltyWindow: policy.repetitionPenaltyWindow,
        presencePenalty: policy.presencePenalty, stopSequences: [...policy.stopSequences], useChatTemplate: false };
      const target = createResidentTargetState(promptTokens, policy, vocabSize, generationId,
        createResidentOutputState(program.createRawIncrementalDecoder(), program.createIncrementalDecoder(), policy, vocabSize));
      let stepResult = null;
      active = true;
      try {
        program.reset();
        resourceBinder.bindSlots(targetPlan.memoryLayout,
          { seqLen: promptTokens.length, maxSeqLen: policy.materializedInputCapacity, batchSize: 1 });
        resourceBinder.writeSlot('input_tokens', Uint32Array.from(promptTokens));
        let phase = await commandExecutor.executePhase('prefill', targetPlan.phases.prefill, {
          signal, modules, context: { prompt: '', promptTokens, generationOptions: options },
        });
        stepResult = phase.results.at(-1);
        while (true) {
          signal?.throwIfAborted();
          resourceBinder.assertDeviceAvailable();
          const parent = target.getCursor();
          const result = parent ? target.advance(parent.cursorId, stepResult?.logits) : target.bootstrap(stepResult?.logits);
          program.releaseStepResult(stepResult); stepResult = null;
          yield result;
          if (closed) throw new Error('Capsule runtime session is closed.');
          signal?.throwIfAborted();
          if (result.successor.terminal) return;
          phase = await commandExecutor.executePhase('decode', targetPlan.phases.decode, {
            signal, modules, context: { contextTokens: [...target.getHistory(), result.successor.pendingTokenId],
              generationOptions: options },
          });
          stepResult = phase.results.at(-1);
        }
      } finally {
        try { program.releaseStepResult(stepResult); }
        finally {
          try { resourceBinder.releaseTransient(); }
          finally { active = false; }
        }
      }
    },

    async *generateTokens(targetPlan, request = {}, { incremental = false } = {}) {
      if (closed) throw new Error('Capsule runtime session is closed.');
      if (active) throw new Error('Capsule generation is already active.');
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
      active = true;
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
            ? sampleCapsuleLogits(stepResult?.logits, contextTokens,
              { ...sampling, seed: samplingSeedForToken(sampling.seed, step) }, tokenContract)
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
            signal, modules, context: { contextTokens,
              generationOptions: { ...options, seed: samplingSeedForToken(sampling.seed, step + 1) } },
          });
          stepResult = decode.results.at(-1);
        }
      } finally {
        try { program.releaseStepResult(stepResult); }
        finally {
          try { resourceBinder.releaseTransient(); }
          finally { active = false; }
        }
      }
    },

    async close() {
      if (closed) return;
      if (active) throw new Error('Finish or cancel active generation before closing its runtime session.');
      closed = true;
      const errors = [];
      try { resourceBinder.releaseAll(); } catch (error) { errors.push(error); }
      try { await program.close(); } catch (error) { errors.push(error); }
      if (errors.length === 1) throw errors[0];
      if (errors.length) throw new AggregateError(errors, 'Capsule session cleanup failed.', { cause: errors[0] });
    },
  };
}

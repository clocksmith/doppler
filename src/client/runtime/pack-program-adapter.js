import { releaseBuffer } from '../../memory/buffer-pool.js';
import { observeInitialExecutionIdentity } from '../../config/initial-execution-identity.js';

function arraysEqual(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function toPipelineOptions(options, signal) {
  return {
    maxTokens: options.maxTokens,
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    repetitionPenalty: options.repetitionPenalty,
    repetitionPenaltyWindow: options.repetitionPenaltyWindow,
    ...(Number.isFinite(options.seed) ? { seed: options.seed } : {}),
    suppressSpecialTokens: options.suppressSpecialTokens,
    suppressSpecialLikeTokens: options.suppressSpecialLikeTokens,
    suppressTokenIds: options.suppressTokenIds,
    stopSequences: options.stopSequences,
    useChatTemplate: false,
    signal,
  };
}

export function createPackProgramAdapter(modelHandle, pack, targetPlan) {
  if (!modelHandle?.advanced) throw new Error('Pack program adapter requires a loaded Doppler model handle.');
  if (modelHandle.manifest?.modelId !== pack.modelId) throw new Error('Loaded program modelId does not match the Pack.');
  const declaredByPhase = Object.fromEntries(['prefill', 'decode'].map((phase) => [
    phase,
    targetPlan.phases[phase].flatMap((command) => command.declaredStepIds || []),
  ]));

  function assertNoPlanMutation() {
    const transitions = modelHandle.advanced.getStats()?.executionPlan?.transitions;
    if (Array.isArray(transitions) && transitions.length > 0) {
      throw new Error('Pack program attempted an undeclared execution-plan transition.');
    }
  }

  return {
    executionGraphHash: pack.program.executionGraphHash,

    getInitialExecutionIdentity() {
      return observeInitialExecutionIdentity(modelHandle.advanced.getResolvedRuntimeSession());
    },

    tokenize(prompt, options = {}) {
      return modelHandle.advanced.tokenizePrompt(prompt, options);
    },

    decodeTokens(tokenIds) {
      return modelHandle.advanced.decodeTokenIds(tokenIds);
    },

    getTokenContract() {
      const special = modelHandle.advanced.getSpecialTokens();
      return {
        padTokenId: Number.isInteger(special.pad) ? special.pad : null,
        eosTokenId: Number.isInteger(special.eos) ? special.eos : null,
        stopTokenIds: modelHandle.advanced.getStopTokenIds(),
      };
    },

    reset() {
      modelHandle.resetGenerationState();
    },

    async executePhase(phase, request) {
      if (!arraysEqual(request.declaredStepIds, declaredByPhase[phase])) {
        throw new Error(`Pack program phase "${phase}" command closure changed after qualification.`);
      }
      let result;
      if (phase === 'prefill') {
        const { prompt, promptTokens, generationOptions } = request.context;
        result = await modelHandle.advanced.prefillWithLogits(prompt, {
          ...toPipelineOptions(generationOptions, request.signal),
          inputIds: promptTokens,
        });
      } else if (phase === 'decode') {
        result = await modelHandle.advanced.decodeStepLogits(request.context.contextTokens, {
          ...toPipelineOptions(request.context.generationOptions, request.signal),
        });
      } else {
        throw new Error(`Pack program adapter does not implement phase "${phase}".`);
      }
      assertNoPlanMutation();
      return result;
    },

    releaseStepResult(result) {
      if (!result) return;
      if (result.logitsBuffer) releaseBuffer(result.logitsBuffer);
      result.cache?.destroy?.();
    },

    async close() {
      await modelHandle.unload();
    },
  };
}

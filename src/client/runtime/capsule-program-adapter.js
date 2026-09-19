import { GENERATION_CONTRACT } from '../../config/generation-contract.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { observeInitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import { validateCapsuleTokenSelection } from '../../config/capsule-token-selection.js';

function arraysEqual(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function toPipelineOptions(options, signal) {
  return {
    ...Object.fromEntries(Object.keys(GENERATION_CONTRACT.options)
      .filter(key => options[key] !== undefined).map(key => [key, options[key]])),
    // The prompt has already been prepared by the session controller.
    useChatTemplate: false,
    signal,
  };
}

export function createCapsuleProgramAdapter(modelHandle, capsule, targetPlan) {
  if (!modelHandle?.advanced) throw new Error('Capsule program adapter requires a loaded Doppler model handle.');
  if (modelHandle.manifest?.modelId !== capsule.modelId) throw new Error('Loaded program modelId does not match the Capsule.');
  const tokenSelection = targetPlan.tokenSelection === undefined ? null
    : validateCapsuleTokenSelection(targetPlan, capsule.wgslModules);
  if (tokenSelection && (typeof modelHandle.advanced.prefillWithToken !== 'function'
    || typeof modelHandle.advanced.decodeStepWithToken !== 'function')) {
    throw new Error('Loaded program does not implement the declared GPU token-selection recipe.');
  }
  const declaredByPhase = Object.fromEntries(['prefill', 'decode'].map((phase) => [
    phase,
    targetPlan.phases[phase].flatMap((command) => command.declaredStepIds || []),
  ]));

  function assertNoPlanMutation() {
    const transitions = modelHandle.advanced.getStats()?.executionPlan?.transitions;
    if (Array.isArray(transitions) && transitions.length > 0) {
      throw new Error('Capsule program attempted an undeclared execution-plan transition.');
    }
  }

  return {
    executionGraphHash: capsule.program.executionGraphHash,

    getActiveAdapterIdentity() { return modelHandle.activeLoRAIdentity; },

    async loadAdapter(manifest, { bytes, signal, weightsLayout }) {
      const read = async path => {
        signal.throwIfAborted();
        if (path !== manifest.weightsPath) throw new Error('Adapter loader requested an undeclared artifact.');
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      };
      if (weightsLayout === undefined) throw new Error('Capsule adapter requires its resolved weight layout.');
      await modelHandle.loadLoRA(manifest, { readFile: read, readOPFS: read, fetchUrl: read, skipVerify: false, weightsLayout });
    },

    async unloadAdapter() { await modelHandle.unloadLoRA(); },

    getInitialExecutionIdentity() {
      return observeInitialExecutionIdentity(modelHandle.advanced.getResolvedRuntimeSession());
    },

    tokenize(prompt, options = {}) {
      return modelHandle.advanced.tokenizePrompt(prompt, options);
    },

    createIncrementalDecoder() {
      return modelHandle.advanced.createIncrementalDecoder();
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

    async rerank(request) {
      if (typeof modelHandle.rerankWithEvidence !== 'function') {
        throw new Error('Loaded Doppler model handle does not implement rerank evidence.');
      }
      try {
        return await modelHandle.rerankWithEvidence(
          request.query,
          request.documents,
          request.options
        );
      } finally {
        assertNoPlanMutation();
      }
    },

    async embed(text, options) {
      if (modelHandle.supportsEmbedding !== true || typeof modelHandle.embedWithEvidence !== 'function') {
        throw new Error('Loaded Doppler Capsule does not declare text embedding execution.');
      }
      try {
        return await modelHandle.embedWithEvidence(text, options);
      } finally { assertNoPlanMutation(); }
    },

    async encodeSequence(sequence, options) {
      if (modelHandle.supportsSequence !== true || typeof modelHandle.encodeSequence !== 'function') {
        throw new Error('Loaded Doppler Capsule does not declare sequence execution.');
      }
      try {
        return await modelHandle.encodeSequence(sequence, options);
      } finally { assertNoPlanMutation(); }
    },

    async executePhase(phase, request) {
      if (!arraysEqual(request.declaredStepIds, declaredByPhase[phase])) {
        throw new Error(`Capsule program phase "${phase}" command closure changed after qualification.`);
      }
      let result;
      if (phase === 'prefill') {
        const { prompt, promptTokens, generationOptions } = request.context;
        const options = {
          ...toPipelineOptions(generationOptions, request.signal),
          inputIds: promptTokens,
        };
        result = tokenSelection
          ? await modelHandle.advanced.prefillWithToken(prompt, options, this.getTokenContract())
          : await modelHandle.advanced.prefillWithLogits(prompt, options);
      } else if (phase === 'decode') {
        const options = {
          ...toPipelineOptions(request.context.generationOptions, request.signal),
        };
        result = tokenSelection
          ? await modelHandle.advanced.decodeStepWithToken(request.context.contextTokens, options, this.getTokenContract())
          : await modelHandle.advanced.decodeStepLogits(request.context.contextTokens, options);
      } else {
        throw new Error(`Capsule program adapter does not implement phase "${phase}".`);
      }
      try { assertNoPlanMutation(); }
      catch (error) { this.releaseStepResult(result); throw error; }
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

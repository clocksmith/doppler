const SESSION_SCHEMA = 'doppler.scoped-session/v1';
const RESULT_SCHEMA = 'doppler.generation-result/v1';
const EVENT_SCHEMA = 'doppler.generation-event/v1';

const OBSERVATION_POLICIES = Object.freeze({
  always: Object.freeze({
    id: 'demo/always-on',
    tier: 'always',
    executionClassification: 'representative',
  }),
  'guided-quality': Object.freeze({
    id: 'demo/guided-quality',
    tier: 'guided-quality',
    executionClassification: 'observed',
  }),
  'deep-xray': Object.freeze({
    id: 'demo/deep-xray',
    tier: 'deep-xray',
    executionClassification: 'deep-diagnostic',
  }),
});

function resolveObservationPolicy(value = 'always') {
  const key = String(value || 'always').trim();
  const policy = OBSERVATION_POLICIES[key]
    || Object.values(OBSERVATION_POLICIES).find((entry) => entry.id === key);
  if (!policy) {
    throw new Error(
      `Unsupported Doppler observation policy "${key}". `
      + `Expected one of: ${Object.keys(OBSERVATION_POLICIES).join(', ')}.`
    );
  }
  return policy;
}

function capabilityMap(handle) {
  const specializedModel = handle?.supportsEmbedding === true || handle?.supportsSequence === true;
  return Object.freeze({
    generate: !specializedModel && typeof handle?.generateWithEvidence === 'function',
    stream: !specializedModel && typeof handle?.generate === 'function',
    embed: handle?.supportsEmbedding === true && typeof handle?.embed === 'function',
    sequence: handle?.supportsSequence === true && typeof handle?.encodeSequence === 'function',
    inspect: typeof handle?.inspect?.generate === 'function',
    lora: typeof handle?.loadLoRA === 'function' && typeof handle?.unloadLoRA === 'function',
    advanced: handle?.advanced && typeof handle.advanced === 'object',
  });
}

function buildObservation(policy, evidence, inspectionReceipt = null) {
  const statsAvailable = Boolean(evidence?.stats && typeof evidence.stats === 'object');
  return {
    policyId: policy.id,
    tier: policy.tier,
    executionClassification: policy.executionClassification,
    executionChanged: inspectionReceipt?.policy?.modifiesExecution === true,
    unavailable: policy.tier === 'guided-quality' && !inspectionReceipt?.quality
      ? ['token-surprisal', 'word-surprisal', 'rolling-perplexity']
      : [],
    deepEvidenceAvailable: policy.tier === 'deep-xray' ? statsAvailable : null,
  };
}

function buildGenerationResult(evidence, policy, inspectionReceipt = null) {
  const outputText = String(evidence?.outputText ?? '');
  const tokenIds = Array.from(evidence?.tokenIds || [], Number);
  return {
    schema: RESULT_SCHEMA,
    outputText,
    content: outputText,
    tokenIds,
    usage: {
      promptTokens: null,
      completionTokens: tokenIds.length,
      totalTokens: null,
    },
    observation: buildObservation(policy, evidence, inspectionReceipt),
    fingerprint: inspectionReceipt?.fingerprint || {
      modelId: evidence?.runtimeProfile?.model?.modelId ?? null,
      manifestHash: evidence?.runtimeProfile?.model?.manifestHash ?? null,
      tokenizerHash: evidence?.stats?.tokenizerHash ?? null,
      executionPlanId: evidence?.backendIdentity?.executionPlanId ?? null,
      kernelPathId: evidence?.backendIdentity?.kernelPathId ?? null,
      runtimeProfileHash: evidence?.runtimeProfileHash ?? null,
      backendIdentityHash: evidence?.backendIdentityHash ?? null,
      observationPolicyId: policy.id,
    },
    inspectionReceipt,
    evidence,
  };
}

export function createScopedModelSession(handle) {
  if (!handle || typeof handle !== 'object') {
    throw new Error('createScopedModelSession requires a Doppler model handle.');
  }
  let closed = false;
  const capabilities = capabilityMap(handle);

  function requireOpen() {
    if (closed) {
      throw new Error('Doppler scoped session is closed.');
    }
  }

  function supports(capability) {
    return capabilities[String(capability || '')] === true;
  }

  function requireCapability(capability) {
    requireOpen();
    if (!supports(capability)) {
      throw new Error(
        `Doppler model "${handle.modelId || 'unknown'}" does not support "${capability}".`
      );
    }
    return true;
  }

  const session = {
    schema: SESSION_SCHEMA,
    capabilities,
    supports,
    require: requireCapability,
    async generate(input, options = {}) {
      requireCapability('generate');
      const policy = resolveObservationPolicy(options.observe);
      const {
        observe: _observe,
        observationPolicy: _observationPolicy,
        ...generationOptions
      } = options;
      if (
        typeof input === 'string'
        && policy.tier === 'always'
        && typeof handle.inspect?.generate === 'function'
      ) {
        const inspectionReceipt = await handle.inspect.generate(input, {
          policyId: policy.id,
          generation: generationOptions,
        });
        return buildGenerationResult(
          inspectionReceipt.generationEvidence,
          policy,
          inspectionReceipt
        );
      }
      const evidence = await handle.generateWithEvidence(input, generationOptions);
      return buildGenerationResult(evidence, policy);
    },
    async *stream(input, options = {}) {
      requireCapability('stream');
      const policy = resolveObservationPolicy(options.observe);
      const {
        observe: _observe,
        observationPolicy: _observationPolicy,
        ...generationOptions
      } = options;
      let outputText = '';
      for await (const text of handle.generate(input, generationOptions)) {
        const delta = String(text ?? '');
        outputText += delta;
        yield {
          schema: EVENT_SCHEMA,
          type: 'text-delta',
          text: delta,
          observationPolicyId: policy.id,
        };
      }
      yield {
        schema: EVENT_SCHEMA,
        type: 'complete',
        outputText,
        observationPolicyId: policy.id,
      };
    },
    async inspect(input, options = {}) {
      requireCapability('inspect');
      const observationPolicy = options.observationPolicy ?? options.observe ?? 'guided-quality';
      const policy = resolveObservationPolicy(observationPolicy);
      if (typeof handle.inspect?.generate !== 'function') {
        throw new Error('Loaded Doppler handle does not expose semantic inspection.');
      }
      const {
        observe: _observe,
        observationPolicy: _observationPolicy,
        topKSize,
        onEvent,
        ...generation
      } = options;
      return handle.inspect.generate(input, {
        policyId: policy.id,
        generation,
        topKSize,
        onEvent,
      });
    },
    async embed(input, options = {}) {
      requireCapability('embed');
      return handle.embed(input, options);
    },
    async encodeSequence(sequence, options = {}) {
      requireCapability('sequence');
      return handle.encodeSequence(sequence, options);
    },
    async loadLoRA(adapter, options = {}) {
      requireCapability('lora');
      return handle.loadLoRA(adapter, options);
    },
    async unloadLoRA() {
      requireCapability('lora');
      return handle.unloadLoRA();
    },
    resetGenerationState() {
      requireOpen();
      return handle.resetGenerationState();
    },
    async close() {
      if (closed) return;
      closed = true;
      await handle.unload();
    },
    async [Symbol.asyncDispose]() {
      await session.close();
    },
    get closed() {
      return closed;
    },
    get loaded() {
      return !closed && handle.loaded === true;
    },
    get modelId() {
      return handle.modelId;
    },
    get manifestHash() {
      return handle.manifestHash ?? null;
    },
    get persistentCache() {
      return handle.persistentCache ?? null;
    },
    get activeLoRA() {
      return handle.activeLoRA ?? null;
    },
    get deviceInfo() {
      return handle.deviceInfo ?? null;
    },
    get advanced() {
      requireCapability('advanced');
      return handle.advanced;
    },
  };
  return session;
}

export { OBSERVATION_POLICIES };

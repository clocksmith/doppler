import { resolveSamplingConfig } from './sampling-config.js';

/** @type {WeakMap<object, import('./sampling-config.js').ResolvedSamplingConfig>} */
const resolvedRequests = new WeakMap();

/** @type {import('./generation-request.js').resolveChatTemplateSetting} */
export function resolveChatTemplateSetting(options, runtimeConfig, modelConfig) {
  const value = options.useChatTemplate === undefined
    ? runtimeConfig.inference.chatTemplate?.enabled ?? modelConfig?.chatTemplateEnabled ?? false
    : options.useChatTemplate;
  if (typeof value !== 'boolean') throw new Error('options.useChatTemplate must be a boolean.');
  return value;
}

/** @type {import('./generation-request.js').resolveTextGenerationRequest} */
export function resolveTextGenerationRequest(options, runtimeConfig, modelConfig) {
  if (resolvedRequests.has(options)) return /** @type {import('./generation-request.js').ResolvedTextGenerationRequest} */ (options);
  const generation = runtimeConfig?.inference?.generation;
  if (!generation) throw new Error('Loaded pipeline must expose resolved generation config.');
  const sampling = resolveSamplingConfig(options, runtimeConfig);
  const maxTokens = options.maxTokens === undefined ? generation.maxTokens : options.maxTokens;
  if (!Number.isSafeInteger(maxTokens) || maxTokens <= 0) throw new Error('Generation maxTokens must be a positive safe integer.');
  const stopSequences = options.stopSequences === undefined ? [] : options.stopSequences;
  if (!Array.isArray(stopSequences) || stopSequences.some(value => typeof value !== 'string')) {
    throw new Error('Generation stopSequences must be an array of strings.');
  }
  const useSpeculative = options.useSpeculative === undefined ? generation.useSpeculative : options.useSpeculative;
  if (useSpeculative !== undefined && typeof useSpeculative !== 'boolean') {
    throw new Error('Generation useSpeculative must be a boolean when supplied.');
  }
  if (options.seed !== undefined && (!Number.isFinite(options.seed) || options.seed < 0)) {
    throw new Error('Generation seed must be a non-negative finite number when supplied.');
  }
  const request = Object.freeze({
    ...options,
    ...sampling,
    maxTokens,
    seed: options.seed,
    useSpeculative,
    useChatTemplate: resolveChatTemplateSetting(options, runtimeConfig, modelConfig),
    suppressTokenIds: Object.freeze([...sampling.suppressTokenIds]),
    stopSequences: Object.freeze([...stopSequences]),
  });
  resolvedRequests.set(request, sampling);
  return request;
}

// Evidence projects the request actually passed to execution. It does not
// consult mutable defaults or resolve a second set of sampling rules.
/** @type {import('./generation-request.js').generationRequestEvidence} */
export function generationRequestEvidence(request) {
  const sampling = resolvedRequests.get(request);
  if (!sampling) throw new Error('Generation evidence requires a resolved request.');
  return Object.freeze({
    ...sampling,
    suppressTokenIds: request.suppressTokenIds,
    greedyThreshold: request.greedyThreshold,
    suppressSpecialTokens: request.suppressSpecialTokens,
    suppressSpecialLikeTokens: request.suppressSpecialLikeTokens,
    maxTokens: request.maxTokens,
    stopSequences: request.stopSequences,
    useChatTemplate: request.useChatTemplate,
    useSpeculative: request.useSpeculative ?? null,
    seed: request.seed ?? null,
  });
}

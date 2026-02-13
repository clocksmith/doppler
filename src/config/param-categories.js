export const ParamCategory = Object.freeze({
  GENERATION: 'generation',
  MODEL: 'model',
  SESSION: 'session',
  HYBRID: 'hybrid',
});

export const PARAM_CATEGORIES = Object.freeze({
  // Generation params
  temperature: ParamCategory.GENERATION,
  topK: ParamCategory.GENERATION,
  topP: ParamCategory.GENERATION,
  repetitionPenalty: ParamCategory.GENERATION,
  maxTokens: ParamCategory.GENERATION,
  stopSequences: ParamCategory.GENERATION,
  embeddingMode: ParamCategory.GENERATION,

  // Model params (manifest-primary, runtime-overridable)
  slidingWindow: ParamCategory.MODEL,
  attnLogitSoftcapping: ParamCategory.MODEL,
  queryPreAttnScalar: ParamCategory.MODEL,
  queryKeyNorm: ParamCategory.MODEL,
  causal: ParamCategory.MODEL,
  ropeTheta: ParamCategory.MODEL,
  ropeLocalTheta: ParamCategory.MODEL,
  ropeScalingType: ParamCategory.MODEL,
  ropeScalingFactor: ParamCategory.MODEL,
  rmsNormEps: ParamCategory.MODEL,
  rmsNormWeightOffset: ParamCategory.MODEL,
  postAttentionNorm: ParamCategory.MODEL,
  preFeedforwardNorm: ParamCategory.MODEL,
  postFeedforwardNorm: ParamCategory.MODEL,
  activation: ParamCategory.MODEL,
  gatedActivation: ParamCategory.MODEL,
  finalLogitSoftcapping: ParamCategory.MODEL,
  tieWordEmbeddings: ParamCategory.MODEL,
  scaleEmbeddings: ParamCategory.MODEL,

  // Session params
  activationDtype: ParamCategory.SESSION,
  kvDtype: ParamCategory.SESSION,
  batchSize: ParamCategory.SESSION,
  readbackInterval: ParamCategory.SESSION,
  ringTokens: ParamCategory.SESSION,
  ringStop: ParamCategory.SESSION,
  ringStaging: ParamCategory.SESSION,
  logLevel: ParamCategory.SESSION,

  // Hybrid params
  useChatTemplate: ParamCategory.HYBRID,
  kernelPath: ParamCategory.HYBRID,
});

export const CategoryRules = Object.freeze({
  [ParamCategory.GENERATION]: { callTime: true, runtime: true, manifest: false },
  [ParamCategory.MODEL]: { callTime: false, runtime: true, manifest: true },
  [ParamCategory.SESSION]: { callTime: false, runtime: true, manifest: false },
  [ParamCategory.HYBRID]: { callTime: true, runtime: true, manifest: true },
});

export function getParamCategory(name) {
  return PARAM_CATEGORIES[name] ?? null;
}

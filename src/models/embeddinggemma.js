// Static metadata for the EmbeddingGemma family of RDRR artifacts.

export const FAMILY_ID = 'embeddinggemma';
export const HF_REPO_ID = 'Clocksmith/rdrr';

export const KNOWN_MODELS = Object.freeze([
  Object.freeze({
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    label: 'EmbeddingGemma 300M (Q4K)',
    sourceModel: 'google/embeddinggemma-300m',
    hfPath: 'models/google-embeddinggemma-300m-q4k-ehf16-af32',
    defaultRuntimeProfile: 'profiles/throughput',
    modes: Object.freeze(['embedding']),
  }),
]);

export function resolveModel(modelId) {
  return KNOWN_MODELS.find((m) => m.modelId === modelId) || null;
}

export function resolveHfBaseUrl(modelId, revision = 'main') {
  const entry = resolveModel(modelId);
  if (!entry) return null;
  return `https://huggingface.co/${HF_REPO_ID}/resolve/${revision}/${entry.hfPath}`;
}

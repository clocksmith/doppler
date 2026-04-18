// Static metadata for the Gemma 3 family of RDRR artifacts.
// See qwen3.js for the contract.

export const FAMILY_ID = 'gemma3';
export const HF_REPO_ID = 'Clocksmith/rdrr';

export const KNOWN_MODELS = Object.freeze([
  Object.freeze({
    modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    label: 'Gemma 3 270M Instruct (Q4K/F32a)',
    sourceModel: 'google/gemma-3-270m-it',
    hfPath: 'models/gemma-3-270m-it-q4k-ehf16-af32',
    defaultRuntimeProfile: 'profiles/throughput',
    modes: Object.freeze(['text', 'vision']),
  }),
  Object.freeze({
    modelId: 'gemma-3-1b-it-q4k-ehf16-af32',
    label: 'Gemma 3 1B Instruct (Q4K/F32a)',
    sourceModel: 'google/gemma-3-1b-it',
    hfPath: 'models/gemma-3-1b-it-q4k-ehf16-af32',
    defaultRuntimeProfile: 'profiles/throughput',
    modes: Object.freeze(['text', 'vision']),
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

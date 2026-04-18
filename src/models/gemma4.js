// Static metadata for the Gemma 4 family of RDRR artifacts.
// See qwen3.js for the contract.

export const FAMILY_ID = 'gemma4';
export const HF_REPO_ID = 'Clocksmith/rdrr';

export const KNOWN_MODELS = Object.freeze([
  Object.freeze({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    label: 'Gemma 4 E2B Instruct (Q4K/F32a)',
    sourceModel: 'google/gemma-4-e2b-it',
    hfPath: 'models/gemma-4-e2b-it-q4k-ehf16-af32',
    defaultRuntimeProfile: 'profiles/gemma4-e2b-throughput',
    modes: Object.freeze(['text', 'vision']),
  }),
  Object.freeze({
    modelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    label: 'Gemma 4 E2B Instruct (Q4K/F32a/INT4 PLE)',
    sourceModel: 'google/gemma-4-e2b-it',
    hfPath: 'models/gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    defaultRuntimeProfile: 'profiles/gemma4-e2b-throughput',
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

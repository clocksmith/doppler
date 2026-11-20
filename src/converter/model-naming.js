// Model names (docs/style/model-naming.md): lowercase letters, digits and single hyphens, ending with the
// suffix the model's format generates. A manifest carries its model ID and a model's identity is the hash of
// its manifest, so a name is checked before it is written.

// The suffix of a model ID, from the variant tag a manifest stores. The stored tag is manifest data: it keeps
// its spelling, because rewriting it would change the identity of every model converted again.
/** @type {typeof import('./model-naming.js').variantNameSuffix} */
export function variantNameSuffix(variantTag) {
  if (!variantTag) return '';
  return String(variantTag).toLowerCase()
    .replace(/q(\d)_(\d)/g, 'q$1$2')
    .replace(/[^a-z0-9-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '');
}

export const MODEL_ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

// Identities issued before the naming rule. A manifest carries its model ID, so renaming one of these is a
// new conversion with a new identity, not an edit. Remove an entry when its model is reissued.
/** @type {readonly string[]} */
export const LEGACY_NONCANONICAL_MODEL_IDS = Object.freeze([
  'qwen-3-6-35b-a3b-text-source-q4_0-af32',
  'qwen3-8-27b-text-source-q4_0-af32',
]);

// Identities issued before the naming rule whose suffix was written by hand and does not match the tag
// their own settings generate (for example "ehaf16" over F32 activations). Same rule as the list above:
// nothing new may be added, and an entry is removed when its model is reissued.
/** @type {readonly string[]} */
export const LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS = Object.freeze([
  'diffusiongemma-26b-a4b-it-q4k-ehf16-af16',
  'esm2-t12-35m-ur50d-f32-af32',
  'esmc-300m-f32-af32',
  'gemma-3-270m-it-q4k-ehf16-af16',
  'gemma-4-12b-it-text-q4k-ehf16-af16',
  'gemma-4-12b-it-text-q4k-ehf16-hq4k-af16',
  'gemma-4-12b-it-text-w4a16-ct-ehf16-af16',
  'gemma-4-31b-it-text-q4k-ehf16-af16',
  'gemma-4-e2b-it-q4k-ehf16-af16-int4ple',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
  'gemma-4-e2b-it-q4k-ehf16-af32',
  'gpt-oss-20b-f16-xmxfp4',
  'nucleotide-transformer-v2-50m-f32-af32',
  'qwen-3-5-0-8b-q4k-ehaf16',
  'qwen-3-5-2b-q4k-ehaf16',
  'qwen-3-5-9b-q4k-ehaf16',
  'qwen-3-6-27b-q4k-ehaf16',
  'qwen-3-reranker-0-6b-f16-true-logit-af32',
  'qwen3-8-27b-text-q4k-ehaf16',
]);

/** @type {typeof import('./model-naming.js').isCanonicalModelId} */
export function isCanonicalModelId(modelId) {
  return typeof modelId === 'string' && MODEL_ID_PATTERN.test(modelId);
}

/** @type {typeof import('./model-naming.js').assertCanonicalModelId} */
export function assertCanonicalModelId(modelId, source = 'model ID') {
  if (isCanonicalModelId(modelId) || LEGACY_NONCANONICAL_MODEL_IDS.includes(modelId)) return modelId;
  throw new Error(`${source} "${modelId}" is not a valid model name: use lowercase letters, digits and single hyphens only (docs/style/model-naming.md).`);
}

// A name must end with the suffix its own format generates, after at least one other part. Returns what is
// wrong, or null.
/** @type {typeof import('./model-naming.js').modelIdSuffixProblem} */
export function modelIdSuffixProblem(modelId, variantTag) {
  const suffix = variantNameSuffix(variantTag);
  if (!suffix) return 'its format is unknown, so its suffix cannot be checked';
  if (typeof modelId === 'string' && modelId.length > suffix.length + 1 && modelId.endsWith(`-${suffix}`)) return null;
  return `it must end with "-${suffix}", the suffix its format generates`;
}

/** @type {typeof import('./model-naming.js').isLegacyModelId} */
export function isLegacyModelId(modelId) {
  return typeof modelId === 'string'
    && (LEGACY_NONCANONICAL_MODEL_IDS.includes(modelId) || LEGACY_HANDWRITTEN_SUFFIX_MODEL_IDS.includes(modelId));
}

/** @type {typeof import('./model-naming.js').assertModelIdDescribesFormat} */
export function assertModelIdDescribesFormat(modelId, variantTag, source = 'model ID') {
  if (isLegacyModelId(modelId)) return modelId;
  const problem = modelIdSuffixProblem(modelId, variantTag);
  if (problem === null) return modelId;
  throw new Error(`${source} "${modelId}" does not describe its model: ${problem} (docs/style/model-naming.md).`);
}

// A label for a model run straight from a file, taken from the file's name. Nobody chose it, so it is brought
// into line (every other character becomes a hyphen) instead of refused.
/** @type {typeof import('./model-naming.js').modelLabelFromFileName} */
export function modelLabelFromFileName(name) {
  const label = String(name ?? '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 64).replace(/-$/, '');
  return label || null;
}

const TENSOR_FORMAT_TOKENS = Object.freeze({ q4_k_m: 'q4k', q4_k: 'q4k', q5_k: 'q5k', q5_k_m: 'q5k', q6_k: 'q6k' });
/** @param {unknown} dtype */
const tensorFormatToken = dtype => {
  const lower = String(dtype ?? '').toLowerCase();
  return TENSOR_FORMAT_TOKENS[/** @type {keyof typeof TENSOR_FORMAT_TOKENS} */ (lower)] ?? lower;
};

// The suffix is generated from quantizationInfo, so quantizationInfo has to be true. This compares what it says
// about the embeddings and the output head with the stored type of the tensors that have those roles. A role
// whose tensors do not share one type is not judged.
/** @type {typeof import('./model-naming.js').formatRecordProblems} */
export function formatRecordProblems(quantizationInfo, tensors) {
  const problems = [];
  const entries = Array.isArray(tensors) ? tensors : Object.values(tensors ?? {});
  for (const [role, field] of /** @type {const} */ ([['embedding', 'embeddings'], ['lm_head', 'lmHead']])) {
    const stored = new Set(entries.filter(tensor => tensor?.role === role).map(tensor => tensorFormatToken(tensor.dtype)));
    if (stored.size !== 1) continue;
    const [actual] = stored;
    const declared = tensorFormatToken(quantizationInfo?.[field] ?? (field === 'lmHead' ? quantizationInfo?.embeddings : null) ?? quantizationInfo?.weights);
    if (declared !== actual) problems.push(`quantizationInfo.${field} says ${declared} but the ${role} tensor is stored as ${actual}`);
  }
  return problems;
}

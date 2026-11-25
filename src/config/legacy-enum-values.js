// Legacy snake_case spellings of Doppler-owned enum values, mapped to their
// kebab-case replacements. Published manifests, runtime configs and command
// requests written before the kebab-case migration still carry the old
// spellings; readers canonicalize them through this table at load time.
// Writers emit only the kebab-case form. Do not add new entries: new values
// are kebab-case from the start (tools/check-kebab-enum-values.js).
export const LEGACY_ENUM_VALUE_ALIASES = Object.freeze({
  // per-layer-input-materialization
  range_backed: 'range-backed',
  cpu_resident: 'cpu-resident',
  gpu_resident: 'gpu-resident',
  gpu_split_tables: 'gpu-split-tables',
  // per-layer-input-prefetch-mode
  next_token: 'next-token',
  // per-layer-input-gpu-upload-mode
  per_step_slices: 'per-step-slices',
  per_batch_slices: 'per-batch-slices',
  // per-layer-input-hot-cache-mode
  prepared_tokens: 'prepared-tokens',
  tokenizer_scores: 'tokenizer-scores',
  // dtype-transition-policy
  require_cast_step: 'require-cast-step',
  // dtype-transition-declared-by
  step_precision: 'step-precision',
  explicit_cast_step: 'explicit-cast-step',
  // layer-pattern-type
  every_n: 'every-n',
  all_attention: 'all-attention',
  alternating_even: 'alternating-even',
  alternating_odd: 'alternating-odd',
  // ffn-branch-mode
  dense_plus_moe: 'dense-plus-moe',
  // kv-layout
  contiguous_quantized: 'contiguous-quantized',
  bdpa_paged: 'bdpa-paged',
  // kv-quant-mode
  turboquant_prod: 'turboquant-prod',
  turboquant_outlier: 'turboquant-outlier',
  // kv-tiering-gating-mode
  force_on: 'force-on',
  force_off: 'force-off',
  // choice-scoring-format
  qwen3_yes_no_logit: 'qwen3-yes-no-logit',
  // choice-score-policy
  logit_difference: 'logit-difference',
  true_logit: 'true-logit',
  // int4-storage-encoding
  offset_binary: 'offset-binary',
  // int4-scale-semantics
  qmax_abs: 'qmax-abs',
  // per-layer-embedding-format
  int4_per_row: 'int4-per-row',
  // lowering-exactness-class
  bit_exact_solo: 'bit-exact-solo',
  algorithm_exact: 'algorithm-exact',
  tolerance_bounded: 'tolerance-bounded',
  // lowering-algorithm-exact-invariant
  reduction_order: 'reduction-order',
  tree_shape: 'tree-shape',
  accum_dtype: 'accum-dtype',
  associativity_grouping: 'associativity-grouping',
  // sequence-alphabet
  amino_acid: 'amino-acid',
  // tooling-diagnostics-mode
  on_failure: 'on-failure',
  // training-stage
  stage1_joint: 'stage1-joint',
  stage2_base: 'stage2-base',
  stage_a: 'stage-a',
  stage_b: 'stage-b',
  // distill-student-graph-mode
  projection_head: 'projection-head',
  transformer_full: 'transformer-full',
  // training-objective
  cross_entropy: 'cross-entropy',
  causal_lm_cross_entropy: 'causal-lm-cross-entropy',
  ul_stage1_joint: 'ul-stage1-joint',
  ul_stage2_base: 'ul-stage2-base',
  // bench-compare-lane
  performance_comparable: 'performance-comparable',
  capability_only: 'capability-only',
  // bench-prefill-semantics
  internal_prefill_phase: 'internal-prefill-phase',
  ttft_proxy: 'ttft-proxy',
  // bench-prefill-token-semantics
  model_input_tokens: 'model-input-tokens',
  // bench-runner-kind
  pipeline_text_generation: 'pipeline-text-generation',
  qwen3_5_conditional_generation: 'qwen3-5-conditional-generation',
  causal_lm_yes_no_logit: 'causal-lm-yes-no-logit',
  // decode-mode
  single_token: 'single-token',
  batched_gpu: 'batched-gpu',
  batched_gpu_stepwise_ple: 'batched-gpu-stepwise-ple',
  self_speculation: 'self-speculation',
  // decode-batch-guard-reason
  multi_token_decode_disabled: 'multi-token-decode-disabled',
  // decode-strategy
  replay_prefill: 'replay-prefill',
  // execution-plan-id
  finiteness_fallback: 'finiteness-fallback',
  // precision-replay-mode
  f32_forward: 'f32-forward',
  f32_reverse: 'f32-reverse',
  f32_tree64: 'f32-tree64',
  f16_forward: 'f16-forward',
  // model-support-status
  partially_verified: 'partially-verified',
  performance_evidence: 'performance-evidence',
  not_implemented: 'not-implemented',
  // sequence-qualification-abort
  before_execution: 'before-execution',
  after_start: 'after-start',
  // operator-event-type
  sequence_mismatch: 'sequence-mismatch',
  drift_check_needed: 'drift-check-needed',
  length_mismatch: 'length-mismatch',
  // translation-artifact-lane
  selective_f16: 'selective-f16',
  // structured-head-model-type
  structured_json_head: 'structured-json-head',
  dream_structured: 'dream-structured',
  dream_intent_posterior_head: 'dream-intent-posterior-head',
  dream_d1_to2_bridge: 'dream-d1-to2-bridge',
  dream_synthesis: 'dream-synthesis',
  dream_energy_compose: 'dream-energy-compose',
  energy_row_head: 'energy-row-head',
  dream_energy_head: 'dream-energy-head',
  // trainer-artifact-format (shared with Gamma handoffs)
  huggingface_safetensors: 'huggingface-safetensors',
  // provider-failure-class
  gpu_device_lost: 'gpu-device-lost',
  gpu_oom: 'gpu-oom',
  gpu_timeout: 'gpu-timeout',
  gpu_unsupported: 'gpu-unsupported',
  gpu_unavailable: 'gpu-unavailable',
  model_load_failed: 'model-load-failed',
  policy_denied: 'policy-denied',
  runtime_internal: 'runtime-internal',
  fallback_failed: 'fallback-failed',
  // diffusion-scheduler-type
  euler_a: 'euler-a',
  dpmpp_2m: 'dpmpp-2m',
  flowmatch_euler: 'flowmatch-euler',
});

// Object keys whose string values (or string-array members) belong to the
// vocabularies above. Canonicalization only rewrites values under these keys.
export const LEGACY_ENUM_VALUE_FIELDS = Object.freeze(new Set([
  'materialization',
  'mode',
  'dtypeTransition',
  'transitionDeclaredBy',
  'type',
  'globalPattern',
  'patternType',
  'patternKind',
  'branchMode',
  'ffnBranchMode',
  'layout',
  'kvLayout',
  'tieringMode',
  'coldQuantMode',
  'quantMode',
  'contiguousQuantMode',
  'format',
  'score',
  'storageEncoding',
  'scaleSemantics',
  'fixedInt4StorageEncoding',
  'perLayerEmbeddings',
  'class',
  'algorithmExactInvariants',
  'alphabet',
  'sequenceAlphabet',
  'diagnostics',
  'receiptPolicy',
  'refactorReceipt',
  'trainingStage',
  'stage',
  'decodeMode',
  'decodeStrategy',
  'compareLane',
  'lane',
  'declared',
  'laneId',
  'modelType',
  'failureClass',
  'fallbackOn',
]));

/**
 * @param {object} object
 * @param {string} key
 * @returns {boolean}
 */
const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object, key);

/**
 * @param {unknown} value
 * @returns {unknown}
 */
export function canonicalizeLegacyEnumValue(value) {
  if (typeof value !== 'string' || !hasOwn(LEGACY_ENUM_VALUE_ALIASES, value)) {
    return value;
  }
  return /** @type {Record<string, string>} */ (LEGACY_ENUM_VALUE_ALIASES)[value];
}

/**
 * @param {unknown} value
 * @returns {unknown}
 */
function canonicalizeFieldValue(value) {
  if (typeof value === 'string') return canonicalizeLegacyEnumValue(value);
  if (Array.isArray(value)) {
    /** @type {unknown[]} */
    const next = value.map((entry) => (
      typeof entry === 'string' ? canonicalizeLegacyEnumValue(entry) : canonicalizeLegacyEnumValues(entry)
    ));
    return next.some((entry, index) => entry !== value[index]) ? next : value;
  }
  return canonicalizeLegacyEnumValues(value);
}

// Returns a structurally shared copy of `value` in which every legacy enum
// spelling under a known enum field is replaced by its kebab-case form.
// Unknown values are returned unchanged so the owning validator still rejects them.
/**
 * @param {unknown} value
 * @returns {unknown}
 */
export function canonicalizeLegacyEnumValues(value) {
  if (Array.isArray(value)) {
    /** @type {unknown[]} */
    const next = value.map((entry) => canonicalizeLegacyEnumValues(entry));
    return next.some((entry, index) => entry !== value[index]) ? next : value;
  }
  if (!value || typeof value !== 'object') return value;
  /** @type {Record<string, unknown> | null} */
  let next = null;
  for (const [key, entry] of Object.entries(value)) {
    /** @type {unknown} */
    const canonical = LEGACY_ENUM_VALUE_FIELDS.has(key)
      ? canonicalizeFieldValue(entry)
      : canonicalizeLegacyEnumValues(entry);
    if (canonical !== entry) {
      next ??= { ...value };
      next[key] = canonical;
    }
  }
  return next ?? value;
}

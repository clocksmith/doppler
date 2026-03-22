
// Canonical semantic stage namespace for operator diffing.
//
// Stage names are dot-notation templates that combine with context
// (layerIdx, phase) to produce deterministic operator IDs.
// These names survive across execution paths, backends, and devices.
//
// Existing probe names (flat, underscore-separated) map to these
// canonical names via PROBE_TO_CANONICAL. The probe system continues
// to use its own names; the operator identity system maps them.

// ============================================================================
// Operator Classes
// ============================================================================

export const OPERATOR_CLASSES = Object.freeze({
  EMBEDDING: 'embedding',
  NORMALIZATION: 'normalization',
  PROJECTION: 'projection',
  ROPE: 'rope',
  ATTENTION: 'attention',
  ACTIVATION: 'activation',
  FFN: 'ffn',
  RESIDUAL: 'residual',
  CAST: 'cast',
  QUANTIZE: 'quantize',
  LOGITS: 'logits',
  CONV: 'conv',
  MOE_ROUTING: 'moe_routing',
});

// ============================================================================
// Canonical Stage Templates
// ============================================================================

export const STAGES = Object.freeze({
  // Embedding
  EMBED_OUT: 'embed.out',

  // Attention stages (per-layer, prefixed with attn.)
  ATTN_INPUT: 'attn.input',
  ATTN_POST_INPUT_NORM: 'attn.post_input_norm',
  ATTN_NORMED: 'attn.normed',
  ATTN_QKV_PROJ: 'attn.qkv_proj',
  ATTN_Q_PROJ: 'attn.q_proj',
  ATTN_K_PROJ: 'attn.k_proj',
  ATTN_V_PROJ: 'attn.v_proj',
  ATTN_Q_NORM: 'attn.q_norm',
  ATTN_K_NORM: 'attn.k_norm',
  ATTN_Q_ROPE: 'attn.q_rope',
  ATTN_K_ROPE: 'attn.k_rope',
  ATTN_SCORES: 'attn.scores',
  ATTN_SOFTMAX: 'attn.softmax',
  ATTN_OUT: 'attn.out',
  ATTN_O_PROJ: 'attn.o_proj',
  ATTN_POST_ATTN: 'attn.post_attn',

  // Linear attention stages
  ATTN_LINEAR_Z_PROJ: 'attn.linear_z_proj',
  ATTN_LINEAR_A_PROJ: 'attn.linear_a_proj',
  ATTN_LINEAR_B_PROJ: 'attn.linear_b_proj',
  ATTN_LINEAR_CORE_OUT: 'attn.linear_core_out',

  // Conv stages (per-layer)
  CONV_INPUT_NORM: 'conv.input_norm',
  CONV_IN_PROJ: 'conv.in_proj',
  CONV_KERNEL: 'conv.kernel',
  CONV_OUT_PROJ: 'conv.out_proj',

  // FFN stages (per-layer)
  FFN_NORMED: 'ffn.normed',
  FFN_IN: 'ffn.in',
  FFN_GATE: 'ffn.gate',
  FFN_UP: 'ffn.up',
  FFN_ACT: 'ffn.act',
  FFN_OUT: 'ffn.out',
  LAYER_OUT: 'layer.out',

  // MoE stages (per-layer)
  MOE_ROUTING: 'moe.routing',
  MOE_EXPERT: 'moe.expert',

  // Residual
  RESIDUAL_ADD: 'residual.add',

  // Cast
  CAST: 'cast',

  // Final stages (not per-layer)
  PRE_FINAL_NORM: 'final_norm.pre',
  FINAL_NORM_OUT: 'final_norm.out',
  LOGITS_OUT: 'logits.out',
  LOGITS_FINAL: 'logits.final',
});

// ============================================================================
// Stage → Operator Class Mapping
// ============================================================================

const STAGE_TO_CLASS_MAP = Object.freeze({
  [STAGES.EMBED_OUT]: OPERATOR_CLASSES.EMBEDDING,

  [STAGES.ATTN_INPUT]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.ATTN_POST_INPUT_NORM]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.ATTN_NORMED]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.ATTN_QKV_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_Q_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_K_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_V_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_Q_NORM]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.ATTN_K_NORM]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.ATTN_Q_ROPE]: OPERATOR_CLASSES.ROPE,
  [STAGES.ATTN_K_ROPE]: OPERATOR_CLASSES.ROPE,
  [STAGES.ATTN_SCORES]: OPERATOR_CLASSES.ATTENTION,
  [STAGES.ATTN_SOFTMAX]: OPERATOR_CLASSES.ATTENTION,
  [STAGES.ATTN_OUT]: OPERATOR_CLASSES.ATTENTION,
  [STAGES.ATTN_O_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_POST_ATTN]: OPERATOR_CLASSES.RESIDUAL,

  [STAGES.ATTN_LINEAR_Z_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_LINEAR_A_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_LINEAR_B_PROJ]: OPERATOR_CLASSES.PROJECTION,
  [STAGES.ATTN_LINEAR_CORE_OUT]: OPERATOR_CLASSES.ATTENTION,

  [STAGES.CONV_INPUT_NORM]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.CONV_IN_PROJ]: OPERATOR_CLASSES.CONV,
  [STAGES.CONV_KERNEL]: OPERATOR_CLASSES.CONV,
  [STAGES.CONV_OUT_PROJ]: OPERATOR_CLASSES.CONV,

  [STAGES.FFN_NORMED]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.FFN_IN]: OPERATOR_CLASSES.FFN,
  [STAGES.FFN_GATE]: OPERATOR_CLASSES.FFN,
  [STAGES.FFN_UP]: OPERATOR_CLASSES.FFN,
  [STAGES.FFN_ACT]: OPERATOR_CLASSES.ACTIVATION,
  [STAGES.FFN_OUT]: OPERATOR_CLASSES.FFN,
  [STAGES.LAYER_OUT]: OPERATOR_CLASSES.RESIDUAL,

  [STAGES.MOE_ROUTING]: OPERATOR_CLASSES.MOE_ROUTING,
  [STAGES.MOE_EXPERT]: OPERATOR_CLASSES.FFN,

  [STAGES.RESIDUAL_ADD]: OPERATOR_CLASSES.RESIDUAL,
  [STAGES.CAST]: OPERATOR_CLASSES.CAST,

  [STAGES.PRE_FINAL_NORM]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.FINAL_NORM_OUT]: OPERATOR_CLASSES.NORMALIZATION,
  [STAGES.LOGITS_OUT]: OPERATOR_CLASSES.LOGITS,
  [STAGES.LOGITS_FINAL]: OPERATOR_CLASSES.LOGITS,
});

// ============================================================================
// Probe Name → Canonical Stage Mapping
// ============================================================================

export const PROBE_TO_CANONICAL = Object.freeze({
  embed_out: STAGES.EMBED_OUT,
  attn_input: STAGES.ATTN_INPUT,
  post_input_norm: STAGES.ATTN_POST_INPUT_NORM,
  attn_normed: STAGES.ATTN_NORMED,
  linear_qkv_proj: STAGES.ATTN_QKV_PROJ,
  linear_z_proj: STAGES.ATTN_LINEAR_Z_PROJ,
  linear_a_proj: STAGES.ATTN_LINEAR_A_PROJ,
  linear_b_proj: STAGES.ATTN_LINEAR_B_PROJ,
  linear_core_out: STAGES.ATTN_LINEAR_CORE_OUT,
  q_proj: STAGES.ATTN_Q_PROJ,
  k_proj: STAGES.ATTN_K_PROJ,
  v_proj: STAGES.ATTN_V_PROJ,
  q_rope: STAGES.ATTN_Q_ROPE,
  k_rope: STAGES.ATTN_K_ROPE,
  attn_scores: STAGES.ATTN_SCORES,
  attn_out: STAGES.ATTN_OUT,
  o_proj: STAGES.ATTN_O_PROJ,
  post_attn: STAGES.ATTN_POST_ATTN,
  ffn_normed: STAGES.FFN_NORMED,
  ffn_in: STAGES.FFN_IN,
  ffn_gate: STAGES.FFN_GATE,
  ffn_up: STAGES.FFN_UP,
  ffn_act: STAGES.FFN_ACT,
  ffn_out: STAGES.FFN_OUT,
  layer_out: STAGES.LAYER_OUT,
  pre_final_norm: STAGES.PRE_FINAL_NORM,
  final_norm: STAGES.FINAL_NORM_OUT,
  logits: STAGES.LOGITS_OUT,
  logits_final: STAGES.LOGITS_FINAL,
});

// ============================================================================
// Lookup Helpers
// ============================================================================

export function getOperatorClass(stageName) {
  return STAGE_TO_CLASS_MAP[stageName] ?? null;
}

export function canonicalizeProbeStage(probeStageName) {
  return PROBE_TO_CANONICAL[probeStageName] ?? null;
}

export function isValidStage(stageName) {
  return stageName in STAGE_TO_CLASS_MAP;
}

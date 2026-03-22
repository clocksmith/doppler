/**
 * Canonical semantic stage namespace for operator diffing.
 *
 * Stage names are dot-notation templates that combine with context
 * (layerIdx, phase) to produce deterministic operator IDs.
 */

export type OperatorClass =
  | 'embedding'
  | 'normalization'
  | 'projection'
  | 'rope'
  | 'attention'
  | 'activation'
  | 'ffn'
  | 'residual'
  | 'cast'
  | 'quantize'
  | 'logits'
  | 'conv'
  | 'moe_routing';

export declare const OPERATOR_CLASSES: Readonly<{
  EMBEDDING: 'embedding';
  NORMALIZATION: 'normalization';
  PROJECTION: 'projection';
  ROPE: 'rope';
  ATTENTION: 'attention';
  ACTIVATION: 'activation';
  FFN: 'ffn';
  RESIDUAL: 'residual';
  CAST: 'cast';
  QUANTIZE: 'quantize';
  LOGITS: 'logits';
  CONV: 'conv';
  MOE_ROUTING: 'moe_routing';
}>;

export type StageName = string;

export declare const STAGES: Readonly<{
  EMBED_OUT: 'embed.out';
  ATTN_INPUT: 'attn.input';
  ATTN_POST_INPUT_NORM: 'attn.post_input_norm';
  ATTN_NORMED: 'attn.normed';
  ATTN_QKV_PROJ: 'attn.qkv_proj';
  ATTN_Q_PROJ: 'attn.q_proj';
  ATTN_K_PROJ: 'attn.k_proj';
  ATTN_V_PROJ: 'attn.v_proj';
  ATTN_Q_NORM: 'attn.q_norm';
  ATTN_K_NORM: 'attn.k_norm';
  ATTN_Q_ROPE: 'attn.q_rope';
  ATTN_K_ROPE: 'attn.k_rope';
  ATTN_SCORES: 'attn.scores';
  ATTN_SOFTMAX: 'attn.softmax';
  ATTN_OUT: 'attn.out';
  ATTN_O_PROJ: 'attn.o_proj';
  ATTN_POST_ATTN: 'attn.post_attn';
  ATTN_LINEAR_Z_PROJ: 'attn.linear_z_proj';
  ATTN_LINEAR_A_PROJ: 'attn.linear_a_proj';
  ATTN_LINEAR_B_PROJ: 'attn.linear_b_proj';
  ATTN_LINEAR_CORE_OUT: 'attn.linear_core_out';
  CONV_INPUT_NORM: 'conv.input_norm';
  CONV_IN_PROJ: 'conv.in_proj';
  CONV_KERNEL: 'conv.kernel';
  CONV_OUT_PROJ: 'conv.out_proj';
  FFN_NORMED: 'ffn.normed';
  FFN_IN: 'ffn.in';
  FFN_GATE: 'ffn.gate';
  FFN_UP: 'ffn.up';
  FFN_ACT: 'ffn.act';
  FFN_OUT: 'ffn.out';
  LAYER_OUT: 'layer.out';
  MOE_ROUTING: 'moe.routing';
  MOE_EXPERT: 'moe.expert';
  RESIDUAL_ADD: 'residual.add';
  CAST: 'cast';
  PRE_FINAL_NORM: 'final_norm.pre';
  FINAL_NORM_OUT: 'final_norm.out';
  LOGITS_OUT: 'logits.out';
  LOGITS_FINAL: 'logits.final';
}>;

export declare const PROBE_TO_CANONICAL: Readonly<Record<string, StageName>>;

export declare function getOperatorClass(stageName: StageName): OperatorClass | null;
export declare function canonicalizeProbeStage(probeStageName: string): StageName | null;
export declare function isValidStage(stageName: StageName): boolean;

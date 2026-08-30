export const LAYER_PREFIXES = (layerIdx) => [
  `model.decoder.layers.${layerIdx}`,
  `model.encoder.language_model.layers.${layerIdx}`,
  `model.language_model.layers.${layerIdx}`,
  `language_model.layers.${layerIdx}`,
  `language_model.model.layers.${layerIdx}`,
  `model.layers.${layerIdx}`,
  `layers.${layerIdx}`,
  `blk.${layerIdx}`,
];


export const ATTN_SUFFIXES = {
  inputNorm: ['input_layernorm.weight', 'attn_norm.weight', 'operator_norm.weight'],
  inputNormBias: ['input_layernorm.bias', 'attn_norm.bias', 'operator_norm.bias'],
  qProj: ['self_attn.q_proj.weight', 'attention.wq.weight', 'attn_q.weight'],
  qGateProj: ['self_attn.gate_proj.weight', 'self_attn.q_gate_proj.weight', 'attention.q_gate.weight'],
  qProjBias: ['self_attn.q_proj.bias', 'attention.wq.bias', 'attn_q.bias'],
  kProj: ['self_attn.k_proj.weight', 'attention.wk.weight', 'attn_k.weight'],
  kProjBias: ['self_attn.k_proj.bias', 'attention.wk.bias', 'attn_k.bias'],
  vProj: ['self_attn.v_proj.weight', 'attention.wv.weight', 'attn_v.weight'],
  vProjBias: ['self_attn.v_proj.bias', 'attention.wv.bias', 'attn_v.bias'],
  oProj: ['self_attn.o_proj.weight', 'self_attn.out_proj.weight', 'attention.wo.weight', 'attn_output.weight'],
  oProjBias: ['self_attn.o_proj.bias', 'self_attn.out_proj.bias', 'attention.wo.bias', 'attn_output.bias'],
  qNorm: ['self_attn.q_norm.weight', 'self_attn.q_layernorm.weight', 'attn_q_norm.weight'],
  kNorm: ['self_attn.k_norm.weight', 'self_attn.k_layernorm.weight', 'attn_k_norm.weight'],
  postAttentionNorm: [
    'post_self_attn_layernorm.weight',
    'post_attention_layernorm.weight',
    'post_attention_norm.weight',
    'ffn_norm.weight',
  ],
  postAttentionNormBias: ['post_attention_layernorm.bias', 'post_attention_norm.bias', 'ffn_norm.bias'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight', 'post_attention_layernorm.weight'],
  preFeedforwardNorm2: ['pre_feedforward_layernorm_2.weight'],
  postFeedforwardNorm: [
    'post_feedforward_layernorm.weight',
    'post_mlp_layernorm.weight',
    'post_ffw_norm.weight',
  ],
  postFeedforwardNorm1: ['post_feedforward_layernorm_1.weight'],
  postFeedforwardNorm2: ['post_feedforward_layernorm_2.weight'],
  postPerLayerInputNorm: ['post_per_layer_input_norm.weight'],
  layerScalar: ['layer_scalar'],
};

export const LINEAR_ATTN_SUFFIXES = {
  qkvProj: ['linear_attn.in_proj_qkv.weight'],
  outProj: ['linear_attn.out_proj.weight'],
  inProjZ: ['linear_attn.in_proj_z.weight'],
  inProjA: ['linear_attn.in_proj_a.weight'],
  inProjB: ['linear_attn.in_proj_b.weight'],
  conv1D: ['linear_attn.conv1d.weight'],
  dtBias: ['linear_attn.dt_bias'],
  aLog: ['linear_attn.A_log'],
  norm: ['linear_attn.norm.weight'],
};

export const CONV_SUFFIXES = {
  convInProj: ['conv.in_proj.weight', 'convolution.in_proj.weight'],
  convKernel: ['conv.conv.weight', 'convolution.conv.weight', 'conv.weight'],
  convOutProj: ['conv.out_proj.weight', 'convolution.out_proj.weight'],
};


export const FFN_SUFFIXES = {
  ffnGateUp: ['mlp.gate_up_proj.weight', 'ffn_gate_up.weight', 'feed_forward.w1_w3.weight'],
  ffnGate: ['mlp.gate_proj.weight', 'feed_forward.w1.weight', 'ffn_gate.weight'],
  ffnGateBias: ['mlp.gate_proj.bias', 'feed_forward.w1.bias', 'ffn_gate.bias'],
  ffnUp: ['mlp.up_proj.weight', 'feed_forward.w3.weight', 'ffn_up.weight'],
  ffnUpBias: ['mlp.up_proj.bias', 'feed_forward.w3.bias', 'ffn_up.bias'],
  ffnDown: ['mlp.down_proj.weight', 'feed_forward.w2.weight', 'ffn_down.weight'],
  ffnDownBias: ['mlp.down_proj.bias', 'feed_forward.w2.bias', 'ffn_down.bias'],
  perLayerInputGate: ['per_layer_input_gate.weight'],
  perLayerProjection: ['per_layer_projection.weight'],
};


export const ROUTER_SUFFIXES = {
  routerWeight: ['mlp.router.weight', 'block_sparse_moe.gate.weight', 'router.proj.weight'],
  routerBias: ['mlp.router.bias'],
  routerScale: ['router.scale'],
  routerPerExpertScale: ['router.per_expert_scale'],
};


export const SINK_SUFFIXES = ['self_attn.sinks'];


export const MATMUL_KEYS = [
  'qProj', 'kProj', 'vProj', 'oProj',
  'qkvProj',
  'linearInProjZ', 'linearInProjA', 'linearInProjB',
  'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp',
  'perLayerInputGate', 'perLayerProjection',
  'convInProj', 'convOutProj',
  'routerWeight',
];


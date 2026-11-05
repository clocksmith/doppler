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
  preFeedforwardNormBias: ['pre_feedforward_layernorm.bias'],
  preFeedforwardNorm: ['pre_feedforward_layernorm.weight', 'post_attention_layernorm.weight'],
  preFeedforwardNorm2: ['pre_feedforward_layernorm_2.weight'],
  postFeedforwardNormBias: ['post_feedforward_layernorm.bias'],
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
  qkvProj: ['linear_attn.in_proj_qkv.weight', 'attn_qkv.weight'],
  outProj: ['linear_attn.out_proj.weight', 'ssm_out.weight'],
  inProjZ: ['linear_attn.in_proj_z.weight', 'attn_gate.weight'],
  inProjA: ['linear_attn.in_proj_a.weight', 'ssm_alpha.weight'],
  inProjB: ['linear_attn.in_proj_b.weight', 'ssm_beta.weight'],
  conv1D: ['linear_attn.conv1d.weight', 'ssm_conv1d.weight'],
  dtBias: ['linear_attn.dt_bias', 'ssm_dt.bias'],
  aLog: ['linear_attn.A_log', 'ssm_a'],
  norm: ['linear_attn.norm.weight', 'ssm_norm.weight'],
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
  routerWeight: ['mlp.gate.weight', 'mlp.router.weight', 'block_sparse_moe.gate.weight', 'router.proj.weight', 'ffn_gate_inp.weight'],
  routerBias: ['mlp.router.bias'],
  routerScale: ['router.scale'],
  routerPerExpertScale: ['router.per_expert_scale'],
  sharedExpertGate: ['mlp.shared_expert.gate_proj.weight', 'ffn_gate_shexp.weight'],
  sharedExpertUp: ['mlp.shared_expert.up_proj.weight', 'ffn_up_shexp.weight'],
  sharedExpertDown: ['mlp.shared_expert.down_proj.weight', 'ffn_down_shexp.weight'],
  sharedExpertRouter: ['mlp.shared_expert_gate.weight', 'ffn_gate_inp_shexp.weight'],
};


export const SINK_SUFFIXES = ['self_attn.sinks'];

export const GATED_RESIDUAL_SUFFIXES = Object.freeze({
  norm: 'hc_norm.weight',
  down: 'input_mix_weight_down.weight',
  up: 'input_mix_weight_up.weight',
  inject: 'block_inject_weight.weight',
});

export const HYPER_CONNECTION_SUFFIXES = Object.freeze(Object.fromEntries(
  ['attention', 'mlp'].map(branch => [branch, Object.freeze(Object.fromEntries(
    Object.entries(GATED_RESIDUAL_SUFFIXES).map(([key, suffix]) =>
      [key, Object.freeze([`${branch === 'attention' ? 'attn' : 'mlp'}_hyper_connection.${suffix}`])])
  ))])
));

export const QSA_SUFFIXES = Object.freeze({
  projection: Object.freeze(['self_attn.indexer.index_qk_proj.weight']),
  qNorm: Object.freeze(['self_attn.indexer.q_layernorm.weight']),
  kNorm: Object.freeze(['self_attn.indexer.k_layernorm.weight']),
});

export const NGRAM_PLE_SUFFIXES = Object.freeze({
  keyProj: Object.freeze(['ple.key_proj.weight']),
  valueProj: Object.freeze(['ple.value_proj.weight']),
  normKey: Object.freeze(['ple.norm_key.weight']),
  normQuery: Object.freeze(['ple.norm_query.weight']),
  normConv: Object.freeze(['ple.norm_conv.weight']),
  conv: Object.freeze(['ple.conv1d.weight']),
});

export const HYPER_CONNECTION_MIXER_PREFIXES = Object.freeze([
  'model.language_model.hyper_connection_mixer',
  'language_model.hyper_connection_mixer',
  'language_model.model.hyper_connection_mixer',
  'model.hyper_connection_mixer',
  'hyper_connection_mixer',
]);

export const ROOT_WEIGHT_CONSUMERS = Object.freeze(['down', 'up'].map(key => Object.freeze({
  role: `hc_mixer_${key}`,
  names: Object.freeze(HYPER_CONNECTION_MIXER_PREFIXES.map(prefix => `${prefix}.${GATED_RESIDUAL_SUFFIXES[key]}`)),
})));


export const MATMUL_KEYS = [
  'qProj', 'kProj', 'vProj', 'oProj',
  'qkvProj',
  'linearInProjZ', 'linearInProjA', 'linearInProjB',
  'ffnGate', 'ffnUp', 'ffnDown', 'ffnGateUp',
  'perLayerInputGate', 'perLayerProjection',
  'convInProj', 'convOutProj',
  'routerWeight', 'sharedExpertGate', 'sharedExpertUp', 'sharedExpertDown', 'sharedExpertRouter',
];


// Shared by loader binding and pre-load representation analysis.
export const LAYER_WEIGHT_CONSUMERS = Object.freeze([
  ['q_proj', ATTN_SUFFIXES.qProj], ['q_gate_proj', ATTN_SUFFIXES.qGateProj],
  ['k_proj', ATTN_SUFFIXES.kProj], ['v_proj', ATTN_SUFFIXES.vProj], ['o_proj', ATTN_SUFFIXES.oProj],
  ['linear_qkv_proj', LINEAR_ATTN_SUFFIXES.qkvProj], ['linear_out_proj', LINEAR_ATTN_SUFFIXES.outProj],
  ['linear_z_proj', LINEAR_ATTN_SUFFIXES.inProjZ], ['linear_a_proj', LINEAR_ATTN_SUFFIXES.inProjA],
  ['linear_b_proj', LINEAR_ATTN_SUFFIXES.inProjB], ['ffn_gate_up', FFN_SUFFIXES.ffnGateUp],
  ['ffn_gate', ['mlp.experts.gate_up_proj']], ['ffn_down', ['mlp.experts.down_proj']],
  ['ffn_gate', ROUTER_SUFFIXES.sharedExpertGate], ['ffn_up', ROUTER_SUFFIXES.sharedExpertUp],
  ['ffn_down', ROUTER_SUFFIXES.sharedExpertDown],
  ['moe_router', ROUTER_SUFFIXES.routerWeight], ['moe_shared_router', ROUTER_SUFFIXES.sharedExpertRouter],
  ['ffn_gate', FFN_SUFFIXES.ffnGate, ['ffn_gate_up']], ['ffn_up', FFN_SUFFIXES.ffnUp, ['ffn_gate_up']], ['ffn_down', FFN_SUFFIXES.ffnDown],
  ...['attention', 'mlp'].flatMap(branch => ['down', 'up', 'inject'].map(key =>
    [`${branch === 'attention' ? 'attn' : 'mlp'}_hc_${key}`, HYPER_CONNECTION_SUFFIXES[branch][key]])),
  ['qsa_index_qk', QSA_SUFFIXES.projection],
  ['ple_key_proj', NGRAM_PLE_SUFFIXES.keyProj], ['ple_value_proj', NGRAM_PLE_SUFFIXES.valueProj],
].map(([role, suffixes, additionalRoles = []]) => Object.freeze({ role, suffixes, additionalRoles: Object.freeze(additionalRoles) })));




export function classifyTensor(name, modelType) {
  const lower = name.toLowerCase();

  // Embeddings
  if (lower.includes('embed_tokens') || lower.includes('token_embd') ||
      lower.includes('wte.weight') || lower.includes('word_embeddings')) {
    return 'embed';
  }

  // Head (LM head + final norm)
  if (lower.includes('lm_head') || lower.includes('output.weight') ||
      lower.endsWith('.output')) {
    return 'head';
  }
  if ((lower.includes('final') || lower.includes('model.norm') || lower.includes('norm_f')) &&
      lower.includes('norm')) {
    return 'head';
  }

  // Extract layer index
  const layerMatch = name.match(/layers?[._](\d+)/i);
  if (!layerMatch) {
    return 'other';
  }
  const layerIdx = parseInt(layerMatch[1]);

  // MoE experts
  const expertMatch = name.match(/experts?[._](\d+)/i);
  if (expertMatch) {
    const expertIdx = parseInt(expertMatch[1]);

    if (lower.includes('shared_expert')) {
      return `layer.${layerIdx}.shared_expert`;
    }

    return `layer.${layerIdx}.expert.${expertIdx}`;
  }

  // Shared MoE components
  if (lower.includes('block_sparse_moe.gate') || lower.includes('router') ||
      lower.includes('moe.gate')) {
    return `layer.${layerIdx}.shared`;
  }

  // Hybrid architectures
  if (modelType === 'jamba' || modelType === 'hybrid') {
    if (lower.includes('self_attn') || lower.includes('attention')) {
      return `layer.${layerIdx}.attn`;
    }
    if (lower.includes('mamba')) {
      return `layer.${layerIdx}.mamba`;
    }
  }

  // Pure Mamba/RWKV
  if (modelType === 'mamba' || modelType === 'rwkv') {
    return `layer.${layerIdx}`;
  }

  // Default: dense transformer layer
  return `layer.${layerIdx}`;
}

export function classifyTensorRole(name) {
  const lower = name.toLowerCase();

  const embeddingPatterns = [
    'embed_tokens.weight',
    'token_embd.weight',
    'wte.weight',
    'transformer.wte.weight',
    'word_embeddings',
  ];
  if (embeddingPatterns.some((pattern) => lower.includes(pattern))) {
    return 'embedding';
  }

  if (lower.includes('lm_head')) return 'lm_head';
  if (lower.endsWith('output.weight') && !lower.includes('attn_')) return 'lm_head';

  if (lower.includes('router') || lower.includes('block_sparse_moe.gate') || lower.includes('moe.gate')) {
    return 'router';
  }

  if (lower.includes('norm') || lower.includes('ln_') || lower.includes('layernorm')) {
    return 'norm';
  }

  const matmulSuffixes = [
    'q_proj.weight',
    'k_proj.weight',
    'v_proj.weight',
    'o_proj.weight',
    'attention.wq.weight',
    'attention.wk.weight',
    'attention.wv.weight',
    'attention.wo.weight',
    'gate_proj.weight',
    'up_proj.weight',
    'down_proj.weight',
    'w1.weight',
    'w2.weight',
    'w3.weight',
    'attn_q.weight',
    'attn_k.weight',
    'attn_v.weight',
    'attn_output.weight',
    'ffn_gate.weight',
    'ffn_up.weight',
    'ffn_down.weight',
    'ffn_gate_up.weight',
  ];
  if (matmulSuffixes.some((suffix) => lower.endsWith(suffix))) {
    return 'matmul';
  }

  return 'other';
}


export function getGroupType(groupId, modelType) {
  if (groupId === 'embed') return 'embed';
  if (groupId === 'head') return 'head';
  if (groupId === 'other') return 'layer';

  if (groupId.includes('.expert.')) return 'expert';
  if (groupId.includes('.shared_expert')) return 'shared';
  if (groupId.includes('.shared')) return 'layer';
  if (groupId.includes('.attn')) return 'attn';
  if (groupId.includes('.mamba')) return 'mamba';

  if (modelType === 'mamba') return 'mamba';
  if (modelType === 'rwkv') return 'rwkv';

  return 'layer';
}


export function parseGroupLayerIndex(groupId) {
  const match = groupId.match(/layer\.(\d+)/);
  return match ? parseInt(match[1]) : undefined;
}


export function parseGroupExpertIndex(groupId) {
  const match = groupId.match(/expert\.(\d+)/);
  return match ? parseInt(match[1]) : undefined;
}


export function sortGroupIds(groupIds) {
  return [...groupIds].sort((a, b) => {
    if (a === 'embed') return -1;
    if (b === 'embed') return 1;
    if (a === 'head') return 1;
    if (b === 'head') return -1;

    const layerA = parseGroupLayerIndex(a) ?? Infinity;
    const layerB = parseGroupLayerIndex(b) ?? Infinity;
    if (layerA !== layerB) return layerA - layerB;

    if (a.includes('.shared') && !a.includes('.shared_expert')) return -1;
    if (b.includes('.shared') && !b.includes('.shared_expert')) return 1;
    if (a.includes('.attn')) return -1;
    if (b.includes('.attn')) return 1;
    if (a.includes('.mamba')) return -1;
    if (b.includes('.mamba')) return 1;

    const expertA = parseGroupExpertIndex(a) ?? Infinity;
    const expertB = parseGroupExpertIndex(b) ?? Infinity;
    return expertA - expertB;
  });
}

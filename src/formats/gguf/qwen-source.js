import families from './qwen-source.json' with { type: 'json' };

/** @type {Readonly<Record<string, {modelType: string}>>} */
const declaredFamilies = families;

/** @param {unknown} value @param {string} key */
function positive(value, key) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    throw new Error(`GGUF Qwen source requires positive ${key}.`);
  }
  return value;
}

/** @type {typeof import('./qwen-source.js').normalizeQwenGGUFSource} */
export function normalizeQwenGGUFSource(architecture, metadata) {
  const family = declaredFamilies[architecture];
  if (!family) return null;
  /** @param {string} key */
  const number = key => positive(metadata[`${architecture}.${key}`], key);
  /** @param {string} key */
  const integer = key => {
    const value = number(key);
    if (!Number.isSafeInteger(value)) throw new Error(`GGUF Qwen ${key} must be an integer.`);
    return value;
  };
  const mtp = metadata[`${architecture}.nextn_predict_layers`] ?? 0;
  if (typeof mtp !== 'number' || !Number.isSafeInteger(mtp) || mtp < 0) {
    throw new Error('GGUF Qwen nextn_predict_layers must be a nonnegative integer.');
  }
  const textLayers = integer('block_count') - mtp;
  const period = integer('full_attention_interval');
  const valueHeads = integer('ssm.time_step_rank');
  const inner = integer('ssm.inner_size');
  const headDim = integer('attention.key_length');
  const rotaryDim = integer('rope.dimension_count');
  if (textLayers < 1 || inner % valueHeads || rotaryDim > headDim
      || integer('attention.value_length') !== headDim) {
    throw new Error('Unsupported GGUF Qwen text, value-head or rotary geometry.');
  }
  const tokens = metadata['tokenizer.ggml.tokens'];
  if (!Array.isArray(tokens) || !tokens.length) throw new Error('GGUF Qwen vocabulary is missing.');
  const sections = metadata[`${architecture}.rope.dimension_sections`];
  if (!Array.isArray(sections) || sections.length !== 4 || sections[3] !== 0
      || sections.some(value => !Number.isSafeInteger(value) || value < 0)
      || sections.slice(0, 3).reduce((a, b) => a + b, 0) * 2 !== rotaryDim) {
    throw new Error('Unsupported GGUF Qwen MRoPE sections.');
  }
  const config = {
    model_type: family.modelType,
    num_hidden_layers: textLayers,
    hidden_size: integer('embedding_length'),
    intermediate_size: architecture === 'qwen35moe' ? integer('expert_feed_forward_length') : integer('feed_forward_length'),
    num_attention_heads: integer('attention.head_count'),
    num_key_value_heads: integer('attention.head_count_kv'),
    head_dim: headDim,
    vocab_size: tokens.length,
    max_position_embeddings: integer('context_length'),
    rms_norm_eps: number('attention.layer_norm_rms_epsilon'),
    rms_norm_weight_offset: false,
    rope_theta: number('rope.freq_base'),
    partial_rotary_factor: rotaryDim / headDim,
    mrope_section: sections.slice(0, 3),
    rope_parameters: { rope_theta: number('rope.freq_base'), partial_rotary_factor: rotaryDim / headDim, mrope_section: sections.slice(0, 3), mrope_interleaved: true },
    layer_types: Array.from({ length: textLayers }, (_, layer) => (layer + 1) % period === 0 ? 'full_attention' : 'linear_attention'),
    linear_num_key_heads: integer('ssm.group_count'),
    linear_num_value_heads: valueHeads,
    linear_key_head_dim: integer('ssm.state_size'),
    linear_value_head_dim: inner / valueHeads,
    linear_conv_kernel_dim: integer('ssm.conv_kernel'),
    linear_norm_mode: 'shared',
    linear_attention_source_layout: 'gguf-v1',
    query_key_norm: true,
    query_pre_attn_scalar: headDim,
    attention_output_gate: true,
    hidden_act: 'silu',
    tie_word_embeddings: false,
    eos_token_id: metadata['tokenizer.ggml.eos_token_id'],
    bos_token_id: metadata['tokenizer.ggml.bos_token_id'],
    pad_token_id: metadata['tokenizer.ggml.padding_token_id'],
    ...(architecture === 'qwen35moe' ? {
      num_experts: integer('expert_count'),
      num_experts_per_tok: integer('expert_used_count'),
      moe_intermediate_size: integer('expert_feed_forward_length'),
      shared_expert_intermediate_size: integer('expert_shared_feed_forward_length'),
    } : {}),
  };
  return { config, textLayers, modelType: 'transformer' };
}

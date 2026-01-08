const DIFFUSERS_PREFIX = 'transformer.model.diffusion_model';
const CANONICAL_PREFIX = 'transformer';

function normalizeFormat(value) {
  if (!value) return null;
  const normalized = String(value).toLowerCase();
  if (normalized === 'diffusers' || normalized === 'diffuser') return 'diffusers';
  if (normalized === 'doppler' || normalized === 'canonical') return 'doppler';
  return null;
}

function mapLinearName(prefix, name, map) {
  if (!name.startsWith(prefix)) return null;
  const suffix = name.slice(prefix.length);
  const slot = map[suffix];
  if (!slot) return null;
  return slot;
}

function mapDiffusersCandidates(name) {
  if (name === 'pos_embed.pos_embed') return ['pos_embed'];

  if (name.startsWith('pos_embed.proj.')) {
    const suffix = name.slice('pos_embed.proj.'.length);
    return [`x_embedder.proj.${suffix}`];
  }

  const timeTextMap = {
    'linear_1.weight': 'mlp.0.weight',
    'linear_1.bias': 'mlp.0.bias',
    'linear_2.weight': 'mlp.2.weight',
    'linear_2.bias': 'mlp.2.bias',
  };
  const timeText = mapLinearName('time_text_embed.timestep_embedder.', name, timeTextMap);
  if (timeText) return [`t_embedder.${timeText}`];

  const textEmbed = mapLinearName('time_text_embed.text_embedder.', name, timeTextMap);
  if (textEmbed) return [`y_embedder.${textEmbed}`];

  if (name.startsWith('context_embedder.')) {
    return [`context_embedder.${name.slice('context_embedder.'.length)}`];
  }

  if (name.startsWith('norm_out.linear.')) {
    return [`final_layer.adaLN_modulation.1.${name.slice('norm_out.linear.'.length)}`];
  }

  if (name.startsWith('proj_out.')) {
    return [`final_layer.linear.${name.slice('proj_out.'.length)}`];
  }

  const blockMatch = name.match(/^transformer_blocks\.(\d+)\.(.+)$/);
  if (blockMatch) {
    const idx = blockMatch[1];
    const rest = blockMatch[2];

    if (rest.startsWith('norm1.linear.')) {
      const suffix = rest.slice('norm1.linear.'.length);
      return [`joint_blocks.${idx}.x_block.adaLN_modulation.1.${suffix}`];
    }
    if (rest.startsWith('norm1_context.linear.')) {
      const suffix = rest.slice('norm1_context.linear.'.length);
      return [`joint_blocks.${idx}.context_block.adaLN_modulation.1.${suffix}`];
    }
    if (rest.startsWith('attn.qkv.')) {
      const suffix = rest.slice('attn.qkv.'.length);
      return [`joint_blocks.${idx}.x_block.attn.qkv.${suffix}`];
    }
    if (rest.startsWith('attn.add_qkv.')) {
      const suffix = rest.slice('attn.add_qkv.'.length);
      return [`joint_blocks.${idx}.context_block.attn.qkv.${suffix}`];
    }
    if (rest.startsWith('attn.to_out.0.')) {
      const suffix = rest.slice('attn.to_out.0.'.length);
      return [`joint_blocks.${idx}.x_block.attn.proj.${suffix}`];
    }
    if (rest.startsWith('attn.to_add_out.')) {
      const suffix = rest.slice('attn.to_add_out.'.length);
      return [`joint_blocks.${idx}.context_block.attn.proj.${suffix}`];
    }
    if (rest.startsWith('attn2.qkv.')) {
      const suffix = rest.slice('attn2.qkv.'.length);
      return [`joint_blocks.${idx}.x_block.attn2.qkv.${suffix}`];
    }
    if (rest.startsWith('attn2.to_out.0.')) {
      const suffix = rest.slice('attn2.to_out.0.'.length);
      return [`joint_blocks.${idx}.x_block.attn2.proj.${suffix}`];
    }
    if (rest.startsWith('ff.net.0.proj.')) {
      const suffix = rest.slice('ff.net.0.proj.'.length);
      return [`joint_blocks.${idx}.x_block.mlp.fc1.${suffix}`];
    }
    if (rest.startsWith('ff.net.2.')) {
      const suffix = rest.slice('ff.net.2.'.length);
      return [`joint_blocks.${idx}.x_block.mlp.fc2.${suffix}`];
    }
    if (rest.startsWith('ff_context.net.0.proj.')) {
      const suffix = rest.slice('ff_context.net.0.proj.'.length);
      return [`joint_blocks.${idx}.context_block.mlp.fc1.${suffix}`];
    }
    if (rest.startsWith('ff_context.net.2.')) {
      const suffix = rest.slice('ff_context.net.2.'.length);
      return [`joint_blocks.${idx}.context_block.mlp.fc2.${suffix}`];
    }

    const candidateMap = [
      {
        prefix: 'attn.to_q.',
        out: [
          `joint_blocks.${idx}.x_block.attn.q_proj.`,
          `joint_blocks.${idx}.x_block.attn.to_q.`,
          `joint_blocks.${idx}.x_block.attn.q.`,
        ],
      },
      {
        prefix: 'attn.to_k.',
        out: [
          `joint_blocks.${idx}.x_block.attn.k_proj.`,
          `joint_blocks.${idx}.x_block.attn.to_k.`,
          `joint_blocks.${idx}.x_block.attn.k.`,
        ],
      },
      {
        prefix: 'attn.to_v.',
        out: [
          `joint_blocks.${idx}.x_block.attn.v_proj.`,
          `joint_blocks.${idx}.x_block.attn.to_v.`,
          `joint_blocks.${idx}.x_block.attn.v.`,
        ],
      },
      {
        prefix: 'attn.add_q_proj.',
        out: [
          `joint_blocks.${idx}.context_block.attn.q_proj.`,
          `joint_blocks.${idx}.context_block.attn.to_q.`,
          `joint_blocks.${idx}.context_block.attn.q.`,
        ],
      },
      {
        prefix: 'attn.add_k_proj.',
        out: [
          `joint_blocks.${idx}.context_block.attn.k_proj.`,
          `joint_blocks.${idx}.context_block.attn.to_k.`,
          `joint_blocks.${idx}.context_block.attn.k.`,
        ],
      },
      {
        prefix: 'attn.add_v_proj.',
        out: [
          `joint_blocks.${idx}.context_block.attn.v_proj.`,
          `joint_blocks.${idx}.context_block.attn.to_v.`,
          `joint_blocks.${idx}.context_block.attn.v.`,
        ],
      },
      {
        prefix: 'attn2.to_q.',
        out: [
          `joint_blocks.${idx}.x_block.attn2.q_proj.`,
          `joint_blocks.${idx}.x_block.attn2.to_q.`,
          `joint_blocks.${idx}.x_block.attn2.q.`,
        ],
      },
      {
        prefix: 'attn2.to_k.',
        out: [
          `joint_blocks.${idx}.x_block.attn2.k_proj.`,
          `joint_blocks.${idx}.x_block.attn2.to_k.`,
          `joint_blocks.${idx}.x_block.attn2.k.`,
        ],
      },
      {
        prefix: 'attn2.to_v.',
        out: [
          `joint_blocks.${idx}.x_block.attn2.v_proj.`,
          `joint_blocks.${idx}.x_block.attn2.to_v.`,
          `joint_blocks.${idx}.x_block.attn2.v.`,
        ],
      },
    ];

    for (const entry of candidateMap) {
      if (rest.startsWith(entry.prefix)) {
        const suffix = rest.slice(entry.prefix.length);
        return entry.out.map((base) => `${base}${suffix}`);
      }
    }

    const normMap = [
      {
        prefix: 'attn.norm_q.',
        out: [
          `joint_blocks.${idx}.x_block.attn.ln_q.`,
          `joint_blocks.${idx}.x_block.attn.q_norm.`,
          `joint_blocks.${idx}.x_block.attn.norm_q.`,
        ],
      },
      {
        prefix: 'attn.norm_k.',
        out: [
          `joint_blocks.${idx}.x_block.attn.ln_k.`,
          `joint_blocks.${idx}.x_block.attn.k_norm.`,
          `joint_blocks.${idx}.x_block.attn.norm_k.`,
        ],
      },
      {
        prefix: 'attn.norm_added_q.',
        out: [
          `joint_blocks.${idx}.context_block.attn.ln_q.`,
          `joint_blocks.${idx}.context_block.attn.q_norm.`,
          `joint_blocks.${idx}.context_block.attn.norm_q.`,
        ],
      },
      {
        prefix: 'attn.norm_added_k.',
        out: [
          `joint_blocks.${idx}.context_block.attn.ln_k.`,
          `joint_blocks.${idx}.context_block.attn.k_norm.`,
          `joint_blocks.${idx}.context_block.attn.norm_k.`,
        ],
      },
      {
        prefix: 'attn2.norm_q.',
        out: [
          `joint_blocks.${idx}.x_block.attn2.ln_q.`,
          `joint_blocks.${idx}.x_block.attn2.q_norm.`,
          `joint_blocks.${idx}.x_block.attn2.norm_q.`,
        ],
      },
      {
        prefix: 'attn2.norm_k.',
        out: [
          `joint_blocks.${idx}.x_block.attn2.ln_k.`,
          `joint_blocks.${idx}.x_block.attn2.k_norm.`,
          `joint_blocks.${idx}.x_block.attn2.norm_k.`,
        ],
      },
    ];

    for (const entry of normMap) {
      if (rest.startsWith(entry.prefix)) {
        const suffix = rest.slice(entry.prefix.length);
        return entry.out.map((base) => `${base}${suffix}`);
      }
    }
  }

  return [name];
}

export function createSD3WeightResolver(weightsEntry, modelConfig) {
  const weights = weightsEntry?.weights ?? new Map();
  const shapes = weightsEntry?.shapes ?? new Map();
  const dtypes = weightsEntry?.dtypes ?? new Map();

  const transformerConfig = modelConfig?.components?.transformer?.config || {};
  const explicitFormat = normalizeFormat(transformerConfig.weight_format || transformerConfig.weightFormat || transformerConfig.tensorFormat);
  const hasDiffusers = weights.has(`${DIFFUSERS_PREFIX}.pos_embed`) || weights.has(`${DIFFUSERS_PREFIX}.x_embedder.proj.weight`);
  const format = explicitFormat || (hasDiffusers ? 'diffusers' : 'doppler');

  const resolve = (name) => {
    const candidates = format === 'diffusers' ? mapDiffusersCandidates(name) : [name];
    if (!candidates.includes(name)) candidates.push(name);
    const prefixes = format === 'diffusers'
      ? [DIFFUSERS_PREFIX, CANONICAL_PREFIX]
      : [CANONICAL_PREFIX, DIFFUSERS_PREFIX];

    for (const prefix of prefixes) {
      for (const candidate of candidates) {
        const key = `${prefix}.${candidate}`;
        if (weights.has(key)) {
          return { key, value: weights.get(key) };
        }
      }
    }

    const fallbackKey = `${prefixes[0]}.${candidates[0]}`;
    return { key: fallbackKey, value: null };
  };

  return {
    get: (name) => resolve(name).value,
    key: (name) => resolve(name).key,
    shape: (name) => shapes.get(resolve(name).key) || null,
    dtype: (name) => dtypes.get(resolve(name).key) || null,
    format,
  };
}

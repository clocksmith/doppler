export async function loadRequiredVisionGpuTensor(loader, name) {
  if (typeof loader.loadGpuTensor !== 'function') {
    throw new Error('Vision weight loading requires loader.loadGpuTensor().');
  }
  const tensor = await loader.loadGpuTensor(name, true);
  if (!tensor) {
    throw new Error(`Vision tensor "${name}" is missing from the converted artifact.`);
  }
  return tensor;
}

export async function loadVisionScalar(loadRequiredTensor, name) {
  const tensor = await loadRequiredTensor(name, false);
  if (tensor instanceof Float32Array) {
    if (tensor.length !== 1) {
      throw new Error(`Vision scalar "${name}" must be a single-element tensor, got length=${tensor.length}.`);
    }
    return tensor[0];
  }
  if (ArrayBuffer.isView(tensor) && tensor.length === 1) return Number(tensor[0]);
  if (typeof tensor === 'number') return tensor;
  throw new Error(
    `Vision scalar "${name}" must decode to a single numeric value, `
    + `got ${tensor?.constructor?.name ?? typeof tensor} length=${tensor?.length ?? 'N/A'}.`
  );
}

export async function loadVisionClipRange(loadRequiredTensor, prefix) {
  return {
    inputMin: await loadVisionScalar(loadRequiredTensor, `${prefix}.input_min`),
    inputMax: await loadVisionScalar(loadRequiredTensor, `${prefix}.input_max`),
    outputMin: await loadVisionScalar(loadRequiredTensor, `${prefix}.output_min`),
    outputMax: await loadVisionScalar(loadRequiredTensor, `${prefix}.output_max`),
  };
}

export async function loadGlmOcrVisionWeights(loader, options) {
  const { textHiddenSize, depth } = options;
  const load = (name) => loadRequiredVisionGpuTensor(loader, name);
  const root = 'model.visual';
  const visionWeights = {
    textHiddenSize,
    patchProjWeight: await load(`${root}.patch_embed.proj.weight`),
    patchProjBias: await load(`${root}.patch_embed.proj.bias`),
    postLayerNorm: await load(`${root}.post_layernorm.weight`),
    downsampleWeight: await load(`${root}.downsample.weight`),
    downsampleBias: await load(`${root}.downsample.bias`),
    merger: {
      projWeight: await load(`${root}.merger.proj.weight`),
      postProjectionNormWeight: await load(`${root}.merger.post_projection_norm.weight`),
      postProjectionNormBias: await load(`${root}.merger.post_projection_norm.bias`),
      gateProjWeight: await load(`${root}.merger.gate_proj.weight`),
      upProjWeight: await load(`${root}.merger.up_proj.weight`),
      downProjWeight: await load(`${root}.merger.down_proj.weight`),
    },
    layers: [],
  };

  for (let index = 0; index < depth; index++) {
    const prefix = `${root}.blocks.${index}`;
    visionWeights.layers.push({
      norm1Weight: await load(`${prefix}.norm1.weight`),
      norm2Weight: await load(`${prefix}.norm2.weight`),
      qNormWeight: await load(`${prefix}.attn.q_norm.weight`),
      kNormWeight: await load(`${prefix}.attn.k_norm.weight`),
      qkvWeight: await load(`${prefix}.attn.qkv.weight`),
      qkvBias: await load(`${prefix}.attn.qkv.bias`),
      projWeight: await load(`${prefix}.attn.proj.weight`),
      projBias: await load(`${prefix}.attn.proj.bias`),
      gateProjWeight: await load(`${prefix}.mlp.gate_proj.weight`),
      gateProjBias: await load(`${prefix}.mlp.gate_proj.bias`),
      upProjWeight: await load(`${prefix}.mlp.up_proj.weight`),
      upProjBias: await load(`${prefix}.mlp.up_proj.bias`),
      downProjWeight: await load(`${prefix}.mlp.down_proj.weight`),
      downProjBias: await load(`${prefix}.mlp.down_proj.bias`),
    });
  }
  return visionWeights;
}

function createRequiredTensorLoader(loader) {
  return async (name, toGPU = true) => {
    const tensor = await loader.loadTensor(name, toGPU, true);
    if (!tensor) {
      throw new Error(`Vision tensor "${name}" is missing from the converted artifact.`);
    }
    return tensor;
  };
}

export async function loadGemma4VisionWeights(loader, options) {
  const { textHiddenSize, depth } = options;
  const loadRequiredTensor = createRequiredTensorLoader(loader);
  const loadClipRange = (prefix) => loadVisionClipRange(loadRequiredTensor, prefix);
  const isEncoderFree = (depth === 0);
  const visionWeights = {
    textHiddenSize,
    patchInputProj: await loadRequiredTensor('model.vision_tower.patch_embedder.input_proj.weight'),
    patchPositionEmbeddingTable: await loadRequiredVisionGpuTensor(loader,
      'model.vision_tower.patch_embedder.position_embedding_table'
    ),
    projector: isEncoderFree
      ? await loader.loadTensor('model.embed_vision.embedding_projection.weight', true, true)
      : await loadRequiredTensor('model.embed_vision.embedding_projection.weight'),
    layers: [],
  };

  for (let i = 0; i < depth; i++) {
    const prefix = `model.vision_tower.encoder.layers.${i}`;
    const attnPrefix = `${prefix}.self_attn`;
    const mlpPrefix = `${prefix}.mlp`;
    visionWeights.layers.push({
      inputLayerNorm: await loadRequiredTensor(`${prefix}.input_layernorm.weight`),
      postAttentionLayerNorm: await loadRequiredTensor(`${prefix}.post_attention_layernorm.weight`),
      preFeedforwardLayerNorm: await loadRequiredTensor(`${prefix}.pre_feedforward_layernorm.weight`),
      postFeedforwardLayerNorm: await loadRequiredTensor(`${prefix}.post_feedforward_layernorm.weight`),
      qNorm: await loadRequiredTensor(`${attnPrefix}.q_norm.weight`),
      kNorm: await loadRequiredTensor(`${attnPrefix}.k_norm.weight`),
      qProj: await loadRequiredTensor(`${attnPrefix}.q_proj.linear.weight`),
      kProj: await loadRequiredTensor(`${attnPrefix}.k_proj.linear.weight`),
      vProj: await loadRequiredTensor(`${attnPrefix}.v_proj.linear.weight`),
      oProj: await loadRequiredTensor(`${attnPrefix}.o_proj.linear.weight`),
      qProjClip: await loadClipRange(`${attnPrefix}.q_proj`),
      kProjClip: await loadClipRange(`${attnPrefix}.k_proj`),
      vProjClip: await loadClipRange(`${attnPrefix}.v_proj`),
      oProjClip: await loadClipRange(`${attnPrefix}.o_proj`),
      gateProj: await loadRequiredTensor(`${mlpPrefix}.gate_proj.linear.weight`),
      upProj: await loadRequiredTensor(`${mlpPrefix}.up_proj.linear.weight`),
      downProj: await loadRequiredTensor(`${mlpPrefix}.down_proj.linear.weight`),
      gateProjClip: await loadClipRange(`${mlpPrefix}.gate_proj`),
      upProjClip: await loadClipRange(`${mlpPrefix}.up_proj`),
      downProjClip: await loadClipRange(`${mlpPrefix}.down_proj`),
    });
  }

  return visionWeights;
}

export async function loadQwen3VlVisionWeights(loader, options) {
  const { depth } = options;
  const visionWeights = {};

  // Patch embedding weights
  const patchProjName = 'visual.patch_embed.proj.weight';
  const patchProjBiasName = 'visual.patch_embed.proj.bias';
  visionWeights.patchProjWeight = await loader.loadTensor(patchProjName, true, true);
  visionWeights.patchProjBias = await loader.loadTensor(patchProjBiasName, true, true);

  // Vision encoder layer weights
  visionWeights.layers = [];
  for (let i = 0; i < depth; i++) {
    const prefix = `visual.blocks.${i}`;
    const layerW = {
      norm1Weight: await loader.loadTensor(`${prefix}.norm1.weight`, true, true),
      norm2Weight: await loader.loadTensor(`${prefix}.norm2.weight`, true, true),
      qkvWeight: await loader.loadTensor(`${prefix}.attn.qkv.weight`, true, true),
      qkvBias: await loader.loadTensor(`${prefix}.attn.qkv.bias`, true, true),
      projWeight: await loader.loadTensor(`${prefix}.attn.proj.weight`, true, true),
      projBias: await loader.loadTensor(`${prefix}.attn.proj.bias`, true, true),
      fc1Weight: await loader.loadTensor(`${prefix}.mlp.fc1.weight`, true, true),
      fc1Bias: await loader.loadTensor(`${prefix}.mlp.fc1.bias`, true, true),
      fc2Weight: await loader.loadTensor(`${prefix}.mlp.fc2.weight`, true, true),
      fc2Bias: await loader.loadTensor(`${prefix}.mlp.fc2.bias`, true, true),
    };
    visionWeights.layers.push(layerW);
  }

  // Spatial merge projection
  visionWeights.mergerLnWeight = await loader.loadTensor('visual.merger.ln_q.weight', true, true);
  visionWeights.mergerMlp0Weight = await loader.loadTensor('visual.merger.mlp.0.weight', true, true);
  visionWeights.mergerMlp0Bias = await loader.loadTensor('visual.merger.mlp.0.bias', true, true);
  visionWeights.mergerMlp2Weight = await loader.loadTensor('visual.merger.mlp.2.weight', true, true);
  visionWeights.mergerMlp2Bias = await loader.loadTensor('visual.merger.mlp.2.bias', true, true);

  return visionWeights;
}

// Manifest vision_config.vision_architecture -> weight loader and log label.
const VISION_WEIGHT_LOADERS = Object.freeze({
  glmocr: Object.freeze({ load: loadGlmOcrVisionWeights, layerLabel: 'GLM-OCR encoder layers' }),
  gemma4: Object.freeze({ load: loadGemma4VisionWeights, layerLabel: 'Gemma 4 encoder layers' }),
  qwen3vl: Object.freeze({ load: loadQwen3VlVisionWeights, layerLabel: 'encoder layers' }),
});

export function resolveVisionWeightLoader(visionArchitecture) {
  if (typeof visionArchitecture !== 'string' || !Object.hasOwn(VISION_WEIGHT_LOADERS, visionArchitecture)) {
    throw new Error(`No vision weight loader for vision_architecture "${visionArchitecture}".`);
  }
  return VISION_WEIGHT_LOADERS[visionArchitecture];
}

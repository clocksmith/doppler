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

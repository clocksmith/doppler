/**
 * Vision encoder weight loader for Qwen3-VL.
 *
 * Loads vision tensors from an RDRR manifest via a loadTensor callback
 * (provided by DopplerLoader). Tensors are loaded to CPU for the
 * CPU-side vision encoder forward pass.
 */

import { log } from '../debug/index.js';

async function loadTensorSilent(loadTensor, name) {
  try {
    return await loadTensor(name, false, true);
  } catch {
    return null;
  }
}

export async function loadVisionWeights(loadTensor, visionConfig) {
  const depth = visionConfig.depth ?? 24;
  const deepstackIndexes = visionConfig.deepstackVisualIndexes ?? [];
  const weights = {
    patchEmbed: {},
    posEmbed: null,
    blocks: [],
    merger: {},
    deepstackMergers: [],
  };

  weights.patchEmbed.projWeight = await loadTensor('model.visual.patch_embed.proj.weight', false, false);
  weights.patchEmbed.projBias = await loadTensorSilent(loadTensor, 'model.visual.patch_embed.proj.bias');
  weights.posEmbed = await loadTensorSilent(loadTensor, 'model.visual.pos_embed.weight');

  for (let i = 0; i < depth; i++) {
    const prefix = `model.visual.blocks.${i}`;
    const block = {
      qkvWeight: await loadTensor(`${prefix}.attn.qkv.weight`, false, false),
      qkvBias: await loadTensorSilent(loadTensor, `${prefix}.attn.qkv.bias`),
      projWeight: await loadTensor(`${prefix}.attn.proj.weight`, false, false),
      projBias: await loadTensorSilent(loadTensor, `${prefix}.attn.proj.bias`),
      norm1Weight: await loadTensor(`${prefix}.norm1.weight`, false, false),
      norm1Bias: await loadTensorSilent(loadTensor, `${prefix}.norm1.bias`),
      norm2Weight: await loadTensor(`${prefix}.norm2.weight`, false, false),
      norm2Bias: await loadTensorSilent(loadTensor, `${prefix}.norm2.bias`),
      mlpFc1Weight: await loadTensor(`${prefix}.mlp.linear_fc1.weight`, false, false),
      mlpFc1Bias: await loadTensorSilent(loadTensor, `${prefix}.mlp.linear_fc1.bias`),
      mlpFc2Weight: await loadTensor(`${prefix}.mlp.linear_fc2.weight`, false, false),
      mlpFc2Bias: await loadTensorSilent(loadTensor, `${prefix}.mlp.linear_fc2.bias`),
    };
    weights.blocks.push(block);
  }

  weights.merger.fc1Weight = await loadTensor('model.visual.merger.linear_fc1.weight', false, false);
  weights.merger.fc1Bias = await loadTensorSilent(loadTensor, 'model.visual.merger.linear_fc1.bias');
  weights.merger.fc2Weight = await loadTensor('model.visual.merger.linear_fc2.weight', false, false);
  weights.merger.fc2Bias = await loadTensorSilent(loadTensor, 'model.visual.merger.linear_fc2.bias');
  weights.merger.normWeight = await loadTensorSilent(loadTensor, 'model.visual.merger.norm.weight');
  weights.merger.normBias = await loadTensorSilent(loadTensor, 'model.visual.merger.norm.bias');

  for (let i = 0; i < deepstackIndexes.length; i++) {
    const prefix = `model.visual.deepstack_merger_list.${i}`;
    const merger = {
      fc1Weight: await loadTensor(`${prefix}.linear_fc1.weight`, false, false),
      fc1Bias: await loadTensorSilent(loadTensor, `${prefix}.linear_fc1.bias`),
      fc2Weight: await loadTensor(`${prefix}.linear_fc2.weight`, false, false),
      fc2Bias: await loadTensorSilent(loadTensor, `${prefix}.linear_fc2.bias`),
      normWeight: await loadTensorSilent(loadTensor, `${prefix}.norm.weight`),
      normBias: await loadTensorSilent(loadTensor, `${prefix}.norm.bias`),
    };
    weights.deepstackMergers.push(merger);
  }

  const totalBlocks = weights.blocks.length;
  const totalMergers = weights.deepstackMergers.length;
  log.info('Vision', `Loaded ${totalBlocks} ViT blocks, ${totalMergers} deepstack mergers`);

  return weights;
}

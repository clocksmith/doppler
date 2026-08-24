import {
  normalizeDiffusionLocationDtype,
  normalizeDiffusionMatmulLocationDtype,
} from '../weight-contract.js';

export function resolveSD3EmbeddingDtype(weightDtype, locationDtype, runtime) {
  if (weightDtype) return weightDtype;
  const mapped = normalizeDiffusionLocationDtype(locationDtype);
  if (!mapped) return null;
  if (mapped !== 'f16') return mapped;
  return runtime?.loading?.allowF32UpcastNonMatmul !== false ? 'f32' : 'f16';
}

export function resolveSD3MatmulDtype(weightDtype, locationDtype) {
  if (weightDtype) return weightDtype;
  return normalizeDiffusionMatmulLocationDtype(locationDtype);
}

export function resolveSD3BiasDtype(weightDtype, locationDtype) {
  if (weightDtype) return weightDtype;
  return normalizeDiffusionLocationDtype(locationDtype) || 'f32';
}

export function resolveSD3LayerNormEps(config, runtime) {
  const modelEps = Number(config?.norm_eps ?? config?.layer_norm_eps);
  if (Number.isFinite(modelEps) && modelEps > 0) {
    return modelEps;
  }
  const runtimeEps = Number(runtime?.backend?.layerNormEps);
  if (Number.isFinite(runtimeEps) && runtimeEps > 0) {
    return runtimeEps;
  }
  throw new Error(
    'Diffusion transformer requires a positive layer norm epsilon from ' +
    'transformer.config.norm_eps (or layer_norm_eps) or runtime.inference.diffusion.backend.layerNormEps.'
  );
}

export function resolveSD3ModulationSegments(shape, hiddenSize, fallbackSegments, name) {
  const rows = Array.isArray(shape) ? shape[0] : null;
  if (Number.isFinite(rows) && Number.isFinite(hiddenSize) && hiddenSize > 0) {
    const segments = rows / hiddenSize;
    if (Number.isInteger(segments) && segments > 0) {
      return segments;
    }
    throw new Error(
      `Modulation segments mismatch for ${name || 'unknown'}: rows=${rows}, hidden=${hiddenSize}, ` +
      `expected an integer multiple instead of falling back to ${fallbackSegments}.`
    );
  }
  throw new Error(
    `Modulation tensor "${name || 'unknown'}" is missing shape metadata. ` +
    `Runtime cannot fall back to ${fallbackSegments} segments.`
  );
}

export function resolveSD3ModulationOffsets(segments, hiddenSize) {
  if (segments === 9) {
    return Object.freeze({
      attn: Object.freeze({ scale: 0, shift: hiddenSize, gate: hiddenSize * 2 }),
      attn2: Object.freeze({ scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 }),
      ff: Object.freeze({ scale: hiddenSize * 6, shift: hiddenSize * 7, gate: hiddenSize * 8 }),
    });
  }
  if (segments === 6) {
    const attn = Object.freeze({ scale: 0, shift: hiddenSize, gate: hiddenSize * 2 });
    return Object.freeze({
      attn,
      attn2: attn,
      ff: Object.freeze({ scale: hiddenSize * 3, shift: hiddenSize * 4, gate: hiddenSize * 5 }),
    });
  }
  throw new Error(`Unsupported modulation segments=${segments} (expected 6 or 9).`);
}

export function createSD3TransformerPlan(config, runtime, latentShape) {
  const hiddenSize = config.num_attention_heads * config.attention_head_dim;
  const patchSize = config.patch_size;
  const latentChannels = latentShape[0];
  const latentHeight = latentShape[1];
  const latentWidth = latentShape[2];
  const gridHeight = Math.floor(latentHeight / patchSize);
  const gridWidth = Math.floor(latentWidth / patchSize);
  const dualAttentionLayers = Object.freeze([...(config.dual_attention_layers || [])]);
  const attn2Layers = Array.isArray(config.attn2_layers)
    ? Object.freeze([...config.attn2_layers])
    : null;
  return Object.freeze({
    hiddenSize,
    numHeads: config.num_attention_heads,
    headDim: config.attention_head_dim,
    patchSize,
    layerNormEps: resolveSD3LayerNormEps(config, runtime),
    latentChannels,
    latentHeight,
    latentWidth,
    gridHeight,
    gridWidth,
    tokenCount: gridHeight * gridWidth,
    numLayers: config.num_layers,
    dualAttentionLayers,
    attn2Layers,
  });
}

export function createSD3PositionPlan(gridHeight, gridWidth, maxTokens) {
  const maxGrid = Math.floor(Math.sqrt(maxTokens));
  const square = maxGrid * maxGrid === maxTokens;
  const indices = [];
  for (let y = 0; y < gridHeight; y++) {
    const srcY = maxGrid * (y / Math.max(1, gridHeight));
    const srcYIdx = Math.min(maxGrid - 1, Math.floor(srcY));
    for (let x = 0; x < gridWidth; x++) {
      const srcX = maxGrid * (x / Math.max(1, gridWidth));
      const srcXIdx = Math.min(maxGrid - 1, Math.floor(srcX));
      indices.push(srcYIdx * maxGrid + srcXIdx);
    }
  }
  return Object.freeze({
    maxTokens,
    maxGrid,
    square,
    indices: Object.freeze(indices),
  });
}

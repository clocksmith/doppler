


export function crossProduct(dimensions) {
  const keys = Object.keys(dimensions);
  if (keys.length === 0) return [{}];

  const [firstKey, ...restKeys] = keys;
  const firstValues = dimensions[firstKey];

  if (restKeys.length === 0) {
    return firstValues.map((v) => ({ [firstKey]: v }));
  }

  const restDimensions = {};
  for (const k of restKeys) restDimensions[k] = dimensions[k];
  const restProduct = crossProduct(restDimensions);

  const result = [];
  for (const v of firstValues) {
    for (const rest of restProduct) {
      result.push({ [firstKey]: v, ...rest });
    }
  }
  return result;
}


export function filterConfigs(configs, predicate) {
  return configs.filter(predicate);
}


export function configName(config) {
  return Object.entries(config)
    .map(([k, v]) => `${k}=${v}`)
    .join(', ');
}


export function runMatrix(testFn, prefix, configs, runner) {
  for (const config of configs) {
    testFn(`${prefix} [${configName(config)}]`, () => runner(config));
  }
}

// ============================================================================
// Pre-built Test Dimensions
// ============================================================================


export const ATTENTION_DIMS = {
  headDim: [32, 64, 128, 256],
  seqLen: [1, 8, 64, 256],
  kvLen: [64, 256, 1024, 4096],
  numHeads: [4, 8, 32],
  numKVHeads: [1, 2, 4, 8], // MQA, GQA ratios
  kvDtype: ['f32', 'f16'],
};


export const ATTENTION_TIER_CONFIGS = [
  // Subgroup tier (decode, small headDim)
  { headDim: 64, seqLen: 1, kvLen: 256, numHeads: 8, numKVHeads: 8, kvDtype: 'f32', expectedTier: 'subgroup' },
  // Tiled large tier (prefill, small headDim)
  { headDim: 64, seqLen: 64, kvLen: 64, numHeads: 8, numKVHeads: 8, kvDtype: 'f32', expectedTier: 'tiled_large' },
  // Tiled small tier (larger headDim)
  { headDim: 128, seqLen: 64, kvLen: 64, numHeads: 8, numKVHeads: 8, kvDtype: 'f32', expectedTier: 'tiled_small' },
  // Streaming tier (decode, f16 kv)
  { headDim: 64, seqLen: 1, kvLen: 256, numHeads: 8, numKVHeads: 8, kvDtype: 'f16', expectedTier: 'streaming' },
  // Chunked f16kv (decode, f16 kv, bounded kvLen)
  { headDim: 128, seqLen: 1, kvLen: 1024, numHeads: 8, numKVHeads: 8, kvDtype: 'f16', expectedTier: 'streaming' },
  // GQA configurations
  { headDim: 64, seqLen: 1, kvLen: 256, numHeads: 32, numKVHeads: 8, kvDtype: 'f32', expectedTier: 'subgroup' },
  { headDim: 64, seqLen: 1, kvLen: 256, numHeads: 8, numKVHeads: 1, kvDtype: 'f32', expectedTier: 'subgroup' }, // MQA
];


export const MATMUL_DIMS = {
  M: [1, 8, 64, 256],
  N: [256, 1024, 4096],
  K: [256, 1024, 4096],
  aDtype: ['f32', 'f16'],
  bDtype: ['f32', 'f16', 'q4k'],
  outputDtype: ['f32', 'f16'],
};


export const MATMUL_KERNEL_CONFIGS = [
  // F32 path
  { M: 1, N: 1024, K: 256, aDtype: 'f32', bDtype: 'f32', outputDtype: 'f32', expectedKernel: 'f32' },
  // F16 path
  { M: 1, N: 1024, K: 256, aDtype: 'f16', bDtype: 'f16', outputDtype: 'f16', expectedKernel: 'f16' },
  // Mixed precision (f16 weights, f32 activations)
  { M: 1, N: 1024, K: 256, aDtype: 'f32', bDtype: 'f16', outputDtype: 'f32', expectedKernel: 'f16w_f32a' },
  // Q4K fused (decode)
  { M: 1, N: 1024, K: 256, aDtype: 'f32', bDtype: 'q4k', outputDtype: 'f32', expectedKernel: 'q4_fused' },
  // Batched variants
  { M: 64, N: 1024, K: 256, aDtype: 'f32', bDtype: 'f32', outputDtype: 'f32', expectedKernel: 'f32' },
  { M: 64, N: 1024, K: 256, aDtype: 'f32', bDtype: 'q4k', outputDtype: 'f32', expectedKernel: 'q4_fused_batched' },
];


export const RMSNORM_DIMS = {
  batchSize: [1, 8, 64],
  hiddenSize: [256, 1024, 4096],
  dtype: ['f32', 'f16'],
  hasResidual: [true, false],
};


export const INFERENCE_MODE_DIMS = {
  batchSize: [1, 4],
  mode: ['prefill', 'decode'],
  useRecorder: [true, false],
};


export const QUANT_PATH_DIMS = {
  weightQuant: ['f32', 'f16', 'q4k', 'q8'],
  embeddingQuant: ['f32', 'f16'],
  kvDtype: ['f32', 'f16'],
  fusedQ4K: [true, false],
};


export const ARCH_DIMS = {
  arch: ['llama', 'gemma', 'gemma2', 'qwen2', 'phi3'],
  hasSandwichNorm: [true, false],
  hasSlidingWindow: [true, false],
  hasAttnSoftcap: [true, false],
};

// ============================================================================
// Config Validators
// ============================================================================


export function isValidMatmulConfig(config) {
  // Q4K only with f32 activations
  if (config.bDtype === 'q4k' && config.aDtype !== 'f32') return false;
  // F16 output only with F16 inputs
  if (config.outputDtype === 'f16' && config.aDtype !== 'f16') return false;
  return true;
}


export function isValidAttentionConfig(config) {
  // numKVHeads must divide numHeads evenly
  if (config.numHeads % config.numKVHeads !== 0) return false;
  // kvLen must be >= seqLen for decode (seqLen=1)
  if (config.seqLen === 1 && config.kvLen < 1) return false;
  return true;
}


export function isValidQuantConfig(config) {
  // Fused Q4K only with q4k weights
  if (config.fusedQ4K && config.weightQuant !== 'q4k') return false;
  return true;
}

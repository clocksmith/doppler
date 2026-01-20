

// Re-export all public API from submodules
export {
  // Types and utilities
  isMoELayerLocal,
  hasLoggedFusedDownNorm,
  setLoggedFusedDownNorm,
  // Sandwich norm FFN (pre+post FFN normalization)
  processFFNWithSandwichNorm,
  // Standard FFN (single post-attention norm)
  processFFNStandard,
  // Dense FFN operations
  runDenseFFNGPU,
  runDenseFFNWithFusedPostNormGPU,
  // MoE FFN operations
  runMoEFFNGPU,
} from './ffn/index.js';

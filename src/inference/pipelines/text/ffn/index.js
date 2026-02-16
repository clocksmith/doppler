

// Types and utilities
export { isMoELayerLocal, hasLoggedFusedDownNorm, setLoggedFusedDownNorm } from './types.js';

// Sandwich norm FFN (pre+post FFN normalization)
export { processFFNWithSandwichNorm } from './sandwich.js';

// Standard FFN (single post-attention norm)
export { processFFNStandard } from './standard.js';

// Dense FFN operations
export { runDenseFFNGPU, runDenseFFNWithFusedPostNormGPU } from './dense.js';

// MoE FFN operations
export { runMoEFFNGPU } from './moe.js';

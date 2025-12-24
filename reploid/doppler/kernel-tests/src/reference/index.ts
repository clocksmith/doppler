/**
 * Reference Implementation Exports
 */

// Matrix operations
export { matmulRef, batchMatmulRef, matvecRef } from './matmul.js';

// Activation functions
export { softmaxRef, logSoftmaxRef, softmaxInplaceRef } from './softmax.js';
export { siluRef, siluGatedRef, siluFusedRef, siluInplaceRef } from './silu.js';

// Normalization
export { rmsNormRef, rmsNormNoWeightRef } from './rmsnorm.js';

// Position embeddings
export { ropeRef, ropeInterleavedRef, computeRopeFreqs, type RopeFrequencies } from './rope.js';

// Attention
export { attentionRef, createCausalMask, flashAttentionRef, mqaRef } from './attention.js';

// MoE operations
export { topkRef, softmaxTopkRef, type TopKResult } from './topk.js';
export { scatterAddRef, scatterAddAccumulateRef } from './scatter-add.js';
export {
  moeGatherRef,
  moeComputeAssignmentsRef,
  type MoeGatherResult,
  type MoeAssignmentResult,
} from './moe-gather.js';

// Memory operations
export { gatherRef, batchGatherRef, gatherWithPosRef } from './gather.js';
export { residualAddRef, residualAddInplaceRef, scaledResidualAddRef } from './residual.js';

// Quantization
export {
  float32ToFloat16,
  dequantInt8Ref,
  dequantInt4Ref,
  dequantQ4_0Ref,
  quantizeQ4_KRef,
  quantizeQ4_KBlockRef,
  dequantQ4_KRef,
  dequantizeQ4_KBlockRef,
} from './dequant.js';

// Sampling
export {
  argmaxRef,
  topkArgmaxRef,
  softmaxWithTemp,
  sampleTopKRef,
  seededRandom,
} from './sample.js';

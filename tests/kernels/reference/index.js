

// Matrix operations
export { matmulRef, batchMatmulRef, matvecRef } from './matmul.js';

// Activation functions
export { softmaxRef, logSoftmaxRef, softmaxInplaceRef } from './softmax.js';
export { siluRef, siluGatedRef, siluFusedRef, siluInplaceRef } from './silu.js';
export { geluRef, geluFastRef, gegluRef } from './gelu.js';

// Normalization
export { rmsNormRef, rmsNormNoWeightRef } from './rmsnorm.js';

// Position embeddings
export { ropeRef, ropeInterleavedRef, computeRopeFreqs } from './rope.js';

// Attention
export { attentionRef, attentionBackwardRef, createCausalMask, flashAttentionRef, mqaRef } from './attention.js';

// MoE operations
export { topkRef, softmaxTopkRef } from './topk.js';
export { scatterAddRef, scatterAddAccumulateRef } from './scatter-add.js';
export { moeGatherRef, moeComputeAssignmentsRef } from './moe-gather.js';

// Memory operations
export { gatherRef, batchGatherRef, gatherWithPosRef } from './gather.js';
export { residualAddRef, residualAddInplaceRef, scaledResidualAddRef } from './residual.js';
export { splitQkvRef, fuseQkvRef } from './split-qkv.js';

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

// Layer normalization
export { layerNormRef } from './layernorm.js';

// Group normalization
export { groupNormRef } from './groupnorm.js';

// Convolution
export { conv2dRef, conv2dBackwardRef } from './conv2d.js';

// Bias add
export { biasAddRef, biasAddBackwardRef } from './bias-add.js';

// Pixel shuffle
export { pixelShuffleRef } from './pixel-shuffle.js';

// Upsample
export { upsample2dRef, upsample2dBackwardRef } from './upsample2d.js';

// Modulate
export { modulateRef } from './modulate.js';

// Cross entropy
export { crossEntropyLossRef, crossEntropyBackwardRef } from './cross-entropy.js';

// Embed backward
export { embedBackwardRef } from './embed.js';

// Adam optimizer
export { adamRef } from './adam.js';

// Backward pass references
export {
  softmaxBackwardRef,
  siluBackwardRef,
  geluBackwardRef,
  scaleBackwardRef,
  ropeBackwardRef,
  rmsNormBackwardRef,
} from './backward.js';

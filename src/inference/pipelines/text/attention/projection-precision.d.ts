export { resolveProjectionMatmulDtype, resolveProjectionSliceOffsetBytes, shouldForceF32AttentionProjectionForRoPE, resolveAttentionProjectionOutputDtype } from './projections.js';
export function coerceProjectionInputTensor(
  recorder: import('../../../../gpu/command-recorder.js').CommandRecorder | null | undefined,
  tensor: import('../../../../gpu/tensor.js').Tensor,
  targetDtype: 'f16' | 'f32' | null | undefined
): Promise<import('../../../../gpu/tensor.js').Tensor>;
export function resolveProjectionInputDtype(
  options: Omit<Parameters<typeof import('./projections.js').resolveProjectionMatmulDtype>[0], 'precisionField'>
): ReturnType<typeof import('./projections.js').resolveProjectionMatmulDtype>;
export function resolveProjectionOutputDtype(
  options: Omit<Parameters<typeof import('./projections.js').resolveProjectionMatmulDtype>[0], 'precisionField'>
): ReturnType<typeof import('./projections.js').resolveProjectionMatmulDtype>;

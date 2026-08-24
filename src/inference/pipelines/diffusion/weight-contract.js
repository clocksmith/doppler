import { getKernelCapabilities } from '../../../gpu/device.js';
import { getBuffer } from '../../../gpu/weight-buffer.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

export function resolveDiffusionActivationDtype(runtime) {
  const caps = getKernelCapabilities();
  return selectRuleValue('inference', 'dtype', 'diffusionActivationDtype', {
    requested: runtime?.latent?.dtype,
    hasF16: caps.hasF16,
  });
}

export function expectDiffusionWeight(weight, label) {
  if (!weight) {
    throw new Error(`Missing diffusion weight: ${label}`);
  }
  return weight;
}

export function normalizeDiffusionLocationDtype(dtype) {
  if (!dtype) return null;
  const normalized = String(dtype).toLowerCase();
  if (normalized === 'f16' || normalized === 'float16') return 'f16';
  if (normalized === 'f32' || normalized === 'float32') return 'f32';
  if (normalized === 'bf16' || normalized === 'bfloat16') return 'f32';
  return null;
}

export function normalizeDiffusionMatmulLocationDtype(dtype) {
  if (!dtype) return null;
  const normalized = String(dtype).toLowerCase();
  if (normalized === 'f16' || normalized === 'float16') return 'f16';
  if (normalized === 'bf16' || normalized === 'bfloat16') return 'bf16';
  if (normalized === 'f32' || normalized === 'float32') return 'f32';
  if (normalized === 'q4_k' || normalized === 'q4_k_m') return 'q4k';
  return normalized;
}

export function inferDiffusionMatmulDtypeFromBuffer(weight, N, K, preferred) {
  const buffer = getBuffer(weight);
  if (!buffer || !Number.isFinite(N) || !Number.isFinite(K)) return preferred;
  if (preferred === 'q4k') return preferred;
  const expectedF16 = N * K * 2;
  const expectedF32 = N * K * 4;
  if (preferred === 'f32' && buffer.size < expectedF32 && buffer.size >= expectedF16) {
    return 'f16';
  }
  if (!preferred) {
    if (buffer.size >= expectedF32) return 'f32';
    if (buffer.size >= expectedF16) return 'f16';
  }
  return preferred;
}

import {
  KERNEL_CONFIGS,
  getKernelConfig,
} from '../../config/kernel-registry-contract.js';
import { validateAttentionLimits } from './feature-check.js';

export { KERNEL_CONFIGS, getKernelConfig };

export function setKernelValidator(operation, variant, validator) {
  const config = KERNEL_CONFIGS[operation]?.[variant];
  if (config) {
    config.validate = validator;
  }
}

const validatedAttentionVariants = [
  'prefill',
  'prefill_small',
  'decode_small',
  'prefill_streaming',
  'prefill_f16',
  'prefill_small_f16',
  'decode_small_f16',
  'prefill_streaming_f16',
  'prefill_f16kv',
  'prefill_small_f16kv',
  'decode_small_f16kv',
  'prefill_streaming_f16kv',
];

for (const variant of validatedAttentionVariants) {
  setKernelValidator('attention', variant, validateAttentionLimits);
}

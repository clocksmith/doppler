import {
  KERNEL_CONFIGS,
  createKernelRegistry as constructKernelRegistry,
  getActiveKernelRegistry,
  enterKernelRegistry,
} from '../../config/kernel-registry-contract.js';
import { validateAttentionLimits } from './feature-check.js';

export { KERNEL_CONFIGS, enterKernelRegistry };

const validatedAttentionVariants = [
  'prefill', 'prefill_small', 'decode_small', 'prefill_streaming',
  'prefill_f16', 'prefill_small_f16', 'decode_small_f16', 'prefill_streaming_f16',
  'prefill_f16kv', 'prefill_small_f16kv', 'decode_small_f16kv', 'prefill_streaming_f16kv',
];
const attentionValidators = Object.fromEntries(validatedAttentionVariants.map(variant => [variant,
  Object.freeze({
    id: 'doppler.attention-limits/v1',
    validate: ({ uniforms }) => validateAttentionLimits(uniforms.seqLen, uniforms.numHeads, uniforms.headDim),
  }),
]));

export function createKernelRegistry({ extensions = {}, validators = {} } = {}) {
  return constructKernelRegistry({ extensions, validators: {
    ...validators, attention: { ...attentionValidators, ...validators.attention },
  } });
}

export const DEFAULT_KERNEL_REGISTRY = createKernelRegistry();
let compatibilityRegistry = DEFAULT_KERNEL_REGISTRY;
let compatibilityValidatorRevision = 0;
export function getDefaultKernelRegistry() { return compatibilityRegistry; }
export function getKernelRegistry() { return getActiveKernelRegistry() ?? compatibilityRegistry; }

export function getKernelConfig(operation, variant) {
  return getKernelRegistry().getKernelConfig(operation, variant);
}
export function getKernelValidator(operation, variant) {
  return getKernelRegistry().getKernelValidator(operation, variant);
}

// Existing instances keep their companion maps; only future construction changes.
export function setKernelValidator(operation, variant, validator) {
  compatibilityRegistry.getKernelConfig(operation, variant);
  compatibilityRegistry = createKernelRegistry({ validators: {
    ...compatibilityRegistry.validators,
    [operation]: { ...compatibilityRegistry.validators[operation],
      [variant]: { id: `legacy:${operation}/${variant}:${++compatibilityValidatorRevision}`, validate: validator } },
  } });
}

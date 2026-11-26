import { canonicalizeLegacyEnumValue } from './legacy-enum-values.js';
import policy from './lora-layouts.json' with { type: 'json' };

export function resolveLoRAWeightLayout(name = policy.legacyLayout) {
  const layout = typeof name === 'string' && Object.hasOwn(policy.layouts, name) ? policy.layouts[name] : null;
  if (policy.schema !== 'doppler.lora-weight-layout-policy/v1' || !layout
    || typeof layout.transposeB !== 'boolean'
    || ![0, 1].includes(layout.aRankAxis) || ![0, 1].includes(layout.bRankAxis)
    || layout.identityLayout === undefined) {
    throw new Error(`Unsupported LoRA weight layout: ${String(name)}.`);
  }
  return Object.freeze({ name, ...layout });
}

export function resolveLoRAFormatLayout(format) {
  // Adapters and target plans signed before 0.2.0 declare peft_safetensors; read throughout 0.x.
  const canonical = canonicalizeLegacyEnumValue(format);
  if (typeof canonical !== 'string' || !Object.hasOwn(policy.formats, canonical)) throw new Error(`Unsupported LoRA weight format: ${String(format)}.`);
  return resolveLoRAWeightLayout(policy.formats[canonical]);
}



import {
  isWeightBuffer,
  isCpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import { LORA_MODULE_ALIASES } from '../inference/pipelines/text/lora.js';

// ============================================================================
// LoRA Tensor Name Parsing
// ============================================================================


export function parseLoRATensorName(name) {
  const match = name.match(/layers?\.?(\d+)\.(.+?)\.lora_([ab])/i);
  if (!match) return null;

  const layer = parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const moduleKey = rawModule.split('.').pop() ?? rawModule;
  const module = LORA_MODULE_ALIASES[moduleKey] ?? LORA_MODULE_ALIASES[rawModule];

  if (!module) return null;

  const kind =  (match[3].toLowerCase() === 'a' ? 'a' : 'b');
  return { layer, module, kind };
}

// ============================================================================
// Tensor Conversion
// ============================================================================


export function toFloat32(value) {
  if (value instanceof Float32Array) return value;
  if (value instanceof ArrayBuffer) return new Float32Array(value);

  if (value instanceof Uint8Array) {
    return new Float32Array(
      value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)
    );
  }

  if (isCpuWeightBuffer(value)) {
    return value.data;
  }

  // WeightBuffer: should not happen for LoRA loading (toGPU=false), but handle for type safety
  if (isWeightBuffer(value)) {
    throw new Error('LoRA tensor load returned WeightBuffer - expected CPU array');
  }

  throw new Error('LoRA tensor load returned unsupported buffer type');
}

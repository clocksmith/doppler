/**
 * LoRA Utilities - LoRA adapter parsing and conversion.
 *
 * Pure functions for parsing LoRA tensor names and converting tensor data.
 *
 * @module loader/lora-utils
 */

import {
  isWeightBuffer,
  isCpuWeightBuffer,
  type WeightBuffer,
  type CpuWeightBuffer,
} from '../gpu/weight-buffer.js';
import { LORA_MODULE_ALIASES, type LoRAModuleName } from '../inference/pipeline/lora.js';

// ============================================================================
// Types
// ============================================================================

export interface ParsedLoRATensorName {
  layer: number;
  module: LoRAModuleName;
  kind: 'a' | 'b';
}

// ============================================================================
// LoRA Tensor Name Parsing
// ============================================================================

/**
 * Parse a LoRA tensor name to extract layer, module, and A/B kind.
 *
 * Handles formats like:
 * - layers.0.self_attn.q_proj.lora_a
 * - layer0.attention.wq.lora_b
 *
 * @param name - Tensor name from LoRA adapter
 * @returns Parsed components, or null if not a valid LoRA tensor name
 */
export function parseLoRATensorName(name: string): ParsedLoRATensorName | null {
  const match = name.match(/layers?\.?(\d+)\.(.+?)\.lora_([ab])/i);
  if (!match) return null;

  const layer = parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const moduleKey = rawModule.split('.').pop() ?? rawModule;
  const module = LORA_MODULE_ALIASES[moduleKey] ?? LORA_MODULE_ALIASES[rawModule];

  if (!module) return null;

  const kind = match[3].toLowerCase() === 'a' ? 'a' : 'b';
  return { layer, module, kind };
}

// ============================================================================
// Tensor Conversion
// ============================================================================

/**
 * Convert various tensor buffer types to Float32Array.
 *
 * Used for LoRA weight loading where CPU arrays are expected.
 *
 * @param value - Tensor data in various formats
 * @returns Float32Array of tensor data
 * @throws If value is a GPU WeightBuffer (not supported for LoRA)
 */
export function toFloat32(
  value: GPUBuffer | Float32Array | Uint8Array | ArrayBuffer | WeightBuffer | CpuWeightBuffer
): Float32Array {
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

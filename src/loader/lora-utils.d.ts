/**
 * LoRA Utilities - LoRA adapter parsing and conversion.
 *
 * Pure functions for parsing LoRA tensor names and converting tensor data.
 *
 * @module loader/lora-utils
 */

import type { WeightBuffer, CpuWeightBuffer } from '../gpu/weight-buffer.js';
import type { LoRAModuleName } from '../inference/pipelines/text/lora.js';

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
export declare function parseLoRATensorName(name: string): ParsedLoRATensorName | null;

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
export declare function toFloat32(
  value: GPUBuffer | Float32Array | Uint8Array | ArrayBuffer | WeightBuffer | CpuWeightBuffer
): Float32Array;

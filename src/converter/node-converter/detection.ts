/**
 * Model format detection for the Node.js Model Converter.
 *
 * Detects whether input is GGUF, SafeTensors, or unknown format.
 * Also handles model type detection from architecture strings.
 *
 * @module converter/node-converter/detection
 */

import { stat, readdir } from 'fs/promises';
import type { InputFormat } from './types.js';
import {
  detectPreset,
  resolvePreset,
  type ModelType,
  type RawModelConfigSchema,
} from '../../config/index.js';

/**
 * Detect input format from path.
 *
 * @param inputPath - Path to input file or directory
 * @returns Detected format ('gguf', 'safetensors', or 'unknown')
 */
export async function detectInputFormat(inputPath: string): Promise<InputFormat> {
  const stats = await stat(inputPath);

  if (stats.isFile()) {
    if (inputPath.endsWith('.gguf')) {
      return 'gguf';
    }
    if (inputPath.endsWith('.safetensors')) {
      return 'safetensors';
    }
  }

  if (stats.isDirectory()) {
    const files = await readdir(inputPath);
    if (files.some(f => f.endsWith('.safetensors') || f.includes('model.safetensors.index.json'))) {
      return 'safetensors';
    }
    if (files.some(f => f.endsWith('.gguf'))) {
      return 'gguf';
    }
  }

  return 'unknown';
}

/**
 * Detect model type from architecture string and config.
 *
 * Uses config-as-code preset detection for model family identification.
 *
 * @param arch - Architecture string (e.g., 'gemma2', 'llama')
 * @param config - Model configuration object
 * @returns Object with presetId and modelType for grouping strategy
 * @throws Error if model family is unknown (no silent defaults)
 */
export function detectModelTypeFromPreset(
  arch: string,
  config: Record<string, unknown>
): { presetId: string; modelType: ModelType } {
  // Cast config to RawModelConfigSchema for preset detection
  const rawConfig = config as RawModelConfigSchema;

  // Use preset detection (config-as-code pattern)
  const presetId = detectPreset(rawConfig, arch);

  // Error on unknown model families - no silent defaults
  // Manifest-first architecture requires explicit inference config
  if (presetId === 'transformer') {
    const modelType = config.model_type ?? 'unknown';
    throw new Error(
      `Unknown model family: architecture="${arch}", model_type="${modelType}"\n\n` +
      `DOPPLER requires a known model preset to generate correct inference config.\n` +
      `The manifest-first architecture does not support generic defaults.\n\n` +
      `Options:\n` +
      `  1. Wait for official support of this model family\n` +
      `  2. Create a custom preset in src/config/presets/models/\n` +
      `  3. File an issue at https://github.com/clocksmith/doppler/issues\n\n` +
      `Supported model families: gemma2, gemma3, llama3, qwen3, mixtral, deepseek, mamba`
    );
  }

  const preset = resolvePreset(presetId);

  // Get modelType from preset, with fallback logic for MoE/hybrid
  let modelType: ModelType = preset.modelType || 'transformer';

  // Override for MoE if config indicates experts (presets may not cover all cases)
  if (config.num_local_experts || config.expertCount || config.num_experts) {
    if (config.n_shared_experts) {
      modelType = 'deepseek';
    } else if (arch.toLowerCase().includes('jamba')) {
      modelType = 'jamba';
    } else if (modelType === 'transformer') {
      modelType = 'mixtral'; // Generic MoE
    }
  }

  return { presetId, modelType };
}

/**
 * Check if a tensor name represents an embedding tensor.
 *
 * @param name - Tensor name
 * @returns True if this is an embedding tensor
 */
export function isEmbeddingTensorName(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    lower.includes('embed') ||
    lower.includes('tok_embeddings') ||
    lower.includes('token_embd')
  );
}

/**
 * Check if a tensor name represents an LM head tensor.
 *
 * @param name - Tensor name
 * @returns True if this is an LM head tensor
 */
export function isLmHeadTensorName(name: string): boolean {
  const lower = name.toLowerCase();
  if (lower.includes('lm_head')) return true;
  if (lower.endsWith('output.weight')) return true;
  return lower.includes('output') && lower.includes('weight') && !lower.includes('attn');
}

/**
 * Find the dtype of a tensor matching a given criteria.
 *
 * @param tensors - Array of tensors with name and dtype
 * @param matcher - Function to match tensor by name
 * @returns dtype string if found, null otherwise
 */
export function findTensorDtype(
  tensors: Array<{ name: string; dtype: string }>,
  matcher: (name: string) => boolean
): string | null {
  const match = tensors.find((t) => matcher(t.name));
  return match?.dtype ?? null;
}

/**
 * Model format detection for the Node.js Model Converter.
 *
 * Detects whether input is GGUF, SafeTensors, or unknown format.
 * Also handles model type detection from architecture strings.
 *
 * @module converter/node-converter/detection
 */

import { stat, readdir } from 'fs/promises';
import { detectPreset, resolvePreset } from '../../config/index.js';

/**
 * Detect input format from path.
 */
export async function detectInputFormat(inputPath) {
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
 */
export function detectModelTypeFromPreset(arch, config) {
  const rawConfig = config;

  const presetId = detectPreset(rawConfig, arch);

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

  let modelType = preset.modelType || 'transformer';

  if (config.num_local_experts || config.expertCount || config.num_experts) {
    if (config.n_shared_experts) {
      modelType = 'deepseek';
    } else if (arch.toLowerCase().includes('jamba')) {
      modelType = 'jamba';
    } else if (modelType === 'transformer') {
      modelType = 'mixtral';
    }
  }

  return { presetId, modelType };
}

/**
 * Check if a tensor name represents an embedding tensor.
 */
export function isEmbeddingTensorName(name) {
  const lower = name.toLowerCase();
  return (
    lower.includes('embed') ||
    lower.includes('tok_embeddings') ||
    lower.includes('token_embd')
  );
}

/**
 * Check if a tensor name represents an LM head tensor.
 */
export function isLmHeadTensorName(name) {
  const lower = name.toLowerCase();
  if (lower.includes('lm_head')) return true;
  if (lower.endsWith('output.weight')) return true;
  return lower.includes('output') && lower.includes('weight') && !lower.includes('attn');
}

/**
 * Find the dtype of a tensor matching a given criteria.
 */
export function findTensorDtype(tensors, matcher) {
  const match = tensors.find((t) => matcher(t.name));
  return match?.dtype ?? null;
}

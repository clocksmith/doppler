/**
 * Diffusion text encoder scaffold.
 *
 * @module inference/pipelines/diffusion/text-encoder
 */

import type { DiffusionModelConfig } from './types.js';
import type { BaseTokenizer } from '../../tokenizers/base.js';

export interface DiffusionPromptTokens {
  prompt: number[];
  negative: number[];
}

export interface DiffusionEncodedPrompts {
  tokens: Record<string, DiffusionPromptTokens>;
  totalTokens: number;
}

export declare function loadDiffusionTokenizers(
  diffusionConfig: DiffusionModelConfig,
  options?: { baseUrl?: string | null }
): Promise<Record<string, BaseTokenizer>>;

export declare function encodePrompt(
  prompts: { prompt: string; negativePrompt?: string },
  tokenizers: Record<string, BaseTokenizer>,
  options?: { maxLength?: number; maxLengthByTokenizer?: Record<string, number> }
): DiffusionEncodedPrompts;

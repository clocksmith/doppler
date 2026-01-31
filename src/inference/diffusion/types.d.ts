/**
 * Diffusion Pipeline Types (scaffold)
 *
 * @module inference/diffusion/types
 */

export interface DiffusionRequest {
  prompt: string;
  negativePrompt?: string;
  seed?: number;
  steps?: number;
  guidanceScale?: number;
  width?: number;
  height?: number;
}

export interface DiffusionResult {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}

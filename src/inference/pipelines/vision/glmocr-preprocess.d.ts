import type { GlmOcrPreprocessResult, GlmOcrVisionConfig } from './glmocr.js';

export declare function resolveGlmOcrImageSize(
  width: number,
  height: number,
  visionConfig: GlmOcrVisionConfig
): { targetWidth: number; targetHeight: number };

export declare function resizeGlmOcrImageBicubic(
  pixels: Uint8Array | Uint8ClampedArray | Float32Array,
  width: number,
  height: number,
  targetWidth: number,
  targetHeight: number
): Uint8Array;

export declare function preprocessGlmOcrImage(
  pixels: Uint8Array | Uint8ClampedArray | Float32Array,
  width: number,
  height: number,
  visionConfig: GlmOcrVisionConfig
): GlmOcrPreprocessResult;

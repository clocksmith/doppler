

import { trace } from '../../../debug/index.js';

/**
 * Preprocess an image for Qwen3-VL vision encoder.
 *
 * Accepts raw pixel data (Uint8Array RGBA or RGB, or Float32Array normalized)
 * and returns a GPU-ready Float32Array of shape [C, H, W] after:
 *   1. Resize to fit min/max pixel constraints
 *   2. Pad to patch-aligned dimensions
 *   3. Normalize with mean/std
 *   4. Extract temporal patches (for video; single frame for images)
 *
 * @param {Uint8Array|Float32Array} pixels   Raw pixel data (RGBA or RGB)
 * @param {number}                  width    Source image width
 * @param {number}                  height   Source image height
 * @param {object}                  config   Vision config from manifest or explicit config
 * @returns {{ data: Float32Array, gridThw: [number, number, number], patchedHeight: number, patchedWidth: number }}
 */
export function preprocessImage(pixels, width, height, config) {
  const {
    patchSize = 16,
    spatialMergeSize = 2,
    temporalPatchSize = 2,
    minPixels = 3136,
    maxPixels = 1003520,
    normalization = {},
  } = config;

  if (!Array.isArray(normalization.mean) || normalization.mean.length !== 3) {
    throw new Error('Vision config normalization.mean is required (array of 3 channel means)');
  }
  if (!Array.isArray(normalization.std) || normalization.std.length !== 3) {
    throw new Error('Vision config normalization.std is required (array of 3 channel stds)');
  }
  const mean = normalization.mean;
  const std = normalization.std;

  // Step 1: Compute target dimensions respecting pixel constraints and patch alignment.
  const mergedPatch = patchSize * spatialMergeSize;
  const { targetWidth, targetHeight } = computeTargetDimensions(
    width, height, minPixels, maxPixels, mergedPatch,
  );

  trace('vision', `preprocess: ${width}x${height} -> ${targetWidth}x${targetHeight} (patch=${patchSize}, merge=${spatialMergeSize})`);

  // Step 2: Resize to target dimensions (bilinear interpolation on CPU).
  const channels = 3;
  const resized = resizeBilinear(pixels, width, height, targetWidth, targetHeight, channels);

  // Step 3: Normalize to [0,1] then apply mean/std normalization.
  const normalized = new Float32Array(channels * targetHeight * targetWidth);
  for (let c = 0; c < channels; c++) {
    const m = mean[c];
    const s = std[c];
    for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
        const srcIdx = (y * targetWidth + x) * channels + c;
        const dstIdx = c * targetHeight * targetWidth + y * targetWidth + x;
        normalized[dstIdx] = (resized[srcIdx] / 255.0 - m) / s;
      }
    }
  }

  // Step 4: Compute grid dimensions for the LLM.
  //   gridT = 1 for single image (temporalPatchSize frames per temporal patch)
  //   gridH = targetHeight / patchSize
  //   gridW = targetWidth / patchSize
  const gridT = 1;
  const gridH = Math.floor(targetHeight / patchSize);
  const gridW = Math.floor(targetWidth / patchSize);

  return {
    data: normalized,
    width: targetWidth,
    height: targetHeight,
    channels,
    gridThw: [gridT, gridH, gridW],
    patchedHeight: targetHeight,
    patchedWidth: targetWidth,
  };
}

/**
 * Compute target dimensions that satisfy:
 *   - Total pixels >= minPixels and <= maxPixels
 *   - Both dimensions are multiples of mergedPatch
 *   - Aspect ratio is preserved as closely as possible
 */
function computeTargetDimensions(width, height, minPixels, maxPixels, mergedPatch) {
  const aspectRatio = width / height;

  // Start from the geometric mean of min/max pixel counts.
  let targetPixels = Math.sqrt(minPixels * maxPixels);
  targetPixels = Math.max(minPixels, Math.min(maxPixels, targetPixels));

  // Compute dimensions preserving aspect ratio.
  let h = Math.sqrt(targetPixels / aspectRatio);
  let w = h * aspectRatio;

  // Round to nearest mergedPatch multiple.
  h = Math.max(mergedPatch, Math.round(h / mergedPatch) * mergedPatch);
  w = Math.max(mergedPatch, Math.round(w / mergedPatch) * mergedPatch);

  // Clamp total pixels.
  if (h * w > maxPixels) {
    const scale = Math.sqrt(maxPixels / (h * w));
    h = Math.max(mergedPatch, Math.round((h * scale) / mergedPatch) * mergedPatch);
    w = Math.max(mergedPatch, Math.round((w * scale) / mergedPatch) * mergedPatch);
  }
  if (h * w < minPixels) {
    const scale = Math.sqrt(minPixels / (h * w));
    h = Math.max(mergedPatch, Math.round((h * scale) / mergedPatch) * mergedPatch);
    w = Math.max(mergedPatch, Math.round((w * scale) / mergedPatch) * mergedPatch);
  }

  return { targetWidth: w, targetHeight: h };
}

/**
 * Bilinear resize of interleaved RGB(A) pixel data.
 * Input: Uint8Array or Float32Array in [H, W, C] layout (C >= 3, only first 3 used).
 * Output: Float32Array in [H, W, 3] layout with values in [0, 255].
 */
function resizeBilinear(src, srcW, srcH, dstW, dstH, channels) {
  const srcChannels = src.length / (srcW * srcH);
  const out = new Float32Array(dstH * dstW * channels);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const srcY = y * scaleY;
    const y0 = Math.min(Math.floor(srcY), srcH - 1);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const fy = srcY - y0;

    for (let x = 0; x < dstW; x++) {
      const srcX = x * scaleX;
      const x0 = Math.min(Math.floor(srcX), srcW - 1);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const fx = srcX - x0;

      for (let c = 0; c < channels; c++) {
        const v00 = src[(y0 * srcW + x0) * srcChannels + c];
        const v01 = src[(y0 * srcW + x1) * srcChannels + c];
        const v10 = src[(y1 * srcW + x0) * srcChannels + c];
        const v11 = src[(y1 * srcW + x1) * srcChannels + c];
        const top = v00 + (v01 - v00) * fx;
        const bot = v10 + (v11 - v10) * fx;
        out[(y * dstW + x) * channels + c] = top + (bot - top) * fy;
      }
    }
  }

  return out;
}

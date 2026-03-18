/**
 * Qwen3-VL image preprocessing.
 *
 * Resizes an image to fit within min/maxPixels, normalizes pixel values,
 * and produces 3-D temporal patches suitable for the vision encoder.
 */

const DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073];
const DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711];

export function smartResize(height, width, minPixels, maxPixels, patchSize, spatialMergeSize) {
  const factor = patchSize * spatialMergeSize;
  if (height < factor || width < factor) {
    throw new Error(`Image too small: ${width}x${height}, minimum ${factor}x${factor}`);
  }

  let h = Math.round(height / factor) * factor;
  let w = Math.round(width / factor) * factor;

  if (h * w > maxPixels) {
    const ratio = Math.sqrt(maxPixels / (h * w));
    h = Math.round((h * ratio) / factor) * factor;
    w = Math.round((w * ratio) / factor) * factor;
  }
  if (h * w < minPixels) {
    const ratio = Math.sqrt(minPixels / (h * w));
    h = Math.round((h * ratio) / factor) * factor;
    w = Math.round((w * ratio) / factor) * factor;
  }

  return { height: Math.max(h, factor), width: Math.max(w, factor) };
}

export function extractNormalizedPixels(imageData, targetWidth, targetHeight, mean, std) {
  const m = mean ?? DEFAULT_MEAN;
  const s = std ?? DEFAULT_STD;
  const pixels = new Float32Array(3 * targetHeight * targetWidth);

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcIdx = (y * targetWidth + x) * 4;
      const dstBase = y * targetWidth + x;
      for (let c = 0; c < 3; c++) {
        const value = imageData[srcIdx + c] / 255.0;
        pixels[c * targetHeight * targetWidth + dstBase] = (value - m[c]) / s[c];
      }
    }
  }
  return pixels;
}

export function patchify(pixels, channels, height, width, temporalPatchSize, patchSize) {
  const gridH = height / patchSize;
  const gridW = width / patchSize;
  const patchDim = temporalPatchSize * patchSize * patchSize * channels;
  const numPatches = gridH * gridW;
  const patches = new Float32Array(numPatches * patchDim);

  for (let gh = 0; gh < gridH; gh++) {
    for (let gw = 0; gw < gridW; gw++) {
      const patchIdx = gh * gridW + gw;
      let offset = 0;
      for (let t = 0; t < temporalPatchSize; t++) {
        for (let c = 0; c < channels; c++) {
          for (let ph = 0; ph < patchSize; ph++) {
            for (let pw = 0; pw < patchSize; pw++) {
              const y = gh * patchSize + ph;
              const x = gw * patchSize + pw;
              patches[patchIdx * patchDim + offset] = pixels[c * height * width + y * width + x];
              offset++;
            }
          }
        }
      }
    }
  }
  return { patches, gridH, gridW, numPatches, patchDim };
}

export function preprocessImage(imageData, width, height, visionConfig) {
  const patchSize = visionConfig.patchSize ?? 16;
  const spatialMergeSize = visionConfig.spatialMergeSize ?? 2;
  const temporalPatchSize = visionConfig.temporalPatchSize ?? 2;
  const minPixels = visionConfig.minPixels ?? 3136;
  const maxPixels = visionConfig.maxPixels ?? 1003520;
  const mean = visionConfig.normalization?.mean ?? DEFAULT_MEAN;
  const std = visionConfig.normalization?.std ?? DEFAULT_STD;

  const target = smartResize(height, width, minPixels, maxPixels, patchSize, spatialMergeSize);
  const pixels = extractNormalizedPixels(imageData, target.width, target.height, mean, std);
  const result = patchify(pixels, 3, target.height, target.width, temporalPatchSize, patchSize);

  const mergedH = result.gridH / spatialMergeSize;
  const mergedW = result.gridW / spatialMergeSize;
  const numVisualTokens = Math.floor(mergedH) * Math.floor(mergedW);

  return {
    patches: result.patches,
    gridH: result.gridH,
    gridW: result.gridW,
    numPatches: result.numPatches,
    patchDim: result.patchDim,
    targetHeight: target.height,
    targetWidth: target.width,
    numVisualTokens,
  };
}

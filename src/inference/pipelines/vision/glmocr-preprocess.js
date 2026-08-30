function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`[Vision] GLM-OCR requires ${label} to be a positive integer, got ${value}.`);
  }
  return value;
}

function resolveSourceChannels(pixels, width, height) {
  const area = width * height;
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error(`[Vision] GLM-OCR received invalid image size ${width}x${height}.`);
  }
  const channels = pixels?.length / area;
  if (!Number.isInteger(channels) || (channels !== 1 && channels !== 3 && channels !== 4)) {
    throw new Error(
      `[Vision] GLM-OCR expects interleaved grayscale, RGB, or RGBA pixels; `
      + `got length=${pixels?.length ?? 'missing'} for ${width}x${height}.`
    );
  }
  return channels;
}

function readPixelUint8(pixels, channels, pixelIndex, channel) {
  const sourceChannel = channels === 1 ? 0 : channel;
  const value = Number(pixels[pixelIndex * channels + sourceChannel]);
  if (pixels instanceof Float32Array || pixels instanceof Float64Array) {
    const scaled = value <= 1 ? value * 255 : value;
    return Math.max(0, Math.min(255, Math.round(scaled)));
  }
  return Math.max(0, Math.min(255, Math.round(value)));
}

function cubicWeight(distance) {
  const a = -0.5;
  const x = Math.abs(distance);
  if (x < 1) {
    return ((a + 2) * x * x * x) - ((a + 3) * x * x) + 1;
  }
  if (x < 2) {
    return (a * x * x * x) - (5 * a * x * x) + (8 * a * x) - (4 * a);
  }
  return 0;
}

function createUint8ResizeAxis(inputSize, outputSize) {
  const scale = inputSize / outputSize;
  const support = scale >= 1 ? 2 * scale : 2;
  const maxInterpolationSize = Math.ceil(support) * 2 + 1;
  const entries = new Array(outputSize);
  let maximumWeight = 0;

  for (let outputIndex = 0; outputIndex < outputSize; outputIndex++) {
    const center = scale * (outputIndex + 0.5);
    const inverseScale = scale >= 1 ? 1 / scale : 1;
    const minimum = Math.max(Math.trunc(center - support + 0.5), 0);
    const size = Math.max(0, Math.min(
      Math.min(Math.trunc(center + support + 0.5), inputSize) - minimum,
      maxInterpolationSize
    ));
    const weights = new Float64Array(size);
    let totalWeight = 0;
    for (let index = 0; index < size; index++) {
      const weight = cubicWeight((index + minimum - center + 0.5) * inverseScale);
      weights[index] = weight;
      totalWeight += weight;
    }
    if (totalWeight !== 0) {
      for (let index = 0; index < size; index++) {
        weights[index] /= totalWeight;
        maximumWeight = Math.max(maximumWeight, weights[index]);
      }
    }
    entries[outputIndex] = { minimum, weights };
  }

  let precision = 0;
  for (; precision < 22; precision++) {
    const nextValue = Math.trunc(0.5 + maximumWeight * (2 ** (precision + 1)));
    if (nextValue >= (1 << 15)) break;
  }
  const coefficientScale = 2 ** precision;
  for (const entry of entries) {
    const coefficients = new Int16Array(entry.weights.length);
    for (let index = 0; index < entry.weights.length; index++) {
      const scaled = entry.weights[index] * coefficientScale;
      coefficients[index] = Math.trunc(scaled < 0 ? scaled - 0.5 : scaled + 0.5);
    }
    entry.coefficients = coefficients;
  }
  return { entries, precision };
}

function applyHorizontalUint8Resize(input, inputWidth, inputHeight, outputWidth) {
  if (inputWidth === outputWidth) return input;
  const { entries, precision } = createUint8ResizeAxis(inputWidth, outputWidth);
  const output = new Uint8Array(outputWidth * inputHeight * 3);
  const rounding = 1 << (precision - 1);
  for (let y = 0; y < inputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      const { minimum, coefficients } = entries[x];
      for (let channel = 0; channel < 3; channel++) {
        let value = rounding;
        for (let index = 0; index < coefficients.length; index++) {
          value += input[((y * inputWidth + minimum + index) * 3) + channel] * coefficients[index];
        }
        output[((y * outputWidth + x) * 3) + channel] = Math.max(
          0,
          Math.min(255, value >> precision)
        );
      }
    }
  }
  return output;
}

function applyVerticalUint8Resize(input, width, inputHeight, outputHeight) {
  if (inputHeight === outputHeight) return input;
  const { entries, precision } = createUint8ResizeAxis(inputHeight, outputHeight);
  const output = new Uint8Array(width * outputHeight * 3);
  const rounding = 1 << (precision - 1);
  for (let y = 0; y < outputHeight; y++) {
    const { minimum, coefficients } = entries[y];
    for (let x = 0; x < width; x++) {
      for (let channel = 0; channel < 3; channel++) {
        let value = rounding;
        for (let index = 0; index < coefficients.length; index++) {
          value += input[(((minimum + index) * width + x) * 3) + channel] * coefficients[index];
        }
        output[((y * width + x) * 3) + channel] = Math.max(
          0,
          Math.min(255, value >> precision)
        );
      }
    }
  }
  return output;
}

export function resizeGlmOcrImageBicubic(pixels, width, height, targetWidth, targetHeight) {
  const sourceChannels = resolveSourceChannels(pixels, width, height);
  requirePositiveInteger(targetWidth, 'targetWidth');
  requirePositiveInteger(targetHeight, 'targetHeight');
  const source = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let channel = 0; channel < 3; channel++) {
        source[((y * width + x) * 3) + channel] = readPixelUint8(
          pixels,
          sourceChannels,
          y * width + x,
          channel
        );
      }
    }
  }
  const horizontal = applyHorizontalUint8Resize(source, width, height, targetWidth);
  return applyVerticalUint8Resize(horizontal, targetWidth, height, targetHeight);
}

function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

export function resolveGlmOcrImageSize(width, height, visionConfig) {
  const patchSize = requirePositiveInteger(Number(visionConfig.patchSize), 'patchSize');
  const mergeSize = requirePositiveInteger(Number(visionConfig.spatialMergeSize), 'spatialMergeSize');
  const temporalPatchSize = requirePositiveInteger(
    Number(visionConfig.temporalPatchSize),
    'temporalPatchSize'
  );
  const minPixels = requirePositiveInteger(Number(visionConfig.minPixels), 'minPixels');
  const maxPixels = requirePositiveInteger(Number(visionConfig.maxPixels), 'maxPixels');
  const factor = patchSize * mergeSize;
  let resizedHeight = height;
  let resizedWidth = width;
  if (resizedHeight < factor || resizedWidth < factor) {
    const scale = Math.max(factor / resizedHeight, factor / resizedWidth);
    resizedHeight = Math.trunc(resizedHeight * scale);
    resizedWidth = Math.trunc(resizedWidth * scale);
  }
  const aspectRatio = Math.max(resizedHeight, resizedWidth) / Math.min(resizedHeight, resizedWidth);
  if (!Number.isFinite(aspectRatio) || aspectRatio > 200) {
    throw new Error(`[Vision] GLM-OCR rejects image aspect ratio ${aspectRatio}; maximum is 200.`);
  }

  let targetHeight = roundHalfEven(resizedHeight / factor) * factor;
  let targetWidth = roundHalfEven(resizedWidth / factor) * factor;
  const temporalExtent = temporalPatchSize;
  if (temporalExtent * targetHeight * targetWidth > maxPixels) {
    const beta = Math.sqrt((temporalExtent * resizedHeight * resizedWidth) / maxPixels);
    targetHeight = Math.max(factor, Math.floor(resizedHeight / beta / factor) * factor);
    targetWidth = Math.max(factor, Math.floor(resizedWidth / beta / factor) * factor);
  } else if (temporalExtent * targetHeight * targetWidth < minPixels) {
    const beta = Math.sqrt(minPixels / (temporalExtent * resizedHeight * resizedWidth));
    targetHeight = Math.ceil(resizedHeight * beta / factor) * factor;
    targetWidth = Math.ceil(resizedWidth * beta / factor) * factor;
  }
  return { targetWidth, targetHeight };
}

export function preprocessGlmOcrImage(pixels, width, height, visionConfig) {
  const patchSize = requirePositiveInteger(Number(visionConfig.patchSize), 'patchSize');
  const mergeSize = requirePositiveInteger(Number(visionConfig.spatialMergeSize), 'spatialMergeSize');
  const temporalPatchSize = requirePositiveInteger(
    Number(visionConfig.temporalPatchSize),
    'temporalPatchSize'
  );
  const inChannels = requirePositiveInteger(Number(visionConfig.inChannels), 'inChannels');
  if (inChannels !== 3) {
    throw new Error(`[Vision] GLM-OCR browser preprocessing requires inChannels=3, got ${inChannels}.`);
  }
  const mean = visionConfig.normalization?.mean;
  const std = visionConfig.normalization?.std;
  if (!Array.isArray(mean) || mean.length !== 3 || !Array.isArray(std) || std.length !== 3) {
    throw new Error('[Vision] GLM-OCR requires three-channel normalization mean and std arrays.');
  }
  if (std.some((value) => !Number.isFinite(Number(value)) || Number(value) <= 0)) {
    throw new Error('[Vision] GLM-OCR normalization std values must be positive numbers.');
  }

  const { targetWidth, targetHeight } = resolveGlmOcrImageSize(width, height, visionConfig);
  const resized = resizeGlmOcrImageBicubic(pixels, width, height, targetWidth, targetHeight);
  const gridHeight = targetHeight / patchSize;
  const gridWidth = targetWidth / patchSize;
  if (gridHeight % mergeSize !== 0 || gridWidth % mergeSize !== 0) {
    throw new Error(
      `[Vision] GLM-OCR resized grid ${gridWidth}x${gridHeight} is not divisible by mergeSize=${mergeSize}.`
    );
  }

  const patchDim = inChannels * temporalPatchSize * patchSize * patchSize;
  const numPatches = gridHeight * gridWidth;
  const patches = new Float32Array(numPatches * patchDim);
  let patchIndex = 0;
  for (let blockY = 0; blockY < gridHeight / mergeSize; blockY++) {
    for (let blockX = 0; blockX < gridWidth / mergeSize; blockX++) {
      for (let localBlockY = 0; localBlockY < mergeSize; localBlockY++) {
        for (let localBlockX = 0; localBlockX < mergeSize; localBlockX++) {
          const patchY = blockY * mergeSize + localBlockY;
          const patchX = blockX * mergeSize + localBlockX;
          let destination = patchIndex * patchDim;
          for (let channel = 0; channel < inChannels; channel++) {
            for (let temporal = 0; temporal < temporalPatchSize; temporal++) {
              for (let localY = 0; localY < patchSize; localY++) {
                for (let localX = 0; localX < patchSize; localX++) {
                  const sourceY = patchY * patchSize + localY;
                  const sourceX = patchX * patchSize + localX;
                  const source = resized[(sourceY * targetWidth + sourceX) * 3 + channel] / 255;
                  patches[destination++] = (source - Number(mean[channel])) / Number(std[channel]);
                }
              }
            }
          }
          patchIndex++;
        }
      }
    }
  }

  return {
    patches,
    patchDim,
    numPatches,
    gridHeight,
    gridWidth,
    targetWidth,
    targetHeight,
  };
}

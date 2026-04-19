

import { log } from '../../../debug/index.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { preprocessImage } from './image-preprocess.js';
import { patchEmbed } from './patch-embed.js';
import { runVisionEncoder } from './encoder.js';
import { encodeGemma4Image } from './gemma4.js';

/**
 * Encode an image through the vision pipeline.
 *
 * Routes to architecture-specific preprocessing based on visionConfig.visionArchitecture.
 *
 * Full flow:
 *   raw pixels -> preprocess -> patch embed -> ViT blocks -> spatial merge -> visual tokens
 *
 * @param {object} params
 * @param {Uint8Array|Float32Array} params.pixels   Raw image pixel data (RGBA or RGB)
 * @param {number}                  params.width    Image width
 * @param {number}                  params.height   Image height
 * @param {object}                  params.visionConfig  Vision config from manifest
 * @param {object}                  params.weights  Vision encoder weight buffers
 * @param {number}                  [params.softTokenBudget]  Per-request soft token budget override (Gemma 4 tiers: 70/140/280/560/1120)
 * @returns {Promise<VisionEncodeResult>}
 */
export async function encodeImage(params) {
  const { pixels, width, height, visionConfig, weights, softTokenBudget } = params;

  const arch = typeof visionConfig?.visionArchitecture === 'string'
    ? visionConfig.visionArchitecture.trim()
    : '';
  if (!arch) {
    throw new Error(
      'Vision encode requires visionConfig.visionArchitecture. ' +
      'Re-convert the model with explicit vision_config.vision_architecture.'
    );
  }
  log.debug('Vision', `encodeImage: ${width}x${height} input, arch=${arch}`);

  // Architecture-specific preprocessing dispatch
  let preprocessed;
  switch (arch) {
    case 'gemma4':
      return encodeGemma4Image(params);
    case 'qwen3vl':
      preprocessed = preprocessImage(pixels, width, height, visionConfig);
      break;
    default:
      throw new Error(
        `Unsupported vision architecture "${arch}". ` +
        'Supported: gemma4, qwen3vl. Check vision_config.vision_architecture.'
      );
  }

  // Step 2: Patch embedding — conv2d patches -> [numPatches, hiddenSize].
  const { patchBuffer, numPatches } = await patchEmbed({
    imageData: preprocessed.data,
    height: preprocessed.height,
    width: preprocessed.width,
    channels: preprocessed.channels,
    visionConfig,
    weights,
  });

  // Step 3: Vision encoder — ViT blocks + spatial merge.
  const { features, numTokens } = await runVisionEncoder({
    patchBuffer,
    numPatches,
    visionConfig,
    weights,
  });

  return {
    features,
    numTokens,
    gridThw: preprocessed.gridThw,
    imageWidth: preprocessed.width,
    imageHeight: preprocessed.height,
  };
}

/**
 * @typedef {object} VisionEncodeResult
 * @property {GPUBuffer}  features     Encoded visual tokens [numTokens, outHiddenSize]
 * @property {number}     numTokens    Number of visual tokens after spatial merge
 * @property {number[]}   gridThw      [temporal, height, width] grid dimensions
 * @property {number}     imageWidth   Processed image width
 * @property {number}     imageHeight  Processed image height
 */

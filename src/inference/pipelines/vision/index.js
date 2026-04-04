

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
 * Currently supported: 'qwen3vl' (default for backward compatibility).
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

  const arch = visionConfig.visionArchitecture ?? 'qwen3vl';
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
        'Supported: gemma4, qwen3vl. Check manifest.visionArchitecture or vision_config.vision_architecture.'
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
 * Inject visual tokens into text token embeddings.
 *
 * Replaces positions in the embedding sequence where image_token_id appears
 * with the encoded visual features from the vision encoder.
 *
 * For Qwen3-VL with DeepStack, visual tokens are injected at specific decoder
 * layers (deepstackVisualIndexes), not at the input embedding level.
 * This function handles the simpler input-level injection case.
 * DeepStack injection is handled in the decoder layer loop.
 *
 * @param {object} params
 * @param {Float32Array}  params.textEmbeddings    [seqLen, hiddenSize]
 * @param {Int32Array}    params.tokenIds          [seqLen]
 * @param {GPUBuffer}     params.visualFeatures    [numVisualTokens, outHiddenSize]
 * @param {number}        params.numVisualTokens   Number of visual tokens
 * @param {number}        params.imageTokenId      Token ID marking image positions
 * @param {number}        params.hiddenSize        Text model hidden size
 * @returns {{ mergedEmbeddings: Float32Array, mergedLength: number }}
 */
export function mergeVisualTokens(params) {
  const {
    textEmbeddings, tokenIds, visualFeatures,
    numVisualTokens, imageTokenId, hiddenSize,
  } = params;

  // Count image token positions.
  const imagePositions = [];
  for (let i = 0; i < tokenIds.length; i++) {
    if (tokenIds[i] === imageTokenId) {
      imagePositions.push(i);
    }
  }

  if (imagePositions.length === 0) {
    log.debug('Vision', 'mergeVisualTokens: no image tokens found, returning text-only');
    return { mergedEmbeddings: textEmbeddings, mergedLength: tokenIds.length };
  }

  log.debug('Vision', `mergeVisualTokens: replacing ${imagePositions.length} image tokens with ${numVisualTokens} visual tokens`);

  // The merged sequence replaces contiguous image_token_id runs with visual features.
  // For Qwen3-VL: image tokens appear as a block between vision_start and vision_end tokens.
  // The visual features replace the entire image token block.

  // Find contiguous image token ranges.
  const ranges = [];
  let rangeStart = imagePositions[0];
  let rangeEnd = imagePositions[0];
  for (let i = 1; i < imagePositions.length; i++) {
    if (imagePositions[i] === rangeEnd + 1) {
      rangeEnd = imagePositions[i];
    } else {
      ranges.push([rangeStart, rangeEnd]);
      rangeStart = imagePositions[i];
      rangeEnd = imagePositions[i];
    }
  }
  ranges.push([rangeStart, rangeEnd]);

  // Build merged sequence: text tokens (non-image) + visual tokens replacing each range.
  const textLen = tokenIds.length;
  const replacedCount = imagePositions.length;
  const mergedLength = textLen - replacedCount + numVisualTokens;
  const merged = new Float32Array(mergedLength * hiddenSize);

  let srcPos = 0;
  let dstPos = 0;
  let visualOffset = 0;

  for (const [start, end] of ranges) {
    // Copy text tokens before this range.
    const textBefore = start - srcPos;
    if (textBefore > 0) {
      merged.set(
        textEmbeddings.subarray(srcPos * hiddenSize, start * hiddenSize),
        dstPos * hiddenSize,
      );
      dstPos += textBefore;
    }

    // Insert visual tokens replacing this range.
    const rangeLen = end - start + 1;
    const tokensToInsert = Math.min(numVisualTokens - visualOffset, rangeLen);
    // Copy from visual features buffer (CPU side).
    for (let i = 0; i < tokensToInsert; i++) {
      for (let d = 0; d < hiddenSize; d++) {
        merged[(dstPos + i) * hiddenSize + d] = visualFeatures[(visualOffset + i) * hiddenSize + d];
      }
    }
    dstPos += tokensToInsert;
    visualOffset += tokensToInsert;
    srcPos = end + 1;
  }

  // Copy remaining text tokens after last range.
  if (srcPos < textLen) {
    merged.set(
      textEmbeddings.subarray(srcPos * hiddenSize, textLen * hiddenSize),
      dstPos * hiddenSize,
    );
    dstPos += textLen - srcPos;
  }

  return { mergedEmbeddings: merged, mergedLength: dstPos };
}

/**
 * @typedef {object} VisionEncodeResult
 * @property {GPUBuffer}  features     Encoded visual tokens [numTokens, outHiddenSize]
 * @property {number}     numTokens    Number of visual tokens after spatial merge
 * @property {number[]}   gridThw      [temporal, height, width] grid dimensions
 * @property {number}     imageWidth   Processed image width
 * @property {number}     imageHeight  Processed image height
 */

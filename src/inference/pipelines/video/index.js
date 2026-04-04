
import { log } from '../../../debug/index.js';
import { encodeImage } from '../vision/index.js';
import { sampleFrames } from './frame-sampler.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { getDevice } from '../../../gpu/device.js';

/**
 * Encode video frames through the vision pipeline.
 *
 * Samples frames uniformly, encodes each through the vision encoder,
 * and concatenates the visual token buffers.
 *
 * @param {object} params
 * @param {Array<{ pixels: Uint8Array|Float32Array, width: number, height: number }>} params.frames
 * @param {object}  params.visionConfig   Vision config from manifest
 * @param {object}  params.weights        Vision encoder weight buffers
 * @param {number}  [params.maxFrames=8]  Maximum frames to sample
 * @param {number}  [params.perFrameSoftTokenBudget]  Soft token budget per frame
 * @returns {Promise<VideoEncodeResult>}
 */
export async function encodeVideo(params) {
  const { frames, visionConfig, weights, maxFrames = 8, perFrameSoftTokenBudget } = params;

  const sampled = sampleFrames(frames, maxFrames);
  log.debug('Video', `encodeVideo: ${sampled.length} frames sampled from ${frames.length} total`);

  const frameResults = [];
  let totalTokens = 0;

  for (const frame of sampled) {
    const result = await encodeImage({
      pixels: frame.pixels,
      width: frame.width,
      height: frame.height,
      visionConfig,
      weights,
      softTokenBudget: perFrameSoftTokenBudget,
    });
    frameResults.push(result);
    totalTokens += result.numTokens;
  }

  if (frameResults.length === 0) {
    throw new Error('[Video] No frames were encoded.');
  }

  // Concatenate visual token buffers from all frames
  const outputDims = visionConfig.outHiddenSize ?? visionConfig.hiddenSize;
  const bytesPerToken = outputDims * Float32Array.BYTES_PER_ELEMENT;
  const totalBytes = totalTokens * bytesPerToken;
  const concatBuffer = acquireBuffer(totalBytes, undefined, 'video_concat_features');

  try {
    const device = getDevice();
    const encoder = device.createCommandEncoder();
    let offset = 0;
    for (const result of frameResults) {
      const frameBytes = result.numTokens * bytesPerToken;
      encoder.copyBufferToBuffer(result.features, 0, concatBuffer, offset, frameBytes);
      offset += frameBytes;
    }
    device.queue.submit([encoder.finish()]);
  } catch (error) {
    releaseBuffer(concatBuffer);
    throw error;
  } finally {
    for (const result of frameResults) {
      releaseBuffer(result.features);
    }
  }

  log.info('Video', `Video encoding complete: ${totalTokens} tokens from ${sampled.length} frames`);

  return {
    features: concatBuffer,
    numTokens: totalTokens,
    numFrames: sampled.length,
  };
}

/**
 * @typedef {object} VideoEncodeResult
 * @property {GPUBuffer}  features     Concatenated visual tokens [totalTokens, outputDims]
 * @property {number}     numTokens    Total visual tokens across all frames
 * @property {number}     numFrames    Number of frames actually encoded
 */

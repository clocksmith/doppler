import { log } from '../../../debug/index.js';
import { encodeImage } from '../vision/index.js';
import { sampleFrames } from './frame-sampler.js';
import { planVideoFeatureConcat } from './feature-concat.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';
import { getDevice } from '../../../gpu/device.js';

export async function encodeVideo(params) {
  const { frames, visionConfig, weights, maxFrames = 8, perFrameSoftTokenBudget } = params;

  const sampled = sampleFrames(frames, maxFrames);
  log.debug('Video', `encodeVideo: ${sampled.length} frames sampled from ${frames.length} total`);

  const frameResults = [];
  try {
    for (const frame of sampled) {
      frameResults.push(await encodeImage({
        pixels: frame.pixels,
        width: frame.width,
        height: frame.height,
        visionConfig,
        weights,
        softTokenBudget: perFrameSoftTokenBudget,
      }));
    }

    // Concatenate visual token buffers from all frames at the width the encoder produced.
    const plan = planVideoFeatureConcat(frameResults);
    const concatBuffer = acquireBuffer(plan.totalBytes, undefined, 'video_concat_features');
    try {
      const device = getDevice();
      const encoder = device.createCommandEncoder();
      plan.copies.forEach((copy, frame) => {
        encoder.copyBufferToBuffer(frameResults[frame].features, 0, concatBuffer, copy.offset, copy.byteLength);
      });
      device.queue.submit([encoder.finish()]);
    } catch (error) {
      releaseBuffer(concatBuffer);
      throw error;
    }

    log.info('Video', `Video encoding complete: ${plan.totalTokens} tokens from ${sampled.length} frames`);

    return {
      features: concatBuffer,
      numTokens: plan.totalTokens,
      numFrames: sampled.length,
      featureSize: plan.featureSize,
    };
  } finally {
    for (const result of frameResults) {
      releaseBuffer(result.features);
    }
  }
}

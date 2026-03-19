

import { trace } from '../../../debug/index.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';

/**
 * Patch embedding for the vision encoder.
 *
 * Qwen3-VL uses a 3D convolution for temporal+spatial patch extraction:
 *   Conv3D(in_channels=3, out_channels=hiddenSize, kernel=[temporalPatchSize, patchSize, patchSize])
 *
 * For single images (T=1), this reduces to a 2D convolution with stride=patchSize.
 * The output is [numPatches, hiddenSize] where numPatches = (H/patchSize) * (W/patchSize).
 *
 * For the initial implementation, this runs on CPU and uploads to GPU.
 * TODO(perf): GPU kernel for patch embedding (conv2d with large stride).
 *
 * @param {object} params
 * @param {Float32Array} params.imageData    Preprocessed image [C, H, W] normalized
 * @param {number}       params.height       Image height (patch-aligned)
 * @param {number}       params.width        Image width (patch-aligned)
 * @param {number}       params.channels     Number of channels (3)
 * @param {object}       params.visionConfig Vision config
 * @param {object}       params.weights      Vision encoder weight buffers
 * @returns {Promise<{ patchBuffer: GPUBuffer, numPatches: number }>}
 */
export async function patchEmbed(params) {
  const {
    imageData, height, width, channels,
    visionConfig, weights,
  } = params;

  const {
    patchSize = 16,
    hiddenSize = 1024,
    temporalPatchSize = 2,
  } = visionConfig;

  const gridH = Math.floor(height / patchSize);
  const gridW = Math.floor(width / patchSize);
  const numPatches = gridH * gridW;

  trace('vision', `patchEmbed: ${height}x${width} -> ${gridH}x${gridW} = ${numPatches} patches (${hiddenSize}d)`);

  // Read conv weight from GPU to CPU for the embedding computation.
  // Weight shape: [hiddenSize, channels * temporalPatchSize * patchSize * patchSize]
  // For single image: effectively [hiddenSize, channels * patchSize * patchSize]
  //
  // Qwen3-VL patch_embed is actually:
  //   proj = Conv3d(3, embed_dim, kernel_size=(tpp, pp, pp), stride=(tpp, pp, pp))
  // For T=1 frame, temporal dim collapses: input is [1, C, 1, H, W]
  // Output: [1, embed_dim, 1, H/pp, W/pp] -> reshape to [numPatches, embed_dim]

  const device = getDevice();
  const patchArea = channels * patchSize * patchSize;

  // Extract patches from image: each patch is [C, patchSize, patchSize] flattened.
  const patches = new Float32Array(numPatches * patchArea);
  for (let ph = 0; ph < gridH; ph++) {
    for (let pw = 0; pw < gridW; pw++) {
      const patchIdx = ph * gridW + pw;
      for (let c = 0; c < channels; c++) {
        for (let py = 0; py < patchSize; py++) {
          for (let px = 0; px < patchSize; px++) {
            const imgY = ph * patchSize + py;
            const imgX = pw * patchSize + px;
            const srcIdx = c * height * width + imgY * width + imgX;
            const dstIdx = patchIdx * patchArea + c * patchSize * patchSize + py * patchSize + px;
            patches[dstIdx] = imageData[srcIdx];
          }
        }
      }
    }
  }

  // Read the projection weight from GPU.
  // The weight tensor name is visual.patch_embed.proj.weight with shape [hiddenSize, C, tpp, pp, pp].
  // For temporal_patch_size=2 and a single frame, we need to handle the temporal dimension.
  // In practice for a single image, we sum over the temporal kernel dimension.
  const weightKey = 'visual.patch_embed.proj.weight';
  const biasKey = 'visual.patch_embed.proj.bias';

  const weightBuffer = weights[weightKey];
  const biasBuffer = weights[biasKey] || null;

  // Full conv weight size: hiddenSize * channels * temporalPatchSize * patchSize * patchSize
  const fullWeightSize = hiddenSize * channels * temporalPatchSize * patchSize * patchSize;
  const weightData = new Float32Array(fullWeightSize);
  {
    const staging = device.createBuffer({
      size: fullWeightSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(weightBuffer, 0, staging, 0, fullWeightSize * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    weightData.set(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    staging.destroy();
  }

  // For single frame: average over temporal kernel dimension to get [hiddenSize, C*pp*pp].
  const spatialWeight = new Float32Array(hiddenSize * patchArea);
  const spatialPatchArea = channels * patchSize * patchSize;
  for (let h = 0; h < hiddenSize; h++) {
    for (let s = 0; s < spatialPatchArea; s++) {
      let sum = 0;
      for (let t = 0; t < temporalPatchSize; t++) {
        sum += weightData[h * temporalPatchSize * spatialPatchArea + t * spatialPatchArea + s];
      }
      spatialWeight[h * spatialPatchArea + s] = sum;
    }
  }

  // Read bias if present.
  let biasData = null;
  if (biasBuffer) {
    biasData = new Float32Array(hiddenSize);
    const staging = device.createBuffer({
      size: hiddenSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(biasBuffer, 0, staging, 0, hiddenSize * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    biasData.set(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    staging.destroy();
  }

  // Compute patch embeddings: patches [numPatches, patchArea] @ spatialWeight^T [patchArea, hiddenSize]
  const embeddings = new Float32Array(numPatches * hiddenSize);
  for (let p = 0; p < numPatches; p++) {
    for (let h = 0; h < hiddenSize; h++) {
      let val = biasData ? biasData[h] : 0;
      for (let k = 0; k < patchArea; k++) {
        val += patches[p * patchArea + k] * spatialWeight[h * patchArea + k];
      }
      embeddings[p * hiddenSize + h] = val;
    }
  }

  // Upload to GPU.
  const patchBuffer = acquireBuffer(numPatches * hiddenSize * 4, 'vision-patch-embed');
  device.queue.writeBuffer(patchBuffer, 0, embeddings);

  return { patchBuffer, numPatches };
}

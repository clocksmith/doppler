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
export function patchEmbed(params: {
    imageData: Float32Array;
    height: number;
    width: number;
    channels: number;
    visionConfig: object;
    weights: object;
}): Promise<{
    patchBuffer: GPUBuffer;
    numPatches: number;
}>;

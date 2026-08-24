/**
 * Run the Qwen3-VL vision encoder on preprocessed image patches.
 *
 * Architecture:
 *   patch_embed (conv2d 3->hidden, stride=patchSize) -> [numPatches, hiddenSize]
 *   for each ViT block:
 *     x = layerNorm(x)
 *     x = x + selfAttention(x)    (no KV cache — full prefill attention)
 *     x = layerNorm(x)
 *     x = x + FFN(x)              (gelu activation)
 *   spatialMerge(x) -> [numMergedPatches, outHiddenSize]
 *
 * @param {object} params
 * @param {GPUBuffer}  params.patchBuffer    Preprocessed patches [numPatches, hiddenSize] on GPU
 * @param {number}     params.numPatches     Total number of patches
 * @param {object}     params.visionConfig   Vision config from manifest
 * @param {object}     params.weights        Vision encoder weight buffers keyed by tensor name
 * @param {object}     params.pipelineState  Shared pipeline state for buffer tracking
 * @returns {Promise<{ features: GPUBuffer, numTokens: number }>}
 */
export function runVisionEncoder(params: {
    patchBuffer: GPUBuffer;
    numPatches: number;
    visionConfig: object;
    weights: object;
    pipelineState: object;
}): Promise<{
    features: GPUBuffer;
    numTokens: number;
}>;

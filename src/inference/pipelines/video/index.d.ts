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
export function encodeVideo(params: {
    frames: Array<{
        pixels: Uint8Array | Float32Array;
        width: number;
        height: number;
    }>;
    visionConfig: object;
    weights: object;
    maxFrames?: number | undefined;
    perFrameSoftTokenBudget?: number | undefined;
}): Promise<VideoEncodeResult>;
export type VideoEncodeResult = {
    /**
     * Concatenated visual tokens [totalTokens, outputDims]
     */
    features: GPUBuffer;
    /**
     * Total visual tokens across all frames
     */
    numTokens: number;
    /**
     * Number of frames actually encoded
     */
    numFrames: number;
};

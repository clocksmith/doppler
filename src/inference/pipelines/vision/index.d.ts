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
export function encodeImage(params: {
    pixels: Uint8Array | Float32Array;
    width: number;
    height: number;
    visionConfig: object;
    weights: object;
    softTokenBudget?: number | undefined;
}): Promise<VisionEncodeResult>;
export type VisionEncodeResult = {
    /**
     * Encoded visual tokens [numTokens, outHiddenSize]
     */
    features: GPUBuffer;
    /**
     * Number of visual tokens after spatial merge
     */
    numTokens: number;
    /**
     * [temporal, height, width] grid dimensions
     */
    gridThw: number[];
    /**
     * Processed image width
     */
    imageWidth: number;
    /**
     * Processed image height
     */
    imageHeight: number;
};

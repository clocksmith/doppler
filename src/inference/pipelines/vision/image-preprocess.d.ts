/**
 * Preprocess an image for Qwen3-VL vision encoder.
 *
 * Accepts raw pixel data (Uint8Array RGBA or RGB, or Float32Array normalized)
 * and returns a GPU-ready Float32Array of shape [C, H, W] after:
 *   1. Resize to fit min/max pixel constraints
 *   2. Pad to patch-aligned dimensions
 *   3. Normalize with mean/std
 *   4. Extract temporal patches (for video; single frame for images)
 *
 * @param {Uint8Array|Float32Array} pixels   Raw pixel data (RGBA or RGB)
 * @param {number}                  width    Source image width
 * @param {number}                  height   Source image height
 * @param {object}                  config   Vision config from manifest or explicit config
 * @returns {{ data: Float32Array, gridThw: [number, number, number], patchedHeight: number, patchedWidth: number }}
 */
export function preprocessImage(pixels: Uint8Array | Float32Array, width: number, height: number, config: object): {
    data: Float32Array;
    gridThw: [number, number, number];
    patchedHeight: number;
    patchedWidth: number;
};

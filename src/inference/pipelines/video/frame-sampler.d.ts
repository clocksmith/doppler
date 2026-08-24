/**
 * Video frame sampling for Gemma 4 video processing.
 *
 * Gemma 4 processes video by sampling frames and running each through the vision encoder.
 * This module handles uniform frame selection from a decoded frame array.
 */
/**
 * Sample N frames uniformly from a video frame array.
 *
 * @param {Array<{ pixels: Uint8Array|Float32Array, width: number, height: number }>} frames
 * @param {number} maxFrames  Maximum number of frames to sample
 * @returns {Array<{ pixels: Uint8Array|Float32Array, width: number, height: number, frameIndex: number }>}
 */
export function sampleFrames(frames: Array<{
    pixels: Uint8Array | Float32Array;
    width: number;
    height: number;
}>, maxFrames: number): Array<{
    pixels: Uint8Array | Float32Array;
    width: number;
    height: number;
    frameIndex: number;
}>;

/**
 * Encode audio through the Gemma 4 conformer audio tower.
 *
 * @param {object} params
 * @param {Float32Array} params.melFeatures  Log-mel spectrogram [numFrames * nMels]
 * @param {number}       params.numFrames    Number of mel frames
 * @param {number}       params.nMels        Number of mel bands
 * @param {object}       params.audioConfig  Resolved audio encoder config
 * @param {object}       params.weights      Audio encoder weight buffers
 * @returns {Promise<{ features: GPUBuffer, numTokens: number }>}
 */
export function encodeGemma4Audio(params: {
    melFeatures: Float32Array;
    numFrames: number;
    nMels: number;
    audioConfig: object;
    weights: object;
}): Promise<{
    features: GPUBuffer;
    numTokens: number;
}>;

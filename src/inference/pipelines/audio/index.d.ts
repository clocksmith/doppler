/**
 * Encode audio through the audio pipeline.
 *
 * Routes to architecture-specific encoder based on audioConfig.audioArchitecture.
 *
 * @param {object} params
 * @param {Float32Array} params.melFeatures  Log-mel spectrogram [numFrames * nMels]
 * @param {number}       params.numFrames    Number of mel frames
 * @param {number}       params.nMels        Number of mel bands
 * @param {object}       params.audioConfig  Audio encoder config from manifest
 * @param {object}       params.weights      Audio encoder weight buffers
 * @returns {Promise<AudioEncodeResult>}
 */
export function encodeAudio(params: {
    melFeatures: Float32Array;
    numFrames: number;
    nMels: number;
    audioConfig: object;
    weights: object;
}): Promise<AudioEncodeResult>;
export type AudioEncodeResult = {
    /**
     * Encoded audio tokens [numTokens, outputDims]
     */
    features: GPUBuffer;
    /**
     * Number of audio tokens after subsampling
     */
    numTokens: number;
};

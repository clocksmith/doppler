/**
 * Extract log-mel spectrogram from raw audio PCM.
 *
 * @param {Float32Array} audio   Mono audio samples (16kHz expected)
 * @param {object}       [opts]
 * @param {number}       [opts.sampleRate=16000]
 * @param {number}       [opts.nFft=512]
 * @param {number}       [opts.hopLength=160]
 * @param {number}       [opts.nMels=80]
 * @param {number}       [opts.windowLength=400]
 * @returns {{ features: Float32Array, numFrames: number, nMels: number }}
 */
export function extractLogMelSpectrogram(audio: Float32Array, opts?: {
    sampleRate?: number | undefined;
    nFft?: number | undefined;
    hopLength?: number | undefined;
    nMels?: number | undefined;
    windowLength?: number | undefined;
}): {
    features: Float32Array;
    numFrames: number;
    nMels: number;
};

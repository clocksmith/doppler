/**
 * Log-mel front end for the Gemma 4 audio encoder (reference feature extractor port).
 *
 * @module inference/pipelines/audio/mel
 */

import type { AudioFeatureExtractorConfig } from '../text/config.js';

export declare function buildMelFilterBank(input: {
  featureSize: number;
  fftLength: number;
  samplingRate: number;
  minFrequency: number;
  maxFrequency: number;
}): { filters: Float32Array; numBins: number };

/**
 * Extract log-mel features from mono PCM samples using the manifest's
 * audio_config.feature_extractor parameters. There are no runtime defaults.
 */
export declare function extractLogMelSpectrogram(
  audio: Float32Array,
  featureExtractor: AudioFeatureExtractorConfig
): {
  features: Float32Array;
  numFrames: number;
  nMels: number;
};

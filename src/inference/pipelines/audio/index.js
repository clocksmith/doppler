
import { log } from '../../../debug/index.js';
import { encodeGemma4Audio } from './gemma4.js';

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
export async function encodeAudio(params) {
  const { audioConfig } = params;
  const arch = audioConfig?.audioArchitecture ?? null;

  log.debug('Audio', `encodeAudio: arch=${arch}`);

  switch (arch) {
    case 'gemma4':
      return encodeGemma4Audio(params);
    default:
      throw new Error(
        `Unsupported audio architecture "${arch}". ` +
        'Supported: gemma4. Check manifest audio_config.model_type.'
      );
  }
}

/**
 * @typedef {object} AudioEncodeResult
 * @property {GPUBuffer}  features     Encoded audio tokens [numTokens, outputDims]
 * @property {number}     numTokens    Number of audio tokens after subsampling
 */

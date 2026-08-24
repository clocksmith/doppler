
import { log } from '../../../debug/index.js';
import { encodeGemma4Audio } from '../../../experimental/audio/gemma4.js';

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
        'Supported: gemma4. Check audio_config.audio_architecture.'
      );
  }
}

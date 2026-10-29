
import { log } from '../../../debug/index.js';

export async function encodeAudio(params) {
  const { audioConfig } = params;
  const arch = audioConfig?.audioArchitecture ?? null;

  log.debug('Audio', `encodeAudio: arch=${arch}`);

  switch (arch) {
    case 'gemma4': {
      // The Gemma4 audio encoder is a quarantined prototype; load it only when
      // a manifest selects it rather than with every audio entrypoint import.
      const { encodeGemma4Audio } = await import('../../../experimental/audio/gemma4.js');
      return encodeGemma4Audio(params);
    }
    default:
      throw new Error(
        `Unsupported audio architecture "${arch}". ` +
        'Supported: gemma4. Check audio_config.audio_architecture.'
      );
  }
}

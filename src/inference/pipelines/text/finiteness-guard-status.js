export function parseFinitenessStatusWords(words, offset = 0) {
  const status = Number(words[offset] ?? 0);
  if (status <= 0) {
    return {
      triggered: false,
      layer: 0,
      step: 0,
      metadata: '',
    };
  }

  const layer = Number(words[offset + 1] ?? 0);
  const step = Number(words[offset + 2] ?? 0);

  return {
    triggered: true,
    layer,
    step,
    metadata: ` (layer ${layer}, step ${step})`,
  };
}

export const FINITENESS_RESET_WORDS = new Uint32Array(4);

export class FinitenessError extends Error {
  constructor(message) {
    super(message);
    this.name = 'FinitenessError';
  }
}

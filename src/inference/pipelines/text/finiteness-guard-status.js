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

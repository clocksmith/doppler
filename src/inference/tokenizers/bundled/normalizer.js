/** @param {unknown} spec
 * @returns {import('./normalizer.js').CompiledTokenizerNormalizer | null} */
export function compileTokenizerNormalizer(spec) {
  if (spec == null) return null;
  if (typeof spec !== 'object' || Array.isArray(spec)) {
    throw new Error('[Tokenizer] Invalid tokenizer normalizer declaration.');
  }
  const record = /** @type {Record<string, unknown>} */ (spec);
  if (record.type === 'Sequence') {
    if (!Array.isArray(record.normalizers)) {
      throw new Error('[Tokenizer] Sequence normalizer requires normalizers[].');
    }
    const steps = record.normalizers.map(value => {
      const normalize = compileTokenizerNormalizer(value);
      if (!normalize) throw new Error('[Tokenizer] Sequence normalizer contains an empty declaration.');
      return normalize;
    });
    return text => steps.reduce((value, normalize) => normalize(value), text);
  }
  if (record.type === 'Prepend') {
    if (typeof record.prepend !== 'string') {
      throw new Error('[Tokenizer] Prepend normalizer requires a string prepend.');
    }
    const prefix = record.prepend;
    return text => prefix + text;
  }
  if (record.type === 'Replace') {
    const pattern = record.pattern;
    if (!pattern || typeof pattern !== 'object' || Array.isArray(pattern)
      || !('String' in pattern) || typeof pattern.String !== 'string'
      || pattern.String.length === 0 || Object.keys(pattern).length !== 1
      || typeof record.content !== 'string') {
      throw new Error('[Tokenizer] Replace normalizer requires a non-empty literal String pattern and string content.');
    }
    const search = pattern.String;
    const replacement = record.content;
    return text => text.replaceAll(search, () => replacement);
  }
  if (record.type === 'NFC' || record.type === 'NFD' || record.type === 'NFKC' || record.type === 'NFKD') {
    const form = record.type;
    return text => text.normalize(form);
  }
  throw new Error(`[Tokenizer] Unsupported tokenizer normalizer: ${String(record.type)}.`);
}

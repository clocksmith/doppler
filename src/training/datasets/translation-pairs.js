function normalizePair(record, sourceKey, targetKey) {
  if (!record || typeof record !== 'object') return null;
  const source = String(record[sourceKey] ?? '').trim();
  const target = String(record[targetKey] ?? '').trim();
  if (!source || !target) return null;
  return { source, target };
}

export function mapTranslationPairs(records, options = {}) {
  const sourceKey = String(options.sourceKey || 'source');
  const targetKey = String(options.targetKey || 'target');
  if (!Array.isArray(records)) return [];
  const pairs = [];
  for (const record of records) {
    const pair = normalizePair(record, sourceKey, targetKey);
    if (pair) pairs.push(pair);
  }
  return pairs;
}

export async function tokenizeTranslationPairs(tokenizer, pairs, options = {}) {
  if (!tokenizer || typeof tokenizer.encode !== 'function') {
    throw new Error('tokenizeTranslationPairs requires a tokenizer with encode()');
  }
  const maxLength = Number.isFinite(options.maxLength)
    ? Math.max(2, Math.floor(options.maxLength))
    : null;
  const promptPrefix = String(options.promptPrefix || 'Translate to target: ');
  const separator = String(options.separator || '\n');

  const samples = [];
  for (const pair of pairs) {
    const source = String(pair?.source || '').trim();
    const target = String(pair?.target || '').trim();
    if (!source || !target) continue;
    const text = `${promptPrefix}${source}${separator}${target}`;
    const tokens = tokenizer.encode(text);
    const clipped = maxLength ? tokens.slice(0, maxLength) : tokens;
    if (clipped.length < 2) continue;
    samples.push({
      inputIds: clipped.slice(0, clipped.length - 1),
      targetIds: clipped.slice(1),
      source,
      target,
      text,
    });
  }
  return samples;
}

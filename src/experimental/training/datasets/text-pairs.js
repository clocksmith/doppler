
function concatTokens(tokens, maxLength) {
  if (!maxLength || tokens.length <= maxLength) {
    return tokens;
  }
  return tokens.slice(0, maxLength);
}

export function buildCausalPair(tokens) {
  if (tokens.length < 2) {
    return { inputIds: [], targetIds: [] };
  }
  return {
    inputIds: tokens.slice(0, tokens.length - 1),
    targetIds: tokens.slice(1),
  };
}

export async function tokenizeTextPairs(tokenizer, pairs, options = {}) {
  if (!tokenizer || typeof tokenizer.encode !== 'function') {
    throw new Error('tokenizeTextPairs requires a tokenizer with encode()');
  }

  const {
    maxLength = null,
    joinWith = '',
  } = options;

  const samples = [];
  for (const pair of pairs) {
    const prompt = pair.prompt ?? '';
    const completion = pair.completion ?? '';
    const fullText = `${prompt}${joinWith}${completion}`;
    const tokens = tokenizer.encode(fullText);
    const clipped = concatTokens(tokens, maxLength);
    const { inputIds, targetIds } = buildCausalPair(clipped);
    if (inputIds.length > 0) {
      samples.push({ inputIds, targetIds, text: fullText });
    }
  }
  return samples;
}

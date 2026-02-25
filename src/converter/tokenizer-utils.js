export function resolveEosTokenId({ config, tokenizer, tokenizerJson }) {
  const candidateSources = [
    tokenizer?.eosTokenId,
    tokenizer?.eos_token_id,
    tokenizerJson?.specialTokens?.eos,
    tokenizerJson?.specialTokens?.eos_token_id,
    tokenizerJson?.special_tokens?.eos,
    tokenizerJson?.special_tokens?.eos_token_id,
    config?.eos_token_id,
    config?.text_config?.eos_token_id,
    config?.eos_token_ids,
    config?.text_config?.eos_token_ids,
  ];

  for (const candidate of candidateSources) {
    const normalized = normalizeEosTokenId(candidate);
    if (normalized != null) return normalized;
  }

  const eosTokenStringCandidates = [
    tokenizer?.eosToken,
    tokenizer?.eos_token,
    tokenizerJson?.specialTokens?.eos_token,
    tokenizerJson?.special_tokens?.eos_token,
    config?.eos_token,
    config?.text_config?.eos_token,
  ];

  for (const candidate of eosTokenStringCandidates) {
    const tokenText = normalizeTokenText(candidate);
    if (!tokenText) continue;
    const resolvedFromConfigDecoder = resolveTokenIdFromAddedTokensDecoder(
      tokenizer?.added_tokens_decoder,
      tokenText
    );
    if (resolvedFromConfigDecoder != null) {
      return resolvedFromConfigDecoder;
    }
    const resolvedFromTokenizerJson = resolveTokenIdFromAddedTokensDecoder(
      tokenizerJson?.added_tokens_decoder,
      tokenText
    );
    if (resolvedFromTokenizerJson != null) {
      return resolvedFromTokenizerJson;
    }
  }

  throw new Error('Missing eos_token_id. Provide eos_token_id in config or tokenizer metadata.');
}

function normalizeEosTokenId(value) {
  if (Array.isArray(value)) {
    if (value.length === 0 || value.some((id) => typeof id !== 'number')) {
      return null;
    }
    return value;
  }
  if (typeof value === 'number') return value;
  return null;
}

function normalizeTokenText(value) {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed || null;
  }
  if (value && typeof value === 'object' && typeof value.content === 'string') {
    const trimmed = value.content.trim();
    return trimmed || null;
  }
  return null;
}

function resolveTokenIdFromAddedTokensDecoder(addedTokensDecoder, tokenText) {
  if (!addedTokensDecoder || typeof addedTokensDecoder !== 'object') {
    return null;
  }
  for (const [rawId, entry] of Object.entries(addedTokensDecoder)) {
    const id = Number(rawId);
    if (!Number.isInteger(id) || id < 0) continue;
    const content = normalizeTokenText(entry);
    if (content === tokenText) {
      return id;
    }
  }
  return null;
}

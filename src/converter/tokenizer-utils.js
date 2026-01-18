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

import { log } from '../debug/index.js';
import { getNestedTextConfig } from './artifact-identity.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';

export function resolveTokenizerId(value) {
  if (typeof value === 'number') return value;
  return null;
}

export function resolveTokenizerIds(value) {
  if (Array.isArray(value) && value.every((id) => typeof id === 'number')) {
    return value;
  }
  if (typeof value === 'number') return [value];
  return null;
}

export function resolveTokenizerField(tokenizerConfig, ...keys) {
  if (!tokenizerConfig) return null;
  for (const key of keys) {
    if (tokenizerConfig[key] != null) {
      return tokenizerConfig[key];
    }
  }
  return null;
}

export function resolveConfigBoolean(rawConfig, ...keys) {
  // Same lookup logic as resolveTokenizerField: return the first non-nullish
  // value from the given keys. Delegates to avoid duplicating the pattern.
  return resolveTokenizerField(rawConfig, ...keys);
}

export function resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture) {
  const nestedTextConfig = getNestedTextConfig(rawConfig);
  const configVocab = rawConfig?.vocab_size ?? nestedTextConfig?.vocab_size;
  const tokenizerVocab = tokenizerConfig?.vocab_size ?? tokenizerConfig?.vocabSize;
  const archVocab = architecture?.vocabSize;

  // Warn if multiple sources provide vocab size and they disagree
  const sources = [
    tokenizerVocab != null ? { label: 'tokenizer', value: tokenizerVocab } : null,
    configVocab != null ? { label: 'config', value: configVocab } : null,
    archVocab != null ? { label: 'architecture', value: archVocab } : null,
  ].filter(Boolean);
  if (sources.length > 1) {
    const distinct = new Set(sources.map((s) => s.value));
    if (distinct.size > 1) {
      const detail = sources.map((s) => `${s.label}=${s.value}`).join(', ');
      log.error(
        'Convert',
        `Vocab size sources disagree: ${detail}. Using first available (${sources[0].label}=${sources[0].value}). ` +
        'This may cause embedding size mismatches at runtime. Verify the correct vocab size in the conversion config.'
      );
    }
  }

  return tokenizerVocab ?? configVocab ?? archVocab ?? null;
}

export function resolveConfigTokenId(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = getNestedTextConfig(rawConfig)?.[key];
  return resolveTokenizerId(direct ?? nested);
}

export function resolveConfigTokenIds(rawConfig, key) {
  const direct = rawConfig?.[key];
  const nested = getNestedTextConfig(rawConfig)?.[key];
  return resolveTokenizerIds(direct ?? nested);
}

export function buildSentencepieceTokenizer(tokenizerConfig, rawConfig, architecture, modelTokenizerModel) {
  if (!modelTokenizerModel) return null;

  const vocabSize = resolveTokenizerVocabSize(tokenizerConfig, rawConfig, architecture);
  const sentencepieceModel = typeof modelTokenizerModel === 'string'
    ? modelTokenizerModel
    : modelTokenizerModel?.file ?? 'tokenizer.model';

  const bosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'bos_token_id', 'bosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'bos_token_id')
  );
  const eosTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'eos_token_id', 'eosTokenId')
    ?? resolveConfigTokenId(rawConfig, 'eos_token_id')
  );
  const eosTokens = resolveTokenizerIds(
    resolveTokenizerField(tokenizerConfig, 'eos_token_ids', 'eosTokens', 'eos_token_id')
    ?? resolveConfigTokenIds(rawConfig, 'eos_token_ids')
  );
  const padTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'pad_token_id', 'padTokenId')
    ?? resolveConfigTokenId(rawConfig, 'pad_token_id')
  );
  const unkTokenId = resolveTokenizerId(
    resolveTokenizerField(tokenizerConfig, 'unk_token_id', 'unkTokenId')
    ?? resolveConfigTokenId(rawConfig, 'unk_token_id')
  );
  const addBosToken = resolveTokenizerField(tokenizerConfig, 'add_bos_token', 'addBosToken');
  const addEosToken = resolveTokenizerField(tokenizerConfig, 'add_eos_token', 'addEosToken');

  const tokenizer = {
    type: 'sentencepiece',
    sentencepieceModel,
    vocabSize: vocabSize ?? 0,
  };

  if (bosTokenId != null) tokenizer.bosTokenId = bosTokenId;
  if (eosTokenId != null) tokenizer.eosTokenId = eosTokenId;
  if (eosTokens) tokenizer.eosTokens = eosTokens;
  if (padTokenId != null) tokenizer.padTokenId = padTokenId;
  if (unkTokenId != null) tokenizer.unkTokenId = unkTokenId;
  if (addBosToken != null) tokenizer.addBosToken = addBosToken;
  if (addEosToken != null) tokenizer.addEosToken = addEosToken;

  return tokenizer;
}

export function resolveBundledTokenizerVocabSize(tokenizerJson) {
  const vocab = tokenizerJson?.model?.vocab;
  let maxTokenId = -1;
  if (Array.isArray(vocab)) {
    maxTokenId = vocab.length - 1;
  }
  if (vocab && typeof vocab === 'object') {
    maxTokenId = Object.keys(vocab).length - 1;
  }
  for (const token of tokenizerJson?.added_tokens ?? []) {
    if (Number.isInteger(token?.id) && token.id >= 0) {
      maxTokenId = Math.max(maxTokenId, token.id);
    }
  }
  return maxTokenId + 1;
}

function normalizeTokenizerTokenText(value) {
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object' && typeof value.content === 'string') {
    return value.content;
  }
  return null;
}

function resolveBundledTokenizerTokenTextId(tokenizerJson, value) {
  const tokenText = normalizeTokenizerTokenText(value);
  if (tokenText == null) return null;
  const vocabId = tokenizerJson?.model?.vocab?.[tokenText];
  if (Number.isInteger(vocabId) && vocabId >= 0) return vocabId;
  const addedToken = tokenizerJson?.added_tokens?.find(
    (entry) => normalizeTokenizerTokenText(entry) === tokenText
  );
  return Number.isInteger(addedToken?.id) && addedToken.id >= 0 ? addedToken.id : null;
}

function resolveBundledTokenizerSpecialId({
  tokenizerJson,
  tokenizerConfig,
  rawConfig,
  generationConfig,
  name,
}) {
  const tokenIdKey = `${name}_token_id`;
  const camelTokenIdKey = `${name}TokenId`;
  const directCandidates = [
    resolveTokenizerField(generationConfig, tokenIdKey, camelTokenIdKey),
    resolveTokenizerField(tokenizerConfig, tokenIdKey, camelTokenIdKey),
    resolveTokenizerField(tokenizerJson, tokenIdKey, camelTokenIdKey),
    resolveConfigTokenId(rawConfig, tokenIdKey),
  ];
  for (const candidate of directCandidates) {
    const id = Array.isArray(candidate) ? candidate[0] : candidate;
    if (Number.isInteger(id) && id >= 0) return id;
  }

  const tokenTextKey = `${name}_token`;
  const camelTokenTextKey = `${name}Token`;
  const textCandidates = [
    resolveTokenizerField(generationConfig, tokenTextKey, camelTokenTextKey),
    resolveTokenizerField(tokenizerConfig, tokenTextKey, camelTokenTextKey),
    resolveTokenizerField(tokenizerJson, tokenTextKey, camelTokenTextKey),
    resolveTokenizerField(tokenizerJson?.special_tokens_map, tokenTextKey, name),
  ];
  for (const candidate of textCandidates) {
    const id = resolveBundledTokenizerTokenTextId(tokenizerJson, candidate);
    if (id != null) return id;
  }
  return null;
}

export function buildBundledTokenizer(
  tokenizerJson,
  tokenizerConfig,
  rawConfig,
  generationConfig = null
) {
  const vocabSize = resolveBundledTokenizerVocabSize(tokenizerJson);
  if (!vocabSize) {
    throw new Error('Tokenizer vocab is missing or empty');
  }

  const tokenizer = {
    type: 'bundled',
    vocabSize,
    file: 'tokenizer.json',
  };
  const addBosToken = (
    resolveTokenizerField(tokenizerJson, 'add_bos_token', 'addBosToken')
    ?? resolveTokenizerField(tokenizerConfig, 'add_bos_token', 'addBosToken')
    ?? resolveConfigBoolean(rawConfig, 'add_bos_token', 'addBosToken')
  );
  const addEosToken = (
    resolveTokenizerField(tokenizerJson, 'add_eos_token', 'addEosToken')
    ?? resolveTokenizerField(tokenizerConfig, 'add_eos_token', 'addEosToken')
    ?? resolveConfigBoolean(rawConfig, 'add_eos_token', 'addEosToken')
  );
  const inferredFlags = inferBundledTokenizerBehaviorFlags(tokenizerJson);

  for (const [name, manifestKey] of [
    ['pad', 'padTokenId'],
    ['bos', 'bosTokenId'],
    ['eos', 'eosTokenId'],
    ['unk', 'unkTokenId'],
  ]) {
    const id = resolveBundledTokenizerSpecialId({
      tokenizerJson,
      tokenizerConfig,
      rawConfig,
      generationConfig,
      name,
    });
    if (id != null) tokenizer[manifestKey] = id;
  }

  if (addBosToken != null) tokenizer.addBosToken = addBosToken;
  else if (inferredFlags.addBosToken != null) tokenizer.addBosToken = inferredFlags.addBosToken;
  if (addEosToken != null) tokenizer.addEosToken = addEosToken;
  else if (inferredFlags.addEosToken != null) tokenizer.addEosToken = inferredFlags.addEosToken;

  return tokenizer;
}


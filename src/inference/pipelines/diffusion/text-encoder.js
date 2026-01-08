import { BPETokenizer } from '../../tokenizers/bpe.js';
import { SentencePieceTokenizer } from '../../tokenizers/sentencepiece.js';
import { loadAuxText, loadAuxFile } from '../../../storage/shard-manager.js';

function parseMerges(text) {
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const merges = [];
  for (const line of lines) {
    if (line.startsWith('#')) continue;
    merges.push(line);
  }
  return merges;
}

function resolveTokenId(value, vocab, addedTokens) {
  if (value == null) return null;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    if (vocab && value in vocab) return vocab[value];
    if (addedTokens) {
      for (const [id, entry] of Object.entries(addedTokens)) {
        if (entry?.content === value) {
          const numId = parseInt(id, 10);
          return Number.isFinite(numId) ? numId : null;
        }
      }
    }
  }
  return null;
}

function resolveSpecialTokenIds(vocab, tokenizerConfig, specialTokensMap) {
  const addedTokens = tokenizerConfig?.added_tokens_decoder ?? null;
  const bosToken = tokenizerConfig?.bos_token ?? specialTokensMap?.bos_token;
  const eosToken = tokenizerConfig?.eos_token ?? specialTokensMap?.eos_token;
  const unkToken = tokenizerConfig?.unk_token ?? specialTokensMap?.unk_token;
  const padToken = tokenizerConfig?.pad_token ?? specialTokensMap?.pad_token;

  return {
    bos: resolveTokenId(tokenizerConfig?.bos_token_id ?? bosToken, vocab, addedTokens),
    eos: resolveTokenId(tokenizerConfig?.eos_token_id ?? eosToken, vocab, addedTokens),
    unk: resolveTokenId(tokenizerConfig?.unk_token_id ?? unkToken, vocab, addedTokens),
    pad: resolveTokenId(tokenizerConfig?.pad_token_id ?? padToken, vocab, addedTokens),
  };
}

async function loadTextAsset(filename, baseUrl) {
  if (baseUrl) {
    const url = `${baseUrl.replace(/\/$/, '')}/${filename}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.status}`);
    }
    return response.text();
  }
  const text = await loadAuxText(filename);
  if (text == null) {
    throw new Error(`Missing ${filename} in model storage.`);
  }
  return text;
}

async function loadBinaryAsset(filename, baseUrl) {
  if (baseUrl) {
    const url = `${baseUrl.replace(/\/$/, '')}/${filename}`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.status}`);
    }
    return response.arrayBuffer();
  }
  const buffer = await loadAuxFile(filename);
  if (!buffer) {
    throw new Error(`Missing ${filename} in model storage.`);
  }
  return buffer;
}

async function loadBpeTokenizer(tokenizerConfig, options = {}) {
  const { baseUrl } = options;
  const vocabText = await loadTextAsset(tokenizerConfig.vocabFile, baseUrl);
  const mergesText = await loadTextAsset(tokenizerConfig.mergesFile, baseUrl);
  const configText = tokenizerConfig.configFile
    ? await loadTextAsset(tokenizerConfig.configFile, baseUrl)
    : null;
  const specialTokensText = tokenizerConfig.specialTokensFile
    ? await loadTextAsset(tokenizerConfig.specialTokensFile, baseUrl)
    : null;

  const vocab = JSON.parse(vocabText);
  const merges = parseMerges(mergesText);
  const tokenizerJson = configText ? JSON.parse(configText) : {};
  const specialTokens = specialTokensText ? JSON.parse(specialTokensText) : {};
  const resolvedSpecials = resolveSpecialTokenIds(vocab, tokenizerJson, specialTokens);

  const tokenizer = new BPETokenizer({
    vocabSize: Object.keys(vocab).length,
    specialTokens: resolvedSpecials,
    addBosToken: tokenizerJson?.add_bos_token ?? true,
    addEosToken: tokenizerJson?.add_eos_token ?? true,
  });
  tokenizer.load(vocab, merges);
  return tokenizer;
}

async function loadSentencePieceTokenizer(tokenizerConfig, options = {}) {
  const { baseUrl } = options;
  const configText = tokenizerConfig.configFile
    ? await loadTextAsset(tokenizerConfig.configFile, baseUrl)
    : null;
  const specialTokensText = tokenizerConfig.specialTokensFile
    ? await loadTextAsset(tokenizerConfig.specialTokensFile, baseUrl)
    : null;
  const tokenizerJsonText = tokenizerConfig.tokenizerFile
    ? await loadTextAsset(tokenizerConfig.tokenizerFile, baseUrl)
    : null;
  const config = configText ? JSON.parse(configText) : {};
  const specialTokens = specialTokensText ? JSON.parse(specialTokensText) : {};
  const tokenizerJson = tokenizerJsonText ? JSON.parse(tokenizerJsonText) : {};
  const addedTokens = config?.added_tokens_decoder ?? tokenizerJson?.added_tokens_decoder ?? null;

  const resolvedSpecials = resolveSpecialTokenIds(null, config, specialTokens);
  if (resolvedSpecials.unk == null && addedTokens) {
    resolvedSpecials.unk = resolveTokenId('<unk>', null, addedTokens);
  }

  const tokenizer = new SentencePieceTokenizer({
    vocabSize: config?.vocab_size ?? 0,
    specialTokens: resolvedSpecials,
    addBosToken: config?.add_bos_token ?? false,
    addEosToken: config?.add_eos_token ?? false,
  });

  const modelData = await loadBinaryAsset(tokenizerConfig.spieceFile, baseUrl);
  await tokenizer.load(modelData);
  return tokenizer;
}

export async function loadDiffusionTokenizers(diffusionConfig, options = {}) {
  const tokenizers = {};
  const config = diffusionConfig?.tokenizers || {};
  if (config.text_encoder) {
    tokenizers.text_encoder = await loadBpeTokenizer(config.text_encoder, options);
  }
  if (config.text_encoder_2) {
    tokenizers.text_encoder_2 = await loadBpeTokenizer(config.text_encoder_2, options);
  }
  if (config.text_encoder_3) {
    tokenizers.text_encoder_3 = await loadSentencePieceTokenizer(config.text_encoder_3, options);
  }
  return tokenizers;
}

function truncateTokens(tokens, maxLength) {
  if (!Number.isFinite(maxLength) || maxLength <= 0) return tokens;
  if (tokens.length <= maxLength) return tokens;
  return tokens.slice(0, maxLength);
}

export function encodePrompt(prompts, tokenizers, options = {}) {
  const { maxLength, maxLengthByTokenizer } = options;
  const result = {};
  let totalTokens = 0;

  for (const [key, tokenizer] of Object.entries(tokenizers || {})) {
    const prompt = prompts?.prompt ?? '';
    const negative = prompts?.negativePrompt ?? '';
    const resolvedMaxLength = (maxLengthByTokenizer && typeof maxLengthByTokenizer === 'object')
      ? (maxLengthByTokenizer[key] ?? maxLength)
      : maxLength;
    const promptIds = truncateTokens(tokenizer.encode(prompt), resolvedMaxLength);
    const negativeIds = truncateTokens(tokenizer.encode(negative), resolvedMaxLength);
    result[key] = { prompt: promptIds, negative: negativeIds };
    totalTokens += promptIds.length + negativeIds.length;
  }

  return { tokens: result, totalTokens };
}

import { isBosLikeLabel, isEosLikeLabel } from '../behavior-flags.js';
import { parseByteTokenValue } from './decoders.js';

function pickCandidate(...values) {
  for (const value of values) {
    if (value != null) return value;
  }
  return null;
}

function resolveTokenId(value, vocab, label) {
  if (value == null) return null;
  if (typeof value === 'number') return value;
  if (typeof value === 'string') {
    const id = vocab.get(value);
    if (id === undefined) {
      throw new Error(`[Tokenizer] Special token "${label}" not found in vocab: "${value}"`);
    }
    return id;
  }
  return null;
}

export function resolveSpecialTokens(specialTokensRaw, fallbackTokens, vocab, options = {}) {
  const padCandidate = pickCandidate(
    specialTokensRaw?.pad,
    specialTokensRaw?.pad_token,
    specialTokensRaw?.pad_token_id,
    fallbackTokens?.pad
  );
  const bosCandidate = pickCandidate(
    specialTokensRaw?.bos,
    specialTokensRaw?.bos_token,
    specialTokensRaw?.bos_token_id,
    fallbackTokens?.bos
  );
  const eosCandidate = pickCandidate(
    specialTokensRaw?.eos,
    specialTokensRaw?.eos_token,
    specialTokensRaw?.eos_token_id,
    fallbackTokens?.eos
  );
  const unkCandidate = pickCandidate(
    specialTokensRaw?.unk,
    specialTokensRaw?.unk_token,
    specialTokensRaw?.unk_token_id,
    fallbackTokens?.unk
  );

  const resolved = {
    pad: resolveTokenId(padCandidate, vocab, 'pad'),
    bos: resolveTokenId(bosCandidate, vocab, 'bos'),
    eos: resolveTokenId(eosCandidate, vocab, 'eos'),
    unk: resolveTokenId(unkCandidate, vocab, 'unk'),
  };

  if (resolved.eos == null && options.allowMissingEos !== true) {
    throw new Error('[Tokenizer] Missing EOS token in tokenizer.json and runtime config.');
  }

  return resolved;
}

export function appendVocabEntry(token, id, vocab, reverseVocab, byteTokens) {
  const numId = typeof id === 'number' ? id : parseInt( (id), 10);
  vocab.set(token, numId);
  reverseVocab.set(numId, token);
  const byteValue = parseByteTokenValue(token);
  if (byteValue !== null) {
    byteTokens.set(byteValue, numId);
  }
  return numId;
}

export function loadObjectVocab(vocabObject, vocab, reverseVocab, byteTokens) {
  let maxId = -1;
  const hasOwn = Object.prototype.hasOwnProperty;
  for (const token in vocabObject) {
    if (!hasOwn.call(vocabObject, token)) {
      continue;
    }
    const numId = appendVocabEntry(token, vocabObject[token], vocab, reverseVocab, byteTokens);
    if (Number.isFinite(numId) && numId > maxId) {
      maxId = numId;
    }
  }
  return maxId;
}

function normalizeMergeKey(merge) {
  if (!Array.isArray(merge)) {
    return merge;
  }
  return merge.length === 2 ? `${merge[0]} ${merge[1]}` : merge.join(' ');
}

export function buildMergeRanks(merges, mergeRanks) {
  const normalizedMerges = new Array(merges.length);
  for (let i = 0; i < merges.length; i++) {
    const mergeKey = normalizeMergeKey(merges[i]);
    normalizedMerges[i] = mergeKey;
    mergeRanks.set(mergeKey, i);
  }
  return normalizedMerges;
}

function isSpecialLikeHotTokenCandidate(token) {
  if (typeof token !== 'string' || token.length === 0) {
    return true;
  }
  if ((token.startsWith('<') && token.endsWith('>')) || (token.startsWith('[') && token.endsWith(']'))) {
    return true;
  }
  return /unused|reserved|multimodal|image|video|audio/i.test(token);
}

function normalizeBpeHotTokenCandidate(token) {
  if (typeof token !== 'string') {
    return '';
  }
  return token
    .replace(/^[▁Ġ]+/, '')
    .replace(/Ċ/g, '\n');
}

function scoreBpeHotTokenCandidate(token, id) {
  if (isSpecialLikeHotTokenCandidate(token)) {
    return Number.NEGATIVE_INFINITY;
  }
  const normalized = normalizeBpeHotTokenCandidate(token);
  if (normalized.length === 0) {
    return Number.NEGATIVE_INFINITY;
  }

  const hasBoundaryMarker = token.startsWith('▁') || token.startsWith('Ġ');
  const isAscii = /^[\x00-\x7F]+$/.test(normalized);
  const isLowerAlpha = /^[a-z]+$/.test(normalized);
  const isTitleAlpha = /^[A-Z][a-z]+$/.test(normalized);
  const isAlpha = /^[A-Za-z]+$/.test(normalized);
  const isPunctuation = /^[.,!?;:'"()%-]+$/.test(normalized);
  const isDigits = /^\d+$/.test(normalized);
  const length = normalized.length;

  let score = 0;
  if (hasBoundaryMarker) score += 40;
  if (isLowerAlpha) score += 32;
  else if (isTitleAlpha) score += 24;
  else if (isAlpha) score += 20;
  if (isPunctuation) score += 20;
  if (isDigits) score += 8;
  if (isAscii) score += 12;

  if (length === 1) score += hasBoundaryMarker ? 8 : 2;
  else if (length <= 4) score += 18 - (length * 2);
  else if (length <= 8) score += 12 - (length - 4);
  else if (length <= 12) score += 4 - (length - 8);
  else score -= Math.min(12, length - 12);

  score -= id / 1e7;
  return score;
}

export function rankFallbackBpeHotTokenIds(reverseVocab, limit, isSpecialToken) {
  const ranked = [];
  for (const [id, token] of reverseVocab.entries()) {
    if (typeof isSpecialToken === 'function' && isSpecialToken(id)) {
      continue;
    }
    const score = scoreBpeHotTokenCandidate(token, id);
    if (!Number.isFinite(score)) {
      continue;
    }
    ranked.push({ id, score });
  }
  ranked.sort((a, b) => b.score - a.score || a.id - b.id);
  return ranked.slice(0, limit).map((entry) => entry.id);
}

export function registerAddedTokens(
  addedTokens,
  vocab,
  reverseVocab,
  patterns,
  specialTokenIds,
  derivedSpecialTokens = null
) {
  let maxId = -1;
  for (const token of addedTokens) {
    const content = token?.content;
    const id = typeof token?.id === 'number' ? token.id : parseInt(token?.id, 10);
    if (!Number.isFinite(id) || !content) continue;
    if (!vocab.has(content)) {
      vocab.set(content, id);
      reverseVocab.set(id, content);
    }
    if (id > maxId) maxId = id;
    if (content.length > 1) {
      patterns.push({ content, id });
    }
    if (token.special) {
      specialTokenIds.add(id);
      if (derivedSpecialTokens) {
        if (derivedSpecialTokens.bos == null && isBosLikeLabel(content)) {
          derivedSpecialTokens.bos = id;
        } else if (derivedSpecialTokens.eos == null && isEosLikeLabel(content)) {
          derivedSpecialTokens.eos = id;
        } else if (derivedSpecialTokens.pad == null && (content === '<pad>' || content.includes('pad'))) {
          derivedSpecialTokens.pad = id;
        } else if (derivedSpecialTokens.unk == null && (content === '<unk>' || content.includes('unk'))) {
          derivedSpecialTokens.unk = id;
        }
      }
    }
  }
  return maxId;
}

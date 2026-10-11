import { validateReasoningOutput } from '../config/reasoning-output.js';
import { createDopplerError, ERROR_CODES } from '../errors/doppler-error.js';

/** @type {import('./generation-output.js').validateGenerationOutputTokenizer} */
export function validateGenerationOutputTokenizer(tokenizer, outputContract, stopTokenIds = []) {
  const contract = validateReasoningOutput(outputContract);
  if (contract == null) return contract;
  for (const marker of [contract.openingToken, contract.closingToken]) {
    if (tokenizer.decode([marker.id], false, false) !== marker.text || stopTokenIds.includes(marker.id)) {
      throw new Error('reasoningOutput markers do not match the tokenizer or overlap stop tokens.');
    }
  }
  if (typeof tokenizer.createRawIncrementalDecoder !== 'function') {
    throw new Error('reasoningOutput requires a tokenizer with raw incremental decoding.');
  }
  return contract;
}

/** @type {import('./generation-output.js').createGenerationOutputDecoder} */
export function createGenerationOutputDecoder(tokenizer, outputContract, stopTokenIds = []) {
  const contract = validateGenerationOutputTokenizer(tokenizer, outputContract, stopTokenIds);
  if (contract == null) return tokenizer.createIncrementalDecoder();
  const raw = tokenizer.createRawIncrementalDecoder?.();
  if (!raw) throw new Error('reasoningOutput requires raw incremental decoding.');
  const visible = tokenizer.createIncrementalDecoder();
  let state = 'opening';
  let rawCharacters = 0;
  let rawTokens = 0;
  let finalCharacters = 0;
  let closed = false;
  let hasFinalContent = false;
  /** @param {string} message */
  function invalid(message) {
    state = 'failed';
    throw createDopplerError(ERROR_CODES.GENERATION_OUTPUT_INVALID, message);
  }
  return {
    push(tokenId) {
      if (closed || state === 'failed' || state === 'stopped') invalid('Generation output is already settled.');
      rawCharacters += raw.push(tokenId).length;
      rawTokens++;
      if (state === 'opening') {
        if (tokenId !== contract.openingToken.id) invalid('Generation did not begin with its declared reasoning marker.');
        state = 'reasoning';
        return '';
      }
      if (tokenId === contract.openingToken.id) invalid('Generation repeated its reasoning opening marker.');
      if (tokenId === contract.closingToken.id) {
        if (state !== 'reasoning') invalid('Generation repeated its reasoning closing marker.');
        state = 'final';
        return '';
      }
      if (stopTokenIds.includes(tokenId)) {
        if (state !== 'final') invalid('Generation stopped before its reasoning closing marker.');
        state = 'stopped';
        return '';
      }
      if (state === 'reasoning') return '';
      const text = visible.push(tokenId);
      finalCharacters += text.length;
      hasFinalContent ||= text.trim().length > 0;
      return text;
    },
    pendingText() { return state === 'final' ? visible.pendingText() : ''; },
    getUsage() { return { rawCharacters: rawCharacters + (closed ? 0 : raw.pendingText().length), rawTokens, finalCharacters, state }; },
    finish(options = {}) {
      if (closed || state === 'failed') invalid('Generation output is already settled.');
      rawCharacters += raw.finish().length;
      closed = true;
      if (options.stopReason === 'max-tokens' || options.stopReason === 'stop-sequence') {
        const text = state === 'final' ? visible.finish() : '';
        finalCharacters += text.length;
        state = 'truncated';
        return text;
      }
      if (state !== 'final' && state !== 'stopped') invalid('Generation ended without its reasoning closing marker.');
      const text = visible.finish();
      finalCharacters += text.length;
      hasFinalContent ||= text.trim().length > 0;
      if (!hasFinalContent) invalid('Generation ended without a final answer.');
      return text;
    },
  };
}

/** @type {import('./generation-output.js').decodeGenerationOutput} */
export function decodeGenerationOutput(tokenizer, contract, stopTokenIds, tokenIds, complete = true, stopReason = null) {
  if (contract == null) return String(tokenizer.decode(tokenIds, true, false));
  const decoder = createGenerationOutputDecoder(tokenizer, contract, stopTokenIds);
  let text = '';
  for (const tokenId of tokenIds) text += decoder.push(tokenId);
  return text + (complete ? decoder.finish({ stopReason }) : decoder.pendingText());
}

// The caller owns the raw iterator and its execution resources. Early return or
// abort discards decoder state; only ordinary completion validates the boundary.
/** @type {import('./generation-output.js').decodeGenerationStream} */
export async function* decodeGenerationStream(tokens, decoder, options = {}) {
  for await (const tokenId of tokens) {
    options.signal?.throwIfAborted();
    const text = decoder.push(tokenId);
    if (text) yield text;
    options.signal?.throwIfAborted();
    options.onToken?.(tokenId, text);
  }
  options.signal?.throwIfAborted();
  const final = decoder.finish({ stopReason: options.getStopReason?.() });
  if (final) yield final;
}

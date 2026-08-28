export function resolveSingleSpecialTokenId(tokenizer, tokenText, label) {
  const rawTokenIds = tokenizer?.encode?.(tokenText);
  const tokenIds = Array.isArray(rawTokenIds)
    ? rawTokenIds
    : (ArrayBuffer.isView(rawTokenIds) ? Array.from(rawTokenIds) : null);
  if (!Array.isArray(tokenIds) || tokenIds.length !== 1) {
    throw new Error(
      `[Pipeline] transcribeImage: tokenizer must encode ${label} "${tokenText}" as exactly one token.`
    );
  }
  const tokenId = Number(tokenIds[0]);
  if (!Number.isFinite(tokenId) || Math.floor(tokenId) !== tokenId || tokenId < 0) {
    throw new Error(
      `[Pipeline] transcribeImage: tokenizer returned invalid ${label} token id "${tokenIds[0]}".`
    );
  }
  return tokenId;
}

export function expandImagePlaceholderTokenIds(tokenIds, imageTokenId, numImageTokens, options = {}) {
  const normalizedTokenIds = Array.isArray(tokenIds)
    ? Int32Array.from(tokenIds)
    : (ArrayBuffer.isView(tokenIds) ? Int32Array.from(tokenIds) : null);
  if (!(normalizedTokenIds instanceof Int32Array)) {
    throw new Error(
      '[Pipeline] transcribeImage: tokenizer.encode() must return an array or typed array of token IDs.'
    );
  }
  if (!Number.isFinite(numImageTokens) || Math.floor(numImageTokens) !== numImageTokens || numImageTokens < 1) {
    throw new Error(
      `[Pipeline] transcribeImage: image token span must be a positive integer, got ${numImageTokens}.`
    );
  }

  let placeholderIndex = -1;
  let placeholderCount = 0;
  for (let index = 0; index < normalizedTokenIds.length; index += 1) {
    if (normalizedTokenIds[index] !== imageTokenId) continue;
    if (placeholderIndex < 0) placeholderIndex = index;
    placeholderCount += 1;
  }

  if (placeholderCount !== 1) {
    throw new Error(
      `[Pipeline] transcribeImage: expected exactly one image_token_id (${imageTokenId}) placeholder ` +
      `from the chat template, got ${placeholderCount}.`
    );
  }

  const boiTokenId = Number.isInteger(options.boiTokenId) ? options.boiTokenId : null;
  const eoiTokenId = Number.isInteger(options.eoiTokenId) ? options.eoiTokenId : null;
  const prefixExtra = boiTokenId == null ? 0 : 1;
  const suffixExtra = eoiTokenId == null ? 0 : 1;
  const expanded = new Int32Array(
    normalizedTokenIds.length - 1 + prefixExtra + numImageTokens + suffixExtra
  );
  expanded.set(normalizedTokenIds.subarray(0, placeholderIndex), 0);
  let writeOffset = placeholderIndex;
  if (boiTokenId != null) expanded[writeOffset++] = boiTokenId;
  expanded.fill(imageTokenId, writeOffset, writeOffset + numImageTokens);
  const imageStartOffset = writeOffset;
  writeOffset += numImageTokens;
  if (eoiTokenId != null) expanded[writeOffset++] = eoiTokenId;
  expanded.set(normalizedTokenIds.subarray(placeholderIndex + 1), writeOffset);

  return { inputIds: expanded, imageStartOffset };
}

export function buildConservativeMultimodalGenerationOptions(options = {}) {
  return {
    ...options,
    disableCommandBatching: true,
    disableMultiTokenDecode: true,
    stopCheckMode: 'per-token',
  };
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return value;
}

export function resolveMultimodalMaxTokens(runtimeConfig, requestedMaxTokens) {
  if (requestedMaxTokens !== undefined) {
    return requirePositiveInteger(requestedMaxTokens, 'maxTokens');
  }
  return requirePositiveInteger(
    runtimeConfig?.inference?.generation?.multimodalMaxTokens,
    'runtime.inference.generation.multimodalMaxTokens'
  );
}

export function assertMultimodalSequenceCapacity({ inputTokenCount, maxTokens, maxSeqLen } = {}) {
  const resolvedInputTokenCount = requirePositiveInteger(
    inputTokenCount,
    'multimodal inputTokenCount'
  );
  const resolvedMaxTokens = requirePositiveInteger(maxTokens, 'multimodal maxTokens');
  const resolvedMaxSeqLen = requirePositiveInteger(
    maxSeqLen,
    'active KV cache maxSeqLen'
  );
  const requiredCapacity = resolvedInputTokenCount + resolvedMaxTokens;
  if (requiredCapacity > resolvedMaxSeqLen) {
    throw new Error(
      `[Pipeline] transcribeImage: multimodal request requires ${requiredCapacity} sequence slots ` +
      `(${resolvedInputTokenCount} input + ${resolvedMaxTokens} output), but the active KV cache ` +
      `maxSeqLen is ${resolvedMaxSeqLen}. Increase runtime.inference.session.kvcache.maxSeqLen ` +
      'or reduce the image or output token budget.'
    );
  }
  return requiredCapacity;
}

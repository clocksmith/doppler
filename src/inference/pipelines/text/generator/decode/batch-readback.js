import { parseFinitenessStatusWords } from "../../policy/finiteness-guard-status.js";

export function createStopTokenLookup(stopTokenIds, eosTokenId) {
  if (!Array.isArray(stopTokenIds)) {
    throw new Error('[Pipeline] stopTokenIds must be an array.');
  }

  const tokenIds = [];
  for (const tokenId of stopTokenIds) {
    if (!tokenIds.includes(tokenId)) {
      tokenIds.push(tokenId);
    }
  }
  if (typeof eosTokenId === 'number' && !tokenIds.includes(eosTokenId)) {
    tokenIds.push(eosTokenId);
  }

  if (tokenIds.length === 0) {
    return { firstTokenId: null, secondTokenId: null, tokenSet: null };
  }
  if (tokenIds.length === 1) {
    return { firstTokenId: tokenIds[0], secondTokenId: null, tokenSet: null };
  }
  if (tokenIds.length === 2) {
    return { firstTokenId: tokenIds[0], secondTokenId: tokenIds[1], tokenSet: null };
  }
  return { firstTokenId: null, secondTokenId: null, tokenSet: new Set(tokenIds) };
}

function lookupHasStopToken(token, stopTokenLookup) {
  if (stopTokenLookup.tokenSet) {
    return stopTokenLookup.tokenSet.has(token);
  }
  if (stopTokenLookup.firstTokenId !== null && token === stopTokenLookup.firstTokenId) {
    return true;
  }
  return stopTokenLookup.secondTokenId !== null && token === stopTokenLookup.secondTokenId;
}

export function resolveBatchStop(tokens, stopFlags, stopTokenLookup) {
  let actualCount = tokens.length;
  if (stopFlags) {
    const maxFlags = Math.min(stopFlags.length, tokens.length);
    for (let i = 0; i < maxFlags; i++) {
      if (stopFlags[i] === 1) {
        actualCount = i + 1;
        break;
      }
    }
  }

  for (let i = 0; i < actualCount; i++) {
    if (lookupHasStopToken(tokens[i], stopTokenLookup)) {
      actualCount = i + 1;
      break;
    }
  }

  return actualCount;
}

export function findInvalidGeneratedToken(tokens, vocabSize, padTokenId = null) {
  for (let i = 0; i < tokens.length; i++) {
    const tokenId = tokens[i];
    const isInvalid = !Number.isFinite(tokenId)
      || tokenId < 0
      || tokenId >= vocabSize
      || (padTokenId != null ? tokenId === padTokenId : tokenId === 0);
    if (isInvalid) {
      return { index: i, tokenId };
    }
  }
  return null;
}

export async function readBatchTokensFromStagingBuffers(options) {
  const {
    tokensStagingBuffer,
    stopStagingBuffer = null,
    finitenessStagingBuffer = null,
    finitenessOffsetBytes = null,
    tokenCount,
    ownsTokensStaging = false,
    ownsStopStaging = false,
    ownsFinitenessStaging = Boolean(finitenessStagingBuffer),
    ring = null,
    cleanupRecorder = null,
  } = options;
  let tokensMapped = false;
  let stopMapped = false;
  let finitenessMapped = false;
  let cleanupCompleted = false;
  const hasPackedFiniteness = Number.isFinite(finitenessOffsetBytes) && finitenessOffsetBytes >= 0;
  const timing = {
    mapWaitMs: 0,
    cleanupMs: 0,
    copyMs: 0,
  };

  try {
    const mapStart = performance.now();
    const mapPromises = [tokensStagingBuffer.mapAsync(GPUMapMode.READ)];
    if (stopStagingBuffer) {
      mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    if (finitenessStagingBuffer && !hasPackedFiniteness) {
      mapPromises.push(finitenessStagingBuffer.mapAsync(GPUMapMode.READ));
    }
    const mapResults = await Promise.allSettled(mapPromises);
    timing.mapWaitMs = performance.now() - mapStart;
    tokensMapped = mapResults[0]?.status === 'fulfilled';
    stopMapped = Boolean(stopStagingBuffer) && mapResults[1]?.status === 'fulfilled';
    finitenessMapped = Boolean(finitenessStagingBuffer) && !hasPackedFiniteness
      && mapResults[stopStagingBuffer ? 2 : 1]?.status === 'fulfilled';
    const mapFailure = mapResults.find((result) => result.status === 'rejected');
    if (mapFailure) {
      throw mapFailure.reason;
    }
    const cleanupStart = performance.now();
    await cleanupRecorder?.completeDeferredCleanup();
    timing.cleanupMs = performance.now() - cleanupStart;
    cleanupCompleted = true;

    const copyStart = performance.now();
    const tokensRange = tokensStagingBuffer.getMappedRange();
    const tokenWords = new Uint32Array(tokensRange, 0, tokenCount);
    const tokens = new Uint32Array(tokenWords.length);
    tokens.set(tokenWords);
    let stopFlags = null;
    if (stopStagingBuffer) {
      const stopWords = new Uint32Array(stopStagingBuffer.getMappedRange()).subarray(0, tokenCount);
      stopFlags = new Uint32Array(stopWords.length);
      stopFlags.set(stopWords);
    }
    const finitenessStatus = hasPackedFiniteness
      ? parseFinitenessStatusWords(new Uint32Array(tokensRange, finitenessOffsetBytes, 4), 0)
      : finitenessStagingBuffer
      ? parseFinitenessStatusWords(new Uint32Array(finitenessStagingBuffer.getMappedRange()), 0)
      : { triggered: false, metadata: '' };
    timing.copyMs = performance.now() - copyStart;

    return {
      tokens,
      stopFlags,
      finitenessStatus,
      timing,
    };
  } finally {
    if (finitenessMapped) {
      finitenessStagingBuffer.unmap();
    }
    if (tokensMapped) {
      tokensStagingBuffer.unmap();
    }
    if (stopMapped) {
      stopStagingBuffer.unmap();
    }
    try {
      if (!cleanupCompleted) {
        await cleanupRecorder?.completeDeferredCleanup({ discardPooled: true });
      }
    } finally {
      if (ownsFinitenessStaging) {
        finitenessStagingBuffer.destroy();
      }
      if (ownsTokensStaging) {
        tokensStagingBuffer.destroy();
      }
      if (ownsStopStaging) {
        stopStagingBuffer?.destroy();
      }
      ring?.advance();
    }
  }
}

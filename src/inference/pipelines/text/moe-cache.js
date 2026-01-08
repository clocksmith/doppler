import { getRuntimeConfig } from '../../../config/runtime.js';
import { QK_K } from '../../../config/schema/index.js';

const dequantCache = new Map();
let dequantCacheMaxEntriesOverride = null;
let dequantCacheHits = 0;
let dequantCacheMisses = 0;

export function resolveMaxTokensPerExpert(numTokens, numExperts, topK, hiddenSize, activationDtype) {
  const routingConfig = getRuntimeConfig().inference.moe.routing;
  const {
    maxTokensPerExpert = 0,
    maxTokensPerExpertHeadroom = 2.0,
    maxTokensPerExpertMin = 4,
    maxTokensPerExpertCap = 0,
  } = routingConfig;

  let target = maxTokensPerExpert > 0
    ? maxTokensPerExpert
    : Math.ceil((numTokens * topK / Math.max(1, numExperts)) * maxTokensPerExpertHeadroom);

  target = Math.max(target, maxTokensPerExpertMin, 1);
  if (activationDtype === 'f16') {
    const bytesPerToken = hiddenSize * 2;
    const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));
    const alignMultiple = QK_K / gcd(QK_K, bytesPerToken);
    let aligned = Math.ceil(target / alignMultiple) * alignMultiple;

    if (maxTokensPerExpertCap > 0) {
      const capAligned = Math.floor(maxTokensPerExpertCap / alignMultiple) * alignMultiple;
      aligned = Math.min(aligned, capAligned || alignMultiple);
    }
    return aligned;
  }

  if (maxTokensPerExpertCap > 0) {
    target = Math.min(target, maxTokensPerExpertCap);
  }
  return Math.min(target, numTokens);
}

function getDequantCacheMaxEntries() {
  return dequantCacheMaxEntriesOverride ?? getRuntimeConfig().inference.moe.cache.dequantCacheMaxEntries;
}

function getDequantCacheKey(layerIdx, expertIdx, outputDtype) {
  return `${layerIdx}_${expertIdx}_${outputDtype}`;
}

export function getCachedDequant(layerIdx, expertIdx, outputDtype) {
  const key = getDequantCacheKey(layerIdx, expertIdx, outputDtype);
  const cached = dequantCache.get(key);
  if (cached) {
    cached.lastUsed = performance.now();
    dequantCacheHits++;
  }
  return cached;
}

export function setCachedDequant(layerIdx, expertIdx, outputDtype, gateUp, down) {
  const key = getDequantCacheKey(layerIdx, expertIdx, outputDtype);
  dequantCacheMisses++;

  if (dequantCache.size >= getDequantCacheMaxEntries()) {
    let oldestKey = '';
    let oldestTime = Infinity;
    for (const [k, v] of dequantCache.entries()) {
      if (v.lastUsed < oldestTime) {
        oldestTime = v.lastUsed;
        oldestKey = k;
      }
    }
    if (oldestKey) {
      const evicted = dequantCache.get(oldestKey);
      if (evicted) {
        evicted.gateUp.destroy();
        evicted.down.destroy();
      }
      dequantCache.delete(oldestKey);
    }
  }

  dequantCache.set(key, { gateUp, down, lastUsed: performance.now() });
}

export function clearDequantCache() {
  for (const cached of dequantCache.values()) {
    cached.gateUp.destroy();
    cached.down.destroy();
  }
  dequantCache.clear();
  dequantCacheHits = 0;
  dequantCacheMisses = 0;
}

export function getDequantCacheStats() {
  return {
    hits: dequantCacheHits,
    misses: dequantCacheMisses,
    size: dequantCache.size,
    maxEntries: getDequantCacheMaxEntries(),
  };
}

export function setDequantCacheMaxEntries(maxEntries) {
  dequantCacheMaxEntriesOverride = maxEntries;
}

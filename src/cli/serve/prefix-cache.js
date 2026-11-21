import crypto from 'node:crypto';

/** @type {Readonly<import('./prefix-cache.js').ServePrefixCachePolicy>} */
export const DEFAULT_PREFIX_CACHE_POLICY = Object.freeze({
  enabled: true,
  maxEntries: 8,
  maxBytes: 256 * 1024 * 1024,
  maxEntryBytes: 256 * 1024 * 1024,
  idleMs: 300_000,
});

/**
 * @param {unknown} value
 * @param {number} fallback
 * @param {string} label
 * @returns {number}
 */
function normalizePositiveInt(value, fallback, label) {
  if (value === undefined) return fallback;
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return n;
}

/**
 * @param {boolean | Partial<import('./prefix-cache.js').ServePrefixCachePolicy>} [input]
 * @returns {Readonly<import('./prefix-cache.js').ServePrefixCachePolicy>}
 */
export function resolvePrefixCachePolicy(input = {}) {
  if (input === false) {
    return Object.freeze({ ...DEFAULT_PREFIX_CACHE_POLICY, enabled: false });
  }
  const source = input && typeof input === 'object' ? input : {};
  const enabled = source.enabled !== false;
  const maxEntries = normalizePositiveInt(source.maxEntries, DEFAULT_PREFIX_CACHE_POLICY.maxEntries, 'prefixCache.maxEntries');
  const maxBytes = normalizePositiveInt(source.maxBytes, DEFAULT_PREFIX_CACHE_POLICY.maxBytes, 'prefixCache.maxBytes');
  const maxEntryBytes = normalizePositiveInt(source.maxEntryBytes, maxBytes, 'prefixCache.maxEntryBytes');
  if (maxEntryBytes > maxBytes) {
    throw new Error('prefixCache.maxEntryBytes must not exceed prefixCache.maxBytes.');
  }
  const idleMs = normalizePositiveInt(source.idleMs, DEFAULT_PREFIX_CACHE_POLICY.idleMs, 'prefixCache.idleMs');
  return Object.freeze({ enabled, maxEntries, maxBytes, maxEntryBytes, idleMs });
}

/**
 * Derive a deterministic conversation-root key from the model ID, leading system
 * messages, and the first non-system turn so multi-turn OpenAI chat requests
 * automatically hit the same GenerationContext without caller-side headers.
 * @param {string} modelId
 * @param {ReadonlyArray<{ role: string; content: string }>} messages
 * @returns {string | null}
 */
export function deriveServePrefixKey(modelId, messages) {
  if (typeof modelId !== 'string' || !modelId.trim() || !Array.isArray(messages) || messages.length === 0) {
    return null;
  }
  /** @type {Array<[string, string]>} */
  const root = [];
  for (const message of messages) {
    if (!message || typeof message.role !== 'string' || typeof message.content !== 'string') {
      continue;
    }
    root.push([message.role, message.content]);
    if (message.role !== 'system') {
      break;
    }
  }
  if (root.length === 0) return null;
  const digest = crypto.createHash('sha256')
    .update(JSON.stringify([modelId.trim(), root]))
    .digest('hex')
    .slice(0, 32);
  return `prefix_${digest}`;
}

/**
 * @typedef {{
 *   owner: object | null;
 *   context: import('../../inference/pipelines/text/generator/session/sequence-state.js').GenerationContext;
 *   used: number;
 *   active: boolean;
 *   reservation: number;
 *   timer: ReturnType<typeof setTimeout> | null;
 * }} PrefixCacheEntry
 */

/**
 * @param {{ policy?: boolean | Partial<import('./prefix-cache.js').ServePrefixCachePolicy>; now?: () => number }} [options]
 * @returns {import('./prefix-cache.js').ServePrefixCache}
 */
export function createServePrefixCache({ policy = {}, now = Date.now } = {}) {
  const resolved = resolvePrefixCachePolicy(policy);
  // Leave room for a second conversation when the loaded model fits half the pool.
  const preferredBytes = typeof policy === 'object' && policy?.maxEntryBytes !== undefined
    ? resolved.maxEntryBytes : Math.ceil(resolved.maxBytes / 2);
  /** @type {Map<string, PrefixCacheEntry>} */
  const entries = new Map();
  let closed = false;
  let hits = 0;
  let misses = 0;
  let evictions = 0;
  let busySkips = 0;
  let budgetSkips = 0;
  const unretained = { budget: 0, discarded: 0, empty: 0 };

  /**
   * @param {string} key
   * @param {{ evicted?: boolean }} [options]
   * @returns {boolean}
   */
  const remove = (key, { evicted = false } = {}) => {
    const entry = entries.get(key);
    if (!entry || entry.active) return false;
    if (entry.timer !== null) clearTimeout(entry.timer);
    entries.delete(key);
    try {
      entry.context.close();
    } catch {
      // Best-effort GPU buffer release on eviction/close.
    }
    if (evicted) evictions += 1;
    return true;
  };

  const sweep = () => {
    const current = now();
    for (const [key, entry] of entries) {
      if (!entry.active && current - entry.used >= resolved.idleMs) {
        remove(key, { evicted: true });
      }
    }
  };

  /**
   * @param {PrefixCacheEntry} entry
   * @returns {number}
   */
  const chargedBytes = (entry) => {
    const bytes = entry.context.getStats?.()?.allocatedBytes;
    return entry.active || typeof bytes !== 'number' || !Number.isSafeInteger(bytes) || bytes < 0
      ? entry.reservation
      : bytes;
  };

  return Object.freeze({
    policy: resolved,
    /**
     * @param {string | null} key
     * @param {(options: { maxBytes: number }) => import('../../inference/pipelines/text/generator/session/sequence-state.js').GenerationContext} createContext
     * @param {object | null} [owner] Loaded pipeline handle that owns the context.
     * @param {number} [minimumBytes] KV bytes already allocated by the loaded model.
     * @returns {import('./prefix-cache.js').ServePrefixCacheLease | null}
     */
    acquire(key, createContext, owner = null, minimumBytes = 0) {
      if (closed || !resolved.enabled || !key || typeof createContext !== 'function') {
        return null;
      }
      sweep();
      if (minimumBytes > resolved.maxEntryBytes) {
        budgetSkips += 1;
        return null;
      }
      let entry = entries.get(key);
      if (entry?.active) {
        // Concurrent request on the same conversation falls back cleanly to uncached prefill.
        busySkips += 1;
        return null;
      }
      if (entry && entry.owner !== owner) {
        remove(key, { evicted: true });
        entry = undefined;
      }
      // A checkpoint also includes token history and any recurrent state.
      const reservation = entry?.reservation
        ?? (minimumBytes >= preferredBytes ? resolved.maxEntryBytes : preferredBytes);
      const otherBytes = () => {
        let total = 0;
        for (const [id, item] of entries) {
          if (id !== key) total += chargedBytes(item);
        }
        return total;
      };
      while ((!entry && entries.size >= resolved.maxEntries) || otherBytes() + reservation > resolved.maxBytes) {
        const victim = [...entries].find(([id, candidate]) => id !== key && !candidate.active);
        if (!victim) {
          return null;
        }
        remove(victim[0], { evicted: true });
      }

      let hit = false;
      if (entry) {
        hits += 1;
        hit = true;
      } else {
        misses += 1;
        entry = {
          owner,
          context: createContext({ maxBytes: reservation }),
          reservation,
          used: now(),
          active: false,
          timer: null,
        };
        entries.set(key, entry);
      }

      if (entry.timer !== null) clearTimeout(entry.timer);
      entry.timer = null;
      entry.active = true;
      let released = false;
      const activeEntry = entry;

      return Object.freeze({
        key,
        hit,
        context: activeEntry.context,
        release({ discard = false } = {}) {
          if (released) return;
          released = true;
          activeEntry.active = false;
          activeEntry.used = now();
          if (discard || closed) {
            remove(key);
            return;
          }
          const stats = activeEntry.context.getStats();
          if (!Number.isSafeInteger(stats.cachedTokens) || stats.cachedTokens <= 0) {
            if (stats.status === 'budget') unretained.budget += 1;
            else if (stats.status === 'discarded') unretained.discarded += 1;
            else unretained.empty += 1;
            remove(key);
            return;
          }
          // Refresh LRU order.
          entries.delete(key);
          entries.set(key, activeEntry);
          activeEntry.timer = setTimeout(() => {
            remove(key, { evicted: true });
          }, resolved.idleMs);
          activeEntry.timer.unref?.();
        },
      });
    },
    getStats() {
      let activeEntries = 0;
      let allocatedBytes = 0;
      let cachedTokens = 0;
      let reusedTokens = 0;
      for (const entry of entries.values()) {
        if (entry.active) activeEntries += 1;
        const stats = entry.context.getStats?.();
        if (typeof stats?.allocatedBytes === 'number' && Number.isSafeInteger(stats.allocatedBytes) && stats.allocatedBytes > 0) {
          allocatedBytes += stats.allocatedBytes;
        }
        if (typeof stats?.cachedTokens === 'number' && Number.isSafeInteger(stats.cachedTokens) && stats.cachedTokens > 0) {
          cachedTokens += stats.cachedTokens;
        }
        if (typeof stats?.reusedTokens === 'number' && Number.isSafeInteger(stats.reusedTokens) && stats.reusedTokens > 0) {
          reusedTokens += stats.reusedTokens;
        }
      }
      return Object.freeze({
        enabled: resolved.enabled,
        entries: entries.size,
        activeEntries,
        allocatedBytes,
        cachedTokens,
        reusedTokens,
        maxEntries: resolved.maxEntries,
        maxBytes: resolved.maxBytes,
        maxEntryBytes: resolved.maxEntryBytes,
        hits,
        misses,
        evictions,
        busySkips,
        budgetSkips,
        unretained: Object.freeze({ ...unretained }),
      });
    },
    close() {
      closed = true;
      for (const key of [...entries.keys()]) {
        remove(key);
      }
    },
  });
}

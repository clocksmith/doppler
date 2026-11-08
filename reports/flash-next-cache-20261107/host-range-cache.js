// Experiment adapter, not an installed runtime feature or implicit storage policy.
export function withHostRangeCache(storage, policy) {
  if (!storage || typeof storage.loadShardRange !== 'function' || typeof policy.enabled !== 'boolean'
      || !Number.isSafeInteger(policy.maxBytes) || policy.maxBytes < 1
      || !Number.isSafeInteger(policy.maxEntryBytes) || policy.maxEntryBytes < 1) {
    throw new Error('Host range experiment requires an explicit reader and bounded policy.');
  }
  const entries = new Map(), pending = new Map();
  const read = storage.loadShardRange.bind(storage), close = storage.close;
  let active = false, closed = false, residentBytes = 0;
  const stats = { hits: 0, misses: 0, sourceBytes: 0, peakResidentBytes: 0, evictions: 0 };
  return {
    storage: { ...storage,
      async loadShardRange(shard, offset, length) {
        if (closed) throw new Error('Host range experiment closed');
        if (!active || !policy.enabled || !Number.isSafeInteger(length)
            || length < 1 || length > policy.maxEntryBytes || length > policy.maxBytes) {
          return read(shard, offset, length);
        }
        const key = `${shard}:${offset}:${length}`;
        let bytes = entries.get(key);
        if (bytes) {
          entries.delete(key); entries.set(key, bytes); stats.hits++;
        } else {
          let task = pending.get(key);
          if (!task) {
            stats.misses++;
            task = (async () => {
              const data = await read(shard, offset, length);
              if (!(data instanceof ArrayBuffer) || data.byteLength !== length) {
                throw new Error('Host range cache requires an exact immutable source range.');
              }
              stats.sourceBytes += data.byteLength;
              // Own a snapshot; callers always receive independent mutable buffers.
              const owned = data.slice(0);
              if (!closed) {
                while (residentBytes + owned.byteLength > policy.maxBytes && entries.size) {
                  const [victim, value] = entries.entries().next().value;
                  entries.delete(victim); residentBytes -= value.byteLength; stats.evictions++;
                }
                entries.set(key, owned); residentBytes += owned.byteLength;
                stats.peakResidentBytes = Math.max(stats.peakResidentBytes, residentBytes);
              }
              return owned;
            })();
            pending.set(key, task);
            task.then(() => pending.delete(key), () => pending.delete(key));
          }
          bytes = await task;
        }
        return bytes.slice(0);
      },
      async close() {
        closed = true;
        await Promise.allSettled([...pending.values()]);
        entries.clear(); residentBytes = 0;
        await close?.call(storage);
      },
    },
    start() { if (closed) throw new Error('Host range experiment closed'); active = true; },
    stats() { return { ...stats, residentBytes, entries: entries.size, pending: pending.size }; },
  };
}

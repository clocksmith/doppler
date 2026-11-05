// Host-resident reader of one qwen4_exp n-gram embedding table.
//
// The table is one logical [rows, rowWidth] matrix stored as row-contiguous
// parts (128 checkpoint shards for Flash-Next). Parts come from authenticated
// headers or manifest spans and must tile [0, rows) exactly. Reads go through a
// caller-supplied range port (an artifact storage context's loadShardRange or
// a file range reader); this module never keeps the table GPU-resident.
//
// Policy is declared, not inferred: the LRU row cache is bounded by maxRows and
// maxBytes; one read batch stages at most staging.maxRows rows; at most
// reads.maxConcurrent range reads are in flight. Cancellation aborts pending
// reads and inserts nothing from an incomplete batch. Row bytes are returned
// as stored (bf16); this reader authenticates the row-to-part mapping, not the
// bytes of individual rows (see `rowBytesVerified` in stats).

const ROW_ELEMENT_BYTES = Object.freeze({ bf16: 2 });

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[NgramTable] ${message}`);
}

/** @param {AbortSignal | null | undefined} signal */
function throwIfAborted(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException('The n-gram table read was aborted.', 'AbortError');
}

export class NgramTableReader {
  /** @type {Map<number, Uint8Array>} */
  #cache = new Map();
  #cachedBytes = 0;
  #stats = { requests: 0, rowsRequested: 0, hits: 0, misses: 0, reads: 0, bytesRead: 0, evictions: 0 };

  /** @param {import('./ngram-table.js').NgramTableReaderOptions} options */
  constructor({ rows, rowWidth, dtype, parts, readRange, session, rowBytesVerified = false }) {
    const elementBytes = ROW_ELEMENT_BYTES[/** @type {'bf16'} */ (dtype)];
    if (!elementBytes) fail(`unsupported table dtype ${JSON.stringify(dtype)}.`);
    if (!Number.isSafeInteger(rows) || rows < 1 || rows > 2 ** 32) fail('rows must be a positive count within u32 addressing.');
    if (!Number.isSafeInteger(rowWidth) || rowWidth < 1) fail('rowWidth must be positive.');
    if (typeof readRange !== 'function') fail('a readRange(part, offset, length, signal) port is required.');
    if (!session?.rowCache || !session.staging || !session.reads) fail('a resolved ngramEmbedding session policy is required.');
    this.rows = rows;
    this.rowWidth = rowWidth;
    this.dtype = dtype;
    this.rowBytes = rowWidth * elementBytes;
    this.readRange = readRange;
    this.session = session;
    this.rowBytesVerified = rowBytesVerified === true;
    if (!Array.isArray(parts) || parts.length === 0) fail('parts must tile the table.');
    let next = 0;
    this.parts = parts.map((part, index) => {
      const { rowBegin, rowEnd, source, byteOffset } = part;
      if (rowBegin !== next || !Number.isSafeInteger(rowEnd) || rowEnd <= rowBegin) {
        fail(`part ${index} covers [${rowBegin}, ${rowEnd}) but the next row is ${next}; parts must tile [0, rows).`);
      }
      if (!Number.isSafeInteger(byteOffset) || byteOffset < 0) fail(`part ${index} needs a non-negative byteOffset.`);
      if (part.byteLength !== undefined && part.byteLength !== (rowEnd - rowBegin) * this.rowBytes) {
        fail(`part ${index} spans ${part.byteLength} bytes for ${rowEnd - rowBegin} rows of ${this.rowBytes} bytes.`);
      }
      next = rowEnd;
      return Object.freeze({ rowBegin, rowEnd, source, byteOffset });
    });
    if (next !== rows) fail(`parts end at row ${next}, but the table has ${rows} rows.`);
  }

  /** @param {number} row */
  #partOf(row) {
    let low = 0;
    let high = this.parts.length - 1;
    while (low < high) {
      const middle = (low + high + 1) >> 1;
      if (this.parts[middle].rowBegin <= row) low = middle;
      else high = middle - 1;
    }
    return this.parts[low];
  }

  /** @param {number} row @param {Uint8Array} bytes */
  #insert(row, bytes) {
    const { maxRows, maxBytes } = this.session.rowCache;
    if (maxRows === 0 || maxBytes < this.rowBytes) return;
    if (this.#cache.has(row)) return;
    this.#cache.set(row, bytes);
    this.#cachedBytes += bytes.byteLength;
    while (this.#cache.size > maxRows || this.#cachedBytes > maxBytes) {
      const oldest = this.#cache.keys().next().value;
      if (oldest === undefined) break;
      this.#cachedBytes -= /** @type {Uint8Array} */ (this.#cache.get(oldest)).byteLength;
      this.#cache.delete(oldest);
      this.#stats.evictions += 1;
    }
  }

  /** @param {number} row */
  #lookup(row) {
    const bytes = this.#cache.get(row);
    if (bytes === undefined) return null;
    this.#cache.delete(row);
    this.#cache.set(row, bytes);
    return bytes;
  }

  /**
   * Read a sorted batch of distinct missing rows: contiguous rows of one part
   * coalesce into a single range read, at most reads.maxConcurrent at a time.
   * @param {number[]} rows @param {AbortSignal | null} signal
   * @returns {Promise<Map<number, Uint8Array>>}
   */
  async #readBatch(rows, signal) {
    /** @type {{ part: ReturnType<NgramTableReader['parts']['at']>, first: number, count: number }[]} */
    const runs = [];
    for (const row of rows) {
      const part = this.#partOf(row);
      const last = runs.at(-1);
      if (last && last.part === part && last.first + last.count === row) last.count += 1;
      else runs.push({ part, first: row, count: 1 });
    }
    const fetched = new Map();
    // One failed read stops the other workers; the caller's signal aborts all of them.
    const batchAbort = new AbortController();
    const linked = signal ? AbortSignal.any([signal, batchAbort.signal]) : batchAbort.signal;
    let cursor = 0;
    const worker = async () => {
      while (cursor < runs.length) {
        throwIfAborted(linked);
        const run = runs[cursor++];
        const { part } = /** @type {{ part: any }} */ (run);
        const length = run.count * this.rowBytes;
        const offset = part.byteOffset + (run.first - part.rowBegin) * this.rowBytes;
        const result = await this.readRange(part.source, offset, length, linked);
        throwIfAborted(linked);
        const bytes = result instanceof Uint8Array ? result : new Uint8Array(result);
        if (bytes.byteLength !== length) fail(`range read returned ${bytes.byteLength} bytes, expected ${length}.`);
        this.#stats.reads += 1;
        this.#stats.bytesRead += length;
        for (let i = 0; i < run.count; i++) {
          fetched.set(run.first + i, bytes.slice(i * this.rowBytes, (i + 1) * this.rowBytes));
        }
      }
    };
    const workers = Math.min(this.session.reads.maxConcurrent, runs.length);
    try {
      const results = await Promise.allSettled(Array.from({ length: workers }, () => worker().catch((error) => {
        batchAbort.abort(error);
        throw error;
      })));
      const failure = results.find(result => result.status === 'rejected');
      if (failure) throw failure.reason;
    } catch (error) {
      throwIfAborted(signal);
      throw error;
    }
    return fetched;
  }

  /**
   * Gather rows for the given addresses, in address order, as stored bytes.
   * @type {import('./ngram-table.js').NgramTableReader['gather']}
   */
  async gather(addresses, { signal = null } = {}) {
    throwIfAborted(signal);
    this.#stats.requests += 1;
    this.#stats.rowsRequested += addresses.length;
    const output = new Uint8Array(addresses.length * this.rowBytes);
    /** @type {Map<number, Uint8Array>} */
    const resolved = new Map();
    const missing = new Set();
    for (let index = 0; index < addresses.length; index++) {
      const row = addresses[index];
      if (!Number.isSafeInteger(row) || row < 0 || row >= this.rows) fail(`row ${row} is outside the ${this.rows}-row table.`);
      if (resolved.has(row) || missing.has(row)) continue;
      const cached = this.#lookup(row);
      if (cached) {
        resolved.set(row, cached);
        this.#stats.hits += 1;
      } else {
        missing.add(row);
        this.#stats.misses += 1;
      }
    }
    const sorted = [...missing].sort((a, b) => a - b);
    const batch = this.session.staging.maxRows;
    for (let start = 0; start < sorted.length; start += batch) {
      const fetched = await this.#readBatch(sorted.slice(start, start + batch), signal);
      for (const [row, bytes] of fetched) {
        resolved.set(row, bytes);
        this.#insert(row, bytes);
      }
    }
    for (let index = 0; index < addresses.length; index++) {
      output.set(/** @type {Uint8Array} */ (resolved.get(addresses[index])), index * this.rowBytes);
    }
    return output;
  }

  /** @type {import('./ngram-table.js').NgramTableReader['stats']} */
  stats() {
    return { ...this.#stats, cachedRows: this.#cache.size, cachedBytes: this.#cachedBytes, rowBytesVerified: this.rowBytesVerified };
  }

  /** @type {import('./ngram-table.js').NgramTableReader['clear']} */
  clear() {
    this.#cache.clear();
    this.#cachedBytes = 0;
  }
}

/**
 * Parts of the n-gram table from the authenticated checkpoint header receipt
 * (artifacts/qwen38-flash-next-20261013/checkpoint-headers.json). Each part's
 * source is the checkpoint file name; byteOffset is absolute in that file.
 * @type {typeof import('./ngram-table.js').ngramTablePartsFromCheckpointHeaders}
 */
export function ngramTablePartsFromCheckpointHeaders(receipt) {
  const table = receipt?.ngramTable;
  if (!table || !Array.isArray(table.shards)) fail('checkpoint header receipt lacks ngramTable.shards.');
  const rowBytes = table.rowWidth * ROW_ELEMENT_BYTES[/** @type {'bf16'} */ (String(table.dtype).toLowerCase())];
  return table.shards.map((shard) => {
    if (shard.fileByteEnd - shard.fileByteBegin !== (shard.rowEnd - shard.rowBegin) * rowBytes) {
      fail(`shard ${shard.shard} byte span does not match its rows.`);
    }
    return {
      rowBegin: shard.rowBegin,
      rowEnd: shard.rowEnd,
      source: shard.file,
      byteOffset: shard.fileByteBegin,
      byteLength: shard.fileByteEnd - shard.fileByteBegin,
    };
  });
}

// Compatibility export; physical tensor bindings are owned by the loader.
export { ngramTablePartsFromTensorLocations } from '../../../../loader/ngram-table-bindings.js';

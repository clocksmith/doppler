import type { ResolvedNgramEmbeddingSession } from '../../../../config/schema/ngram-embedding.schema.js';

export type NgramTablePartSource = string | number;

/** A row-contiguous part of the table: rows [rowBegin, rowEnd) at byteOffset in `source`. */
export interface NgramTablePart {
  rowBegin: number;
  rowEnd: number;
  source: NgramTablePartSource;
  byteOffset: number;
  /** When present, must equal (rowEnd - rowBegin) * rowBytes. */
  byteLength?: number;
}

export type NgramTableRangePort = (
  source: NgramTablePartSource,
  offset: number,
  length: number,
  signal: AbortSignal | null
) => Promise<Uint8Array | ArrayBuffer>;

export interface NgramTableReaderOptions {
  rows: number;
  rowWidth: number;
  dtype: 'bf16';
  /** Must tile [0, rows) in order. */
  parts: NgramTablePart[];
  readRange: NgramTableRangePort;
  session: ResolvedNgramEmbeddingSession;
  /** Whether the range port authenticates the returned bytes (for example a verified piece reader). */
  rowBytesVerified?: boolean;
}

export interface NgramTableStats {
  requests: number;
  rowsRequested: number;
  hits: number;
  misses: number;
  reads: number;
  bytesRead: number;
  evictions: number;
  cachedRows: number;
  cachedBytes: number;
  rowBytesVerified: boolean;
}

export declare class NgramTableReader {
  constructor(options: NgramTableReaderOptions);
  readonly rows: number;
  readonly rowWidth: number;
  readonly dtype: 'bf16';
  readonly rowBytes: number;
  readonly parts: readonly Readonly<Omit<NgramTablePart, 'byteLength'>>[];
  readonly rowBytesVerified: boolean;
  /** Rows for `addresses`, in order, as stored bytes ([addresses.length, rowBytes]). */
  gather(addresses: ArrayLike<number>, options?: { signal?: AbortSignal | null }): Promise<Uint8Array>;
  stats(): NgramTableStats;
  clear(): void;
}

/** The n-gram table section of doppler.qwen4-exp-checkpoint-headers/v1. */
export interface NgramCheckpointHeaderReceipt {
  ngramTable: {
    rowWidth: number;
    dtype: string;
    shards: Array<{ shard: number; file: string; rowBegin: number; rowEnd: number; fileByteBegin: number; fileByteEnd: number }>;
  };
}

/** Table parts from the checkpoint header receipt (file names and absolute byte offsets). */
export declare function ngramTablePartsFromCheckpointHeaders(receipt: NgramCheckpointHeaderReceipt): NgramTablePart[];

/** The loader's normalized location of one retained table tensor. */
export interface NgramTableTensorLocation {
  shardIndex: number;
  offset: number;
  size: number;
  shape: number[];
  dtype: string;
  role: string;
  spans?: ReadonlyArray<unknown> | null;
}

/**
 * Table parts from a loaded artifact's tensor locations: retained tensors
 * `${tensorPrefix}.shard_<k>.weight`, sources are shard indices for
 * loadShardRange. Fails unless the parts tile [0, rows) exactly.
 */
export { ngramTablePartsFromTensorLocations } from '../../../../loader/ngram-table-bindings.js';

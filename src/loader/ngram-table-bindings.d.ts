import type { DopplerLoader } from './doppler-loader.js';
import type { ResolvedNgramEmbedding, ResolvedNgramEmbeddingSession } from '../config/schema/ngram-embedding.schema.js';
import type { NgramTableReaderOptions } from '../inference/pipelines/text/ngram/ngram-table.js';
export declare function bindNgramTables(loader: Pick<DopplerLoader, 'tensorLocations' | 'shardCache'>,
  contract: ResolvedNgramEmbedding, session: ResolvedNgramEmbeddingSession): Map<number, NgramTableReaderOptions>;

/** Retained BF16 tensors must tile the declared rows in contiguous part order.
 * Sources are numeric artifact-shard indices; offsets refer to physical file bytes.
 * Transforms and multi-span tensors cannot provide this direct row layout. */
export declare function ngramTablePartsFromTensorLocations(tensorLocations: ReadonlyMap<string, unknown>,
  options: { tensorPrefix: string; rows: number; rowWidth: number }): import('../inference/pipelines/text/ngram/ngram-table.js').NgramTablePart[];

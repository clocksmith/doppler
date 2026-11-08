/**
 * Loading policies resolved before storage or GPU memory is touched.
 *
 * @module inference/pipelines/text/load-policy
 */

/**
 * Resolve the shard hash verification policy for a load: explicit option, then
 * runtime.loading.shardCache.verifyHashes. A storage context may satisfy the
 * policy (hashesTrusted) but never relax it.
 */
export declare function resolveShardHashVerification(input?: {
  options?: { verifyHashes?: boolean | null } | null;
  loadingConfig?: { shardCache?: { verifyHashes?: boolean | null } | null } | null;
  storageContext?: { verifyHashes?: boolean; hashesTrusted?: boolean } | null;
}): { verifyHashes: boolean; source: string };

/**
 * Decide whether a pipeline allocates an attention KV cache at all. Embedding
 * models with inference.session.skipEmbeddingKVCacheWrites=true never touch it.
 */
export declare function resolveKVCacheAllocation(
  manifest: { modelType?: string } | null | undefined,
  runtimeConfig: { inference?: { session?: { skipEmbeddingKVCacheWrites?: boolean } } } | null | undefined
): { allocate: boolean; reason: string | null };

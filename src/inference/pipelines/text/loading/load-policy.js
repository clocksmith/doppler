// Loading policies the text pipeline resolves before it touches storage or GPU
// memory. Both are decided from the runtime config and the manifest, never
// from whichever object happens to be present at load time.

// Shard hash verification is a loading policy. The runtime config owns it, an
// explicit loadWeights option overrides it, and a storage context may only
// satisfy it (hashesTrusted after a verified download), never relax it.
/**
 * @param {{
 *   options?: { verifyHashes?: boolean | null } | null,
 *   loadingConfig?: { shardCache?: { verifyHashes?: boolean | null } | null } | null,
 *   storageContext?: { verifyHashes?: boolean, hashesTrusted?: boolean } | null,
 * }} [input]
 * @returns {{ verifyHashes: boolean, source: string }}
 */
export function resolveShardHashVerification({ options = {}, loadingConfig = null, storageContext = null } = {}) {
  const configured = options?.verifyHashes ?? loadingConfig?.shardCache?.verifyHashes ?? null;
  if (typeof configured !== 'boolean') {
    throw new Error('runtime.loading.shardCache.verifyHashes is required.');
  }
  const source = options?.verifyHashes != null ? 'options.verifyHashes' : 'runtime.loading.shardCache.verifyHashes';
  if (!configured || !storageContext) {
    return { verifyHashes: configured, source };
  }
  if (storageContext.hashesTrusted === true) {
    return { verifyHashes: false, source: 'storageContext.hashesTrusted' };
  }
  if (storageContext.verifyHashes === false) {
    throw new Error(
      `${source} is true, but the supplied storage context was created with verifyHashes=false and does not ` +
      'mark its bytes as trusted. Create the context with verifyHashes=true or set the loading policy to false explicitly.'
    );
  }
  return { verifyHashes: true, source };
}

// Embedding models never decode. When the session policy skips KV writes for
// embedding-only prefill, no layer reads or writes attention KV, so allocating
// numLayers x maxSeqLen of key/value storage would reserve memory for nothing.
/**
 * @param {{ modelType?: string } | null | undefined} manifest
 * @param {{ inference?: { session?: { skipEmbeddingKVCacheWrites?: boolean } } } | null | undefined} runtimeConfig
 * @returns {{ allocate: boolean, reason: string | null }}
 */
export function resolveKVCacheAllocation(manifest, runtimeConfig) {
  const modelType = typeof manifest?.modelType === 'string' ? manifest.modelType.trim().toLowerCase() : '';
  const skipWrites = runtimeConfig?.inference?.session?.skipEmbeddingKVCacheWrites === true;
  if (modelType === 'embedding' && skipWrites) {
    return {
      allocate: false,
      reason: 'embedding model with inference.session.skipEmbeddingKVCacheWrites=true never reads or writes attention KV',
    };
  }
  return { allocate: true, reason: null };
}

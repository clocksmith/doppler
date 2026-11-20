import type { GenerationContext } from '../../inference/pipelines/text/generator/session/sequence-state.js';

export interface ServePrefixCachePolicy {
  enabled: boolean;
  maxEntries: number;
  maxBytes: number;
  maxEntryBytes: number;
  idleMs: number;
}

export interface ServePrefixCacheStats {
  enabled: boolean;
  entries: number;
  activeEntries: number;
  allocatedBytes: number;
  cachedTokens: number;
  reusedTokens: number;
  maxEntries: number;
  maxBytes: number;
  maxEntryBytes: number;
  hits: number;
  misses: number;
  evictions: number;
  busySkips: number;
}

export interface ServePrefixCacheLease {
  readonly key: string;
  readonly hit: boolean;
  readonly context: GenerationContext;
  release(options?: { discard?: boolean }): void;
}

export interface ServePrefixCache {
  readonly policy: Readonly<ServePrefixCachePolicy>;
  acquire(
    key: string | null,
    createContext: (options: { maxBytes: number }) => GenerationContext,
    owner?: object | null
  ): ServePrefixCacheLease | null;
  getStats(): Readonly<ServePrefixCacheStats>;
  close(): void;
}

export declare const DEFAULT_PREFIX_CACHE_POLICY: Readonly<ServePrefixCachePolicy>;
export declare function resolvePrefixCachePolicy(
  input?: boolean | Partial<ServePrefixCachePolicy>
): Readonly<ServePrefixCachePolicy>;
export declare function deriveServePrefixKey(
  modelId: string,
  messages: ReadonlyArray<{ role: string; content: string }>
): string | null;
export declare function createServePrefixCache(options?: {
  policy?: boolean | Partial<ServePrefixCachePolicy>;
  now?: () => number;
}): ServePrefixCache;

export interface NodeQuickstartCachedSource {
  modelId: string;
  baseUrl: string;
  manifest: object;
  trace?: Array<Record<string, unknown>>;
  cache: {
    state: 'hit' | 'imported';
    dir: string;
  };
}

export interface NodeQuickstartCacheProgress {
  phase: 'cache';
  percent: number | null;
  message: string;
}

export function resolveNodeQuickstartCachedSource(
  resolved: object,
  manifestPayload: { text?: string; manifest?: object },
  options?: { onProgress?: (event: NodeQuickstartCacheProgress) => void },
): Promise<NodeQuickstartCachedSource | null>;

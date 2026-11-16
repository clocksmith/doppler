/** Scheduling only. Execution and resource lifetime belong to the supplied operation. */
export function executeEmbeddingBatch<T>(prompts: readonly string[], request: { signal?: AbortSignal },
  execute: (prompt: string) => Promise<T>): Promise<T[]>;

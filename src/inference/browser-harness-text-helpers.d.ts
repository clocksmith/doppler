export declare function resolvePrompt(runtimeConfig: Record<string, unknown>): string;
export declare function getDefaultEmbeddingSemanticFixtures(): {
  retrievalCases: Array<Record<string, unknown>>;
  pairCases: Array<Record<string, unknown>>;
  minRetrievalTop1Acc: number;
  minPairAcc: number;
  pairMargin: number;
};
export declare function resolveBenchmarkRunSettings(
  runtimeConfig: Record<string, unknown>,
  source?: Record<string, unknown> | null
): {
  warmupRuns: number;
  timedRuns: number;
  prompt: string | Record<string, unknown>;
  promptLabel: string;
  maxTokens: number;
  sampling: Record<string, unknown>;
  seed?: number;
};
export declare function runEmbeddingSemanticChecks(
  pipeline: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function isCoherentOutput(tokens: Array<unknown>, output: unknown): boolean;
export declare function runGeneration(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runEmbedding(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;

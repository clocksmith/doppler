export declare function runEmbeddingSemanticChecks(
  pipeline: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runRerank(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runRerankSemanticChecks(
  pipeline: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
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
export declare function runSequenceEncoding(
  pipeline: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runImageTranscription(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;
export declare function runTextInference(
  pipeline: Record<string, unknown>,
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Promise<Record<string, unknown>>;

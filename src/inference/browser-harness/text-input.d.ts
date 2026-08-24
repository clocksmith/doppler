export declare const DEFAULT_IMAGE_TRANSCRIPTION_PROMPT: string;
export declare const DEFAULT_IMAGE_TRANSCRIPTION_SOFT_TOKEN_BUDGET: number;
export declare function getDefaultEmbeddingSemanticFixtures(): Record<string, unknown>;
export declare function getDefaultRerankSemanticFixtures(): Record<string, unknown>;
export declare function resolveEmbeddingSemanticFixtures(
  runtimeConfig: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Record<string, unknown>;
export declare function resolveEmbeddingSemanticStyle(pipeline: Record<string, unknown>): string;
export declare function formatEmbeddingSemanticText(text: string, kind: string, style: string): string;
export declare function resolvePrompt(runtimeConfig: Record<string, unknown>): string;
export declare function resolveRerankInput(
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Record<string, unknown>;
export declare function resolveRerankSemanticFixtures(
  runtimeConfig: Record<string, unknown>,
  options?: Record<string, unknown> | null
): Record<string, unknown>;
export declare function isStructuredPromptInput(value: unknown): boolean;
export declare function describePromptInput(promptInput: unknown): string;
export declare function resolveInferenceImagePayload(imageInput: Record<string, unknown>): Promise<Record<string, unknown>>;
export declare function resolveGenerationPromptInput(
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null,
  source?: Record<string, unknown> | null
): string | Record<string, unknown>;
export declare function resolveMaxTokens(runtimeConfig: Record<string, unknown>): number;
export declare function resolveAutomaticGenerationDiagnostics(
  runtimeConfig: Record<string, unknown>,
  runOverrides?: Record<string, unknown> | null
): Record<string, unknown> | null;
export declare function resolveBenchmarkRunSettings(
  runtimeConfig: Record<string, unknown>,
  source?: Record<string, unknown> | null,
  runOverrides?: { prompt?: string; maxTokens?: number } | null
): {
  warmupRuns: number;
  timedRuns: number;
  prompt: string | Record<string, unknown>;
  promptLabel: string;
  maxTokens: number;
  sampling: Record<string, unknown>;
  seed?: number;
};

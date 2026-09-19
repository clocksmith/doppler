import type { ResolvedSamplingConfig, SamplingCallOptions } from './sampling-config.js';

export interface GenerationRequestOptions extends SamplingCallOptions {
  maxTokens?: number;
  seed?: number;
  useSpeculative?: boolean;
  useChatTemplate?: boolean;
  stopSequences?: readonly string[];
}
export interface GenerationRequestRuntime {
  inference: {
    generation: { maxTokens: number; useSpeculative?: boolean };
    sampling: ResolvedSamplingConfig;
    chatTemplate?: { enabled?: boolean | null };
  };
}
export type ResolvedTextGenerationRequest = Readonly<Omit<ResolvedSamplingConfig, 'suppressTokenIds'> & {
  suppressTokenIds: readonly number[];
  maxTokens: number;
  seed: number | undefined;
  useSpeculative: boolean | undefined;
  useChatTemplate: boolean;
  stopSequences: readonly string[];
}>;
export type GenerationRequestEvidence = Omit<ResolvedTextGenerationRequest, 'seed' | 'useSpeculative'> & {
  readonly seed: number | null; readonly useSpeculative: boolean | null;
};
export function resolveChatTemplateSetting(options: Pick<GenerationRequestOptions, 'useChatTemplate'>,
  runtimeConfig: GenerationRequestRuntime, modelConfig?: { chatTemplateEnabled?: boolean }): boolean;
export function resolveTextGenerationRequest(options: GenerationRequestOptions, runtimeConfig: GenerationRequestRuntime,
  modelConfig?: { chatTemplateEnabled?: boolean }): ResolvedTextGenerationRequest;
export function generationRequestEvidence(request: ResolvedTextGenerationRequest): GenerationRequestEvidence;

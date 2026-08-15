import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../../generation/index.js';
import type { LogitsStepResult, PrefillResult } from '../../inference/pipelines/text/types.d.ts';
import type {
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerModelSource,
  DopplerModelSourceResolution,
  DopplerPersistentCacheReceipt,
} from './model-source.js';
import type { DopplerGenerationEvidence, DopplerModelHandle } from './model-session.js';
import type {
  DopplerGenerationResult,
  DopplerPromptInput,
  DopplerScopedGenerateOptions,
  DopplerScopedModelSession,
} from './scoped-session.js';

export type DopplerGenerateOptions = Omit<GenerateOptions, 'stopTokens'>;

export interface DopplerChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  evidence: DopplerGenerationEvidence;
}

export interface DopplerNamespace {
  (prompt: string, options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  load(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerModelHandle>;
  open(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerScopedModelSession>;
  generate(
    model: DopplerModelSource,
    input: DopplerPromptInput,
    options?: DopplerLoadOptions & DopplerScopedGenerateOptions
  ): Promise<DopplerGenerationResult>;
  text(prompt: string, options: DopplerCallOptions): Promise<string>;
  chat(messages: ChatMessage[], options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options: DopplerCallOptions): Promise<DopplerChatResponse>;
  evict(model: DopplerModelSource): Promise<boolean>;
  evictAll(): Promise<void>;
  listModels(): Promise<string[]>;
  listModelDetails(): Promise<Array<Record<string, unknown> & { modelId: string }>>;
  listPersistentModels(): Promise<Array<Record<string, unknown> & { modelId: string }>>;
  removePersistentModel(model: DopplerModelSource): Promise<boolean>;
}

export interface DopplerCallOptions extends DopplerGenerateOptions {
  model: DopplerModelSource;
  onProgress?: (event: DopplerLoadProgress) => void;
}

export interface DopplerRuntimeService {
  doppler: DopplerNamespace;
  load(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerModelHandle>;
  open(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerScopedModelSession>;
  generate(
    model: DopplerModelSource,
    input: DopplerPromptInput,
    options?: DopplerLoadOptions & DopplerScopedGenerateOptions
  ): Promise<DopplerGenerationResult>;
  clearModelCache(): void;
  resolveLoadProgressHandlers(options?: DopplerLoadOptions): {
    userProgress: ((event: DopplerLoadProgress) => void) | null;
    pipelineProgress: ((event: DopplerLoadProgress) => void) | null;
  };
  createDefaultNodeLoadProgressLogger(): (event: DopplerLoadProgress) => void;
}

export declare function createDopplerRuntimeService(options: {
  ensureWebGPUAvailable: () => Promise<void>;
  defaultLoadProgressLogger?: ((event: DopplerLoadProgress) => void) | null;
}): DopplerRuntimeService;

export declare function resolvePersistentBrowserLoadSource(
  loadSource: DopplerModelSourceResolution,
  cache: false | 'opfs' | undefined,
  onProgress?: ((event: DopplerLoadProgress) => void) | null,
  cacheSource?: (
    modelId: string,
    modelBaseUrl: string,
    onProgress: ((event: Record<string, unknown>) => void) | null,
    options: { expectedManifestHash: string }
  ) => Promise<{
    storageContext: Record<string, unknown>;
    storageBackend: 'opfs';
    cacheState: 'hit' | 'verified-hit' | 'manifest-refresh' | 'imported';
    fromCache: boolean;
    manifestHash: string;
    totalBytes: number;
  }>
): Promise<DopplerModelSourceResolution & {
  persistentCache?: DopplerPersistentCacheReceipt;
}>;

export type {
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerPersistentCacheReceipt,
  DopplerModelSource,
  DopplerModelSourceResolution,
} from './model-source.js';
export type {
  DopplerResolutionPolicy,
  ResolvedDopplerResolutionPolicy,
} from './resolution-policy.js';

export type { DopplerModelHandle } from './model-session.js';
export type {
  DopplerGenerationEvent,
  DopplerGenerationResult,
  DopplerObservationPolicyId,
  DopplerObservationTier,
  DopplerPromptInput,
  DopplerScopedGenerateOptions,
  DopplerScopedModelSession,
} from './scoped-session.js';
export type { PrefillResult, LogitsStepResult } from '../../inference/pipelines/text/types.d.ts';
export type { LoRAManifest, ExtensionBridgeClient } from './types.js';

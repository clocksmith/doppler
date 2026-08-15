import type { ChatMessage } from '../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../generation/index.js';
import type { LogitsStepResult, PrefillResult } from '../inference/pipelines/text/types.d.ts';
import type { RDRRManifest } from '../formats/rdrr/index.js';
import type {
  DopplerChatResponse,
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerResolutionPolicy,
  DopplerModelHandle,
  DopplerModelSource,
  DopplerGenerationResult,
  DopplerPromptInput,
  DopplerScopedGenerateOptions,
  DopplerScopedModelSession,
  LoRAManifest,
} from './runtime/index.js';

export type DopplerGenerateOptions = Omit<GenerateOptions, 'stopTokens'>;

export interface DopplerCallOptions extends DopplerGenerateOptions {
  model: DopplerModelSource;
  onProgress?: (event: DopplerLoadProgress) => void;
}

export type DopplerModel = DopplerModelHandle;

export interface DopplerNamespace {
  (prompt: string, options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  load(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerModel>;
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

export declare function load(
  model: DopplerModelSource,
  options?: DopplerLoadOptions
): Promise<DopplerModel>;

export declare function open(
  model: DopplerModelSource,
  options?: DopplerLoadOptions
): Promise<DopplerScopedModelSession>;

export declare function generate(
  model: DopplerModelSource,
  input: DopplerPromptInput,
  options?: DopplerLoadOptions & DopplerScopedGenerateOptions
): Promise<DopplerGenerationResult>;

export declare function createDefaultNodeLoadProgressLogger(): (event: DopplerLoadProgress) => void;

export declare function resolveLoadProgressHandlers(options?: DopplerLoadOptions): {
  userProgress: ((event: DopplerLoadProgress) => void) | null;
  pipelineProgress: ((event: DopplerLoadProgress) => void) | null;
};

export declare function clearModelCache(): void;

export declare const doppler: DopplerNamespace;

export type {
  DopplerChatResponse,
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerResolutionPolicy,
  DopplerModelSource,
  DopplerGenerationResult,
  DopplerPromptInput,
  DopplerScopedGenerateOptions,
  DopplerScopedModelSession,
};

export type {
  LogitsStepResult,
  PrefillResult,
  LoRAManifest,
  RDRRManifest,
};

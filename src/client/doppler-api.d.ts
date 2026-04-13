import type { ChatMessage } from '../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../generation/index.js';
import type { LogitsStepResult, PrefillResult } from '../inference/pipelines/text/types.d.ts';
import type { LoRAManifest } from '../experimental/adapters/lora-loader.js';
import type { RDRRManifest } from '../formats/rdrr/index.js';
import type {
  DopplerChatResponse,
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerModelHandle,
  DopplerModelSource,
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
  text(prompt: string, options: DopplerCallOptions): Promise<string>;
  chat(messages: ChatMessage[], options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options: DopplerCallOptions): Promise<DopplerChatResponse>;
  evict(model: DopplerModelSource): Promise<boolean>;
  evictAll(): Promise<void>;
  listModels(): Promise<string[]>;
}

export declare function load(
  model: DopplerModelSource,
  options?: DopplerLoadOptions
): Promise<DopplerModel>;

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
  DopplerModelSource,
};

export type {
  LogitsStepResult,
  PrefillResult,
  LoRAManifest,
  RDRRManifest,
};

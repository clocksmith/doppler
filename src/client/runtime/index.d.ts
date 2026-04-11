import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../../generation/index.js';
import type { LogitsStepResult, PrefillResult } from '../../inference/pipelines/text/types.d.ts';
import type {
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerModelSource,
} from './model-source.js';
import type { DopplerModelHandle } from './model-session.js';

export type DopplerGenerateOptions = Omit<GenerateOptions, 'stopTokens'>;

export interface DopplerChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface DopplerNamespace {
  (prompt: string, options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  load(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerModelHandle>;
  text(prompt: string, options: DopplerCallOptions): Promise<string>;
  chat(messages: ChatMessage[], options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options: DopplerCallOptions): Promise<DopplerChatResponse>;
  evict(model: DopplerModelSource): Promise<boolean>;
  evictAll(): Promise<void>;
  listModels(): Promise<string[]>;
}

export interface DopplerCallOptions extends DopplerGenerateOptions {
  model: DopplerModelSource;
  onProgress?: (event: DopplerLoadProgress) => void;
}

export interface DopplerRuntimeService {
  doppler: DopplerNamespace;
  load(model: DopplerModelSource, options?: DopplerLoadOptions): Promise<DopplerModelHandle>;
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

export type {
  DopplerLoadOptions,
  DopplerLoadProgress,
  DopplerModelSource,
  DopplerModelSourceResolution,
} from './model-source.js';

export type { DopplerModelHandle } from './model-session.js';
export type { PrefillResult, LogitsStepResult } from '../../inference/pipelines/text/types.d.ts';

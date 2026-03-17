import type { RDRRManifest } from '../formats/rdrr/index.js';
import type { GenerateOptions, KVCacheSnapshot } from '../generation/index.js';
import type { ChatMessage } from '../inference/pipelines/text/chat-format.js';
import type { LoRAManifest } from '../adapters/lora-loader.js';
import type { LogitsStepResult, PrefillResult } from '../inference/pipelines/text/types.d.ts';

export interface DopplerLoadProgress {
  phase: 'resolve' | 'manifest' | 'load' | 'ready';
  percent: number;
  message: string;
}

export interface DopplerLoadOptions {
  onProgress?: (event: DopplerLoadProgress) => void;
  runtimeConfig?: Record<string, unknown>;
}

export interface DopplerCallOptions extends GenerateOptions {
  model: string | { url: string };
  onProgress?: (event: DopplerLoadProgress) => void;
}

export interface DopplerChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface DopplerModel {
  generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string, void, void>;
  generateText(prompt: string, options?: GenerateOptions): Promise<string>;
  chat(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options?: GenerateOptions): Promise<DopplerChatResponse>;
  loadLoRA(adapter: string | LoRAManifest): Promise<void>;
  unloadLoRA(): Promise<void>;
  unload(): Promise<void>;
  readonly activeLoRA: string | null;
  readonly loaded: boolean;
  readonly modelId: string;
  readonly manifest: RDRRManifest;
  readonly deviceInfo: Record<string, unknown> | null;
  readonly advanced: {
    prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
    prefillWithLogits(prompt: string | ChatMessage[] | { messages: ChatMessage[] }, options?: GenerateOptions): Promise<PrefillResult>;
    decodeStepLogits(currentIds: number[], options?: GenerateOptions): Promise<LogitsStepResult>;
    generateWithPrefixKV(
      prefix: KVCacheSnapshot,
      prompt: string,
      options?: GenerateOptions
    ): AsyncGenerator<string, void, void>;
  };
}

export interface DopplerNamespace {
  (prompt: string, options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  load(
    model: string | { url: string } | { manifest: RDRRManifest; baseUrl?: string },
    options?: DopplerLoadOptions
  ): Promise<DopplerModel>;
  text(prompt: string, options: DopplerCallOptions): Promise<string>;
  chat(messages: ChatMessage[], options: DopplerCallOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options: DopplerCallOptions): Promise<DopplerChatResponse>;
  evict(model: string | { url: string }): Promise<boolean>;
  evictAll(): Promise<void>;
  listModels(): Promise<string[]>;
}

export declare function load(
  model: string | { url: string } | { manifest: RDRRManifest; baseUrl?: string },
  options?: DopplerLoadOptions
): Promise<DopplerModel>;

export declare function createDefaultNodeLoadProgressLogger(): (event: DopplerLoadProgress) => void;

export declare function resolveLoadProgressHandlers(options?: DopplerLoadOptions): {
  userProgress: ((event: DopplerLoadProgress) => void) | null;
  pipelineProgress: ((event: DopplerLoadProgress) => void) | null;
};

export declare const doppler: DopplerNamespace;

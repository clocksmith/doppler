import type { InferencePipeline, KVCacheSnapshot } from '../../inference/pipelines/text.js';
import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../../generation/index.js';
import type { LoRAManifest } from '../../experimental/adapters/lora-loader.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { LogitsStepResult, PrefillResult } from '../../inference/pipelines/text/types.d.ts';

export type DopplerGenerateOptions = Omit<GenerateOptions, 'stopTokens'>;

export interface DopplerChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface DopplerModelHandle {
  generate(prompt: string, options?: DopplerGenerateOptions): AsyncGenerator<string, void, void>;
  generateText(prompt: string, options?: DopplerGenerateOptions): Promise<string>;
  chat(messages: ChatMessage[], options?: DopplerGenerateOptions): AsyncGenerator<string, void, void>;
  chatText(messages: ChatMessage[], options?: DopplerGenerateOptions): Promise<DopplerChatResponse>;
  loadLoRA(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;
  activateLoRAFromTrainingOutput(
    trainingOutput:
      | string
      | {
        adapter?: LoRAManifest | RDRRManifest | string;
        adapterManifest?: LoRAManifest | RDRRManifest;
        adapterManifestJson?: string;
        adapterManifestUrl?: string;
        adapterManifestPath?: string;
      }
      | null
      | undefined
  ): Promise<{
    activated: boolean;
    adapterName: string | null;
    source: string | null;
    reason: string | null;
  }>;
  unloadLoRA(): Promise<void>;
  unload(): Promise<void>;
  readonly activeLoRA: string | null;
  readonly loaded: boolean;
  readonly modelId: string;
  readonly manifest: unknown;
  readonly deviceInfo: Record<string, unknown> | null;
  readonly advanced: {
    prefillKV(prompt: string, options?: DopplerGenerateOptions): Promise<KVCacheSnapshot>;
    prefillWithLogits(
      prompt: string | ChatMessage[] | { messages: ChatMessage[] },
      options?: DopplerGenerateOptions
    ): Promise<PrefillResult>;
    decodeStepLogits(currentIds: number[], options?: DopplerGenerateOptions): Promise<LogitsStepResult>;
    generateWithPrefixKV(
      prefix: KVCacheSnapshot,
      prompt: string,
      options?: DopplerGenerateOptions
    ): AsyncGenerator<string, void, void>;
  };
}

export declare function assertSupportedGenerationOptions(options?: Record<string, unknown>): void;

export declare function createModelHandle(
  pipeline: InferencePipeline,
  resolved: {
    modelId: string;
  }
): DopplerModelHandle;

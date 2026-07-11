import type { InferencePipeline, KVCacheSnapshot, PromptInput } from '../../inference/pipelines/text.js';
import type { ChatMessage } from '../../inference/pipelines/text/chat-format.js';
import type { GenerateOptions } from '../../generation/index.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';
import type { LogitsStepResult, PrefillResult } from '../../inference/pipelines/text/types.d.ts';
import type { LoRAManifest } from './types.js';
import type { LoRALoadOptions } from './lora.js';

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
  resetGenerationState(): void;
  loadLoRA(adapter: LoRAManifest | RDRRManifest | string, loadOptions?: LoRALoadOptions): Promise<void>;
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
  readonly manifestHash: string | null;
  readonly manifest: unknown;
  readonly deviceInfo: Record<string, unknown> | null;
  readonly advanced: {
    tokenizeText(text: string): number[];
    prefillKV(prompt: string, options?: DopplerGenerateOptions): Promise<KVCacheSnapshot>;
    resetToSeqLen(seqLen: number): void;
    prefillWithLogits(
      prompt: string | ChatMessage[] | { messages: ChatMessage[] },
      options?: DopplerGenerateOptions
    ): Promise<PrefillResult>;
    prefillWithTokenLogits(
      prompt: PromptInput,
      tokenIds: readonly number[],
      options?: DopplerGenerateOptions
    ): Promise<{
      seqLen: number;
      tokens: number[];
      tokenIds: number[];
      logits: Float32Array;
      logitsByTokenId: Record<number, number>;
      phase?: Record<string, unknown> | null;
    }>;
    prefillWithTokenLogitsFromKV(
      prefix: KVCacheSnapshot,
      prompt: PromptInput,
      tokenIds: readonly number[],
      options?: DopplerGenerateOptions
    ): Promise<{
      seqLen: number;
      prefixTokens: number[];
      tokens: number[];
      tokenIds: number[];
      logits: Float32Array;
      logitsByTokenId: Record<number, number>;
      phase?: Record<string, unknown> | null;
    }>;
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
    manifestHash?: string | null;
  }
): DopplerModelHandle;

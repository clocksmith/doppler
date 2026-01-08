import type { KVCacheSnapshot } from '../../inference/pipeline.js';
import type { GenerateOptions, ChatMessage, ChatResponse } from './types.js';

export declare function generate(prompt: string, options?: GenerateOptions): AsyncGenerator<string>;

export declare function prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;

export declare function generateWithPrefixKV(
  prefix: KVCacheSnapshot,
  prompt: string,
  options?: GenerateOptions
): AsyncGenerator<string>;

export declare function formatGemmaChat(messages: ChatMessage[]): string;

export declare function formatLlama3Chat(messages: ChatMessage[]): string;

export declare function formatChatMessages(messages: ChatMessage[]): string;

export declare function dopplerChat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;

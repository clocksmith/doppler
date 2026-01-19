/**
 * chat-controller.d.ts - Chat state and generation controller
 *
 * @module app/chat-controller
 */

import type { Pipeline } from '../src/inference/pipeline.js';

export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

export interface SamplingParams {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
}

export interface GenerationStats {
  tokensPerSec: number;
}

export interface GenerationResult {
  text: string;
  tokenCount: number;
  tokensPerSec: number;
}

export interface ChatControllerCallbacks {
  onUserMessage?: (message: string) => void;
  onGenerationStart?: () => void;
  onToken?: (token: string) => void;
  onStats?: (stats: GenerationStats) => void;
  onGenerationComplete?: (result: GenerationResult) => void;
  onGenerationAborted?: (text: string) => void;
  onGenerationError?: (error: Error) => void;
}

export declare class ChatController {
  constructor(callbacks?: ChatControllerCallbacks);

  get isGenerating(): boolean;
  get messages(): ChatMessage[];

  setSamplingParams(params: SamplingParams): void;
  getSamplingParams(): Required<SamplingParams>;

  generate(message: string, pipeline: Pipeline): Promise<string>;
  stop(): void;
  clear(pipeline?: Pipeline | null): void;
}

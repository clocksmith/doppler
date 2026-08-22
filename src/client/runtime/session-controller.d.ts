import type { TargetPlan } from '../../config/target-plan.js';
import type { CommandExecutor } from './command-executor.js';
import type { ResourceBinder } from './resource-binder.js';

export interface GenerationRunOptions {
  prompt?: unknown;
  promptTokens?: number[];
  maxTokens: number;
  maxSeqLen: number;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  repetitionPenaltyWindow: number;
  useChatTemplate: boolean;
  seed?: number;
  suppressTokenIds?: number[];
  stopSequences?: string[];
  signal?: AbortSignal | null;
}

export interface SessionController {
  generateTokens(targetPlan: TargetPlan, options: GenerationRunOptions): AsyncGenerator<number, void, void>;
  close(): Promise<void>;
}

export declare function samplePackLogits(logits: ArrayLike<number>, contextTokens: number[], options: GenerationRunOptions, tokenContract?: Record<string, unknown>): number;
export declare function createSessionController(commandExecutor: CommandExecutor, resourceBinder: ResourceBinder, program: object): SessionController;

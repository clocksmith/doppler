import type { TargetPlan } from '../../config/target-plan.js';
import type { CommandExecutor } from './command-executor.js';
import type { ResourceBinder } from './resource-binder.js';

export interface GenerationRunOptions {
  promptTokens?: number[];
  maxTokens?: number;
  signal?: AbortSignal | null;
}

export interface SessionController {
  generateTokens(
    targetPlan: TargetPlan,
    options?: GenerationRunOptions
  ): AsyncGenerator<number, void, void>;
}

export declare function createSessionController(
  commandExecutor: CommandExecutor,
  resourceBinder: ResourceBinder
): SessionController;

import type { ResourceBinder } from './resource-binder.js';

export interface PhaseExecutionResult {
  ok: boolean;
  phase: string;
  commandCount: number;
  executedAt: number;
}

export interface CommandExecutor {
  executePhase(
    phase: string,
    commands?: unknown[],
    options?: { signal?: AbortSignal }
  ): Promise<PhaseExecutionResult>;
}

export declare function createCommandExecutor(
  device: unknown,
  resourceBinder: ResourceBinder
): CommandExecutor;

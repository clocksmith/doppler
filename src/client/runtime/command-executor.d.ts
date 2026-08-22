import type { ResourceBinder } from './resource-binder.js';

export interface PhaseExecutionResult {
  ok: true;
  phase: string;
  commandCount: number;
  results: unknown[];
}

export interface CommandExecutor {
  executePhase(phase: string, commands: Array<Record<string, unknown>>, options?: Record<string, unknown>): Promise<PhaseExecutionResult>;
  clearPipelineCache(): void;
}

export declare function createCommandExecutor(device: unknown, resourceBinder: ResourceBinder, program?: unknown): CommandExecutor;

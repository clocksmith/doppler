import type { ResourceBinder } from './resource-binder.js';

export interface DispatchExecutionResult {
  kind: 'dispatch';
  moduleId: string;
  workgroups: [number, number, number];
  outcome: 'submitted' | 'completed';
}

/** Aborting after submission suppresses completion; it does not interrupt GPU work. */
export interface CommandExecutionFailure extends Error {
  code: 'COMMAND_ABORTED' | 'COMMAND_DEVICE_LOST';
  submission: 'not-submitted' | 'submitted';
}

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

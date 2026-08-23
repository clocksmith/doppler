import type { DopplerPackV2 } from '../../config/pack-v2.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DopplerModelHandle } from './model-session.js';

export interface PackProgramAdapter {
  executionGraphHash: string;
  getInitialExecutionIdentity(): InitialExecutionIdentity;
  tokenize(prompt: unknown, options?: Record<string, unknown>): number[];
  decodeTokens(tokenIds: number[]): string;
  getTokenContract(): Record<string, unknown>;
  reset(): void;
  executePhase(phase: string, request: Record<string, unknown>): Promise<unknown>;
  releaseStepResult(result: Record<string, unknown> | null): void;
  close(): Promise<void>;
}

export declare function createPackProgramAdapter(
  modelHandle: DopplerModelHandle,
  pack: DopplerPackV2,
  targetPlan: TargetPlan
): PackProgramAdapter;

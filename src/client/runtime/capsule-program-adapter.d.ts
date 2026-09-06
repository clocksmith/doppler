import type { DopplerCapsule } from '../../config/capsule.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DopplerModelHandle } from './model-session.js';
import type { CapsuleRerankRequest } from './capsule-rerank.js';

export interface CapsuleProgramAdapter {
  executionGraphHash: string;
  getActiveAdapterIdentity(): Readonly<Record<string, unknown>> | null;
  loadAdapter(manifest: Record<string, unknown>, control: { bytes: Uint8Array; signal: AbortSignal }): Promise<void>;
  unloadAdapter(): Promise<void>;
  getInitialExecutionIdentity(): InitialExecutionIdentity;
  tokenize(prompt: unknown, options?: Record<string, unknown>): number[];
  decodeTokens(tokenIds: number[]): string;
  getTokenContract(): Record<string, unknown>;
  reset(): void;
  rerank(request: Omit<CapsuleRerankRequest, 'application'>): ReturnType<DopplerModelHandle['rerankWithEvidence']>;
  embed(text: string, options?: { signal?: AbortSignal }): ReturnType<DopplerModelHandle['embedWithEvidence']>;
  encodeSequence(sequence: string, options?: Record<string, unknown>): ReturnType<DopplerModelHandle['encodeSequence']>;
  executePhase(phase: string, request: Record<string, unknown>): Promise<unknown>;
  releaseStepResult(result: Record<string, unknown> | null): void;
  close(): Promise<void>;
}

export declare function createCapsuleProgramAdapter(
  modelHandle: DopplerModelHandle,
  capsule: DopplerCapsule,
  targetPlan: TargetPlan
): CapsuleProgramAdapter;

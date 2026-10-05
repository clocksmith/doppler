import type { DopplerCapsule } from '../../config/capsule.js';
import type { TargetPlan } from '../../config/target-plan.js';
import type { InitialExecutionIdentity } from '../../config/initial-execution-identity.js';
import type { DopplerModelHandle } from './model-session.js';
import type { CapsuleRerankRequest } from './capsule-rerank.js';
import type { LoRAWeightLayoutName } from '../../config/lora-layouts.js';

export interface CapsuleProgramAdapter {
  residentPartition?: import('../../inference/pipelines/text/resident-partition-contract.js').ResidentPartitionSession;
  executionGraphHash: string;
  getActiveAdapterIdentity(): Readonly<Record<string, unknown>> | null;
  loadAdapter(manifest: Record<string, unknown>, control: { bytes: Uint8Array; signal: AbortSignal; weightsLayout: LoRAWeightLayoutName }): Promise<void>;
  unloadAdapter(): Promise<void>;
  getInitialExecutionIdentity(): InitialExecutionIdentity;
  tokenize(prompt: unknown, options?: Record<string, unknown>): number[];
  createIncrementalDecoder(): import('../../inference/tokenizers/bundled/incremental-decoder.js').IncrementalTokenDecoder;
  decodeTokens(tokenIds: number[]): string;
  getTokenContract(): import('../../inference/generation-step.js').GenerationTokenContract;
  reset(): void;
  scoreChoices(request: import('../../config/choice-scoring.js').ChoiceScoringRequest,
    control?: { signal?: AbortSignal }): ReturnType<DopplerModelHandle['scoreChoices']>;
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
  targetPlan: TargetPlan,
  registries?: import('../../config/execution-registry-contract.js').ResolvedExecutionRegistries | null
): CapsuleProgramAdapter;

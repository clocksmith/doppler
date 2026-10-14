import type { ActivationLayout, AnyLayerPartitionPlan } from './layer-partition-contract.js';
import type { GenerationOptions, ResolvedGenerationOptions, CanonicalGenerationPolicy } from '../../../config/generation-contract.js';
import type { ResidentTargetTransition } from '../../resident-target-state.js';
import type { AdmissionOnlyResult } from '../../resident-state-contract.js';
import type { IncrementalTokenDecoder } from '../../tokenizers/bundled/incremental-decoder.js';
import type { GenerationTokenContract } from '../../generation-step.js';
import type { KernelPipelineIdentity } from '../../../gpu/kernels/pipeline-cache.js';
export interface PartitionTiming {
  inputUploadMs: number; encodeMs: number; submitWaitMs: number; activationReadbackMs: number;
  logitsMs: number; gpuKernelsMs: number | null;
}
export interface ResidentPartitionLimits {
  maxTokens: number; maxPromptTokens: number; maxActivationBytes: number;
  maxOutputCharacters: number; maxAttempts: number; maxConcurrentAttempts: number;
  /**
   * Host-declared bound on opening pipeline preparation, in milliseconds. Exceeding it fails
   * opening; a provider creation already in flight stays owned until it settles.
   */
  maxPreparationMs: number;
}
export interface ResidentPartitionAllocation {
  model: { id: string; identity: string };
  plan: AnyLayerPartitionPlan; planId: string; index: number; participantId: string;
  limits: ResidentPartitionLimits; generation: GenerationOptions;
  /** Explicit opt-in; never inferred from a legacy generation object. */
  canonicalGeneration?: { policy: CanonicalGenerationPolicy; executionIdentity: string };
}
export interface ResidentPartitionIdentity {
  modelId: string; modelIdentity: string; planId: string; threadId: string;
  attemptId: string; participantA: string; participantB: string;
  requesterId?: string; placementGeneration?: number;
}
/** Route metadata binds attempt continuity; the host separately authenticates grants and transport. */
export interface ResidentOrderedPartitionIdentity {
  schema: 'doppler.resident-partition-attempt/v2';
  modelId: string; modelIdentity: string; planId: string; threadId: string; attemptId: string;
  requesterId: string; placementGeneration: number;
  stages: readonly { index: number; participantId: string; transportId: string; residentId: string; executionId: string }[];
}
export type ResidentExecutionIdentity = ResidentPartitionIdentity | ResidentOrderedPartitionIdentity;
/** A part of a stage program whose pipelines are not enumerated and are still created on first use. */
export interface ResidentLazyPipelineScope {
  scope: 'program' | 'stage';
  reason: string;
}
/** Opening receipt: every enumerated pipeline was created on the stage device before it reported ready. */
export interface ResidentPipelinePreparation {
  schema: 'doppler.resident-pipeline-preparation/v1';
  /** True when no part of the stage program remains lazy. */
  complete: boolean;
  count: number;
  durationMs: number;
  /** Compiled programs in creation order; `key` is `shaderFile#entryPoint` plus canonical constants. */
  pipelines: Array<KernelPipelineIdentity & { key: string; roles: string[] }>;
  lazy: ResidentLazyPipelineScope[];
}
/**
 * Opening progress for pipeline preparation, delivered through the opener's `onLoadProgress` port.
 * `outstanding` means a started provider creation has not settled: the opening still owns it and
 * has not released the stage, whatever the status.
 */
export interface ResidentPipelinePreparationProgress {
  phase: 'pipelines';
  /** `preparing` creates `key`; `cancelling` waits for `key` after an abort; the rest are terminal. */
  status: 'preparing' | 'cancelling' | 'cancelled' | 'timed-out' | 'failed' | 'prepared';
  /** Pipelines created so far, of `total`. */
  prepared: number;
  total: number;
  key: string | null;
  elapsedMs: number;
  /** Wall-clock start (ms since the epoch), so observers can show live elapsed time between events. */
  startedAt: number;
  /** The allocation's `limits.maxPreparationMs`; null when the stage has nothing to prepare. */
  maxDurationMs: number | null;
  outstanding: boolean;
  /** False when part of the stage program remains lazy. */
  complete: boolean;
  error?: string;
}
export interface ResidentPartitionDescriptor {
  schema: 'doppler.resident-partition/v1' | 'doppler.resident-partition/v2'; ready: boolean;
  stageCount?: number;
  modelId: string; modelIdentity: string; planId: string; index: number;
  layerRange: readonly number[]; residentWeightBytes: number;
  generationDigest: string;
  canonicalGenerationId?: string;
  /** Pipelines created before readiness; absent from residents that predate opening preparation. */
  pipelinePreparation?: ResidentPipelinePreparation;
}
export interface ResidentRecoveryCapabilities {
  readonly schema: 'doppler.resident-recovery/v1';
  readonly inputReplay: boolean;
  readonly checkpointExport: boolean;
  readonly checkpointImport: boolean;
}
export interface ResidentPartitionTokenizationRequest {
  messages: unknown; identity: ResidentExecutionIdentity; signal: AbortSignal;
}
export interface ResidentPartitionTokenizationResult {
  modelIdentity: string; tokenIds: number[];
  generation: Readonly<ResolvedGenerationOptions>;
}
export interface ResidentPartitionStep {
  identity: ResidentExecutionIdentity; step: number; tokenPosition: number; inputTokenCount: number;
  maxTokens: number; generation: GenerationOptions; continuation: unknown; signal: AbortSignal;
  generationId?: string; baseCursorId?: string | null;
}
export interface ResidentPartitionARequest extends ResidentPartitionStep { tokenIds: number[]; }
export interface ResidentPartitionBRequest extends ResidentPartitionStep {
  inputTokenIds: number[];
  activation: ActivationLayout & { shape: number[]; dtype: string; seqOffset: number; step: number;
    tensorData: Float32Array | Uint16Array };
}
export interface ResidentPartitionAResult {
  activationTensor: ActivationLayout & { shape: number[]; dtype: 'f16' | 'f32'; data: ArrayBuffer; step: number; seqOffset: number };
  continuation: unknown;
  metrics: ResidentPartitionMetrics;
}
export interface ResidentPartitionMetrics extends PartitionTiming {
  outputUsage?: { rawCharacters: number; rawTokens: number; finalCharacters: number; state: string };
  executionMs: number; logitsReadbackMs: number; samplingMs: number;
  memory: Pick<import('../../../memory/device-budget.js').DeviceMemorySnapshot,
    'maxBytes' | 'liveBytes' | 'peakBytes' | 'categories'> | null;
}
export interface ResidentPartitionBResult {
  identity: ResidentExecutionIdentity; step: number; tokenPosition: number;
  tokenId: number; delta: string; done: boolean; stopReason: string | null;
  continuation: unknown; logits: Float32Array; metrics: ResidentPartitionMetrics;
}
export type ResidentCanonicalARequest = Omit<ResidentPartitionARequest, 'generation' | 'maxTokens'> & {
  generationId: string; baseCursorId: string | null;
};
export type ResidentCanonicalBRequest = Omit<ResidentPartitionBRequest, 'generation' | 'maxTokens'> & {
  generationId: string; baseCursorId: string | null;
};
export interface ResidentCanonicalBResult extends ResidentPartitionBResult { transition: Readonly<ResidentTargetTransition>; }
export interface ResidentCanonicalGeneration {
  readonly policy: Readonly<CanonicalGenerationPolicy>; readonly executionIdentity: string; readonly generationId: string;
}
export interface ResidentStageRequest extends Omit<ResidentPartitionStep, 'identity'> {
  identity: ResidentOrderedPartitionIdentity; partitionIndex: number; inputTokenIds: number[];
  generationDigest: string; activation?: ResidentPartitionBRequest['activation'];
}
export type ResidentStageResult = (ResidentPartitionAResult | ResidentPartitionBResult) & {
  identity: ResidentOrderedPartitionIdentity; partitionIndex: number; step: number; tokenPosition: number;
};
export interface ResidentPartitionSession {
  getDescriptor(): ResidentPartitionDescriptor;
  /** Live continuation is not portable recovery. This implementation reports all flags false. */
  getRecoveryCapabilities(): ResidentRecoveryCapabilities;
  getCanonicalGeneration(): ResidentCanonicalGeneration | null;
  admitCanonicalPrompt(request: { identity: ResidentExecutionIdentity; tokenIds: number[]; generationId: string; signal: AbortSignal }):
    Promise<Readonly<AdmissionOnlyResult> | Readonly<{ kind: 'prefill'; tokenIds: readonly number[]; generationId: string }>>;
  executeCanonicalGroup0(request: ResidentCanonicalARequest): Promise<ResidentPartitionAResult>;
  executeCanonicalGroup1(request: ResidentCanonicalBRequest): Promise<ResidentCanonicalBResult>;
  /** Settled numerical input-stage commit; not an authenticated transport acknowledgment. */
  commitCanonicalTransition(request: { identity: ResidentExecutionIdentity; baseCursorId: string | null;
    transition: ResidentTargetTransition; signal: AbortSignal }): Promise<Readonly<ResidentTargetTransition>>;
  tokenize(request: ResidentPartitionTokenizationRequest): Promise<ResidentPartitionTokenizationResult>;
  /** Ordered v3 plan only. Entry embeds, intermediate returns activation, final alone samples. */
  executeStage(request: ResidentStageRequest): Promise<ResidentStageResult>;
  executeGroup0(request: ResidentPartitionARequest): Promise<ResidentPartitionAResult>;
  executeGroup1(request: ResidentPartitionBRequest): Promise<ResidentPartitionBResult>;
  closeAttempt(request: { identity: ResidentExecutionIdentity }): Promise<void>;
  close(): Promise<void>;
}
export interface ResidentPartitionTokenPorts {
  tokenize(prompt: unknown, options: { useChatTemplate: boolean }): number[] | Promise<number[]>;
  createIncrementalDecoder(): IncrementalTokenDecoder;
  createRawIncrementalDecoder?(): IncrementalTokenDecoder;
  getTokenContract(): GenerationTokenContract;
}
export function resolveResidentPartitionAllocation(manifest: { modelId?: unknown; architecture?: unknown },
  manifestHash: string, allocation: ResidentPartitionAllocation): ResidentPartitionAllocation;
export function assertResidentPartitionIdentity(identity: ResidentExecutionIdentity,
  allocation: ResidentPartitionAllocation): string;
export function resolveResidentCanonicalGeneration(allocation: ResidentPartitionAllocation): ResidentCanonicalGeneration | null;

export function hashPartitionGeneration(generation: GenerationOptions): `sha256:${string}`;

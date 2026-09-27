import type { LayerPartitionPlan } from './layer-partition-contract.js';
import type { GenerationOptions, ResolvedGenerationOptions } from '../../../config/generation-contract.js';
import type { IncrementalTokenDecoder } from '../../tokenizers/bundled/incremental-decoder.js';
import type { GenerationTokenContract } from '../../generation-step.js';
export interface ResidentPartitionLimits {
  maxTokens: number; maxPromptTokens: number; maxActivationBytes: number;
  maxOutputCharacters: number; maxAttempts: number; maxConcurrentAttempts: number;
}
export interface ResidentPartitionAllocation {
  model: { id: string; identity: string };
  plan: LayerPartitionPlan; planId: string; index: 0 | 1; participantId: string;
  limits: ResidentPartitionLimits; generation: GenerationOptions;
}
export interface ResidentPartitionIdentity {
  modelId: string; modelIdentity: string; planId: string; threadId: string;
  attemptId: string; participantA: string; participantB: string;
}
export interface ResidentPartitionDescriptor {
  schema: 'doppler.resident-partition/v1'; ready: boolean;
  modelId: string; modelIdentity: string; planId: string; index: 0 | 1;
  layerRange: readonly number[]; residentWeightBytes: number;
  generationDigest: string;
}
export interface ResidentPartitionTokenizationRequest {
  messages: unknown; identity: ResidentPartitionIdentity; signal: AbortSignal;
}
export interface ResidentPartitionStep {
  identity: ResidentPartitionIdentity; step: number; tokenPosition: number; inputTokenCount: number;
  maxTokens: number; generation: GenerationOptions; continuation: unknown; signal: AbortSignal;
}
export interface ResidentPartitionARequest extends ResidentPartitionStep { tokenIds: number[]; }
export interface ResidentPartitionBRequest extends ResidentPartitionStep {
  inputTokenIds: number[];
  activation: { shape: number[]; dtype: string; seqOffset: number; step: number;
    tensorData: Float32Array | Uint16Array };
}
export interface ResidentPartitionAResult {
  activationTensor: { shape: number[]; dtype: 'f16' | 'f32'; data: ArrayBuffer; step: number; seqOffset: number };
  continuation: unknown;
}
export interface ResidentPartitionBResult {
  identity: ResidentPartitionIdentity; step: number; tokenPosition: number;
  tokenId: number; delta: string; done: boolean; stopReason: string | null;
  continuation: unknown; logits: Float32Array;
}
export interface ResidentPartitionSession {
  getDescriptor(): ResidentPartitionDescriptor;
  tokenize(request: ResidentPartitionTokenizationRequest): Promise<{ modelIdentity: string; tokenIds: number[];
    generation: Readonly<ResolvedGenerationOptions> }>;
  executeGroup0(request: ResidentPartitionARequest): Promise<ResidentPartitionAResult>;
  executeGroup1(request: ResidentPartitionBRequest): Promise<ResidentPartitionBResult>;
  closeAttempt(request: { identity: ResidentPartitionIdentity }): Promise<void>;
  close(): Promise<void>;
}
export interface ResidentPartitionTokenPorts {
  tokenize(prompt: unknown, options: { useChatTemplate: boolean }): number[];
  createIncrementalDecoder(): IncrementalTokenDecoder;
  getTokenContract(): GenerationTokenContract;
}
export function resolveResidentPartitionAllocation(manifest: { modelId?: unknown; architecture?: unknown },
  manifestHash: string, allocation: ResidentPartitionAllocation): ResidentPartitionAllocation;
export function assertResidentPartitionIdentity(identity: ResidentPartitionIdentity,
  allocation: ResidentPartitionAllocation): string;

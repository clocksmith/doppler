import type { PipelineStats } from '../../inference/pipelines/text/types.js';
import type { ResolvedDopplerResolutionPolicy } from '../runtime/resolution-policy.js';
import type {
  DopplerEmbeddingEvidence,
  DopplerGenerationBackendIdentity,
  DopplerGenerationConfigEvidence,
  DopplerGenerationEvidence,
  DopplerResolutionIdentity,
  DopplerResolvedExecutionIdentity,
} from './model-session.js';

export const RERANK_EVIDENCE_SCHEMA: 'doppler_rerank_evidence/v1';
export declare function hashEvidenceValue(value: unknown): `sha256:${string}`;
export declare function normalizeSha256Identity(value: unknown, label: string): `sha256:${string}`;
export declare function buildGenerationBackendIdentity(input?: {
  deviceInfo?: Record<string, unknown> | null;
  kernelCapabilities?: Record<string, unknown> | null;
  stats?: PipelineStats | null;
}): DopplerGenerationBackendIdentity;

interface ResolutionEvidenceInput {
  logicalModelId: string;
  modelId: string;
  manifestHash: string | null;
  resolvedRuntimeSessionId: string | null;
  activeAdapter: { name?: string; id?: string; digest?: string } | null;
  backendIdentity: DopplerGenerationBackendIdentity;
  resolutionPolicy: ResolvedDopplerResolutionPolicy;
}

export declare function buildResolutionIdentity(input: ResolutionEvidenceInput): Promise<{
  resolvedModelId: string;
  resolvedArtifactVariantId: `sha256:${string}`;
  runtimeSessionId: `sha256:${string}`;
  runtimeIdentity: DopplerResolvedExecutionIdentity['runtime'];
  executionIdentity: DopplerResolvedExecutionIdentity;
  resolution: DopplerResolutionIdentity;
}>;
export declare function buildGenerationEvidence(input: ResolutionEvidenceInput & {
  outputText: string;
  tokenIds: number[];
  generationConfig: DopplerGenerationConfigEvidence;
  stats: PipelineStats | null;
}): Promise<DopplerGenerationEvidence>;
export declare function buildEmbeddingEvidence(input: ResolutionEvidenceInput & {
  prompt: string;
  result: { embedding: ArrayLike<number>; tokens: ArrayLike<number>; seqLen: number; embeddingMode: string };
  stats: PipelineStats | null;
}): Promise<DopplerEmbeddingEvidence>;

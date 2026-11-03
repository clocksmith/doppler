import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { PipelineStats } from '../types.js';

export interface PrefillCacheObservation {
  currentSeqLen: number;
  kvCache?: { hasGPUCache?: () => boolean } | null;
}

export declare function logPrefillCacheState(state: PrefillCacheObservation): void;

export declare function snapshotPrefillPhase(
  stats: Partial<PipelineStats>,
  phase: { inputMs: number; prefillMs: number; tokens: number }
): { inputMs: number; prefillMs: number; tokens: number } & Record<
  'prefillRecordMs' | 'prefillRecordOps' | 'prefillRecordPasses' | 'prefillSubmitWaitMs' | 'gpuPrefillMs',
  number | null
> & { prefillRecordOpLabels: PipelineStats['prefillRecordOpLabels'] | null };

export declare function tracePrefillEmbeddingIds(
  embeddingInputIds: ArrayLike<number>,
  numTokens: number,
  embeddingOverride: { offset: number; prefixLength: number } | null
): void;

export declare function probePrefillEmbedding(
  stage: string,
  tensor: Tensor,
  options: {
    numTokens: number;
    hiddenSize: number;
    state: Record<string, unknown>;
    recorder: CommandRecorder | null;
  }
): Promise<void>;

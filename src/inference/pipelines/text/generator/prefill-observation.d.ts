import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';

export interface PrefillCacheObservation {
  currentSeqLen: number;
  kvCache?: { hasGPUCache?: () => boolean } | null;
}

export declare function logPrefillCacheState(state: PrefillCacheObservation): void;

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

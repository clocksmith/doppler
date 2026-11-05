import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { QsaSelection } from '../../../../gpu/kernels/qsa-indexer.js';

/** Selection prepared by the hyper-connection layer executor for one QSA layer call. */
export interface QsaAttentionState extends QsaSelection {
  layerIdx: number;
  numTokens: number;
  /** Cache rows visible to the last query (startPos + numTokens). */
  visibleRows: number;
}

export interface QsaIndexedAttentionOptions {
  state: { qsa?: QsaAttentionState | null } | null;
  qTensor: Tensor;
  cachedKTensor: Tensor;
  cachedVTensor: Tensor;
  cachedKDtype: string;
  cachedVDtype: string;
  kvState: { kvLayout?: string; kvStart?: number; kvLenForAttention: number };
  numTokens: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  attnScale: number;
  attnSoftcap: number;
  effectiveSlidingWindow: number | null;
  causalForAttention: boolean;
  layerIdx: number;
}

/** Attention over the selected cache rows ([numTokens, numHeads * headDim] f32). */
export declare function runQsaIndexedAttention(recorder: CommandRecorder, options: QsaIndexedAttentionOptions): Promise<Tensor>;

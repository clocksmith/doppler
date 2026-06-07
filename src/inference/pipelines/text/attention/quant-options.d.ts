export interface QuantAttentionRuntimeState {
  coldLen?: number;
  hotLen?: number;
  hotWindow?: number;
  hotStart?: number;
  coldPackedStride?: number;
  coldQuantMode?: string;
  prodMode?: boolean;
  rotationMatrixBuffer?: GPUBuffer | null;
  codebookCentroidsBuffer?: GPUBuffer | null;
  residualKGPU?: GPUBuffer | null;
  residualVGPU?: GPUBuffer | null;
  residualNormsKGPU?: GPUBuffer | null;
  residualNormsVGPU?: GPUBuffer | null;
  qjlMatrixBuffer?: GPUBuffer | null;
  residualPackedStride?: number;
}

export interface QuantAttentionInputOptions {
  seqLen?: number;
  kvLen?: number;
  numKVHeads?: number;
  causal?: boolean;
  startPos?: number;
  slidingWindow?: number;
  attnSoftcap?: number;
  scale?: number;
}

export declare function buildTieredQuantAttentionOptions(
  kvState: QuantAttentionRuntimeState | null | undefined,
  options?: QuantAttentionInputOptions
): Record<string, unknown>;

export declare function buildContiguousQuantAttentionOptions(
  kvState: QuantAttentionRuntimeState | null | undefined,
  options?: QuantAttentionInputOptions
): Record<string, unknown>;

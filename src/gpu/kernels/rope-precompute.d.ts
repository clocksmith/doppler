export interface RoPEPrecomputeOptions {
  theta: number;
  rotaryDim: number;
  frequencyBaseDim: number;
  maxSeqLen: number;
  ropeScale: number;
  scalingType?: 'linear' | 'yarn' | 'longrope' | null;
  scaling?: Record<string, any> | null;
  positionPlan?: {
    temporal: Int32Array;
    height: Int32Array;
    width: Int32Array;
  } | null;
  mropeSection?: readonly [number, number, number] | null;
}

export interface RoPEPrecomputeDispatchPlan {
  dispatchStride: number;
  workgroups: [number, number, number];
}

export interface RoPEPrecomputeAuxiliaryData {
  code: number;
  factors: Float32Array | Uint32Array;
  yarn: Record<string, any> | null;
  originalMaxPosition: number;
  mropeSection: number[];
}

export declare function buildRoPEPrecomputeAuxiliaryData(
  options: RoPEPrecomputeOptions,
  halfDim: number
): RoPEPrecomputeAuxiliaryData;

export declare function planRoPEPrecomputeDispatch(
  device: Pick<GPUDevice, 'limits'> | null | undefined,
  count: number
): RoPEPrecomputeDispatchPlan;

export declare function runRoPEPrecompute(
  options: RoPEPrecomputeOptions
): Promise<{ cos: GPUBuffer; sin: GPUBuffer }>;

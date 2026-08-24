export interface RoPEPrecomputeOptions {
  theta: number;
  rotaryDim: number;
  frequencyBaseDim: number;
  maxSeqLen: number;
  ropeScale: number;
  scalingType?: 'linear' | 'yarn' | 'longrope' | null;
  scaling?: Record<string, any> | null;
}

export declare function runRoPEPrecompute(
  options: RoPEPrecomputeOptions
): Promise<{ cos: GPUBuffer; sin: GPUBuffer }>;

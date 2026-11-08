export interface DiffusionGemmaCanvasStatsOptions {
  canvasLength: number;
  vocabSize: number;
  temperature: number;
  padTokenId: number | null;
  logitSoftcap: number;
  /** One uniform draw in [0, 1) per canvas position; the kernel samples by inverse CDF. */
  uniformDraws: Float32Array;
  argmaxBuffer?: GPUBuffer | null;
  entropyBuffer?: GPUBuffer | null;
  sampledBuffer?: GPUBuffer | null;
}

export interface DiffusionGemmaCanvasStatsResult {
  argmaxBuffer: GPUBuffer;
  entropyBuffer: GPUBuffer;
  sampledBuffer: GPUBuffer;
}

export function runDiffusionGemmaCanvasStats(
  logitsBuffer: GPUBuffer,
  options: DiffusionGemmaCanvasStatsOptions
): Promise<DiffusionGemmaCanvasStatsResult>;

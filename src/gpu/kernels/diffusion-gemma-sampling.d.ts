export interface DiffusionGemmaCanvasStatsOptions {
  canvasLength: number;
  vocabSize: number;
  temperature: number;
  padTokenId: number | null;
  logitSoftcap: number;
  argmaxBuffer?: GPUBuffer | null;
  entropyBuffer?: GPUBuffer | null;
}

export interface DiffusionGemmaCanvasStatsResult {
  argmaxBuffer: GPUBuffer;
  entropyBuffer: GPUBuffer;
}

export function runDiffusionGemmaCanvasStats(
  logitsBuffer: GPUBuffer,
  options: DiffusionGemmaCanvasStatsOptions
): Promise<DiffusionGemmaCanvasStatsResult>;

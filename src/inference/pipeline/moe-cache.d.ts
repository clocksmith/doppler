export declare function resolveMaxTokensPerExpert(
  numTokens: number,
  numExperts: number,
  topK: number,
  hiddenSize: number,
  activationDtype: 'f16' | 'f32'
): number;

export declare function getCachedDequant(
  layerIdx: number,
  expertIdx: number,
  outputDtype: 'f16' | 'f32'
): { gateUp: GPUBuffer; down: GPUBuffer; lastUsed: number } | undefined;

export declare function setCachedDequant(
  layerIdx: number,
  expertIdx: number,
  outputDtype: 'f16' | 'f32',
  gateUp: GPUBuffer,
  down: GPUBuffer
): void;

export declare function clearDequantCache(): void;

export declare function getDequantCacheStats(): {
  hits: number;
  misses: number;
  size: number;
  maxEntries: number;
};

export declare function setDequantCacheMaxEntries(maxEntries: number): void;

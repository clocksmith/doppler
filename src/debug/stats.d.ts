export interface SampleStats {
  mean: number;
  median: number;
  min: number;
  max: number;
  p95: number;
  p99: number;
  stdDev: number;
  ci95: number;
  samples: number;
  samplesAfterOutlierRemoval: number;
  outliersRemoved: number;
}

export interface ArrayStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  maxAbs: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  validCount: number;
}

export interface BasicStats {
  mean: number;
  min: number;
  max: number;
  total: number;
  count: number;
}

export declare function percentile(sorted: number[], p: number): number;
export declare function median(sorted: number[]): number;
export declare function removeOutliersIQR(values: number[], multiplier?: number): {
  filtered: number[];
  removed: number;
  lower: number;
  upper: number;
};
export declare function sampleStdDev(values: number[], meanValue: number): number;
export declare function confidenceInterval95(stdDev: number, n: number): number;
export declare function computeSampleStats(values: number[], options?: { outlierIqrMultiplier?: number }): SampleStats;
export declare function computeArrayStats(values: ArrayLike<number>, limit?: number): ArrayStats;
export declare function computeBasicStats(values: number[]): BasicStats;

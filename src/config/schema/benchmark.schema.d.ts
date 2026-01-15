export interface BenchmarkOutputConfig {
  schemaVersion: number;
}

export interface BenchmarkRunConfig {
  promptName: string;
  customPrompt: string | null;
  maxNewTokens: number;
  runType: string;
  warmupRuns: number;
  timedRuns: number;
  sampling: {
    temperature: number;
    topK: number;
    topP: number;
  };
  debug: boolean;
  profile: boolean;
  useChatTemplate?: boolean;
}

export interface BenchmarkStatsConfig {
  outlierIqrMultiplier: number;
  warmupStabilityPercent: number;
  thermalSlowdownPercent: number;
  minSamplesForComparison: number;
}

export interface BenchmarkComparisonConfig {
  regressionThresholdPercent: number;
  failOnRegression: boolean;
}

export interface BenchmarkConfig {
  output: BenchmarkOutputConfig;
  run: BenchmarkRunConfig;
  stats: BenchmarkStatsConfig;
  comparison: BenchmarkComparisonConfig;
}

export declare const DEFAULT_BENCHMARK_OUTPUT_CONFIG: BenchmarkOutputConfig;
export declare const DEFAULT_BENCHMARK_RUN_CONFIG: BenchmarkRunConfig;
export declare const DEFAULT_BENCHMARK_STATS_CONFIG: BenchmarkStatsConfig;
export declare const DEFAULT_BENCHMARK_COMPARISON_CONFIG: BenchmarkComparisonConfig;
export declare const DEFAULT_BENCHMARK_CONFIG: BenchmarkConfig;

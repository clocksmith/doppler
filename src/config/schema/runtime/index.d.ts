// Runtime schema contracts.
export {
  type BrowserSuiteMetricsSchema,
  BROWSER_SUITE_METRICS_SCHEMA_VERSION,
  DEFAULT_BROWSER_SUITE_METRICS,
  validateBrowserSuiteMetrics,
} from "./browser-suite-metrics.schema.js";

export {
  // Types
  type ShardCacheConfigSchema,
  type LoaderMemoryBudgetConfigSchema,
  type MemoryManagementConfigSchema,
  type PrefetchConfigSchema,
  type OpfsPathConfigSchema,
  type ExpertCacheConfigSchema,
  type LoadingConfigSchema,

  // Defaults
  DEFAULT_Q4K_LAYOUT,
  DEFAULT_SHARD_CACHE_CONFIG,
  DEFAULT_LOADER_MEMORY_BUDGET_CONFIG,
  DEFAULT_MEMORY_MANAGEMENT_CONFIG,
  DEFAULT_PREFETCH_CONFIG,
  DEFAULT_OPFS_PATH_CONFIG,
  DEFAULT_EXPERT_CACHE_CONFIG,
  DEFAULT_LOADING_CONFIG,
} from "./loading.schema.js";

export {
  // Types
  type QuotaConfigSchema,
  type VramEstimationConfigSchema,
  type StorageAlignmentConfigSchema,
  type StorageFullConfigSchema,

  // Defaults
  DEFAULT_QUOTA_CONFIG,
  DEFAULT_VRAM_ESTIMATION_CONFIG,
  DEFAULT_STORAGE_ALIGNMENT_CONFIG,
  DEFAULT_STORAGE_FULL_CONFIG,
} from "./storage.schema.js";

export {
  // Types
  type DistributionConfigSchema,

  // Defaults
  DEFAULT_DISTRIBUTION_CONFIG,
} from "./distribution.schema.js";

export {
  // Types
  type LogOutputConfigSchema,
  type LogHistoryConfigSchema,
  type LogLevelConfigSchema,
  type LogLevel,
  type TraceCategory,
  type TraceConfigSchema,
  type PipelineDebugCategory,
  type PipelineDebugConfigSchema,
  type ProfilerConfigSchema,
  type PerfGuardsConfigSchema,
  type ProbeStage,
  type ProbeConfigSchema,
  type DebugConfigSchema,

  // Constants
  LOG_LEVELS,

  // Defaults
  DEFAULT_LOG_OUTPUT_CONFIG,
  DEFAULT_LOG_HISTORY_CONFIG,
  DEFAULT_LOG_LEVEL_CONFIG,
  DEFAULT_TRACE_CONFIG,
  DEFAULT_LOADER_DEBUG_CONFIG,
  DEFAULT_MATMUL_DEBUG_CONFIG,
  DEFAULT_KERNEL_TRACE_CONFIG,
  DEFAULT_PIPELINE_DEBUG_CONFIG,
  DEFAULT_PROFILER_CONFIG,
  DEFAULT_PERF_GUARDS_CONFIG,
  DEFAULT_DEBUG_CONFIG,
} from "./debug.schema.js";

export {
  // Types
  type BenchmarkOutputConfig,
  type BenchmarkRunConfig,
  type BenchmarkStatsConfig,
  type BenchmarkComparisonConfig,
  type BenchmarkConfig,

  // Defaults
  DEFAULT_BENCHMARK_OUTPUT_CONFIG,
  DEFAULT_BENCHMARK_RUN_CONFIG,
  DEFAULT_BENCHMARK_STATS_CONFIG,
  DEFAULT_BENCHMARK_COMPARISON_CONFIG,
  DEFAULT_BENCHMARK_BASELINE_CONFIG,
  DEFAULT_BENCHMARK_CONFIG,
} from "./benchmark.schema.js";

export {
  // Types
  type HotSwapSignerSchema,
  type HotSwapConfigSchema,

  // Defaults
  DEFAULT_HOTSWAP_CONFIG,
} from "./hotswap.schema.js";

export {
  // Types
  type BufferPoolBucketConfigSchema,
  type BufferPoolLimitsConfigSchema,
  type BufferPoolAlignmentConfigSchema,
  type BufferPoolBudgetConfigSchema,
  type BufferPoolConfigSchema,

  // Defaults
  DEFAULT_BUFFER_POOL_BUCKET_CONFIG,
  DEFAULT_BUFFER_POOL_LIMITS_CONFIG,
  DEFAULT_BUFFER_POOL_ALIGNMENT_CONFIG,
  DEFAULT_BUFFER_POOL_BUDGET_CONFIG,
  DEFAULT_BUFFER_POOL_CONFIG,
} from "./buffer-pool.schema.js";

export {
  // Types
  type BridgeConfigSchema,

  // Defaults
  DEFAULT_BRIDGE_TIMEOUT_CONFIG,
  DEFAULT_BRIDGE_CONFIG,
} from "./bridge.schema.js";

export {
  KB,
  MB,
  GB,
  formatBytes,
  formatBytesCompact,
} from "./units.schema.js";

export {
  // Types
  type KernelRegistryConfigSchema,
  type SharedRuntimeConfigSchema,

  // Defaults
  DEFAULT_KERNEL_REGISTRY_CONFIG,
  DEFAULT_SHARED_RUNTIME_CONFIG,
} from "./shared-runtime.schema.js";

export {
  // Types
  type ToolingConfigSchema,
  type ToolingIntent,
  type ToolingDiagnosticsMode,
  type RefactorReceiptPolicy,

  // Constants
  TOOLING_INTENTS,
  TOOLING_DIAGNOSTICS,
  REFACTOR_RECEIPT_POLICIES,

  // Defaults
  DEFAULT_TOOLING_CONFIG,
} from "./tooling.schema.js";

export {
  type RuntimeProfileMetadata,
  type RuntimeProfileStability,
  validateRuntimeProfileMetadata,
  assertRuntimeProfileIntentCompatibility,
} from "./runtime-profile.schema.js";

export {
  // Types
  type EcosystemStabilityMode,
  type EcosystemRankingMode,
  type EcosystemIncentiveMode,
  type EcosystemAntiSybilCost,
  type EcosystemEnforcementMode,
  type EcosystemFailoverTier,
  type EcosystemNotarizationAlgorithm,
  type EcosystemConfigSchema,
  type EcosystemConfigOverrides,

  // Constants
  ECOSYSTEM_STABILITY_MODES,
  ECOSYSTEM_RANKING_MODES,
  ECOSYSTEM_INCENTIVE_MODES,
  ECOSYSTEM_ANTISYBIL_COST,
  ECOSYSTEM_ENFORCEMENT_MODES,
  ECOSYSTEM_FAILOVER_TIERS,
  ECOSYSTEM_NOTARIZATION_ALGORITHMS,

  // Defaults
  DEFAULT_ECOSYSTEM_CONFIG,

  // Factory and validation
  mergeEcosystemConfig,
  createEcosystemConfig,
  validateEcosystemConfig,
} from "./ecosystem.schema.js";

export {
  // Types
  type HarnessConfigSchema,
  type HarnessMode,

  // Defaults
  DEFAULT_HARNESS_CONFIG,
} from "./harness.schema.js";

export {
  // Types
  type LargeWeightConfigSchema,
  type RuntimeConfigSchema,
  type DopplerConfigSchema,

  // Defaults
  DEFAULT_LARGE_WEIGHT_CONFIG,
  DEFAULT_CHAT_TEMPLATE_CONFIG,
  DEFAULT_RUNTIME_CONFIG,
  DEFAULT_DOPPLER_CONFIG,

  // Factory
  createDopplerConfig,
} from "./doppler.schema.js";

export {
  VALID_LORA_TARGET_MODULES,
  DEFAULT_ADAPTER_VALIDATION_CONFIG,
  DEFAULT_ADAPTER_STACK_CONFIG,
  DEFAULT_ADAPTER_REGISTRY_CONFIG,
  DEFAULT_ADAPTER_CONFIG,
} from "./adapter.schema.js";

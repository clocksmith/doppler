// Execution schema contracts.
export {
  type KernelPathSchema,
  type KernelPathRef,
  type KernelStepSchema,
  type LayerKernelPathSchema,
  type LayerOverrideSchema,
  DEFAULT_ENTRY,
  DEFAULT_INPUT,
  DEFAULT_OUTPUT,
} from "./kernel-path.schema.js";

export {
  type RoPEConfigSchema,
  type AttentionSchema,
  type NormalizationSchema,
  type FFNSchema,
  type LayerPipelineOp,
  type LayerPipelinePhase,
  type LayerPipelineDtype,
  type LayerPipelineNormWeight,
  type LayerPipelineStepSchema,
  type LayerPipelineOverrideSchema,
  type LayerPipelineSchema,
  type OutputSchema,
  type LayerType,
  type LinearNormMode,
  type GlobalLayerPattern,
  type LayerPatternSchema,
  type InferenceConfigSchema,
  type SamplingSchema,
  type TokenizerConfigSchema,
  computeGlobalLayers,
} from "./inference.schema.js";

export {
  type ExecutionV1Dtype,
  type ExecutionV1KernelSchema,
  type ExecutionV1KernelMap,
  type ExecutionV1StepTuple,
  type ExecutionV1LayerGroupSchema,
  type ExecutionV1StepEntry,
  type ExecutionV1BoundaryStep,
  type ExecutionV1ComputeDefaultsSchema,
  type ExecutionV1DecodeLoopSchema,
  type ExecutionV1SessionSchema,
  type ExecutionV1PoliciesSchema,
  type ExecutionV1GraphSchema,
  type ExecutionV1ConfigSchema,
  type ExecutionV1PatchSetSchema,
  type ExecutionV1PatchRemoveSchema,
  type ExecutionV1PatchAddSchema,
  type ExecutionV1PatchAddKernelSchema,
  type ExecutionV1PatchSchema,
  type ExecutionV1ExpandedStepSchema,
  EXECUTION_V1_SCHEMA_ID,
  DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS,
  DEFAULT_EXECUTION_V1_SESSION,
  DEFAULT_EXECUTION_V1_POLICIES,
  DEFAULT_EXECUTION_V1_PATCH,
  isExecutionV1Digest,
  hasExecutionV1,
  expandExecutionV1,
  type ExpandExecutionV1Options,
} from "./execution-v1.schema.js";

export {
  type DiffusionConfigSchema,
  type DiffusionSchedulerConfigSchema,
  type DiffusionLatentConfigSchema,
  type DiffusionTextEncoderConfigSchema,
  type DiffusionDecodeConfigSchema,
  type DiffusionTilingConfigSchema,
  type DiffusionSwapperConfigSchema,
  type DiffusionQuantizationConfigSchema,
  type DiffusionBackendConfigSchema,
  type DiffusionSchedulerType,
  type DiffusionDtype,
  type DiffusionQuantDtype,
  type DiffusionSwapperStrategy,
  type DiffusionBackendPipeline,
  DEFAULT_DIFFUSION_SCHEDULER_CONFIG,
  DEFAULT_DIFFUSION_LATENT_CONFIG,
  DEFAULT_DIFFUSION_TEXT_ENCODER_CONFIG,
  DEFAULT_DIFFUSION_DECODE_CONFIG,
  DEFAULT_DIFFUSION_TILING_CONFIG,
  DEFAULT_DIFFUSION_SWAPPER_CONFIG,
  DEFAULT_DIFFUSION_QUANTIZATION_CONFIG,
  DEFAULT_DIFFUSION_BACKEND_CONFIG,
  DEFAULT_DIFFUSION_CONFIG,
} from "./diffusion.schema.js";

export {
  type EnergyStateConfigSchema,
  type EnergyInitConfigSchema,
  type EnergyTargetConfigSchema,
  type EnergyLoopConfigSchema,
  type EnergyDiagnosticsConfigSchema,
  type EnergyQuintelRulesConfigSchema,
  type EnergyQuintelWeightsConfigSchema,
  type EnergyQuintelClampConfigSchema,
  type EnergyQuintelConfigSchema,
  type EnergyConfigSchema,
  type EnergyModelConfigSchema,
  DEFAULT_ENERGY_STATE_CONFIG,
  DEFAULT_ENERGY_INIT_CONFIG,
  DEFAULT_ENERGY_TARGET_CONFIG,
  DEFAULT_ENERGY_LOOP_CONFIG,
  DEFAULT_ENERGY_DIAGNOSTICS_CONFIG,
  DEFAULT_ENERGY_QUINTEL_RULES,
  DEFAULT_ENERGY_QUINTEL_WEIGHTS,
  DEFAULT_ENERGY_QUINTEL_CLAMP,
  DEFAULT_ENERGY_QUINTEL_CONFIG,
  DEFAULT_ENERGY_CONFIG,
} from "./energy.schema.js";

export {
  // Types
  type GpuFeature,
  type BindingType,
  type BindingSchema,
  type UniformFieldType,
  type UniformFieldSchema,
  type UniformsSchema,
  type WgslOverridesSchema,
  type KernelVariantSchema,
  type OperationSchema,
  type KernelRegistrySchema,
  type ResolvedKernelConfig,

  // Functions
  mergeBindings,
  resolveKernelConfig,
} from "./kernel-registry.schema.js";

export {
  // Types
  type KVDtype,
  type KVLayout,
  type KVCacheConfigSchema,

  // Defaults
  DEFAULT_KVCACHE_CONFIG,

  // Thresholds
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} from "./kvcache.schema.js";

export {
  // Types
  type MatmulThresholdsSchema,
  type RmsnormThresholdsSchema,
  type RopeDefaultsSchema,
  type AttentionThresholdsSchema,
  type CastThresholdsSchema,
  type KernelThresholdsConfigSchema,
  type SoftmaxThresholdsSchema,
  type FfnThresholdsSchema,
  type SampleThresholdsSchema,

  // Constants
  DTYPE_SIZES,
  getDtypeSize,

  // Defaults
  DEFAULT_MATMUL_THRESHOLDS,
  DEFAULT_RMSNORM_THRESHOLDS,
  DEFAULT_SOFTMAX_THRESHOLDS,
  DEFAULT_FFN_THRESHOLDS,
  DEFAULT_SAMPLE_THRESHOLDS,
  DEFAULT_ROPE_DEFAULTS,
  DEFAULT_ATTENTION_THRESHOLDS,
  DEFAULT_CAST_THRESHOLDS,
  DEFAULT_KERNEL_THRESHOLDS,

  getKernelThresholds,
  getKernelThresholdSnapshot,
  setKernelThresholds,
  resetKernelThresholds,
} from "./kernel-thresholds.schema.js";

export {
  // Types
  type KernelWarmupConfigSchema,

  // Defaults
  DEFAULT_KERNEL_WARMUP_CONFIG,
} from "./kernel-warmup.schema.js";

export {
  // Types
  type SpeculativeConfigSchema,

  // Defaults
  DEFAULT_SPECULATIVE_CONFIG,
} from "./speculative.schema.js";

export {
  SPECULATION_MODES,
  SPECULATION_VERIFY_MODES,
  DEFAULT_SELF_SPECULATION_CONFIG,
  validateSelfSpeculationConfig,
} from "./speculation-self.schema.js";

// Execution schema contracts.
export {
  DEFAULT_ENTRY,
  DEFAULT_INPUT,
  DEFAULT_OUTPUT,
} from "./kernel-path.schema.js";

export {
  computeGlobalLayers,
} from "./inference.schema.js";

export {
  EXECUTION_V1_SCHEMA_ID,
  DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS,
  DEFAULT_EXECUTION_V1_SESSION,
  DEFAULT_EXECUTION_V1_POLICIES,
  DEFAULT_EXECUTION_V1_PATCH,
  isExecutionV1Digest,
  hasExecutionV1,
  expandExecutionV1,
} from "./execution-v1.schema.js";

export {
  mergeBindings,
  resolveKernelConfig,
} from "./kernel-registry.schema.js";

export {
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
  DEFAULT_KVCACHE_CONFIG,
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} from "./kvcache.schema.js";

export {
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

  // Functions
  getKernelThresholds,
  getKernelThresholdSnapshot,
  setKernelThresholds,
  resetKernelThresholds,
} from "./kernel-thresholds.schema.js";

export {
  DEFAULT_KERNEL_WARMUP_CONFIG,
} from "./kernel-warmup.schema.js";

export {
  DEFAULT_SPECULATIVE_CONFIG,
} from "./speculative.schema.js";

export {
  SPECULATION_MODES,
  SPECULATION_VERIFY_MODES,
  DEFAULT_SELF_SPECULATION_CONFIG,
  validateSelfSpeculationConfig,
} from "./speculation-self.schema.js";

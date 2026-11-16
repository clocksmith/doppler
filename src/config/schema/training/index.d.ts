// Training schema contracts.
export {
  type LoraConfigSchema,
  DEFAULT_LORA_CONFIG,
} from "./lora.schema.js";

export {
  type TrainingOptimizerConfigSchema,
  type TrainingGradientConfigSchema,
  type TrainingLossScalingConfigSchema,
  type TrainingPrecisionConfigSchema,
  type TrainingAttentionConfigSchema,
  type TrainingTelemetryConfigSchema,
  type TrainingSettingsSchema,
  DEFAULT_TRAINING_OPTIMIZER_CONFIG,
  DEFAULT_TRAINING_GRADIENT_CONFIG,
  DEFAULT_TRAINING_LOSS_SCALING_CONFIG,
  DEFAULT_TRAINING_PRECISION_CONFIG,
  DEFAULT_TRAINING_ATTENTION_CONFIG,
  DEFAULT_TRAINING_TELEMETRY_CONFIG,
  DEFAULT_TRAINING_SETTINGS,
} from "./training.schema.js";

export {
  type DistillTrainingStage,
  type DistillFreezeConfigSchema,
  type DistillTrainingConfigSchema,
  DISTILL_STAGE_VALUES,
  DISTILL_TRAINING_SCHEMA_VERSION,
  DEFAULT_DISTILL_FREEZE_GROUPS,
  DEFAULT_DISTILL_TRAINING_CONFIG,
  validateDistillTrainingConfig,
} from "./distill-training.schema.js";

export {
  type UlTrainingStage,
  type UlNoiseScheduleSchema,
  type UlPriorAlignmentSchema,
  type UlDecoderSigmoidWeightSchema,
  type UlFreezeGroupsSchema,
  type UlTrainingConfigSchema,
  UL_STAGE_VALUES,
  UL_TRAINING_SCHEMA_VERSION,
  DEFAULT_UL_NOISE_SCHEDULE,
  DEFAULT_UL_PRIOR_ALIGNMENT,
  DEFAULT_UL_DECODER_SIGMOID_WEIGHT,
  DEFAULT_UL_FREEZE_GROUPS,
  DEFAULT_UL_TRAINING_CONFIG,
  validateUlTrainingConfig,
} from "./ul-training.schema.js";

export {
  type TrainingMetricsReportSchema,
  DEFAULT_TRAINING_METRICS_REPORT,
  validateTrainingMetricsEntry,
  validateTrainingMetricsReport,
} from "./training-metrics.schema.js";

export {
  type BackwardRegistryOpSchema,
  type BackwardRegistrySchema,
  validateBackwardRegistry,
} from "./backward-registry.schema.js";

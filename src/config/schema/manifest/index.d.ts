// Manifest schema contracts.
export {
  // Constants
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,
  MAX_HEADER_SIZE,
  HEADER_READ_SIZE,
  DEFAULT_RMS_NORM_EPS,
  DEFAULT_HIGH_PRECISION_EPS,

  // Types
  type HashAlgorithm,
  type ModelType,
  type ComponentGroupType,
  type WeightLayout,
  type QuantizationValue,

  // Schemas
  type ArchitectureSchema,
  type ShardSchema,
  type TensorSpanSchema,
  type TensorSchema,
  type TensorMapSchema,
  type TensorRole,
  type ComponentGroupSchema,
  type MoEConfigSchema,
  type TokenizerSchema,
  type RuntimeOptimizationsSchema,
  type QuantizationInfoSchema,
  type ConversionInfoSchema,
  type SourceArtifactFormat,
  type ArtifactCompleteness,
  type ManifestArtifactIdentitySchema,
  type ManifestWeightsRefSchema,
  type ManifestSchema,
  type AdapterConfigSchema,
  type ProvenanceSchema,

  // Inference config (embedded in manifest)
  type ManifestInferenceSchema,
  type ManifestAttentionSchema,
  type ManifestNormalizationSchema,
  type ManifestFFNSchema,
  type ManifestRoPESchema,
  type ManifestOutputSchema,
  type ManifestEmbeddingProjectionSchema,
  type ManifestEmbeddingPostprocessorSchema,
  type ManifestLayerPatternSchema,
  type ManifestChatTemplateSchema,
  DEFAULT_MANIFEST_INFERENCE,

  // Helpers
  isV1Manifest,
  hasMoEConfig,
  validateManifestInference,
  hasInferenceConfig,
} from "./manifest.schema.js";

export {
  // Types
  type QuantizationType,
  type ConversionStageType,

  // Constants
  ConversionStage,

  // Schemas
  type TensorInfoSchema,
  type ParsedModelSchema,
  type RawModelConfigSchema,
  type ConversionOptionsSchema,
  type ConversionProgressSchema,
  type WriterOptionsSchema,
  type TensorLocationSchema,
  type WriteResultSchema,
  type ConversionIOSchema,
} from "./conversion.schema.js";

export {
  type ConversionReportResultSchema,
  type ConversionReportManifestSchema,
  type ConversionReportSchema,
  CONVERSION_REPORT_SCHEMA_VERSION,
  DEFAULT_CONVERSION_REPORT,
  validateConversionReport,
} from "./conversion-report.schema.js";

export {
  // Types
  type ComputePrecision,
  type ConverterQuantizationConfigSchema,
  type ConverterShardingConfigSchema,
  type ConverterStreamingConfigSchema,
  type ConverterHttpConfigSchema,
  type ConverterWeightLayoutConfigSchema,
  type ConverterManifestConfigSchema,
  type ConverterInferenceConfigSchema,
  type ConverterOutputConfigSchema,
  type ConverterExecutionConfigSchema,
  type ConverterWorkerCountPolicy,
  type GGUFParserDefaultsSchema,
  type ConverterConfigSchema,

  // Defaults
  DEFAULT_CONVERTER_QUANTIZATION_CONFIG,
  DEFAULT_CONVERTER_SHARDING_CONFIG,
  DEFAULT_CONVERTER_STREAMING_CONFIG,
  DEFAULT_CONVERTER_HTTP_CONFIG,
  DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG,
  DEFAULT_CONVERTER_MANIFEST_CONFIG,
  DEFAULT_CONVERTER_INFERENCE_CONFIG,
  DEFAULT_CONVERTER_OUTPUT_CONFIG,
  DEFAULT_CONVERTER_EXECUTION_CONFIG,
  DEFAULT_GGUF_PARSER_DEFAULTS,
  DEFAULT_CONVERTER_CONFIG,

  // Factory
  createConverterConfig,
} from "./converter.schema.js";

export {
  // Types
  type WeightQuantType,
  type EmbeddingQuantType,
  type QuantizationDefaultsSchema,

  // Defaults
  DEFAULT_QUANTIZATION_DEFAULTS,
} from "./quantization-defaults.schema.js";

export {
  QK_K,
  Q4_0_BLOCK_SIZE, Q4_0_BLOCK_BYTES, Q4K_BLOCK_BYTES,
  Q6K_BLOCK_BYTES,
  Q8_0_BLOCK_BYTES,
  Q8_0_BLOCK_SIZE,
  K_SCALE_SIZE,
  QK4_K_BLOCK_SIZE,
  padToQ4KBlock,
  q4kBlockCount,
} from "./quantization.schema.js";

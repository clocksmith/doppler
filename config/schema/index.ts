/**
 * Schema Index
 *
 * Re-exports all schema definitions for easy importing.
 *
 * Naming Convention:
 * - *Schema: Type definitions (interface structure)
 * - *Config: Runtime instances (validated values)
 * - *Raw: Unparsed input (from manifest/file)
 * - *Options: Function parameters
 *
 * @module config/schema
 */

// =============================================================================
// Manifest Schema
// =============================================================================
export {
  // Constants
  RDRR_VERSION,
  SHARD_SIZE,
  TENSORS_FILENAME,

  // Types
  type HashAlgorithm,
  type ModelType,
  type ComponentGroupType,
  type WeightLayout,

  // Schemas
  type ArchitectureSchema,
  type ShardSchema,
  type TensorSpanSchema,
  type TensorSchema,
  type TensorMapSchema,
  type ComponentGroupSchema,
  type MoEConfigSchema,
  type TokenizerSchema,
  type KernelHintsSchema,
  type RuntimeOptimizationsSchema,
  type ConversionInfoSchema,
  type ManifestSchema,

  // Helpers
  isV1Manifest,
  hasMoEConfig,
} from './manifest.schema.js';

// =============================================================================
// Inference Schema
// =============================================================================
export {
  type AttentionSchema,
  type NormalizationSchema,
  type FFNSchema,
  type OutputSchema,
  type LayerType,
  type GlobalLayerPattern,
  type LayerPatternSchema,
  type InferenceConfigSchema,
  type SamplingSchema,
  type TokenizerConfigSchema,
  // Functions
  computeGlobalLayers,
} from './inference.schema.js';

// =============================================================================
// Conversion Schema
// =============================================================================
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
} from './conversion.schema.js';

// =============================================================================
// Preset Schema
// =============================================================================
export {
  type PresetSchema,
  type TensorPatternSchema,
  type DetectionPatternSchema,
  type ResolvedConfigSchema,
} from './preset.schema.js';

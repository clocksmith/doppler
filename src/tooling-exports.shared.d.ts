/**
 * Shared Tooling Surface Exports
 *
 * Browser-safe tooling exports shared by browser and node-facing surfaces.
 * Storage / device / manifest symbols are sourced from the narrow slice files
 * under ./tooling-exports/ so those slices remain the single source of truth
 * for their respective groups.
 *
 * @module tooling-exports.shared
 */

// Debug
export { log } from './debug/index.js';

// Config
export {
  createConverterConfig,
} from './config/index.js';
export { getRuntimeConfig, setRuntimeConfig } from './config/runtime.js';
export { TOOLING_INTENTS } from './config/schema/tooling.schema.js';
export {
  PROGRAM_BUNDLE_SCHEMA_VERSION,
  PROGRAM_BUNDLE_SCHEMA_ID,
  PROGRAM_BUNDLE_HOST_SCHEMA_ID,
  PROGRAM_BUNDLE_HOST_JS_SUBSET,
  PROGRAM_BUNDLE_CAPTURE_PROFILE_SCHEMA_ID,
  PROGRAM_BUNDLE_REFERENCE_TRANSCRIPT_SCHEMA_ID,
  validateProgramBundle,
} from './config/schema/program-bundle.schema.js';
export type {
  ProgramBundle,
  ProgramBundleArtifact,
  ProgramBundleReferenceTranscript,
  ProgramBundleWgslModule,
} from './config/schema/program-bundle.schema.js';

// Storage + manifests (sourced from narrow slices)
export * from './tooling-exports/storage.js';
export * from './tooling-exports/manifest.js';
export { inferEmbeddingOutputConfig } from './converter/core.js';

// GPU init + capabilities (sourced from narrow slice)
export * from './tooling-exports/device.js';

// Memory tooling
export { captureMemorySnapshot } from './loader/memory-monitor.js';
export { destroyBufferPool } from './memory/buffer-pool.js';

// Browser-safe runtime profile helpers
export {
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimeProfile,
  applyRuntimeProfile,
} from './inference/browser-harness-runtime-helpers.js';

export {
  MODEL_CHECKPOINT_EVIDENCE_SCHEMA,
  MODEL_CHECKPOINT_STAGES,
  buildModelCheckpointEvidence,
  flattenModelCheckpointDigests,
} from './inference/model-checkpoint-evidence.js';
export type {
  ModelCheckpointStage,
  ModelCheckpointRecord,
  ModelCheckpointDigest,
  ModelCheckpointStep,
  ModelCheckpointEvidence,
} from './inference/model-checkpoint-evidence.js';

// Shared command contract (browser + CLI parity)
export {
  TOOLING_COMMANDS,
  TOOLING_SURFACES,
  TOOLING_WORKLOADS,
  TOOLING_VERIFY_WORKLOADS,
  TOOLING_TRAINING_COMMAND_SCHEMA_VERSION,
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './tooling/command-api.js';
export { runBrowserCommand, normalizeBrowserCommand } from './tooling/browser-command-runner.js';
export {
  RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA,
  RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA,
  RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA,
  RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA,
  validateRuntimeOptimizationContract,
  hashRuntimeOptimizationContract,
  enumerateRuntimeOptimizationCandidates,
  validateRuntimeOptimizationCandidate,
  materializeRuntimeOptimizationCandidate,
  validateRuntimeOptimizationCandidateRegistry,
  evaluateBrowserRuntimeOptimizationCandidate,
} from './tooling/runtime-optimization.js';
export {
  RUNTIME_OPTIMIZATION_RESULTS_INDEX_SCHEMA,
  buildRuntimeOptimizationResultsIndex,
} from './tooling/runtime-optimization-index.js';
export {
  SOURCE_BOUNDARY_PACK_SCHEMA,
  RUNTIME_BOUNDARY_CAPTURE_SCHEMA,
  BOUNDARY_COMPARISON_RECEIPT_SCHEMA,
  DETERMINISTIC_TOKEN_EVIDENCE_SCHEMA,
  BOUNDARY_PROVIDER_CAPTURE_SCHEMA,
  buildRuntimeBoundaryCapture,
  buildSourceBoundaryPack,
  buildSourceBoundaryPackFromProviderCapture,
  buildDeterministicTokenEvidenceFromReferenceTranscript,
  compareBoundaryEvidence,
} from './tooling/boundary-evidence.js';
export type { BoundaryEvidence } from './tooling/boundary-evidence.js';
export {
  TOKEN_COST_LEDGER_SCHEMA,
  isExecutionObservationRequested,
  buildTokenCostLedger,
  classifyTokenCostLedger,
} from './tooling/execution-cost-ledger.js';
export {
  REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA,
  REGISTERED_VARIANT_CALIBRATION_RECEIPT_SCHEMA,
  digestRegisteredVariantDescriptor,
  validateRegisteredVariantCalibrationPlan,
  calibrateRegisteredVariants,
} from './tooling/registered-variant-calibration.js';
export type {
  RuntimeOptimizationWorkload,
  RuntimeOptimizationDirection,
  RuntimeOptimizationContract,
  RuntimeOptimizationCandidate,
  RuntimeOptimizationReceipt,
  RuntimeOptimizationEvaluationOptions,
} from './tooling/runtime-optimization.js';

export type {
  ToolingCommand,
  ToolingSurface,
  ToolingWorkload,
  ToolingIntent,
  ToolingCommandRequestInput,
  ToolingCommandRequest,
} from './tooling/command-api.js';
export type {
  BrowserCommandRunOptions,
  BrowserCommandRunResult,
} from './tooling/browser-command-runner.js';

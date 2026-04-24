// ============================================================================
// Shared Tooling Surface Exports
//
// Browser-safe tooling exports shared by browser and node-facing surfaces.
// Storage / device / manifest symbols are sourced from the narrow slice files
// under ./tooling-exports/ so those slices remain the single source of truth
// for their respective groups.
// ============================================================================

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

// Reference-transcript capture helpers — used by demo and verify flows to emit
// doppler.reference-transcript/v1 seeds without depending on the full harness.
export {
  buildReferenceTranscriptSeed,
  resolveExecutionGraphHash,
} from './inference/browser-harness.js';
export {
  captureKvCacheByteProof,
  digestLogitsForTranscript,
} from './inference/browser-harness-text-helpers.js';

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

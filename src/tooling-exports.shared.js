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

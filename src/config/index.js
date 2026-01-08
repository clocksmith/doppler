/**
 * Config Module Index
 *
 * Central export for config-as-code architecture.
 *
 * Platform and kernel registry initialization:
 * - Platform detection and registry loading happen automatically in gpu/device.ts
 * - Call initDevice() to initialize both GPU and config systems
 * - Use getPlatformConfig() from device.ts to access resolved platform config
 * - Kernel selection in kernels/selection.js uses platform preferences automatically
 *
 * @module config
 */

// Schema types
export * from './schema/index.js';

// Preset loader
export {
  getPreset,
  listPresets,
  resolvePreset,
  detectPreset,
  resolveConfig,
  PRESET_REGISTRY,
} from './loader.js';

// Runtime config registry
export {
  getRuntimeConfig,
  setRuntimeConfig,
  resetRuntimeConfig,
} from './runtime.js';

// Config merge (manifest + runtime → merged with source tracking)
export {
  mergeConfig,
  formatConfigSources,
  getValuesBySource,
  summarizeSources,
} from './merge.js';

// Kernel registry (JS modules - dynamic import for hotswap)
// Auto-initialized by gpu/device.ts during initDevice()
// Use: const { getRegistry, getRegistrySync } = await import('./kernels/registry.js');
// Use: const { selectMatmul, selectAttention } = await import('./kernels/selection.js');

// Platform loader (JS modules - dynamic import for hotswap)
// Auto-initialized by gpu/device.ts during initDevice()
// Use: const { getPlatform, getCapabilities } = await import('./platforms/loader.js');
// Alternative: import { getPlatformConfig } from '../gpu/device.js';

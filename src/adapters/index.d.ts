/**
 * LoRA Adapter Infrastructure
 *
 * Provides complete infrastructure for loading, managing, and applying
 * LoRA (Low-Rank Adaptation) adapters at runtime. This enables RSI
 * (Recursive Self-Improvement) through adapter-based self-modification.
 *
 * Components:
 * - Adapter Manifest: JSON schema and types for adapter definitions
 * - LoRA Loader: Weight loading from OPFS/URL with format support
 * - Adapter Manager: Runtime enable/disable and stacking
 * - Adapter Registry: Persistent storage and discovery
 *
 * @module adapters
 */

// Manifest types and schema
export {
  // Schema
  ADAPTER_MANIFEST_SCHEMA,
  // Functions
  validateManifest,
  parseManifest,
  serializeManifest,
  createManifest,
  computeLoRAScale,
} from './adapter-manifest.js';

export type {
  // Types
  AdapterManifest,
  AdapterMetadata,
  AdapterTensorSpec,
  MinimalAdapterManifest,
  ManifestValidationResult,
  ManifestValidationError,
} from './adapter-manifest.js';

// LoRA loading
export {
  // Functions
  loadLoRAWeights,
  loadLoRAFromManifest,
  loadLoRAFromUrl,
  loadLoRAFromSafetensors,
  applyDeltaWeights,
} from './lora-loader.js';

export type {
  // Types
  LoRAManifest,
  LoRATensorSpec,
  LoRALoadOptions,
  LoRAWeightsResult,
} from './lora-loader.js';

// Adapter management
export {
  // Class
  AdapterManager,
  // Default instance
  getAdapterManager,
  resetAdapterManager,
} from './adapter-manager.js';

export type {
  // Types
  AdapterState,
  EnableAdapterOptions,
  AdapterStackOptions,
  AdapterManagerEvents,
} from './adapter-manager.js';

// Adapter registry
export {
  // Class
  AdapterRegistry,
  // Default instance
  getAdapterRegistry,
  resetAdapterRegistry,
  createMemoryRegistry,
} from './adapter-registry.js';

export type {
  // Types
  AdapterRegistryEntry,
  AdapterQueryOptions,
  RegistryStorage,
} from './adapter-registry.js';

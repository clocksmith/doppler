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
  // Types
  type AdapterManifest,
  type AdapterMetadata,
  type AdapterTensorSpec,
  type MinimalAdapterManifest,
  type ManifestValidationResult,
  type ManifestValidationError,
  // Functions
  validateManifest,
  parseManifest,
  serializeManifest,
  createManifest,
  computeLoRAScale,
} from './adapter-manifest.js';

// LoRA loading
export {
  // Types
  type LoRAManifest,
  type LoRATensorSpec,
  type LoRALoadOptions,
  type LoRAWeightsResult,
  // Functions
  loadLoRAWeights,
  loadLoRAFromManifest,
  loadLoRAFromUrl,
  loadLoRAFromSafetensors,
  applyDeltaWeights,
} from './lora-loader.js';

// Adapter management
export {
  // Class
  AdapterManager,
  // Types
  type AdapterState,
  type EnableAdapterOptions,
  type AdapterStackOptions,
  type AdapterManagerEvents,
  // Default instance
  getAdapterManager,
  resetAdapterManager,
} from './adapter-manager.js';

// Adapter registry
export {
  // Class
  AdapterRegistry,
  // Types
  type AdapterRegistryEntry,
  type AdapterQueryOptions,
  type RegistryStorage,
  // Default instance
  getAdapterRegistry,
  resetAdapterRegistry,
  createMemoryRegistry,
} from './adapter-registry.js';

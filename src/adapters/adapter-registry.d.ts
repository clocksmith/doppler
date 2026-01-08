/**
 * Local Adapter Registry
 *
 * Persists adapter metadata to OPFS/IndexedDB for offline discovery.
 * Tracks available adapters without loading full weights into memory.
 *
 * @module adapters/adapter-registry
 */

import type { AdapterManifest, AdapterMetadata } from './adapter-manifest.js';
import type { LoRAModuleName } from '../inference/pipeline/lora-types.js';

/**
 * Registry entry for a stored adapter.
 */
export interface AdapterRegistryEntry {
  /** Unique adapter ID */
  id: string;
  /** Human-readable name */
  name: string;
  /** Version string */
  version: string;
  /** Base model this adapter is for */
  baseModel: string;
  /** LoRA rank */
  rank: number;
  /** LoRA alpha */
  alpha: number;
  /** Target modules */
  targetModules: LoRAModuleName[];
  /** Storage location type */
  storageType: 'opfs' | 'indexeddb' | 'url';
  /** Path to manifest */
  manifestPath: string;
  /** Path to weights (if separate from manifest) */
  weightsPath?: string;
  /** Size of weights in bytes */
  weightsSize?: number;
  /** SHA-256 checksum */
  checksum?: string;
  /** Additional metadata */
  metadata?: AdapterMetadata;
  /** Registration timestamp */
  registeredAt: number;
  /** Last access timestamp */
  lastAccessedAt: number;
}

/**
 * Query options for listing adapters.
 */
export interface AdapterQueryOptions {
  /** Filter by base model */
  baseModel?: string;
  /** Filter by target modules (adapter must include all) */
  targetModules?: LoRAModuleName[];
  /** Filter by tags (adapter must include at least one) */
  tags?: string[];
  /** Sort field */
  sortBy?: 'name' | 'registeredAt' | 'lastAccessedAt';
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
  /** Maximum number of results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Registry storage interface.
 */
export interface RegistryStorage {
  /** Get all entries */
  getAll(): Promise<AdapterRegistryEntry[]>;
  /** Get entry by ID */
  get(id: string): Promise<AdapterRegistryEntry | null>;
  /** Set entry */
  set(id: string, entry: AdapterRegistryEntry): Promise<void>;
  /** Delete entry */
  delete(id: string): Promise<boolean>;
  /** Clear all entries */
  clear(): Promise<void>;
}

/**
 * Local registry for tracking available LoRA adapters.
 */
export declare class AdapterRegistry {
  constructor(storage?: RegistryStorage);

  /**
   * Registers an adapter in the registry.
   */
  register(
    manifest: AdapterManifest,
    location: {
      storageType: 'opfs' | 'indexeddb' | 'url';
      manifestPath: string;
      weightsPath?: string;
    }
  ): Promise<AdapterRegistryEntry>;

  /**
   * Registers an adapter from a URL (fetches manifest first).
   */
  registerFromUrl(url: string): Promise<AdapterRegistryEntry>;

  /**
   * Unregisters an adapter from the registry.
   */
  unregister(id: string): Promise<boolean>;

  /**
   * Clears all entries from the registry.
   */
  clear(): Promise<void>;

  /**
   * Gets an adapter entry by ID.
   */
  get(id: string): Promise<AdapterRegistryEntry | null>;

  /**
   * Lists adapters matching the given query.
   */
  list(options?: AdapterQueryOptions): Promise<AdapterRegistryEntry[]>;

  /**
   * Gets count of registered adapters.
   */
  count(options?: Omit<AdapterQueryOptions, 'sortBy' | 'sortOrder' | 'limit' | 'offset'>): Promise<number>;

  /**
   * Checks if an adapter is registered.
   */
  has(id: string): Promise<boolean>;

  /**
   * Gets all unique base models in the registry.
   */
  getBaseModels(): Promise<string[]>;

  /**
   * Gets all unique tags in the registry.
   */
  getTags(): Promise<string[]>;

  /**
   * Updates metadata for an adapter.
   */
  updateMetadata(id: string, metadata: Partial<AdapterMetadata>): Promise<AdapterRegistryEntry | null>;

  /**
   * Updates storage location for an adapter.
   */
  updateLocation(
    id: string,
    location: {
      storageType?: 'opfs' | 'indexeddb' | 'url';
      manifestPath?: string;
      weightsPath?: string;
    }
  ): Promise<AdapterRegistryEntry | null>;

  /**
   * Exports all registry entries as JSON.
   */
  exportToJSON(): Promise<string>;

  /**
   * Imports registry entries from JSON.
   */
  importFromJSON(
    json: string,
    options?: { overwrite?: boolean; merge?: boolean }
  ): Promise<{ imported: number; skipped: number; errors: string[] }>;
}

/**
 * Gets the default adapter registry instance.
 */
export declare function getAdapterRegistry(): AdapterRegistry;

/**
 * Resets the default adapter registry (useful for testing).
 */
export declare function resetAdapterRegistry(): void;

/**
 * Creates an in-memory registry for testing.
 */
export declare function createMemoryRegistry(): AdapterRegistry;

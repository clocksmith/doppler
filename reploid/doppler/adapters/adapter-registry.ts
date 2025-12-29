/**
 * Local Adapter Registry
 *
 * Persists adapter metadata to OPFS/IndexedDB for offline discovery.
 * Tracks available adapters without loading full weights into memory.
 *
 * @module adapters/adapter-registry
 */

import type { AdapterManifest, AdapterMetadata } from './adapter-manifest.js';
import { validateManifest } from './adapter-manifest.js';
import type { LoRAModuleName } from '../inference/pipeline/lora-types.js';

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// IndexedDB Storage Implementation
// ============================================================================

const DB_NAME = 'doppler-adapter-registry';
const DB_VERSION = 1;
const STORE_NAME = 'adapters';

/**
 * IndexedDB-backed registry storage.
 */
class IndexedDBStorage implements RegistryStorage {
  private db: IDBDatabase | null = null;
  private initPromise: Promise<void> | null = null;

  private async init(): Promise<void> {
    if (this.db) return;

    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        reject(new Error(`Failed to open IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('baseModel', 'baseModel', { unique: false });
          store.createIndex('registeredAt', 'registeredAt', { unique: false });
          store.createIndex('lastAccessedAt', 'lastAccessedAt', { unique: false });
        }
      };
    });

    await this.initPromise;
  }

  async getAll(): Promise<AdapterRegistryEntry[]> {
    await this.init();

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(new Error(`Failed to get all: ${request.error?.message}`));
    });
  }

  async get(id: string): Promise<AdapterRegistryEntry | null> {
    await this.init();

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(new Error(`Failed to get ${id}: ${request.error?.message}`));
    });
  }

  async set(id: string, entry: AdapterRegistryEntry): Promise<void> {
    await this.init();

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error(`Failed to set ${id}: ${request.error?.message}`));
    });
  }

  async delete(id: string): Promise<boolean> {
    await this.init();

    const existing = await this.get(id);
    if (!existing) return false;

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(new Error(`Failed to delete ${id}: ${request.error?.message}`));
    });
  }

  async clear(): Promise<void> {
    await this.init();

    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error(`Failed to clear: ${request.error?.message}`));
    });
  }
}

// ============================================================================
// In-Memory Storage (for Node.js or testing)
// ============================================================================

/**
 * In-memory registry storage (fallback for non-browser environments).
 */
class MemoryStorage implements RegistryStorage {
  private data: Map<string, AdapterRegistryEntry> = new Map();

  async getAll(): Promise<AdapterRegistryEntry[]> {
    return [...this.data.values()];
  }

  async get(id: string): Promise<AdapterRegistryEntry | null> {
    return this.data.get(id) || null;
  }

  async set(id: string, entry: AdapterRegistryEntry): Promise<void> {
    this.data.set(id, entry);
  }

  async delete(id: string): Promise<boolean> {
    return this.data.delete(id);
  }

  async clear(): Promise<void> {
    this.data.clear();
  }
}

// ============================================================================
// Adapter Registry Class
// ============================================================================

/**
 * Local registry for tracking available LoRA adapters.
 *
 * Provides persistent storage and discovery of adapters without
 * loading full weights into memory. Supports OPFS and IndexedDB backends.
 *
 * Usage:
 * ```typescript
 * const registry = new AdapterRegistry();
 *
 * // Register an adapter from manifest
 * await registry.register(manifest, {
 *   storageType: 'opfs',
 *   manifestPath: '/adapters/coding/manifest.json'
 * });
 *
 * // List all adapters for a base model
 * const adapters = await registry.list({ baseModel: 'gemma-3-1b' });
 *
 * // Get adapter info
 * const info = await registry.get('coding-assistant');
 * ```
 */
export class AdapterRegistry {
  private storage: RegistryStorage;
  private cache: Map<string, AdapterRegistryEntry> = new Map();
  private cacheValid = false;

  constructor(storage?: RegistryStorage) {
    // Use IndexedDB in browser, memory storage elsewhere
    if (storage) {
      this.storage = storage;
    } else if (typeof indexedDB !== 'undefined') {
      this.storage = new IndexedDBStorage();
    } else {
      this.storage = new MemoryStorage();
    }
  }

  // ==========================================================================
  // Registration
  // ==========================================================================

  /**
   * Registers an adapter in the registry.
   *
   * @param manifest - Adapter manifest
   * @param location - Storage location info
   */
  async register(
    manifest: AdapterManifest,
    location: {
      storageType: 'opfs' | 'indexeddb' | 'url';
      manifestPath: string;
      weightsPath?: string;
    }
  ): Promise<AdapterRegistryEntry> {
    // Validate manifest
    const validation = validateManifest(manifest);
    if (!validation.valid) {
      const errors = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
      throw new Error(`Invalid manifest: ${errors}`);
    }

    const now = Date.now();

    const entry: AdapterRegistryEntry = {
      id: manifest.id,
      name: manifest.name,
      version: manifest.version || '1.0.0',
      baseModel: manifest.baseModel,
      rank: manifest.rank,
      alpha: manifest.alpha,
      targetModules: manifest.targetModules,
      storageType: location.storageType,
      manifestPath: location.manifestPath,
      weightsPath: location.weightsPath,
      weightsSize: manifest.weightsSize,
      checksum: manifest.checksum,
      metadata: manifest.metadata,
      registeredAt: now,
      lastAccessedAt: now,
    };

    await this.storage.set(manifest.id, entry);
    this.cache.set(manifest.id, entry);

    return entry;
  }

  /**
   * Registers an adapter from a URL (fetches manifest first).
   */
  async registerFromUrl(url: string): Promise<AdapterRegistryEntry> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
    }

    const manifest = await res.json() as AdapterManifest;

    return this.register(manifest, {
      storageType: 'url',
      manifestPath: url,
    });
  }

  // ==========================================================================
  // Unregistration
  // ==========================================================================

  /**
   * Unregisters an adapter from the registry.
   *
   * This only removes the registry entry, not the actual adapter files.
   */
  async unregister(id: string): Promise<boolean> {
    const deleted = await this.storage.delete(id);
    this.cache.delete(id);
    return deleted;
  }

  /**
   * Clears all entries from the registry.
   */
  async clear(): Promise<void> {
    await this.storage.clear();
    this.cache.clear();
    this.cacheValid = false;
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  /**
   * Gets an adapter entry by ID.
   *
   * Updates lastAccessedAt timestamp.
   */
  async get(id: string): Promise<AdapterRegistryEntry | null> {
    // Check cache first
    let entry = this.cache.get(id);

    if (!entry) {
      entry = await this.storage.get(id);
      if (entry) {
        this.cache.set(id, entry);
      }
    }

    if (entry) {
      // Update last accessed time
      entry.lastAccessedAt = Date.now();
      await this.storage.set(id, entry);
    }

    return entry || null;
  }

  /**
   * Lists adapters matching the given query.
   */
  async list(options: AdapterQueryOptions = {}): Promise<AdapterRegistryEntry[]> {
    let entries = await this.storage.getAll();

    // Apply filters
    if (options.baseModel) {
      entries = entries.filter(e => e.baseModel === options.baseModel);
    }

    if (options.targetModules && options.targetModules.length > 0) {
      entries = entries.filter(e =>
        options.targetModules!.every(mod => e.targetModules.includes(mod))
      );
    }

    if (options.tags && options.tags.length > 0) {
      entries = entries.filter(e =>
        e.metadata?.tags?.some(tag => options.tags!.includes(tag))
      );
    }

    // Apply sorting
    const sortField = options.sortBy || 'name';
    const sortOrder = options.sortOrder || 'asc';
    const sortMultiplier = sortOrder === 'asc' ? 1 : -1;

    entries.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortMultiplier * aVal.localeCompare(bVal);
      }
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortMultiplier * (aVal - bVal);
      }
      return 0;
    });

    // Apply pagination
    if (options.offset) {
      entries = entries.slice(options.offset);
    }
    if (options.limit) {
      entries = entries.slice(0, options.limit);
    }

    return entries;
  }

  /**
   * Gets count of registered adapters.
   */
  async count(options: Omit<AdapterQueryOptions, 'sortBy' | 'sortOrder' | 'limit' | 'offset'> = {}): Promise<number> {
    const entries = await this.list(options);
    return entries.length;
  }

  /**
   * Checks if an adapter is registered.
   */
  async has(id: string): Promise<boolean> {
    const entry = await this.storage.get(id);
    return entry !== null;
  }

  /**
   * Gets all unique base models in the registry.
   */
  async getBaseModels(): Promise<string[]> {
    const entries = await this.storage.getAll();
    const models = new Set(entries.map(e => e.baseModel));
    return [...models].sort();
  }

  /**
   * Gets all unique tags in the registry.
   */
  async getTags(): Promise<string[]> {
    const entries = await this.storage.getAll();
    const tags = new Set<string>();
    for (const entry of entries) {
      if (entry.metadata?.tags) {
        for (const tag of entry.metadata.tags) {
          tags.add(tag);
        }
      }
    }
    return [...tags].sort();
  }

  // ==========================================================================
  // Update Methods
  // ==========================================================================

  /**
   * Updates metadata for an adapter.
   */
  async updateMetadata(id: string, metadata: Partial<AdapterMetadata>): Promise<AdapterRegistryEntry | null> {
    const entry = await this.storage.get(id);
    if (!entry) return null;

    entry.metadata = {
      ...entry.metadata,
      ...metadata,
      updatedAt: new Date().toISOString(),
    };

    await this.storage.set(id, entry);
    this.cache.set(id, entry);

    return entry;
  }

  /**
   * Updates storage location for an adapter.
   */
  async updateLocation(
    id: string,
    location: {
      storageType?: 'opfs' | 'indexeddb' | 'url';
      manifestPath?: string;
      weightsPath?: string;
    }
  ): Promise<AdapterRegistryEntry | null> {
    const entry = await this.storage.get(id);
    if (!entry) return null;

    if (location.storageType) entry.storageType = location.storageType;
    if (location.manifestPath) entry.manifestPath = location.manifestPath;
    if (location.weightsPath) entry.weightsPath = location.weightsPath;

    await this.storage.set(id, entry);
    this.cache.set(id, entry);

    return entry;
  }

  // ==========================================================================
  // Import/Export
  // ==========================================================================

  /**
   * Exports all registry entries as JSON.
   */
  async exportToJSON(): Promise<string> {
    const entries = await this.storage.getAll();
    return JSON.stringify(entries, null, 2);
  }

  /**
   * Imports registry entries from JSON.
   *
   * @param json - JSON string with array of entries
   * @param options - Import options
   */
  async importFromJSON(
    json: string,
    options: { overwrite?: boolean; merge?: boolean } = {}
  ): Promise<{ imported: number; skipped: number; errors: string[] }> {
    let entries: AdapterRegistryEntry[];
    try {
      entries = JSON.parse(json);
    } catch (e) {
      throw new Error(`Invalid JSON: ${(e as Error).message}`);
    }

    if (!Array.isArray(entries)) {
      throw new Error('JSON must be an array of entries');
    }

    let imported = 0;
    let skipped = 0;
    const errors: string[] = [];

    for (const entry of entries) {
      try {
        const existing = await this.storage.get(entry.id);

        if (existing && !options.overwrite) {
          skipped++;
          continue;
        }

        if (existing && options.merge) {
          // Merge metadata
          entry.metadata = { ...existing.metadata, ...entry.metadata };
          entry.registeredAt = existing.registeredAt;
        }

        await this.storage.set(entry.id, entry);
        imported++;
      } catch (e) {
        errors.push(`${entry.id}: ${(e as Error).message}`);
      }
    }

    // Invalidate cache
    this.cache.clear();
    this.cacheValid = false;

    return { imported, skipped, errors };
  }
}

// ============================================================================
// Default Instance
// ============================================================================

/**
 * Default global adapter registry instance.
 */
let defaultRegistry: AdapterRegistry | null = null;

/**
 * Gets the default adapter registry instance.
 */
export function getAdapterRegistry(): AdapterRegistry {
  if (!defaultRegistry) {
    defaultRegistry = new AdapterRegistry();
  }
  return defaultRegistry;
}

/**
 * Resets the default adapter registry (useful for testing).
 */
export function resetAdapterRegistry(): void {
  defaultRegistry = null;
}

/**
 * Creates an in-memory registry for testing.
 */
export function createMemoryRegistry(): AdapterRegistry {
  return new AdapterRegistry(new MemoryStorage());
}

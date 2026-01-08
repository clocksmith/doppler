/**
 * Local Adapter Registry
 *
 * Persists adapter metadata to OPFS/IndexedDB for offline discovery.
 * Tracks available adapters without loading full weights into memory.
 *
 * @module adapters/adapter-registry
 */

import { validateManifest } from './adapter-manifest.js';

// ============================================================================
// IndexedDB Storage Implementation
// ============================================================================

const DB_NAME = 'doppler-adapter-registry';
const DB_VERSION = 1;
const STORE_NAME = 'adapters';

/**
 * IndexedDB-backed registry storage.
 */
class IndexedDBStorage {
  #db = null;
  #initPromise = null;

  async #init() {
    if (this.#db) return;

    if (this.#initPromise) {
      await this.#initPromise;
      return;
    }

    this.#initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        reject(new Error(`Failed to open IndexedDB: ${request.error?.message}`));
      };

      request.onsuccess = () => {
        this.#db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('baseModel', 'baseModel', { unique: false });
          store.createIndex('registeredAt', 'registeredAt', { unique: false });
          store.createIndex('lastAccessedAt', 'lastAccessedAt', { unique: false });
        }
      };
    });

    await this.#initPromise;
  }

  async getAll() {
    await this.#init();

    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(new Error(`Failed to get all: ${request.error?.message}`));
    });
  }

  async get(id) {
    await this.#init();

    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(new Error(`Failed to get ${id}: ${request.error?.message}`));
    });
  }

  async set(id, entry) {
    await this.#init();

    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.put(entry);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error(`Failed to set ${id}: ${request.error?.message}`));
    });
  }

  async delete(id) {
    await this.#init();

    const existing = await this.get(id);
    if (!existing) return false;

    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(new Error(`Failed to delete ${id}: ${request.error?.message}`));
    });
  }

  async clear() {
    await this.#init();

    return new Promise((resolve, reject) => {
      const tx = this.#db.transaction(STORE_NAME, 'readwrite');
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
class MemoryStorage {
  #data = new Map();

  async getAll() {
    return [...this.#data.values()];
  }

  async get(id) {
    return this.#data.get(id) || null;
  }

  async set(id, entry) {
    this.#data.set(id, entry);
  }

  async delete(id) {
    return this.#data.delete(id);
  }

  async clear() {
    this.#data.clear();
  }
}

// ============================================================================
// Adapter Registry Class
// ============================================================================

/**
 * Local registry for tracking available LoRA adapters.
 */
export class AdapterRegistry {
  #storage;
  #cache = new Map();
  #cacheValid = false;

  constructor(storage) {
    // Use IndexedDB in browser, memory storage elsewhere
    if (storage) {
      this.#storage = storage;
    } else if (typeof indexedDB !== 'undefined') {
      this.#storage = new IndexedDBStorage();
    } else {
      this.#storage = new MemoryStorage();
    }
  }

  // ==========================================================================
  // Registration
  // ==========================================================================

  /**
   * Registers an adapter in the registry.
   */
  async register(manifest, location) {
    // Validate manifest
    const validation = validateManifest(manifest);
    if (!validation.valid) {
      const errors = validation.errors.map(e => `${e.field}: ${e.message}`).join('; ');
      throw new Error(`Invalid manifest: ${errors}`);
    }

    const now = Date.now();

    const entry = {
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

    await this.#storage.set(manifest.id, entry);
    this.#cache.set(manifest.id, entry);

    return entry;
  }

  /**
   * Registers an adapter from a URL (fetches manifest first).
   */
  async registerFromUrl(url) {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch manifest: ${res.status} ${res.statusText}`);
    }

    const manifest = await res.json();

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
   */
  async unregister(id) {
    const deleted = await this.#storage.delete(id);
    this.#cache.delete(id);
    return deleted;
  }

  /**
   * Clears all entries from the registry.
   */
  async clear() {
    await this.#storage.clear();
    this.#cache.clear();
    this.#cacheValid = false;
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  /**
   * Gets an adapter entry by ID.
   */
  async get(id) {
    // Check cache first
    let entry = this.#cache.get(id);

    if (!entry) {
      entry = await this.#storage.get(id);
      if (entry) {
        this.#cache.set(id, entry);
      }
    }

    if (entry) {
      // Update last accessed time
      entry.lastAccessedAt = Date.now();
      await this.#storage.set(id, entry);
    }

    return entry || null;
  }

  /**
   * Lists adapters matching the given query.
   */
  async list(options = {}) {
    let entries = await this.#storage.getAll();

    // Apply filters
    if (options.baseModel) {
      entries = entries.filter(e => e.baseModel === options.baseModel);
    }

    if (options.targetModules && options.targetModules.length > 0) {
      entries = entries.filter(e =>
        options.targetModules.every(mod => e.targetModules.includes(mod))
      );
    }

    if (options.tags && options.tags.length > 0) {
      entries = entries.filter(e =>
        e.metadata?.tags?.some(tag => options.tags.includes(tag))
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
  async count(options = {}) {
    const entries = await this.list(options);
    return entries.length;
  }

  /**
   * Checks if an adapter is registered.
   */
  async has(id) {
    const entry = await this.#storage.get(id);
    return entry !== null;
  }

  /**
   * Gets all unique base models in the registry.
   */
  async getBaseModels() {
    const entries = await this.#storage.getAll();
    const models = new Set(entries.map(e => e.baseModel));
    return [...models].sort();
  }

  /**
   * Gets all unique tags in the registry.
   */
  async getTags() {
    const entries = await this.#storage.getAll();
    const tags = new Set();
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
  async updateMetadata(id, metadata) {
    const entry = await this.#storage.get(id);
    if (!entry) return null;

    entry.metadata = {
      ...entry.metadata,
      ...metadata,
      updatedAt: new Date().toISOString(),
    };

    await this.#storage.set(id, entry);
    this.#cache.set(id, entry);

    return entry;
  }

  /**
   * Updates storage location for an adapter.
   */
  async updateLocation(id, location) {
    const entry = await this.#storage.get(id);
    if (!entry) return null;

    if (location.storageType) entry.storageType = location.storageType;
    if (location.manifestPath) entry.manifestPath = location.manifestPath;
    if (location.weightsPath) entry.weightsPath = location.weightsPath;

    await this.#storage.set(id, entry);
    this.#cache.set(id, entry);

    return entry;
  }

  // ==========================================================================
  // Import/Export
  // ==========================================================================

  /**
   * Exports all registry entries as JSON.
   */
  async exportToJSON() {
    const entries = await this.#storage.getAll();
    return JSON.stringify(entries, null, 2);
  }

  /**
   * Imports registry entries from JSON.
   */
  async importFromJSON(json, options = {}) {
    let entries;
    try {
      entries = JSON.parse(json);
    } catch (e) {
      throw new Error(`Invalid JSON: ${e.message}`);
    }

    if (!Array.isArray(entries)) {
      throw new Error('JSON must be an array of entries');
    }

    let imported = 0;
    let skipped = 0;
    const errors = [];

    for (const entry of entries) {
      try {
        const existing = await this.#storage.get(entry.id);

        if (existing && !options.overwrite) {
          skipped++;
          continue;
        }

        if (existing && options.merge) {
          // Merge metadata
          entry.metadata = { ...existing.metadata, ...entry.metadata };
          entry.registeredAt = existing.registeredAt;
        }

        await this.#storage.set(entry.id, entry);
        imported++;
      } catch (e) {
        errors.push(`${entry.id}: ${e.message}`);
      }
    }

    // Invalidate cache
    this.#cache.clear();
    this.#cacheValid = false;

    return { imported, skipped, errors };
  }
}

// ============================================================================
// Default Instance
// ============================================================================

/**
 * Default global adapter registry instance.
 */
let defaultRegistry = null;

/**
 * Gets the default adapter registry instance.
 */
export function getAdapterRegistry() {
  if (!defaultRegistry) {
    defaultRegistry = new AdapterRegistry();
  }
  return defaultRegistry;
}

/**
 * Resets the default adapter registry (useful for testing).
 */
export function resetAdapterRegistry() {
  defaultRegistry = null;
}

/**
 * Creates an in-memory registry for testing.
 */
export function createMemoryRegistry() {
  return new AdapterRegistry(new MemoryStorage());
}

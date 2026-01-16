

import { validateManifest } from './adapter-manifest.js';
import { DEFAULT_ADAPTER_REGISTRY_CONFIG } from '../config/schema/index.js';

// ============================================================================
// IndexedDB Storage Implementation
// ============================================================================

const { dbName: DB_NAME, dbVersion: DB_VERSION, storeName: STORE_NAME } = DEFAULT_ADAPTER_REGISTRY_CONFIG;


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

  
  async unregister(id) {
    const deleted = await this.#storage.delete(id);
    this.#cache.delete(id);
    return deleted;
  }

  
  async clear() {
    await this.#storage.clear();
    this.#cache.clear();
    this.#cacheValid = false;
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  
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
    const sortField = options.sortBy || DEFAULT_ADAPTER_REGISTRY_CONFIG.defaultSortBy;
    const sortOrder = options.sortOrder || DEFAULT_ADAPTER_REGISTRY_CONFIG.defaultSortOrder;
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

  
  async count(options = {}) {
    const entries = await this.list(options);
    return entries.length;
  }

  
  async has(id) {
    const entry = await this.#storage.get(id);
    return entry !== null;
  }

  
  async getBaseModels() {
    const entries = await this.#storage.getAll();
    const models = new Set(entries.map(e => e.baseModel));
    return [...models].sort();
  }

  
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

  
  async exportToJSON() {
    const entries = await this.#storage.getAll();
    return JSON.stringify(entries, null, 2);
  }

  
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


let defaultRegistry = null;


export function getAdapterRegistry() {
  if (!defaultRegistry) {
    defaultRegistry = new AdapterRegistry();
  }
  return defaultRegistry;
}


export function resetAdapterRegistry() {
  defaultRegistry = null;
}


export function createMemoryRegistry() {
  return new AdapterRegistry(new MemoryStorage());
}

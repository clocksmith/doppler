

import { log } from '../src/debug/index.js';
import {
  listModels,
  openModelStore,
  loadManifestFromStore,
} from '../src/storage/shard-manager.js';
import { parseManifest } from '../src/storage/rdrr-format.js';
import { QUICKSTART_MODELS } from '../src/storage/quickstart-downloader.js';
import { listRegisteredModels } from '../src/storage/registry.js';

export class ModelRegistry {
  #models = [];

  #remoteModels = [];

  constructor(remoteModels = []) {
    this.#remoteModels = remoteModels;
  }

  async discover() {
    log.debug('ModelRegistry', 'Discovering models...');

    const modelMap = new Map();

    // 1. Discover server models (local HTTP)
    const serverModels = await this.#discoverServerModels();
    log.debug('ModelRegistry', `Found ${serverModels.length} server models`);

    for (const model of serverModels) {
      const key = this.#getModelKey(model.architecture, model.quantization, model.downloadSize);
      this.#addModel(modelMap, key, {
        name: model.name,
        size: model.size,
        quantization: model.quantization,
        downloadSize: model.downloadSize,
        architecture: model.architecture,
      }, 'server', { id: model.id, url: model.url });
    }

    // 2. Discover browser-cached models (OPFS)
    const cachedModels = await this.#discoverCachedModels();
    for (const cached of cachedModels) {
      const key = this.#getModelKey(cached.architecture, cached.quantization, cached.downloadSize);
      this.#addModel(modelMap, key, {
        name: cached.name,
        architecture: cached.architecture,
        size: cached.size,
        quantization: cached.quantization,
        downloadSize: cached.downloadSize,
      }, 'browser', { id: cached.id });
    }

    // 3. Add remote models
    for (const remote of this.#remoteModels) {
      const key = this.#getModelKey(remote.architecture || remote.id, remote.quantization, remote.downloadSize);
      this.#addModel(modelMap, key, {
        name: remote.name,
        size: remote.size,
        quantization: remote.quantization,
        downloadSize: remote.downloadSize,
        architecture: remote.architecture,
      }, 'remote', { id: remote.id, url: remote.url });
    }

    // 4. Add Quick Start models
    for (const [modelId, config] of Object.entries(QUICKSTART_MODELS)) {
      const req = config.requirements;
      const key = this.#getModelKey(req.architecture || modelId, req.quantization, req.downloadSize);
      const existing = modelMap.get(key);
      if (existing) {
        existing.quickStartAvailable = true;
      } else {
        this.#addModel(modelMap, key, {
          name: config.displayName,
          size: req.paramCount,
          quantization: req.quantization,
          downloadSize: req.downloadSize,
          architecture: req.architecture,
          quickStartAvailable: true,
        }, 'remote', { id: modelId, url: config.baseUrl });
      }
    }

    // Sort by availability (server+browser > server > browser > remote)
    this.#models = Array.from(modelMap.values()).sort((a, b) => {
      return this.#getAvailabilityScore(b) - this.#getAvailabilityScore(a);
    });

    log.info('ModelRegistry', `Registry: ${this.#models.length} unique models`);
    return this.#models;
  }

  async #discoverServerModels() {
    const baseUrl = window.location.origin;

    try {
      const response = await fetch(`${baseUrl}/api/models`);
      if (!response.ok) return [];

      const models = await response.json();
      return models.map((m) => {
        let modelName = m.name
        .replace(/-rdrr$/, '')
        .replace(/-q4$/, '')
        .split('-')
        .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
        .join(' ');

        const inferredParams = m.size || (m.numLayers ? `${m.numLayers}L` : 'Unknown');

        return {
          id: m.name,
          name: modelName,
          size: inferredParams,
          quantization: m.quantization || 'Unknown',
          downloadSize: m.downloadSize || 0,
          url: `${baseUrl}/${m.path}`,
          architecture: m.architecture || 'Unknown',
        };
      });
    } catch (e) {
      log.debug('ModelRegistry', 'Failed to fetch models from API:', e);
      return [];
    }
  }

  async #discoverCachedModels() {
    const results = [];
    const registryEntries = [];

    let cachedIds = [];
    try {
      const registered = await listRegisteredModels();
      if (Array.isArray(registered)) {
        registryEntries.push(...registered);
      }
    } catch (err) {
      log.debug('ModelRegistry', `No registry entries available: ${err.message}`);
    }

    const registryMap = new Map(
      registryEntries.map((entry) => [entry.modelId, entry])
    );

    try {
      cachedIds = await listModels();
      log.debug('ModelRegistry', 'Found cached models in OPFS:', cachedIds);
    } catch (err) {
      log.warn('ModelRegistry', 'Could not query cached models:', err.message);
      if (registryEntries.length === 0) {
        return results;
      }
      cachedIds = registryEntries.map((entry) => entry.modelId);
    }

    for (const cachedId of cachedIds) {
      try {
        await openModelStore(cachedId);
        const manifestText = await loadManifestFromStore();
        if (!manifestText) continue;

        const manifest = parseManifest(manifestText);
        const archInfo = manifest.architecture;
        const archLabel = typeof archInfo === 'string' ? archInfo : manifest.modelType;
        const registryEntry = registryMap.get(cachedId);
        const quant = manifest.quantization || registryEntry?.quantization || 'Unknown';
        const totalSize = manifest.totalSize
          ?? (manifest.shards || []).reduce((sum, s) => sum + (s.size || 0), 0)
          ?? registryEntry?.totalSize
          ?? 0;

        // Estimate param count from hidden size
        const hiddenSize = typeof archInfo === 'object' && archInfo !== null
        ? archInfo.hiddenSize || 0
        : 0;
        let paramStr = 'Unknown';
        if (hiddenSize >= 4096) paramStr = '7B+';
        else if (hiddenSize >= 2048) paramStr = '1-3B';
        else if (hiddenSize >= 1024) paramStr = '<1B';

        results.push({
          id: cachedId,
          name: manifest.modelId || this.#formatModelName(cachedId),
          architecture: archLabel || 'Unknown',
          size: paramStr,
          quantization: quant,
          downloadSize: totalSize,
        });
      } catch (e) {
        const registryEntry = registryMap.get(cachedId);
        if (registryEntry) {
          results.push({
            id: cachedId,
            name: registryEntry.modelId || this.#formatModelName(cachedId),
            architecture: 'Unknown',
            size: 'Unknown',
            quantization: registryEntry.quantization || 'Unknown',
            downloadSize: registryEntry.totalSize || 0,
          });
        } else {
          log.warn('ModelRegistry', `Could not load manifest for cached model ${cachedId}:`, e.message);
        }
      }
    }

    return results;
  }

  #addModel(map, key, info, sourceType, sourceData) {
    if (map.has(key)) {
      const existing = map.get(key);
      existing.sources[sourceType] = sourceData;
      // Prefer better metadata (server > browser > remote)
      if (sourceType === 'server' || (sourceType === 'browser' && !existing.sources.server)) {
        existing.name = info.name || existing.name;
        existing.size = info.size || existing.size;
        existing.downloadSize = info.downloadSize || existing.downloadSize;
      }
    } else {
      map.set(key, {
        ...info,
        key,
        sources: { [sourceType]: sourceData },
      });
    }
  }

  #getModelKey(arch, quant, _size) {
    const normArch = (arch || 'unknown')
    .toLowerCase()
    .replace(/forcausallm|forconditionalgeneration|model/gi, '')
    .replace(/[^a-z0-9]/g, '');

    const normQuant = (quant || 'unknown').toLowerCase().replace(/[^a-z0-9]/g, '');

    return `${normArch}:${normQuant}`;
  }

  #getAvailabilityScore(model) {
    let score = 0;
    if (model.sources.server) score += 2;
    if (model.sources.browser) score += 1;
    return score;
  }

  #formatModelName(modelId) {
    let name = modelId
    .replace(/^custom-\d+$/, 'Custom Model')
    .replace(/^tools\//, '')
    .replace(/-rdrr$/, '')
    .replace(/-q4$/, '')
    .replace(/-q4_k_m$/i, '');

    if (/^custom-\d+$/.test(modelId)) {
      return 'Custom Model';
    }

    return name
    .split(/[-_]/)
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1).toLowerCase())
    .join(' ');
  }

  getModels() {
    return this.#models;
  }

  findByKey(key) {
    return this.#models.find((m) => m.key === key);
  }

  findByBrowserId(id) {
    return this.#models.find((m) => m.sources.browser?.id === id);
  }

  isAvailableLocally(model) {
    return !!(model.sources.server || model.sources.browser);
  }
}





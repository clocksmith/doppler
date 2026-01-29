

import { loadLoRAWeights } from './lora-loader.js';
import { log } from '../debug/index.js';
import { DEFAULT_ADAPTER_STACK_CONFIG } from '../config/schema/index.js';


function isFloat32Array(buf) {
  return buf instanceof Float32Array;
}

// ============================================================================
// Adapter Manager Class
// ============================================================================


export class AdapterManager {
  
  #adapters = new Map();

  
  #activeAdapterIds = [];

  
  #events = {};

  
  #defaultLoadOptions = {};

  
  #stackOptions = { ...DEFAULT_ADAPTER_STACK_CONFIG };

  // ==========================================================================
  // Configuration
  // ==========================================================================

  
  setDefaultLoadOptions(options) {
    this.#defaultLoadOptions = { ...options };
  }

  
  setEvents(events) {
    this.#events = { ...this.#events, ...events };
  }

  
  setStackOptions(options) {
    this.#stackOptions = { ...this.#stackOptions, ...options };
  }

  // ==========================================================================
  // Loading
  // ==========================================================================

  
  async loadAdapter(id, path, options = {}) {
    // Check if already loaded
    if (this.#adapters.has(id)) {
      throw new Error(`Adapter '${id}' is already loaded. Unload it first.`);
    }

    // Merge options with defaults
    const mergedOptions = { ...this.#defaultLoadOptions, ...options };

    // Load the adapter
    const result = await loadLoRAWeights(path, mergedOptions);

    // Create state
    const state = {
      id,
      adapter: result.adapter,
      manifest: result.manifest,
      enabled: false,
      weight: 1.0,
      loadedAt: Date.now(),
      lastToggled: 0,
    };

    // Store it
    this.#adapters.set(id, state);

    // Fire event
    this.#events.onAdapterLoaded?.(id, result.adapter);

    return state;
  }

  
  registerAdapter(id, adapter, manifest) {
    if (this.#adapters.has(id)) {
      throw new Error(`Adapter '${id}' is already loaded. Unload it first.`);
    }

    const state = {
      id,
      adapter,
      manifest,
      enabled: false,
      weight: 1.0,
      loadedAt: Date.now(),
      lastToggled: 0,
    };

    this.#adapters.set(id, state);
    this.#events.onAdapterLoaded?.(id, adapter);

    return state;
  }

  // ==========================================================================
  // Enable/Disable API
  // ==========================================================================

  
  enableAdapter(id, options = {}) {
    const state = this.#adapters.get(id);
    if (!state) {
      throw new Error(`Adapter '${id}' not found. Load it first.`);
    }

    // Validate base model if requested
    if (options.validateBaseModel && options.expectedBaseModel) {
      if (state.manifest.baseModel !== options.expectedBaseModel) {
        throw new Error(
          `Adapter '${id}' is for base model '${state.manifest.baseModel}' ` +
          `but expected '${options.expectedBaseModel}'`
        );
      }
    }

    // Set weight
    if (options.weight !== undefined) {
      const { minWeight, maxWeight } = DEFAULT_ADAPTER_STACK_CONFIG;
      if (options.weight < minWeight || options.weight > maxWeight) {
        throw new Error(`Adapter weight must be between ${minWeight} and ${maxWeight}`);
      }
      state.weight = options.weight;
    }

    // Already enabled?
    if (state.enabled) {
      return;
    }

    // Enable it
    state.enabled = true;
    state.lastToggled = Date.now();

    // Add to active list
    if (!this.#activeAdapterIds.includes(id)) {
      this.#activeAdapterIds.push(id);
    }

    // Fire events
    this.#events.onAdapterEnabled?.(id);
    this.#events.onActiveAdaptersChanged?.([...this.#activeAdapterIds]);
  }

  
  disableAdapter(id) {
    const state = this.#adapters.get(id);
    if (!state) {
      throw new Error(`Adapter '${id}' not found.`);
    }

    // Already disabled?
    if (!state.enabled) {
      return;
    }

    // Disable it
    state.enabled = false;
    state.lastToggled = Date.now();

    // Remove from active list
    const idx = this.#activeAdapterIds.indexOf(id);
    if (idx >= 0) {
      this.#activeAdapterIds.splice(idx, 1);
    }

    // Fire events
    this.#events.onAdapterDisabled?.(id);
    this.#events.onActiveAdaptersChanged?.([...this.#activeAdapterIds]);
  }

  
  toggleAdapter(id) {
    const state = this.#adapters.get(id);
    if (!state) {
      throw new Error(`Adapter '${id}' not found.`);
    }

    if (state.enabled) {
      this.disableAdapter(id);
      return false;
    } else {
      this.enableAdapter(id);
      return true;
    }
  }

  
  disableAll() {
    for (const id of [...this.#activeAdapterIds]) {
      this.disableAdapter(id);
    }
  }

  
  enableOnly(id, options) {
    this.disableAll();
    this.enableAdapter(id, options);
  }

  
  setAdapterWeight(id, weight) {
    const state = this.#adapters.get(id);
    if (!state) {
      throw new Error(`Adapter '${id}' not found.`);
    }
    const { minWeight, maxWeight } = DEFAULT_ADAPTER_STACK_CONFIG;
    if (weight < minWeight || weight > maxWeight) {
      throw new Error(`Adapter weight must be between ${minWeight} and ${maxWeight}`);
    }
    state.weight = weight;
  }

  // ==========================================================================
  // Unloading
  // ==========================================================================

  
  unloadAdapter(id) {
    const state = this.#adapters.get(id);
    if (!state) {
      return;
    }

    // Disable if active
    if (state.enabled) {
      this.disableAdapter(id);
    }

    // Remove from map
    this.#adapters.delete(id);

    // Fire event
    this.#events.onAdapterUnloaded?.(id);
  }

  
  unloadAll() {
    for (const id of [...this.#adapters.keys()]) {
      this.unloadAdapter(id);
    }
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  
  getActiveAdapter() {
    if (this.#activeAdapterIds.length === 0) {
      return null;
    }

    if (this.#activeAdapterIds.length === 1) {
      const state = this.#adapters.get(this.#activeAdapterIds[0]);
      if (!state) return null;

      // Apply weight if not 1.0
      if (state.weight !== 1.0) {
        return this.#applyWeight(state.adapter, state.weight);
      }
      return state.adapter;
    }

    // Multiple adapters - merge them
    return this.#mergeActiveAdapters();
  }

  
  getActiveAdapterIds() {
    return [...this.#activeAdapterIds];
  }

  
  getAdapterState(id) {
    return this.#adapters.get(id);
  }

  
  getAllAdapters() {
    return [...this.#adapters.values()];
  }

  
  getEnabledAdapters() {
    return this.getAllAdapters().filter(s => s.enabled);
  }

  
  isLoaded(id) {
    return this.#adapters.has(id);
  }

  
  isEnabled(id) {
    return this.#adapters.get(id)?.enabled ?? false;
  }

  
  get loadedCount() {
    return this.#adapters.size;
  }

  
  get enabledCount() {
    return this.#activeAdapterIds.length;
  }

  // ==========================================================================
  // Merging Logic
  // ==========================================================================

  
  #mergeActiveAdapters() {
    const activeStates = this.#activeAdapterIds
      .map(id => this.#adapters.get(id))
      .filter((s) => s !== undefined);

    if (activeStates.length === 0) return null;
    if (activeStates.length === 1) {
      return this.#applyWeight(activeStates[0].adapter, activeStates[0].weight);
    }

    // Compute weights
    let weights = activeStates.map(s => s.weight);
    if (this.#stackOptions.normalizeWeights) {
      const sum = weights.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        weights = weights.map(w => w / sum);
      }
    }

    // Merge based on strategy
    switch (this.#stackOptions.strategy) {
      case 'sum':
      case 'weighted_sum':
        return this.#mergeByWeightedSum(activeStates, weights);
      case 'sequential':
        // For sequential, just use the last adapter
        return this.#applyWeight(
          activeStates[activeStates.length - 1].adapter,
          weights[weights.length - 1]
        );
      default:
        return activeStates[0].adapter;
    }
  }

  
  #mergeByWeightedSum(states, weights) {
    // Use first adapter as template
    const first = states[0].adapter;

    const merged = {
      name: `merged(${states.map(s => s.id).join('+')})`,
      rank: first.rank,
      alpha: first.alpha,
      targetModules: first.targetModules,
      layers: new Map(),
    };

    // Collect all layer indices
    const allLayers = new Set();
    for (const state of states) {
      for (const layerIdx of state.adapter.layers.keys()) {
        allLayers.add(layerIdx);
      }
    }

    // Merge each layer
    for (const layerIdx of allLayers) {
      const mergedLayer = {};

      // Get all modules in this layer across all adapters
      const allModules = new Set();
      for (const state of states) {
        const layer = state.adapter.layers.get(layerIdx);
        if (layer) {
          for (const mod of Object.keys(layer)) {
            allModules.add(mod);
          }
        }
      }

      // Merge each module
      for (const modName of allModules) {
        let mergedA = null;
        let mergedB = null;
        let mergedRank = 0;
        let mergedAlpha = 0;

        for (let i = 0; i < states.length; i++) {
          const state = states[i];
          const weight = weights[i];
          const layer = state.adapter.layers.get(layerIdx);
          const mod = layer?.[modName];

          if (!mod) continue;

          // Only Float32Array can be merged on CPU
          if (!isFloat32Array(mod.a) || !isFloat32Array(mod.b)) {
            log.warn('AdapterManager', 'Cannot merge GPUBuffer weights on CPU, skipping');
            continue;
          }

          if (!mergedA) {
            // First adapter with this module - initialize
            mergedA = new Float32Array(mod.a.length);
            mergedB = new Float32Array(mod.b.length);
            mergedRank = mod.rank;
            mergedAlpha = mod.alpha * weight;
          } else {
            // Accumulate alpha
            mergedAlpha += mod.alpha * weight;
          }

          // Weighted add to merged arrays
          for (let j = 0; j < mod.a.length; j++) {
            mergedA[j] += mod.a[j] * weight;
          }
          for (let j = 0; j < mod.b.length; j++) {
            mergedB[j] += mod.b[j] * weight;
          }
        }

        if (mergedA && mergedB) {
          mergedLayer[modName] = {
            a: mergedA,
            b: mergedB,
            rank: mergedRank,
            alpha: mergedAlpha,
            scale: mergedRank > 0 ? mergedAlpha / mergedRank : 1,
          };
        }
      }

      if (Object.keys(mergedLayer).length > 0) {
        merged.layers.set(layerIdx, mergedLayer);
      }
    }

    return merged;
  }

  
  #applyWeight(adapter, weight) {
    if (weight === 1.0) return adapter;

    const weighted = {
      ...adapter,
      alpha: adapter.alpha * weight,
      layers: new Map(),
    };

    for (const [layerIdx, layer] of adapter.layers) {
      const weightedLayer = {};

      for (const [modName, mod] of Object.entries(layer)) {
        weightedLayer[modName] = {
          ...mod,
          alpha: mod.alpha * weight,
          scale: mod.scale * weight,
        };
      }

      weighted.layers.set(layerIdx, weightedLayer);
    }

    return weighted;
  }
}

// ============================================================================
// Default Instance
// ============================================================================


let defaultManager = null;


export function getAdapterManager() {
  if (!defaultManager) {
    defaultManager = new AdapterManager();
  }
  return defaultManager;
}


export function resetAdapterManager() {
  if (defaultManager) {
    defaultManager.unloadAll();
  }
  defaultManager = null;
}

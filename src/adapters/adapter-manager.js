/**
 * Adapter Manager - Runtime adapter enable/disable API
 *
 * Manages active LoRA adapters with support for:
 * - Runtime switching without full model reload
 * - Multiple adapter stacking (with merge strategies)
 * - State tracking and validation
 *
 * @module adapters/adapter-manager
 */

import { loadLoRAWeights } from './lora-loader.js';
import { log } from '../debug/index.js';

/**
 * Helper to check if buffer is Float32Array.
 */
function isFloat32Array(buf) {
  return buf instanceof Float32Array;
}

// ============================================================================
// Adapter Manager Class
// ============================================================================

/**
 * Manages runtime loading, enabling, and disabling of LoRA adapters.
 */
export class AdapterManager {
  /** Map of adapter ID to state */
  #adapters = new Map();

  /** Currently active adapter IDs (in order) */
  #activeAdapterIds = [];

  /** Event callbacks */
  #events = {};

  /** Default loading options */
  #defaultLoadOptions = {};

  /** Stack options for combining multiple adapters */
  #stackOptions = {
    strategy: 'sum',
    normalizeWeights: false,
  };

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Sets default loading options for all adapter loads.
   */
  setDefaultLoadOptions(options) {
    this.#defaultLoadOptions = { ...options };
  }

  /**
   * Sets event callbacks.
   */
  setEvents(events) {
    this.#events = { ...this.#events, ...events };
  }

  /**
   * Sets adapter stacking options.
   */
  setStackOptions(options) {
    this.#stackOptions = { ...this.#stackOptions, ...options };
  }

  // ==========================================================================
  // Loading
  // ==========================================================================

  /**
   * Loads an adapter from a path (URL or OPFS).
   */
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

  /**
   * Loads an adapter from an already-parsed manifest and adapter.
   */
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

  /**
   * Enables an adapter for inference.
   */
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
      if (options.weight < 0 || options.weight > 2) {
        throw new Error('Adapter weight must be between 0.0 and 2.0');
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

  /**
   * Disables an adapter.
   */
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

  /**
   * Toggles an adapter's enabled state.
   */
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

  /**
   * Disables all adapters.
   */
  disableAll() {
    for (const id of [...this.#activeAdapterIds]) {
      this.disableAdapter(id);
    }
  }

  /**
   * Enables only the specified adapter, disabling all others.
   */
  enableOnly(id, options) {
    this.disableAll();
    this.enableAdapter(id, options);
  }

  /**
   * Sets the weight for an adapter.
   */
  setAdapterWeight(id, weight) {
    const state = this.#adapters.get(id);
    if (!state) {
      throw new Error(`Adapter '${id}' not found.`);
    }
    if (weight < 0 || weight > 2) {
      throw new Error('Adapter weight must be between 0.0 and 2.0');
    }
    state.weight = weight;
  }

  // ==========================================================================
  // Unloading
  // ==========================================================================

  /**
   * Unloads an adapter, freeing its memory.
   */
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

  /**
   * Unloads all adapters.
   */
  unloadAll() {
    for (const id of [...this.#adapters.keys()]) {
      this.unloadAdapter(id);
    }
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  /**
   * Gets the currently active adapter for use with pipeline.
   */
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

  /**
   * Gets all active adapter IDs.
   */
  getActiveAdapterIds() {
    return [...this.#activeAdapterIds];
  }

  /**
   * Gets state of a specific adapter.
   */
  getAdapterState(id) {
    return this.#adapters.get(id);
  }

  /**
   * Gets all loaded adapter states.
   */
  getAllAdapters() {
    return [...this.#adapters.values()];
  }

  /**
   * Gets all enabled adapter states.
   */
  getEnabledAdapters() {
    return this.getAllAdapters().filter(s => s.enabled);
  }

  /**
   * Checks if an adapter is loaded.
   */
  isLoaded(id) {
    return this.#adapters.has(id);
  }

  /**
   * Checks if an adapter is enabled.
   */
  isEnabled(id) {
    return this.#adapters.get(id)?.enabled ?? false;
  }

  /**
   * Gets count of loaded adapters.
   */
  get loadedCount() {
    return this.#adapters.size;
  }

  /**
   * Gets count of enabled adapters.
   */
  get enabledCount() {
    return this.#activeAdapterIds.length;
  }

  // ==========================================================================
  // Merging Logic
  // ==========================================================================

  /**
   * Merges multiple active adapters into a single virtual adapter.
   */
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

  /**
   * Merges adapters by weighted sum of their layers.
   */
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

  /**
   * Applies a weight multiplier to an adapter.
   */
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

/**
 * Default global adapter manager instance.
 */
let defaultManager = null;

/**
 * Gets the default adapter manager instance.
 */
export function getAdapterManager() {
  if (!defaultManager) {
    defaultManager = new AdapterManager();
  }
  return defaultManager;
}

/**
 * Resets the default adapter manager (useful for testing).
 */
export function resetAdapterManager() {
  if (defaultManager) {
    defaultManager.unloadAll();
  }
  defaultManager = null;
}

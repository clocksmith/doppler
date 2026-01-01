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

import type { LoRAAdapter, LoRAModuleWeights } from '../inference/pipeline/lora-types.js';
import type { AdapterManifest } from './adapter-manifest.js';
import { loadLoRAWeights, type LoRALoadOptions, type LoRAWeightsResult } from './lora-loader.js';

/**
 * Helper to get the length of a MaybeGPUBuffer.
 * For Float32Array, returns length. For GPUBuffer, estimates from size.
 */
function getBufferLength(buf: GPUBuffer | Float32Array): number {
  if (buf instanceof Float32Array) {
    return buf.length;
  }
  // GPUBuffer: size is in bytes, divide by 4 for float32 count
  return buf.size / 4;
}

/**
 * Helper to check if buffer is Float32Array.
 */
function isFloat32Array(buf: GPUBuffer | Float32Array): buf is Float32Array {
  return buf instanceof Float32Array;
}

/**
 * Helper to get element at index from MaybeGPUBuffer.
 * Only works for Float32Array; throws for GPUBuffer.
 */
function getElement(buf: GPUBuffer | Float32Array, idx: number): number {
  if (buf instanceof Float32Array) {
    return buf[idx];
  }
  throw new Error('Cannot read elements from GPUBuffer - use Float32Array for CPU merging');
}

// ============================================================================
// Types
// ============================================================================

/**
 * State of a loaded adapter.
 */
export interface AdapterState {
  /** Unique adapter identifier */
  id: string;
  /** The loaded adapter data */
  adapter: LoRAAdapter;
  /** Original manifest */
  manifest: AdapterManifest;
  /** Whether adapter is currently active */
  enabled: boolean;
  /** Weight multiplier for this adapter (default: 1.0) */
  weight: number;
  /** Load timestamp */
  loadedAt: number;
  /** Last enabled/disabled timestamp */
  lastToggled: number;
}

/**
 * Options for enabling an adapter.
 */
export interface EnableAdapterOptions {
  /** Weight multiplier (0.0 - 2.0, default: 1.0) */
  weight?: number;
  /** Whether to validate base model compatibility */
  validateBaseModel?: boolean;
  /** Expected base model ID */
  expectedBaseModel?: string;
}

/**
 * Options for adapter stacking/merging.
 */
export interface AdapterStackOptions {
  /** How to combine multiple adapters */
  strategy: 'sum' | 'weighted_sum' | 'sequential';
  /** Normalize weights to sum to 1.0 */
  normalizeWeights?: boolean;
}

/**
 * Adapter manager events.
 */
export interface AdapterManagerEvents {
  onAdapterLoaded?: (id: string, adapter: LoRAAdapter) => void;
  onAdapterEnabled?: (id: string) => void;
  onAdapterDisabled?: (id: string) => void;
  onAdapterUnloaded?: (id: string) => void;
  onActiveAdaptersChanged?: (activeIds: string[]) => void;
}

// ============================================================================
// Adapter Manager Class
// ============================================================================

/**
 * Manages runtime loading, enabling, and disabling of LoRA adapters.
 *
 * Usage:
 * ```typescript
 * const manager = new AdapterManager();
 *
 * // Load an adapter
 * await manager.loadAdapter('coding-assistant', 'https://example.com/adapters/coding.json');
 *
 * // Enable it for inference
 * manager.enableAdapter('coding-assistant');
 *
 * // Get the active adapter for pipeline
 * const adapter = manager.getActiveAdapter();
 * pipeline.setLoRAAdapter(adapter);
 *
 * // Switch to a different adapter at runtime
 * manager.disableAdapter('coding-assistant');
 * manager.enableAdapter('creative-writing');
 * ```
 */
export class AdapterManager {
  /** Map of adapter ID to state */
  private adapters: Map<string, AdapterState> = new Map();

  /** Currently active adapter IDs (in order) */
  private activeAdapterIds: string[] = [];

  /** Event callbacks */
  private events: AdapterManagerEvents = {};

  /** Default loading options */
  private defaultLoadOptions: LoRALoadOptions = {};

  /** Stack options for combining multiple adapters */
  private stackOptions: AdapterStackOptions = {
    strategy: 'sum',
    normalizeWeights: false,
  };

  // ==========================================================================
  // Configuration
  // ==========================================================================

  /**
   * Sets default loading options for all adapter loads.
   */
  setDefaultLoadOptions(options: LoRALoadOptions): void {
    this.defaultLoadOptions = { ...options };
  }

  /**
   * Sets event callbacks.
   */
  setEvents(events: AdapterManagerEvents): void {
    this.events = { ...this.events, ...events };
  }

  /**
   * Sets adapter stacking options.
   */
  setStackOptions(options: Partial<AdapterStackOptions>): void {
    this.stackOptions = { ...this.stackOptions, ...options };
  }

  // ==========================================================================
  // Loading
  // ==========================================================================

  /**
   * Loads an adapter from a path (URL or OPFS).
   *
   * @param id - Unique identifier for this adapter
   * @param path - Path to adapter manifest
   * @param options - Loading options
   * @returns Loaded adapter state
   */
  async loadAdapter(
    id: string,
    path: string,
    options: LoRALoadOptions = {}
  ): Promise<AdapterState> {
    // Check if already loaded
    if (this.adapters.has(id)) {
      throw new Error(`Adapter '${id}' is already loaded. Unload it first.`);
    }

    // Merge options with defaults
    const mergedOptions = { ...this.defaultLoadOptions, ...options };

    // Load the adapter
    const result = await loadLoRAWeights(path, mergedOptions);

    // Create state
    const state: AdapterState = {
      id,
      adapter: result.adapter,
      manifest: result.manifest,
      enabled: false,
      weight: 1.0,
      loadedAt: Date.now(),
      lastToggled: 0,
    };

    // Store it
    this.adapters.set(id, state);

    // Fire event
    this.events.onAdapterLoaded?.(id, result.adapter);

    return state;
  }

  /**
   * Loads an adapter from an already-parsed manifest and adapter.
   */
  registerAdapter(
    id: string,
    adapter: LoRAAdapter,
    manifest: AdapterManifest
  ): AdapterState {
    if (this.adapters.has(id)) {
      throw new Error(`Adapter '${id}' is already loaded. Unload it first.`);
    }

    const state: AdapterState = {
      id,
      adapter,
      manifest,
      enabled: false,
      weight: 1.0,
      loadedAt: Date.now(),
      lastToggled: 0,
    };

    this.adapters.set(id, state);
    this.events.onAdapterLoaded?.(id, adapter);

    return state;
  }

  // ==========================================================================
  // Enable/Disable API
  // ==========================================================================

  /**
   * Enables an adapter for inference.
   *
   * @param id - Adapter ID to enable
   * @param options - Enable options
   */
  enableAdapter(id: string, options: EnableAdapterOptions = {}): void {
    const state = this.adapters.get(id);
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
    if (!this.activeAdapterIds.includes(id)) {
      this.activeAdapterIds.push(id);
    }

    // Fire events
    this.events.onAdapterEnabled?.(id);
    this.events.onActiveAdaptersChanged?.([...this.activeAdapterIds]);
  }

  /**
   * Disables an adapter.
   *
   * @param id - Adapter ID to disable
   */
  disableAdapter(id: string): void {
    const state = this.adapters.get(id);
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
    const idx = this.activeAdapterIds.indexOf(id);
    if (idx >= 0) {
      this.activeAdapterIds.splice(idx, 1);
    }

    // Fire events
    this.events.onAdapterDisabled?.(id);
    this.events.onActiveAdaptersChanged?.([...this.activeAdapterIds]);
  }

  /**
   * Toggles an adapter's enabled state.
   */
  toggleAdapter(id: string): boolean {
    const state = this.adapters.get(id);
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
  disableAll(): void {
    for (const id of [...this.activeAdapterIds]) {
      this.disableAdapter(id);
    }
  }

  /**
   * Enables only the specified adapter, disabling all others.
   */
  enableOnly(id: string, options?: EnableAdapterOptions): void {
    this.disableAll();
    this.enableAdapter(id, options);
  }

  /**
   * Sets the weight for an adapter.
   */
  setAdapterWeight(id: string, weight: number): void {
    const state = this.adapters.get(id);
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
  unloadAdapter(id: string): void {
    const state = this.adapters.get(id);
    if (!state) {
      return;
    }

    // Disable if active
    if (state.enabled) {
      this.disableAdapter(id);
    }

    // Remove from map
    this.adapters.delete(id);

    // Fire event
    this.events.onAdapterUnloaded?.(id);
  }

  /**
   * Unloads all adapters.
   */
  unloadAll(): void {
    for (const id of [...this.adapters.keys()]) {
      this.unloadAdapter(id);
    }
  }

  // ==========================================================================
  // Query Methods
  // ==========================================================================

  /**
   * Gets the currently active adapter for use with pipeline.
   *
   * If multiple adapters are active, merges them according to stack options.
   * Returns null if no adapters are active.
   */
  getActiveAdapter(): LoRAAdapter | null {
    if (this.activeAdapterIds.length === 0) {
      return null;
    }

    if (this.activeAdapterIds.length === 1) {
      const state = this.adapters.get(this.activeAdapterIds[0]);
      if (!state) return null;

      // Apply weight if not 1.0
      if (state.weight !== 1.0) {
        return this.applyWeight(state.adapter, state.weight);
      }
      return state.adapter;
    }

    // Multiple adapters - merge them
    return this.mergeActiveAdapters();
  }

  /**
   * Gets all active adapter IDs.
   */
  getActiveAdapterIds(): string[] {
    return [...this.activeAdapterIds];
  }

  /**
   * Gets state of a specific adapter.
   */
  getAdapterState(id: string): AdapterState | undefined {
    return this.adapters.get(id);
  }

  /**
   * Gets all loaded adapter states.
   */
  getAllAdapters(): AdapterState[] {
    return [...this.adapters.values()];
  }

  /**
   * Gets all enabled adapter states.
   */
  getEnabledAdapters(): AdapterState[] {
    return this.getAllAdapters().filter(s => s.enabled);
  }

  /**
   * Checks if an adapter is loaded.
   */
  isLoaded(id: string): boolean {
    return this.adapters.has(id);
  }

  /**
   * Checks if an adapter is enabled.
   */
  isEnabled(id: string): boolean {
    return this.adapters.get(id)?.enabled ?? false;
  }

  /**
   * Gets count of loaded adapters.
   */
  get loadedCount(): number {
    return this.adapters.size;
  }

  /**
   * Gets count of enabled adapters.
   */
  get enabledCount(): number {
    return this.activeAdapterIds.length;
  }

  // ==========================================================================
  // Merging Logic
  // ==========================================================================

  /**
   * Merges multiple active adapters into a single virtual adapter.
   */
  private mergeActiveAdapters(): LoRAAdapter | null {
    const activeStates = this.activeAdapterIds
      .map(id => this.adapters.get(id))
      .filter((s): s is AdapterState => s !== undefined);

    if (activeStates.length === 0) return null;
    if (activeStates.length === 1) {
      return this.applyWeight(activeStates[0].adapter, activeStates[0].weight);
    }

    // Compute weights
    let weights = activeStates.map(s => s.weight);
    if (this.stackOptions.normalizeWeights) {
      const sum = weights.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        weights = weights.map(w => w / sum);
      }
    }

    // Merge based on strategy
    switch (this.stackOptions.strategy) {
      case 'sum':
      case 'weighted_sum':
        return this.mergeByWeightedSum(activeStates, weights);
      case 'sequential':
        // For sequential, just use the last adapter
        return this.applyWeight(
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
  private mergeByWeightedSum(states: AdapterState[], weights: number[]): LoRAAdapter {
    // Use first adapter as template
    const first = states[0].adapter;

    const merged: LoRAAdapter = {
      name: `merged(${states.map(s => s.id).join('+')})`,
      rank: first.rank,
      alpha: first.alpha,
      targetModules: first.targetModules,
      layers: new Map(),
    };

    // Collect all layer indices
    const allLayers = new Set<number>();
    for (const state of states) {
      for (const layerIdx of state.adapter.layers.keys()) {
        allLayers.add(layerIdx);
      }
    }

    // Merge each layer
    for (const layerIdx of allLayers) {
      const mergedLayer: Record<string, {
        a: Float32Array;
        b: Float32Array;
        rank: number;
        alpha: number;
        scale: number;
      }> = {};

      // Get all modules in this layer across all adapters
      const allModules = new Set<string>();
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
        let mergedA: Float32Array | null = null;
        let mergedB: Float32Array | null = null;
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
            console.warn('[AdapterManager] Cannot merge GPUBuffer weights on CPU, skipping');
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
            mergedB![j] += mod.b[j] * weight;
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
   * Note: This modifies scale/alpha but preserves weight buffers as-is.
   * The scale adjustment will be applied during forward pass.
   */
  private applyWeight(adapter: LoRAAdapter, weight: number): LoRAAdapter {
    if (weight === 1.0) return adapter;

    const weighted: LoRAAdapter = {
      ...adapter,
      alpha: adapter.alpha * weight,
      layers: new Map(),
    };

    for (const [layerIdx, layer] of adapter.layers) {
      const weightedLayer: Record<string, LoRAModuleWeights> = {};

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
let defaultManager: AdapterManager | null = null;

/**
 * Gets the default adapter manager instance.
 */
export function getAdapterManager(): AdapterManager {
  if (!defaultManager) {
    defaultManager = new AdapterManager();
  }
  return defaultManager;
}

/**
 * Resets the default adapter manager (useful for testing).
 */
export function resetAdapterManager(): void {
  if (defaultManager) {
    defaultManager.unloadAll();
  }
  defaultManager = null;
}

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

import type { LoRAAdapter, LoRAModuleWeights } from '../inference/pipelines/text/lora-types.js';
import type { AdapterManifest } from './adapter-manifest.js';
import type { LoRALoadOptions } from './lora-loader.js';

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

/**
 * Manages runtime loading, enabling, and disabling of LoRA adapters.
 */
export declare class AdapterManager {
  /**
   * Sets default loading options for all adapter loads.
   */
  setDefaultLoadOptions(options: LoRALoadOptions): void;

  /**
   * Sets event callbacks.
   */
  setEvents(events: AdapterManagerEvents): void;

  /**
   * Sets adapter stacking options.
   */
  setStackOptions(options: Partial<AdapterStackOptions>): void;

  /**
   * Loads an adapter from a path (URL or OPFS).
   */
  loadAdapter(
    id: string,
    path: string,
    options?: LoRALoadOptions
  ): Promise<AdapterState>;

  /**
   * Loads an adapter from an already-parsed manifest and adapter.
   */
  registerAdapter(
    id: string,
    adapter: LoRAAdapter,
    manifest: AdapterManifest
  ): AdapterState;

  /**
   * Enables an adapter for inference.
   */
  enableAdapter(id: string, options?: EnableAdapterOptions): void;

  /**
   * Disables an adapter.
   */
  disableAdapter(id: string): void;

  /**
   * Toggles an adapter's enabled state.
   */
  toggleAdapter(id: string): boolean;

  /**
   * Disables all adapters.
   */
  disableAll(): void;

  /**
   * Enables only the specified adapter, disabling all others.
   */
  enableOnly(id: string, options?: EnableAdapterOptions): void;

  /**
   * Sets the weight for an adapter.
   */
  setAdapterWeight(id: string, weight: number): void;

  /**
   * Unloads an adapter, freeing its memory.
   */
  unloadAdapter(id: string): void;

  /**
   * Unloads all adapters.
   */
  unloadAll(): void;

  /**
   * Gets the currently active adapter for use with pipeline.
   */
  getActiveAdapter(): LoRAAdapter | null;

  /**
   * Gets all active adapter IDs.
   */
  getActiveAdapterIds(): string[];

  /**
   * Gets state of a specific adapter.
   */
  getAdapterState(id: string): AdapterState | undefined;

  /**
   * Gets all loaded adapter states.
   */
  getAllAdapters(): AdapterState[];

  /**
   * Gets all enabled adapter states.
   */
  getEnabledAdapters(): AdapterState[];

  /**
   * Checks if an adapter is loaded.
   */
  isLoaded(id: string): boolean;

  /**
   * Checks if an adapter is enabled.
   */
  isEnabled(id: string): boolean;

  /**
   * Gets count of loaded adapters.
   */
  get loadedCount(): number;

  /**
   * Gets count of enabled adapters.
   */
  get enabledCount(): number;
}

/**
 * Gets the default adapter manager instance.
 */
export declare function getAdapterManager(): AdapterManager;

/**
 * Resets the default adapter manager (useful for testing).
 */
export declare function resetAdapterManager(): void;

/**
 * Config Loader
 *
 * Loads, validates, and converts config to RuntimeConfigSchema.
 * Main entry point for CLI config handling.
 *
 * @module cli/config/config-loader
 */

import type { RuntimeConfigSchema } from '../../src/config/schema/index.js';
import { ConfigComposer, type ComposedConfig } from './config-composer.js';

export interface LoadedConfig {
  /** Validated runtime config */
  runtime: RuntimeConfigSchema;
  /** Source chain (for debugging) */
  chain: string[];
  /** Raw composed config (before validation) */
  raw: Record<string, unknown>;
}

export interface LoadOptions {
  /** Merge with defaults (default: true) */
  mergeDefaults?: boolean;
  /** Validate config (default: true) */
  validate?: boolean;
}

export declare class ConfigLoader {
  constructor(composer?: ConfigComposer);
  /**
   * Load and validate a config.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @param options - Load options
   * @returns Validated runtime config
   */
  load(ref: string, options?: LoadOptions): Promise<LoadedConfig>;
}

/**
 * Load a config by reference.
 */
export declare function loadConfig(
  ref: string,
  options?: LoadOptions
): Promise<LoadedConfig>;

/**
 * List available presets.
 */
export declare function listPresets(): Promise<{ name: string; source: string; path: string }[]>;

/**
 * Dump a loaded config for debugging.
 */
export declare function dumpConfig(loaded: LoadedConfig): string;

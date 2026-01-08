/**
 * Config Composer
 *
 * Handles config inheritance via "extends" field.
 * Deep-merges parent configs with cycle detection.
 *
 * @module cli/config/config-composer
 */

import { ConfigResolver } from './config-resolver.js';

export interface RawConfig {
  extends?: string;
  [key: string]: unknown;
}

export interface ComposedConfig {
  /** Merged config object (extends resolved) */
  config: Record<string, unknown>;
  /** Chain of configs that were merged (root first) */
  chain: string[];
}

export declare class ConfigComposer {
  constructor(resolver?: ConfigResolver, maxDepth?: number);
  /**
   * Compose a config by resolving its extends chain.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @returns Composed config with all extends merged
   */
  compose(ref: string): Promise<ComposedConfig>;
}

export declare function composeConfig(ref: string): Promise<ComposedConfig>;

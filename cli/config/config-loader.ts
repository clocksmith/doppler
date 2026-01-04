/**
 * Config Loader
 *
 * Loads, validates, and converts config to RuntimeConfigSchema.
 * Main entry point for CLI config handling.
 *
 * @module cli/config/config-loader
 */

import type { RuntimeConfigSchema } from '../../src/config/schema/index.js';
import { DEFAULT_RUNTIME_CONFIG, LOG_LEVELS } from '../../src/config/schema/index.js';
import { ConfigComposer, type ComposedConfig } from './config-composer.js';
import { ConfigResolver, listPresets as listPresetsFromResolver } from './config-resolver.js';

// =============================================================================
// Types
// =============================================================================

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

// =============================================================================
// Config Loader
// =============================================================================

export class ConfigLoader {
  private composer: ConfigComposer;

  constructor(composer?: ConfigComposer) {
    this.composer = composer ?? new ConfigComposer();
  }

  /**
   * Load and validate a config.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @param options - Load options
   * @returns Validated runtime config
   */
  async load(ref: string, options: LoadOptions = {}): Promise<LoadedConfig> {
    const { mergeDefaults = true, validate = true } = options;

    // Compose config (resolve extends chain)
    const composed = await this.composer.compose(ref);

    // Extract runtime config
    const rawRuntime = (composed.config.runtime ?? {}) as Record<string, unknown>;

    // Merge with defaults if requested
    const runtime = mergeDefaults
      ? this.mergeWithDefaults(rawRuntime)
      : (rawRuntime as unknown as RuntimeConfigSchema);

    // Validate if requested
    if (validate) {
      this.validateRuntime(runtime);
    }

    return {
      runtime,
      chain: composed.chain,
      raw: composed.config,
    };
  }

  /**
   * Deep merge config with defaults.
   *
   * Note: This merge happens in CLI before passing to browser. The browser-side
   * setRuntimeConfig() does another merge with defaults. This double-merge is
   * intentional: CLI merge ensures valid structure for validation/logging,
   * browser merge provides safety net if config is modified or partially applied.
   * Both merges are idempotent (cumulative deep merge), so double-applying is safe.
   */
  private mergeWithDefaults(config: Record<string, unknown>): RuntimeConfigSchema {
    return this.deepMerge(
      DEFAULT_RUNTIME_CONFIG as unknown as Record<string, unknown>,
      config
    ) as unknown as RuntimeConfigSchema;
  }

  /**
   * Validate runtime config structure.
   * Throws on invalid config.
   */
  private validateRuntime(config: RuntimeConfigSchema): void {
    // Basic structure validation
    if (typeof config !== 'object' || config === null) {
      throw new Error('Config must be an object');
    }

    // Validate known sections
    const sections = [
      'inference',
      'kvcache',
      'debug',
      'moe',
      'storage',
      'distribution',
      'bridge',
      'memory',
      'bufferPool',
      'tuner',
      'loading',
      'gpuCache',
    ];

    for (const section of sections) {
      const value = (config as unknown as Record<string, unknown>)[section];
      if (value !== undefined && (typeof value !== 'object' || value === null)) {
        throw new Error(`Config section "${section}" must be an object`);
      }
    }

    // Validate specific values
    this.validateDebugConfig(config.debug);
    this.validateSamplingConfig(config.inference?.sampling);
  }

  /**
   * Validate debug config section.
   */
  private validateDebugConfig(debug: RuntimeConfigSchema['debug']): void {
    if (!debug) return;

    // Validate log level (uses LOG_LEVELS from schema to stay in sync)
    if (debug.logLevel?.defaultLogLevel) {
      const level = debug.logLevel.defaultLogLevel;
      if (!LOG_LEVELS.includes(level as typeof LOG_LEVELS[number])) {
        throw new Error(
          `Invalid log level "${level}". Valid: ${LOG_LEVELS.join(', ')}`
        );
      }
    }

    // Validate trace categories
    const validCategories = [
      'loader', 'kernels', 'logits', 'embed', 'attn',
      'ffn', 'kv', 'sample', 'buffers', 'perf', 'all'
    ];
    if (debug.trace?.categories) {
      for (const cat of debug.trace.categories) {
        if (!validCategories.includes(cat)) {
          throw new Error(
            `Invalid trace category "${cat}". Valid: ${validCategories.join(', ')}`
          );
        }
      }
    }

    // Validate pipeline debug categories
    const validPipelineCategories = [
      'embed', 'layer', 'attn', 'ffn', 'kv', 'logits',
      'sample', 'io', 'perf', 'kernel', 'all'
    ];
    if (debug.pipeline?.categories) {
      for (const cat of debug.pipeline.categories) {
        if (!validPipelineCategories.includes(cat)) {
          throw new Error(
            `Invalid pipeline debug category "${cat}". Valid: ${validPipelineCategories.join(', ')}`
          );
        }
      }
    }
    if (debug.pipeline?.maxDecodeSteps !== undefined && debug.pipeline.maxDecodeSteps < 0) {
      throw new Error('debug.pipeline.maxDecodeSteps must be >= 0');
    }
    if (debug.pipeline?.maxAbsThreshold !== undefined && debug.pipeline.maxAbsThreshold <= 0) {
      throw new Error('debug.pipeline.maxAbsThreshold must be > 0');
    }

    // Validate probes
    const validProbeStages = [
      'embed_out', 'attn_out', 'post_attn', 'ffn_in', 'ffn_out',
      'layer_out', 'pre_final_norm', 'final_norm', 'logits', 'logits_final'
    ];
    if (debug.probes !== undefined) {
      if (!Array.isArray(debug.probes)) {
        throw new Error('debug.probes must be an array');
      }
      for (const probe of debug.probes) {
        if (!probe || typeof probe !== 'object') {
          throw new Error('debug.probes entries must be objects');
        }
        const stage = (probe as { stage?: string }).stage;
        if (!stage || !validProbeStages.includes(stage)) {
          throw new Error(
            `Invalid probe stage "${stage}". Valid: ${validProbeStages.join(', ')}`
          );
        }
        const dims = (probe as { dims?: unknown }).dims;
        if (!Array.isArray(dims) || dims.some((d) => typeof d !== 'number')) {
          throw new Error('debug.probes.dims must be an array of numbers');
        }
        const layers = (probe as { layers?: unknown }).layers;
        if (layers !== null && layers !== undefined) {
          if (!Array.isArray(layers) || layers.some((d) => typeof d !== 'number')) {
            throw new Error('debug.probes.layers must be null or an array of numbers');
          }
        }
        const tokens = (probe as { tokens?: unknown }).tokens;
        if (tokens !== null && tokens !== undefined) {
          if (!Array.isArray(tokens) || tokens.some((d) => typeof d !== 'number')) {
            throw new Error('debug.probes.tokens must be null or an array of numbers');
          }
        }
        const category = (probe as { category?: string }).category;
        if (category && !validCategories.includes(category)) {
          throw new Error(
            `Invalid probe category "${category}". Valid: ${validCategories.join(', ')}`
          );
        }
      }
    }
  }

  /**
   * Validate sampling config section.
   */
  private validateSamplingConfig(sampling: RuntimeConfigSchema['inference']['sampling'] | undefined): void {
    if (!sampling) return;

    if (sampling.temperature !== undefined) {
      if (typeof sampling.temperature !== 'number' || sampling.temperature < 0) {
        throw new Error('sampling.temperature must be a non-negative number');
      }
    }

    if (sampling.topK !== undefined) {
      if (typeof sampling.topK !== 'number' || sampling.topK < 1) {
        throw new Error('sampling.topK must be a positive integer');
      }
    }

    if (sampling.topP !== undefined) {
      if (typeof sampling.topP !== 'number' || sampling.topP < 0 || sampling.topP > 1) {
        throw new Error('sampling.topP must be between 0 and 1');
      }
    }
  }

  /**
   * Deep merge two objects.
   */
  private deepMerge(
    parent: Record<string, unknown>,
    child: Record<string, unknown>
  ): Record<string, unknown> {
    const result = { ...parent };

    for (const key of Object.keys(child)) {
      const childVal = child[key];
      const parentVal = parent[key];

      if (childVal === undefined) {
        continue;
      }

      if (
        childVal !== null &&
        typeof childVal === 'object' &&
        !Array.isArray(childVal) &&
        parentVal !== null &&
        typeof parentVal === 'object' &&
        !Array.isArray(parentVal)
      ) {
        result[key] = this.deepMerge(
          parentVal as Record<string, unknown>,
          childVal as Record<string, unknown>
        );
      } else {
        result[key] = childVal;
      }
    }

    return result;
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

const defaultLoader = new ConfigLoader();

/**
 * Load a config by reference.
 */
export async function loadConfig(
  ref: string,
  options?: LoadOptions
): Promise<LoadedConfig> {
  return defaultLoader.load(ref, options);
}

/**
 * List available presets.
 */
export async function listPresets(): Promise<{ name: string; source: string; path: string }[]> {
  return listPresetsFromResolver();
}

/**
 * Dump a loaded config for debugging.
 */
export function dumpConfig(loaded: LoadedConfig): string {
  return JSON.stringify(
    {
      chain: loaded.chain,
      runtime: loaded.runtime,
    },
    null,
    2
  );
}

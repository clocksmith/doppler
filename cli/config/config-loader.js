/**
 * Config Loader
 *
 * Loads, validates, and converts config to RuntimeConfigSchema.
 * Main entry point for CLI config handling.
 *
 * @module cli/config/config-loader
 */

import { DEFAULT_RUNTIME_CONFIG, LOG_LEVELS } from '../../src/config/schema/index.js';
import { ConfigComposer } from './config-composer.js';
import { listPresets as listPresetsFromResolver } from './config-resolver.js';

// =============================================================================
// Config Loader
// =============================================================================

export class ConfigLoader {
  /** @type {ConfigComposer} */
  #composer;

  /**
   * @param {ConfigComposer} [composer]
   */
  constructor(composer) {
    this.#composer = composer ?? new ConfigComposer();
  }

  /**
   * Load and validate a config.
   *
   * @param {string} ref - Config reference (name, path, URL, or inline JSON)
   * @param {import('./config-loader.js').LoadOptions} [options] - Load options
   * @returns {Promise<import('./config-loader.js').LoadedConfig>} Validated runtime config
   */
  async load(ref, options = {}) {
    const { mergeDefaults = true, validate = true } = options;

    // Compose config (resolve extends chain)
    const composed = await this.#composer.compose(ref);

    // Extract runtime config
    /** @type {Record<string, unknown>} */
    const rawRuntime = (composed.config.runtime ?? {});

    // Merge with defaults if requested
    const runtime = mergeDefaults
      ? this.#mergeWithDefaults(rawRuntime)
      : /** @type {import('../../src/config/schema/index.js').RuntimeConfigSchema} */ (rawRuntime);

    // Validate if requested
    if (validate) {
      this.#validateRuntime(runtime);
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
   *
   * @param {Record<string, unknown>} config
   * @returns {import('../../src/config/schema/index.js').RuntimeConfigSchema}
   */
  #mergeWithDefaults(config) {
    return /** @type {import('../../src/config/schema/index.js').RuntimeConfigSchema} */ (
      this.#deepMerge(
        /** @type {Record<string, unknown>} */ (DEFAULT_RUNTIME_CONFIG),
        config
      )
    );
  }

  /**
   * Validate runtime config structure.
   * Throws on invalid config.
   *
   * @param {import('../../src/config/schema/index.js').RuntimeConfigSchema} config
   */
  #validateRuntime(config) {
    // Basic structure validation
    if (typeof config !== 'object' || config === null) {
      throw new Error('Config must be an object');
    }

    // Validate known sections
    const sections = [
      'shared',
      'loading',
      'inference',
    ];

    for (const section of sections) {
      const value = /** @type {Record<string, unknown>} */ (config)[section];
      if (value !== undefined && (typeof value !== 'object' || value === null)) {
        throw new Error(`Config section "${section}" must be an object`);
      }
    }

    // Reject deprecated keys
    if (/** @type {Record<string, unknown>} */ (config).debug !== undefined) {
      throw new Error('runtime.debug is removed; use runtime.shared.debug');
    }
    if (config.loading?.debug !== undefined) {
      throw new Error('runtime.loading.debug is removed; use runtime.shared.debug');
    }
    if (config.inference?.debug !== undefined) {
      throw new Error('runtime.inference.debug is removed; use runtime.shared.debug');
    }

    // Validate specific values
    this.#validateDebugConfig(config.shared?.debug);
    this.#validateSamplingConfig(config.inference?.sampling);
  }

  /**
   * Validate debug config section.
   *
   * @param {import('../../src/config/schema/index.js').RuntimeConfigSchema['shared']['debug']} debug
   */
  #validateDebugConfig(debug) {
    if (!debug) return;

    // Validate log level (uses LOG_LEVELS from schema to stay in sync)
    if (debug.logLevel?.defaultLogLevel) {
      const level = debug.logLevel.defaultLogLevel;
      if (!LOG_LEVELS.includes(/** @type {typeof LOG_LEVELS[number]} */ (level))) {
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
        const stage = /** @type {{ stage?: string }} */ (probe).stage;
        if (!stage || !validProbeStages.includes(stage)) {
          throw new Error(
            `Invalid probe stage "${stage}". Valid: ${validProbeStages.join(', ')}`
          );
        }
        const dims = /** @type {{ dims?: unknown }} */ (probe).dims;
        if (!Array.isArray(dims) || dims.some((d) => typeof d !== 'number')) {
          throw new Error('debug.probes.dims must be an array of numbers');
        }
        const layers = /** @type {{ layers?: unknown }} */ (probe).layers;
        if (layers !== null && layers !== undefined) {
          if (!Array.isArray(layers) || layers.some((d) => typeof d !== 'number')) {
            throw new Error('debug.probes.layers must be null or an array of numbers');
          }
        }
        const tokens = /** @type {{ tokens?: unknown }} */ (probe).tokens;
        if (tokens !== null && tokens !== undefined) {
          if (!Array.isArray(tokens) || tokens.some((d) => typeof d !== 'number')) {
            throw new Error('debug.probes.tokens must be null or an array of numbers');
          }
        }
        const category = /** @type {{ category?: string }} */ (probe).category;
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
   *
   * @param {import('../../src/config/schema/index.js').RuntimeConfigSchema['inference']['sampling'] | undefined} sampling
   */
  #validateSamplingConfig(sampling) {
    if (!sampling) return;

    const deprecatedMaxTokens = /** @type {{ maxTokens?: unknown }} */ (sampling).maxTokens;
    if (deprecatedMaxTokens !== undefined) {
      throw new Error('sampling.maxTokens is removed; use inference.batching.maxTokens');
    }

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
   *
   * @param {Record<string, unknown>} parent
   * @param {Record<string, unknown>} child
   * @returns {Record<string, unknown>}
   */
  #deepMerge(parent, child) {
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
        result[key] = this.#deepMerge(
          /** @type {Record<string, unknown>} */ (parentVal),
          /** @type {Record<string, unknown>} */ (childVal)
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
 *
 * @param {string} ref
 * @param {import('./config-loader.js').LoadOptions} [options]
 * @returns {Promise<import('./config-loader.js').LoadedConfig>}
 */
export async function loadConfig(ref, options) {
  return defaultLoader.load(ref, options);
}

/**
 * List available presets.
 *
 * @returns {Promise<{ name: string; source: string; path: string }[]>}
 */
export async function listPresets() {
  return listPresetsFromResolver();
}

/**
 * Dump a loaded config for debugging.
 *
 * @param {import('./config-loader.js').LoadedConfig} loaded
 * @returns {string}
 */
export function dumpConfig(loaded) {
  return JSON.stringify(
    {
      chain: loaded.chain,
      runtime: loaded.runtime,
    },
    null,
    2
  );
}

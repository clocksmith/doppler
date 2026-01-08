/**
 * Config Composer
 *
 * Handles config inheritance via "extends" field.
 * Deep-merges parent configs with cycle detection.
 *
 * @module cli/config/config-composer
 */

import { ConfigResolver } from './config-resolver.js';

// =============================================================================
// Config Composer
// =============================================================================

export class ConfigComposer {
  /** @type {ConfigResolver} */
  #resolver;
  /** @type {number} */
  #maxDepth;

  /**
   * @param {ConfigResolver} [resolver]
   * @param {number} [maxDepth]
   */
  constructor(resolver, maxDepth = 10) {
    this.#resolver = resolver ?? new ConfigResolver();
    this.#maxDepth = maxDepth;
  }

  /**
   * Compose a config by resolving its extends chain.
   *
   * @param {string} ref - Config reference (name, path, URL, or inline JSON)
   * @returns {Promise<import('./config-composer.js').ComposedConfig>} Composed config with all extends merged
   */
  async compose(ref) {
    /** @type {Set<string>} */
    const visited = new Set();
    /** @type {string[]} */
    const stack = [];

    return this.#composeRecursive(ref, visited, stack, 0);
  }

  /**
   * @param {string} ref
   * @param {Set<string>} visited
   * @param {string[]} stack
   * @param {number} depth
   * @returns {Promise<import('./config-composer.js').ComposedConfig>}
   */
  async #composeRecursive(ref, visited, stack, depth) {
    // Cycle detection
    const normalizedRef = this.#normalizeRef(ref);
    if (visited.has(normalizedRef)) {
      throw new Error(
        `Circular extends detected: ${[...stack, normalizedRef].join(' -> ')}`
      );
    }

    // Depth limit
    if (depth > this.#maxDepth) {
      throw new Error(
        `Extends chain too deep (max ${this.#maxDepth}): ${[...stack, normalizedRef].join(' -> ')}`
      );
    }

    visited.add(normalizedRef);
    stack.push(normalizedRef);

    // Resolve and parse config
    const resolved = await this.#resolver.resolve(ref);
    const raw = this.#parseConfig(resolved);

    // If no extends, return as-is
    if (!raw.extends) {
      const { extends: _, ...config } = raw;
      stack.pop();
      return { config, chain: [normalizedRef] };
    }

    // Recursively resolve parent
    const parent = await this.#composeRecursive(
      raw.extends,
      visited,
      stack,
      depth + 1
    );

    // Deep merge: parent first, child overrides
    const { extends: _, ...childConfig } = raw;
    const merged = this.#deepMerge(parent.config, childConfig);

    stack.pop();
    return { config: merged, chain: [...parent.chain, normalizedRef] };
  }

  /**
   * Parse config content to object.
   *
   * @param {import('./config-resolver.js').ResolvedConfig} resolved
   * @returns {import('./config-composer.js').RawConfig}
   */
  #parseConfig(resolved) {
    try {
      return JSON.parse(resolved.content);
    } catch (err) {
      const source = resolved.name ?? resolved.path ?? 'inline';
      throw new Error(`Invalid JSON in config "${source}": ${/** @type {Error} */ (err).message}`);
    }
  }

  /**
   * Normalize a ref for cycle detection.
   * Note: Preserves case for file paths (case-sensitive filesystems).
   *
   * @param {string} ref
   * @returns {string}
   */
  #normalizeRef(ref) {
    // Inline JSON: use hash-like identifier
    if (ref.trim().startsWith('{')) {
      return `inline:${ref.length}:${ref.slice(0, 50)}`;
    }
    // Preserve case - on case-sensitive FS, "Foo.json" and "foo.json" are different
    return ref.trim();
  }

  /**
   * Deep merge two objects. Child values override parent.
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
        // Recursively merge nested objects
        result[key] = this.#deepMerge(
          /** @type {Record<string, unknown>} */ (parentVal),
          /** @type {Record<string, unknown>} */ (childVal)
        );
      } else {
        // Override with child value
        result[key] = childVal;
      }
    }

    return result;
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

const defaultComposer = new ConfigComposer();

/**
 * @param {string} ref
 * @returns {Promise<import('./config-composer.js').ComposedConfig>}
 */
export async function composeConfig(ref) {
  return defaultComposer.compose(ref);
}

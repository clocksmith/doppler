/**
 * Config Composer
 *
 * Handles config inheritance via "extends" field.
 * Deep-merges parent configs with cycle detection.
 *
 * @module cli/config/config-composer
 */

import { ConfigResolver, type ResolvedConfig } from './config-resolver.js';

// =============================================================================
// Types
// =============================================================================

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

// =============================================================================
// Config Composer
// =============================================================================

export class ConfigComposer {
  private resolver: ConfigResolver;
  private maxDepth: number;

  constructor(resolver?: ConfigResolver, maxDepth = 10) {
    this.resolver = resolver ?? new ConfigResolver();
    this.maxDepth = maxDepth;
  }

  /**
   * Compose a config by resolving its extends chain.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @returns Composed config with all extends merged
   */
  async compose(ref: string): Promise<ComposedConfig> {
    const visited = new Set<string>();
    const stack: string[] = [];

    return this.composeRecursive(ref, visited, stack, 0);
  }

  private async composeRecursive(
    ref: string,
    visited: Set<string>,
    stack: string[],
    depth: number
  ): Promise<ComposedConfig> {
    // Cycle detection
    const normalizedRef = this.normalizeRef(ref);
    if (visited.has(normalizedRef)) {
      throw new Error(
        `Circular extends detected: ${[...stack, normalizedRef].join(' -> ')}`
      );
    }

    // Depth limit
    if (depth > this.maxDepth) {
      throw new Error(
        `Extends chain too deep (max ${this.maxDepth}): ${stack.join(' -> ')}`
      );
    }

    visited.add(normalizedRef);
    stack.push(normalizedRef);

    // Resolve and parse config
    const resolved = await this.resolver.resolve(ref);
    const raw = this.parseConfig(resolved);

    // If no extends, return as-is
    if (!raw.extends) {
      const { extends: _, ...config } = raw;
      stack.pop();
      return { config, chain: [normalizedRef] };
    }

    // Recursively resolve parent
    const parent = await this.composeRecursive(
      raw.extends,
      visited,
      stack,
      depth + 1
    );

    // Deep merge: parent first, child overrides
    const { extends: _, ...childConfig } = raw;
    const merged = this.deepMerge(parent.config, childConfig);

    stack.pop();
    return { config: merged, chain: [...parent.chain, normalizedRef] };
  }

  /**
   * Parse config content to object.
   */
  private parseConfig(resolved: ResolvedConfig): RawConfig {
    try {
      return JSON.parse(resolved.content);
    } catch (err) {
      const source = resolved.name ?? resolved.path ?? 'inline';
      throw new Error(`Invalid JSON in config "${source}": ${(err as Error).message}`);
    }
  }

  /**
   * Normalize a ref for cycle detection.
   */
  private normalizeRef(ref: string): string {
    // Inline JSON: use hash-like identifier
    if (ref.trim().startsWith('{')) {
      return `inline:${ref.length}:${ref.slice(0, 50)}`;
    }
    return ref.toLowerCase();
  }

  /**
   * Deep merge two objects. Child values override parent.
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
        // Recursively merge nested objects
        result[key] = this.deepMerge(
          parentVal as Record<string, unknown>,
          childVal as Record<string, unknown>
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

export async function composeConfig(ref: string): Promise<ComposedConfig> {
  return defaultComposer.compose(ref);
}

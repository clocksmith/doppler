/**
 * Config Resolver
 *
 * Resolves config references to their sources:
 * - Named presets (built-in, user, project)
 * - File paths (absolute or relative)
 * - URLs (fetched and cached)
 * - Inline JSON
 *
 * @module cli/config/config-resolver
 */

export interface ResolvedConfig {
  source: 'builtin' | 'user' | 'project' | 'file' | 'url' | 'inline';
  path: string | null;
  content: string;
  name: string | null;
}

export interface ResolverOptions {
  /** Project root directory (default: cwd) */
  projectRoot?: string;
  /** User home directory (default: homedir) */
  userHome?: string;
  /** URL cache directory (default: ~/.doppler/cache) */
  cacheDir?: string;
  /** Cache TTL in ms (default: 1 hour) */
  cacheTTL?: number;
}

export declare class ConfigResolver {
  constructor(options?: ResolverOptions);
  /**
   * Resolve a config reference to its content.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @returns Resolved config with source info and content
   */
  resolve(ref: string): Promise<ResolvedConfig>;
  /**
   * List available presets from all sources.
   */
  listPresets(): Promise<{ name: string; source: string; path: string }[]>;
}

export declare function resolveConfig(ref: string): Promise<ResolvedConfig>;

export declare function listPresets(): Promise<{ name: string; source: string; path: string }[]>;

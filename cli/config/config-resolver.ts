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

import { readFile, access, mkdir, writeFile } from 'fs/promises';
import { resolve, join } from 'path';
import { homedir } from 'os';

// =============================================================================
// Types
// =============================================================================

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

// =============================================================================
// Constants
// =============================================================================

const BUILTIN_PRESETS_DIR = resolve(import.meta.dirname, '../../src/config/presets/runtime');
const DEFAULT_CACHE_TTL = 60 * 60 * 1000; // 1 hour

// =============================================================================
// Config Resolver
// =============================================================================

export class ConfigResolver {
  private projectRoot: string;
  private userHome: string;
  private cacheDir: string;
  private cacheTTL: number;

  constructor(options: ResolverOptions = {}) {
    this.projectRoot = options.projectRoot ?? process.cwd();
    this.userHome = options.userHome ?? homedir();
    this.cacheDir = options.cacheDir ?? join(this.userHome, '.doppler', 'cache');
    this.cacheTTL = options.cacheTTL ?? DEFAULT_CACHE_TTL;
  }

  /**
   * Resolve a config reference to its content.
   *
   * @param ref - Config reference (name, path, URL, or inline JSON)
   * @returns Resolved config with source info and content
   */
  async resolve(ref: string): Promise<ResolvedConfig> {
    // 1. Inline JSON (starts with '{')
    if (ref.trim().startsWith('{')) {
      return {
        source: 'inline',
        path: null,
        content: ref,
        name: null,
      };
    }

    // 2. URL (starts with http:// or https://)
    if (ref.startsWith('http://') || ref.startsWith('https://')) {
      return this.resolveUrl(ref);
    }

    // 3. File path (contains / or \ or ends with .json)
    if (ref.includes('/') || ref.includes('\\') || ref.endsWith('.json')) {
      return this.resolveFile(ref);
    }

    // 4. Named preset (search in order: project -> user -> builtin)
    return this.resolvePreset(ref);
  }

  /**
   * Resolve a named preset from preset directories.
   */
  private async resolvePreset(name: string): Promise<ResolvedConfig> {
    const filename = name.endsWith('.json') ? name : `${name}.json`;

    // Search order: project -> user -> builtin
    const searchPaths = [
      { source: 'project' as const, dir: join(this.projectRoot, '.doppler') },
      { source: 'user' as const, dir: join(this.userHome, '.doppler', 'presets') },
      { source: 'builtin' as const, dir: BUILTIN_PRESETS_DIR },
    ];

    for (const { source, dir } of searchPaths) {
      const path = join(dir, filename);
      try {
        await access(path);
        const content = await readFile(path, 'utf-8');

        // Warn if project/user preset shadows a builtin
        if (source !== 'builtin') {
          const builtinPath = join(BUILTIN_PRESETS_DIR, filename);
          try {
            await access(builtinPath);
            console.warn(
              `[Config] ${source} preset "${name}" shadows built-in preset`
            );
          } catch {
            // No builtin with same name, no warning needed
          }
        }

        return { source, path, content, name };
      } catch {
        // Continue to next search path
      }
    }

    throw new Error(
      `Config preset "${name}" not found. Searched:\n` +
      searchPaths.map(p => `  - ${p.dir}/${filename}`).join('\n')
    );
  }

  /**
   * Resolve a file path to config content.
   */
  private async resolveFile(ref: string): Promise<ResolvedConfig> {
    const path = resolve(this.projectRoot, ref);
    try {
      const content = await readFile(path, 'utf-8');
      return { source: 'file', path, content, name: null };
    } catch (err) {
      throw new Error(`Config file not found: ${path}`);
    }
  }

  /**
   * Resolve a URL to config content (with caching).
   */
  private async resolveUrl(url: string): Promise<ResolvedConfig> {
    // Check cache first
    const cacheKey = this.urlToCacheKey(url);
    const cachePath = join(this.cacheDir, cacheKey);
    const metaPath = `${cachePath}.meta`;

    try {
      const meta = JSON.parse(await readFile(metaPath, 'utf-8'));
      const age = Date.now() - meta.timestamp;
      if (age < this.cacheTTL) {
        const content = await readFile(cachePath, 'utf-8');
        return { source: 'url', path: url, content, name: null };
      }
    } catch {
      // Cache miss or expired, fetch fresh
    }

    // Fetch from URL
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch config from ${url}: ${response.status}`);
    }
    const content = await response.text();

    // Cache the result
    try {
      await mkdir(this.cacheDir, { recursive: true });
      await writeFile(cachePath, content);
      await writeFile(metaPath, JSON.stringify({
        url,
        timestamp: Date.now(),
        etag: response.headers.get('etag'),
      }));
    } catch {
      // Cache write failed, continue anyway
    }

    return { source: 'url', path: url, content, name: null };
  }

  /**
   * Convert URL to a safe cache key filename.
   */
  private urlToCacheKey(url: string): string {
    return url
      .replace(/^https?:\/\//, '')
      .replace(/[^a-zA-Z0-9.-]/g, '_')
      .slice(0, 200);
  }

  /**
   * List available presets from all sources.
   */
  async listPresets(): Promise<{ name: string; source: string; path: string }[]> {
    const presets: { name: string; source: string; path: string }[] = [];
    const seen = new Set<string>();

    const searchPaths = [
      { source: 'project', dir: join(this.projectRoot, '.doppler') },
      { source: 'user', dir: join(this.userHome, '.doppler', 'presets') },
      { source: 'builtin', dir: BUILTIN_PRESETS_DIR },
    ];

    for (const { source, dir } of searchPaths) {
      try {
        const { readdir } = await import('fs/promises');
        const files = await readdir(dir);
        for (const file of files) {
          if (file.endsWith('.json')) {
            const name = file.replace('.json', '');
            if (!seen.has(name)) {
              seen.add(name);
              presets.push({ name, source, path: join(dir, file) });
            }
          }
        }
      } catch {
        // Directory doesn't exist, skip
      }
    }

    return presets;
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

const defaultResolver = new ConfigResolver();

export async function resolveConfig(ref: string): Promise<ResolvedConfig> {
  return defaultResolver.resolve(ref);
}

export async function listPresets(): Promise<{ name: string; source: string; path: string }[]> {
  return defaultResolver.listPresets();
}

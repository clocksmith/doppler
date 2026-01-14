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

import { readFile, access, mkdir, writeFile, readdir } from 'fs/promises';
import { resolve, join, relative, basename, sep } from 'path';
import { homedir } from 'os';

// =============================================================================
// Constants
// =============================================================================

const BUILTIN_PRESETS_DIR = resolve(import.meta.dirname, '../../src/config/presets/runtime');
const DEFAULT_CACHE_TTL = 60 * 60 * 1000; // 1 hour
const PRESET_EXTENSION = '.json';
const PATH_SEPARATOR_PATTERN = /[\\/]/g;

// =============================================================================
// Config Resolver
// =============================================================================

export class ConfigResolver {
  /** @type {string} */
  #projectRoot;
  /** @type {string} */
  #userHome;
  /** @type {string} */
  #cacheDir;
  /** @type {number} */
  #cacheTTL;

  /**
   * @param {import('./config-resolver.js').ResolverOptions} [options]
   */
  constructor(options = {}) {
    this.#projectRoot = options.projectRoot ?? process.cwd();
    this.#userHome = options.userHome ?? homedir();
    this.#cacheDir = options.cacheDir ?? join(this.#userHome, '.doppler', 'cache');
    this.#cacheTTL = options.cacheTTL ?? DEFAULT_CACHE_TTL;
  }

  /**
   * Resolve a config reference to its content.
   *
   * @param {string} ref - Config reference (name, path, URL, or inline JSON)
   * @returns {Promise<import('./config-resolver.js').ResolvedConfig>} Resolved config with source info and content
   */
  async resolve(ref) {
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
      return this.#resolveUrl(ref);
    }

    // 3. File path or nested preset (contains / or \)
    if (ref.includes('/') || ref.includes('\\')) {
      if (!ref.endsWith(PRESET_EXTENSION)) {
        try {
          return await this.#resolvePreset(ref);
        } catch {
          // Fall back to file resolution below
        }
      }
      return this.#resolveFile(ref);
    }

    // 4. File path (ends with .json)
    if (ref.endsWith(PRESET_EXTENSION)) {
      return this.#resolveFile(ref);
    }

    // 5. Named preset (search in order: project -> user -> builtin)
    return this.#resolvePreset(ref);
  }

  /**
   * Resolve a named preset from preset directories.
   *
   * @param {string} name
   * @returns {Promise<import('./config-resolver.js').ResolvedConfig>}
   */
  async #resolvePreset(name) {
    const normalized = name.replace(PATH_SEPARATOR_PATTERN, '/');
    const filename = normalized.endsWith(PRESET_EXTENSION)
      ? normalized
      : `${normalized}${PRESET_EXTENSION}`;

    // Search order: project -> user -> builtin
    /** @type {{ source: 'project' | 'user' | 'builtin', dir: string }[]} */
    const searchPaths = [
      { source: 'project', dir: join(this.#projectRoot, '.doppler') },
      { source: 'user', dir: join(this.#userHome, '.doppler', 'presets') },
      { source: 'builtin', dir: BUILTIN_PRESETS_DIR },
    ];

    for (const { source, dir } of searchPaths) {
      const resolved = await findPresetPath(dir, normalized);
      if (!resolved) continue;

      const content = await readFile(resolved, 'utf-8');

      // Warn if project/user preset shadows a builtin
      if (source !== 'builtin') {
        const builtinPath = await findPresetPath(BUILTIN_PRESETS_DIR, normalized);
        if (builtinPath) {
          console.warn(
            `[Config] ${source} preset "${name}" shadows built-in preset`
          );
        }
      }

      return { source, path: resolved, content, name };
    }

    throw new Error(
      `Config preset "${name}" not found. Searched:\n` +
      searchPaths.map(p => `  - ${p.dir}/${filename}`).join('\n')
    );
  }

  /**
   * Resolve a file path to config content.
   *
   * @param {string} ref
   * @returns {Promise<import('./config-resolver.js').ResolvedConfig>}
   */
  async #resolveFile(ref) {
    const path = resolve(this.#projectRoot, ref);
    try {
      const content = await readFile(path, 'utf-8');
      return { source: 'file', path, content, name: null };
    } catch (err) {
      throw new Error(`Config file not found: ${path}`);
    }
  }

  /**
   * Resolve a URL to config content (with caching).
   *
   * @param {string} url
   * @returns {Promise<import('./config-resolver.js').ResolvedConfig>}
   */
  async #resolveUrl(url) {
    // Check cache first
    const cacheKey = this.#urlToCacheKey(url);
    const cachePath = join(this.#cacheDir, cacheKey);
    const metaPath = `${cachePath}.meta`;

    try {
      const meta = JSON.parse(await readFile(metaPath, 'utf-8'));
      const age = Date.now() - meta.timestamp;
      if (age < this.#cacheTTL) {
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
      await mkdir(this.#cacheDir, { recursive: true });
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
   *
   * @param {string} url
   * @returns {string}
   */
  #urlToCacheKey(url) {
    return url
      .replace(/^https?:\/\//, '')
      .replace(/[^a-zA-Z0-9.-]/g, '_')
      .slice(0, 200);
  }

  /**
   * List available presets from all sources.
   *
   * @returns {Promise<{ name: string; source: string; path: string }[]>}
   */
  async listPresets() {
    /** @type {{ name: string; source: string; path: string }[]} */
    const presets = [];
    /** @type {Set<string>} */
    const seen = new Set();

    const searchPaths = [
      { source: 'project', dir: join(this.#projectRoot, '.doppler') },
      { source: 'user', dir: join(this.#userHome, '.doppler', 'presets') },
      { source: 'builtin', dir: BUILTIN_PRESETS_DIR },
    ];

    for (const { source, dir } of searchPaths) {
      try {
        const files = await listPresetFiles(dir);
        for (const file of files) {
          if (!seen.has(file.name)) {
            seen.add(file.name);
            presets.push({ name: file.name, source, path: file.path });
          }
        }
      } catch {
        // Directory doesn't exist, skip
      }
    }

    return presets;
  }
}

async function listPresetFiles(rootDir, baseDir = rootDir) {
  /** @type {{ name: string; path: string }[]} */
  const results = [];
  const entries = await readdir(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = join(rootDir, entry.name);
    if (entry.isDirectory()) {
      const nested = await listPresetFiles(fullPath, baseDir);
      results.push(...nested);
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith(PRESET_EXTENSION)) continue;
    const name = relative(baseDir, fullPath)
      .slice(0, -PRESET_EXTENSION.length)
      .split(sep)
      .join('/');
    results.push({ name, path: fullPath });
  }
  return results;
}

async function findPresetPath(rootDir, normalizedName) {
  try {
    await access(rootDir);
  } catch {
    return null;
  }

  if (normalizedName.includes('/')) {
    const candidate = join(rootDir, `${normalizedName}${PRESET_EXTENSION}`);
    try {
      await access(candidate);
      return candidate;
    } catch {
      return null;
    }
  }

  const files = await listPresetFiles(rootDir);
  const exact = files.find((file) => file.name === normalizedName);
  if (exact) return exact.path;
  const byBase = files.find((file) => basename(file.name) === normalizedName);
  return byBase?.path ?? null;
}

// =============================================================================
// Convenience Functions
// =============================================================================

const defaultResolver = new ConfigResolver();

/**
 * @param {string} ref
 * @returns {Promise<import('./config-resolver.js').ResolvedConfig>}
 */
export async function resolveConfig(ref) {
  return defaultResolver.resolve(ref);
}

/**
 * @returns {Promise<{ name: string; source: string; path: string }[]>}
 */
export async function listPresets() {
  return defaultResolver.listPresets();
}

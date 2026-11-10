import os from 'node:os';
import path from 'node:path';

/**
 * Name a local conversion input without machine-specific prefixes, so the same
 * input converted on two machines records the same provenance and identity.
 * Inside `baseDir` the path is relative; under the home directory it starts
 * with `~/`; anything else stays absolute. Separators are always `/`.
 * @param {unknown} filePath
 * @param {{ baseDir?: string, homeDir?: string | null }} [options]
 * @returns {string | null}
 */
export function portableSourcePath(filePath, { baseDir = process.cwd(), homeDir = os.homedir() } = {}) {
  if (typeof filePath !== 'string' || !filePath.trim()) return null;
  const absolute = path.resolve(filePath);
  /** @param {string} root */
  const inside = (root) => {
    const relative = path.relative(path.resolve(root), absolute);
    return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ? relative : null;
  };
  const fromBase = inside(baseDir);
  if (fromBase) return fromBase.split(path.sep).join('/');
  const fromHome = homeDir ? inside(homeDir) : null;
  if (fromHome) return `~/${fromHome.split(path.sep).join('/')}`;
  return absolute.split(path.sep).join('/');
}

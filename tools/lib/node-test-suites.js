import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';

const { suites } = JSON.parse(readFileSync(
  new URL('../policies/test-coverage-policy.json', import.meta.url), 'utf8'
));

export function resolveTestFiles(suiteName, directories = [], { includePending = false, root = process.cwd() } = {}) {
  if (!Object.hasOwn(suites, suiteName)) {
    throw new Error(`Unknown --suite "${suiteName}". Valid suites: ${Object.keys(suites).join(', ')}`);
  }
  const suite = suites[suiteName];
  const roots = directories.length ? directories : suite.roots;
  const excluded = directories.length ? [] : suite.excludeRoots.map((dir) => resolve(root, dir));
  const files = new Set();
  function collect(file, explicit = false) {
    if (excluded.some((dir) => file === dir || file.startsWith(`${dir}${sep}`))) return;
    if (!existsSync(file)) throw new Error(`Test path not found: ${file}`);
    if (statSync(file).isFile()) {
      if (!file.endsWith('.test.js')) {
        if (explicit) throw new Error(`Test file must end with .test.js: ${file}`);
        return;
      }
      if (explicit || includePending || !file.endsWith('.pending.test.js')) files.add(file);
      return;
    }
    for (const entry of readdirSync(file, { withFileTypes: true })) {
      if (entry.name.startsWith('.') || entry.isSymbolicLink()) continue;
      if (entry.isDirectory() || entry.isFile()) collect(join(file, entry.name));
    }
  }
  for (const path of roots) collect(resolve(root, path), true);
  return [...files].sort();
}

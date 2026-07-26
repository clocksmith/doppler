#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ROOTS = ['demo/demo.js', 'demo/sw.js'];
const ALLOWED_BARE = new Set([
  'doppler-gpu',
  'doppler-gpu/tooling/runtime',
  'doppler-gpu/tooling/evidence',
]);
const PATTERNS = [
  /\b(?:import|export)\s+(?:[^'"]*?\sfrom\s*)?['"]([^'"]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g,
];

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function resolveLocal(importer, specifier) {
  const base = path.resolve(ROOT, path.dirname(importer), specifier);
  const candidates = path.extname(base) ? [base] : [`${base}.js`, path.join(base, 'index.js')];
  for (const candidate of candidates) {
    if (await exists(candidate)) {
      return path.relative(ROOT, candidate).split(path.sep).join('/');
    }
  }
  return null;
}

async function collectDemoJs(directory = path.join(ROOT, 'demo')) {
  const output = [];
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      output.push(...await collectDemoJs(target));
    } else if (entry.name.endsWith('.js') && entry.name !== 'generated-shell-manifest.js') {
      output.push(path.relative(ROOT, target).split(path.sep).join('/'));
    }
  }
  return output;
}

async function main() {
  const seen = new Set();
  const pending = [...ROOTS];
  const errors = [];
  while (pending.length > 0) {
    const file = pending.pop();
    if (!file || seen.has(file)) continue;
    seen.add(file);
    const source = await fs.readFile(path.join(ROOT, file), 'utf8');
    if (/(?:\.\.\/)+src\//.test(source)) {
      errors.push(`${file}: demo modules must not import private src paths`);
    }
    for (const pattern of PATTERNS) {
      pattern.lastIndex = 0;
      for (const match of source.matchAll(pattern)) {
        const specifier = match[1];
        if (!specifier.startsWith('.')) {
          if (!ALLOWED_BARE.has(specifier)) {
            errors.push(`${file}: unsupported package import ${specifier}`);
          }
          continue;
        }
        const resolved = await resolveLocal(file, specifier);
        if (!resolved) {
          errors.push(`${file}: unresolved local import ${specifier}`);
        } else if (resolved.startsWith('demo/')) {
          pending.push(resolved);
        }
      }
    }
  }
  const all = await collectDemoJs();
  const unreachable = all.filter((file) => !seen.has(file));
  if (unreachable.length > 0) {
    errors.push(`unreachable demo JavaScript: ${unreachable.join(', ')}`);
  }
  if (errors.length > 0) {
    throw new Error(errors.join('\n'));
  }
  console.log(`demo reachability: ${seen.size} live modules, no duplicate implementation`);
}

await main();

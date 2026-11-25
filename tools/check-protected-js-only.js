#!/usr/bin/env node
// Guard: the confidential-compute tools (tools/privacy-protected, tools/secret-shared) are JavaScript only. Owner
// decision 2026-11-25: no Go, no Python (and no shell drivers) in Doppler's protected and secret-shared routes.
// Fails with the offending paths. Evidence data (JSON, logs) and WGSL kernels are allowed.
//
//   node tools/check-protected-js-only.js

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const GUARDED = ['tools/privacy-protected', 'tools/secret-shared'];
const FORBIDDEN_NAMES = new Set(['go.mod', 'go.sum', 'go.work', '__pycache__', 'requirements.txt', 'pyproject.toml', 'Pipfile']);
const FORBIDDEN_EXT = /\.(go|py|pyc|pyi|ipynb|sh|bash|zsh|c|cc|cpp|h|rs|zig|wat|wasm|so|dylib)$/i;

function shebangLanguage(file) {
  try {
    const head = readFileSync(file, { encoding: 'utf8', flag: 'r' }).slice(0, 64);
    const m = /^#!\s*\S*?(?:env\s+)?(\w+)/.exec(head);
    return m && m[1] !== 'node' ? m[1] : null;
  } catch {
    return null;
  }
}

export function findNonJavaScript(root) {
  const bad = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      const rel = relative(root, p);
      if (name === 'node_modules') continue;
      if (FORBIDDEN_NAMES.has(name) || FORBIDDEN_EXT.test(name)) {
        bad.push(rel);
        continue;
      }
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else if (st.mode & 0o111 && shebangLanguage(p)) bad.push(`${rel} (executable ${shebangLanguage(p)} script)`);
    }
  };
  for (const g of GUARDED) if (existsSync(join(root, g))) walk(join(root, g));
  return bad.sort();
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
  const bad = findNonJavaScript(root);
  if (bad.length) {
    console.error(`JavaScript only (tools/privacy-protected, tools/secret-shared): remove or port\n  ${bad.join('\n  ')}`);
    process.exit(1);
  }
  console.log('protected and secret-shared tools: JavaScript only');
}

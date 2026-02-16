#!/usr/bin/env node

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(new URL(import.meta.url));
const __dirname = path.dirname(__filename);
const demoRoot = path.resolve(__dirname, '..', 'demo');
const forbiddenImport = '../src/';
const allowedExts = new Set(['.js', '.ts']);

async function walkJsTsFiles(dir, out = []) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === 'node_modules') continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await walkJsTsFiles(fullPath, out);
      continue;
    }
    if (!entry.isFile()) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (!allowedExts.has(ext)) continue;
    out.push(fullPath);
  }
  return out;
}

async function main() {
  const files = await walkJsTsFiles(demoRoot);
  const matches = [];

  for (const filePath of files) {
    const text = await fs.readFile(filePath, 'utf8');
    if (text.includes(forbiddenImport)) {
      matches.push(path.relative(path.resolve(__dirname, '..'), filePath));
    }
  }

  if (matches.length > 0) {
    console.log('');
    console.log('ERROR: demo contains forbidden imports of ../src/*');
    console.log("Use '@doppler/core' (and only deep-import '@doppler/core/...' if truly necessary).");
    process.exit(1);
  }

  console.log('OK: demo imports are core-only.');
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exit(1);
});

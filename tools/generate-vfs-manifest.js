#!/usr/bin/env node

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const SRC_ROOT = path.join(PROJECT_ROOT, 'src');
const OUTPUT_PATH = path.join(PROJECT_ROOT, 'config', 'vfs-manifest.json');

const EXTENSIONS = new Set(['.js', '.json', '.wgsl']);
const CONTENT_TYPES = {
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
};

function normalizePath(p) {
  return `/${p.split(path.sep).join('/')}`;
}

function guessContentType(p) {
  for (const [ext, type] of Object.entries(CONTENT_TYPES)) {
    if (p.endsWith(ext)) return type;
  }
  return 'application/octet-stream';
}

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const resolved = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...await walk(resolved));
    } else if (entry.isFile()) {
      files.push(resolved);
    }
  }
  return files;
}

async function main() {
  const allFiles = await walk(SRC_ROOT);
  const filtered = [];

  for (const file of allFiles) {
    if (file.endsWith('.d.ts')) continue;
    const ext = path.extname(file);
    if (!EXTENSIONS.has(ext)) continue;
    const rel = path.relative(PROJECT_ROOT, file);
    const stats = await fs.stat(file);
    filtered.push({
      path: normalizePath(rel),
      url: normalizePath(rel),
      contentType: guessContentType(rel),
      size: stats.size,
    });
  }

  filtered.sort((a, b) => a.path.localeCompare(b.path));

  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    root: '/',
    files: filtered,
  };

  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.writeFile(OUTPUT_PATH, JSON.stringify(manifest, null, 2) + '\n', 'utf8');

  console.log(`Wrote ${filtered.length} entries to ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
});

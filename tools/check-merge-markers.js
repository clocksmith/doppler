#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.resolve(__dirname, '..');

const SEARCH_ROOTS = Object.freeze([
  'src',
  'tools',
  'tests',
  'demo',
  'docs',
  'models',
]);

const ALLOWED_EXTENSIONS = new Set([
  '.js',
  '.mjs',
  '.cjs',
  '.d.ts',
  '.json',
  '.md',
  '.wgsl',
  '.html',
  '.css',
  '.txt',
]);

const CONFLICT_MARKER_REGEX = /^\s*(<<<<<<<|=======|>>>>>>>)\b/;

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function shouldInspectFile(filePath) {
  const ext = path.extname(filePath);
  return ALLOWED_EXTENSIONS.has(ext);
}

async function collectFiles(rootDir, relativeRoot) {
  const absoluteRoot = path.join(rootDir, relativeRoot);
  if (!await exists(absoluteRoot)) return [];

  const files = [];
  async function walk(currentPath) {
    const entries = await fs.readdir(currentPath, { withFileTypes: true });
    for (const entry of entries) {
      const absolute = path.join(currentPath, entry.name);
      if (entry.isDirectory()) {
        await walk(absolute);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!shouldInspectFile(absolute)) continue;
      files.push(absolute);
    }
  }

  await walk(absoluteRoot);
  return files;
}

async function findConflictMarkers(filePath) {
  const content = await fs.readFile(filePath, 'utf-8');
  const lines = content.split('\n');
  const hits = [];
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (CONFLICT_MARKER_REGEX.test(line)) {
      hits.push({ line: i + 1, text: line.trim() });
    }
  }
  return hits;
}

async function main() {
  const issues = [];

  for (const relativeRoot of SEARCH_ROOTS) {
    const files = await collectFiles(ROOT_DIR, relativeRoot);
    for (const filePath of files) {
      const hits = await findConflictMarkers(filePath);
      if (hits.length === 0) continue;
      const relativePath = path.relative(ROOT_DIR, filePath);
      for (const hit of hits) {
        issues.push(`${relativePath}:${hit.line} ${hit.text}`);
      }
    }
  }

  if (issues.length > 0) {
    console.error('merge conflict marker check failed:');
    for (const issue of issues) {
      console.error(`- ${issue}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log('merge conflict marker check passed');
}

await main();

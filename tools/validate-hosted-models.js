#!/usr/bin/env node

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(new URL(import.meta.url)));
const modelsDir = path.resolve(scriptDir, '..', 'models');
const repoRoot = path.resolve(scriptDir, '..', '..', '..');
const firebaseConfig = path.resolve(repoRoot, 'firebase.json');

async function collectFiles(rootDir, out = []) {
  const entries = await fs.readdir(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      await collectFiles(fullPath, out);
      continue;
    }
    if (entry.isFile()) {
      out.push(fullPath);
    }
  }
  return out;
}

async function main() {
  try {
    await fs.access(path.join(modelsDir, 'catalog.json'));
  } catch {
    console.error('[models-validate] Missing required file: models/catalog.json');
    process.exit(1);
  }

  const files = await collectFiles(modelsDir);
  const disallowed = [];

  for (const filePath of files.sort((a, b) => a.localeCompare(b))) {
    const relPath = path.relative(modelsDir, filePath);
    if (
      relPath === 'catalog.json' ||
      relPath === 'README.md' ||
      relPath.startsWith('curated/') ||
      relPath.startsWith('local/')
    ) {
      continue;
    }
    disallowed.push(relPath);
  }

  if (disallowed.length > 0) {
    console.log('[models-validate] Only the following model paths are allowed under models/:');
    console.log('  - models/catalog.json');
    console.log('  - models/README.md');
    console.log('  - models/curated/** (hosted)');
    console.log('  - models/local/** (local only; not deployed)');
    console.error('[models-validate] Disallowed files found:');
    for (const file of disallowed) {
      console.error(`  - ${file}`);
    }
    process.exit(1);
  }

  try {
    const config = await fs.readFile(firebaseConfig, 'utf8');
    if (!config.includes('"models/local/**"')) {
      console.error('[models-validate] firebase.json must ignore models/local/** for the doppler host target.');
      process.exit(1);
    }
  } catch {
  }

  console.log('[models-validate] Model layout OK (curated hosted, local excluded).');
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exit(1);
});

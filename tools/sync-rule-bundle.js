#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const rulesRoot = resolve(repoRoot, 'src/rules');
const outputPath = resolve(rulesRoot, 'generated/rule-bundle.json');
const checkOnly = process.argv.includes('--check');
const standaloneRulePaths = new Set([
  'inference/capability-transforms.rules.json',
]);

function sha256(text) {
  return `sha256:${createHash('sha256').update(text).digest('hex')}`;
}

async function collectJsonFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      if (path === dirname(outputPath)) continue;
      files.push(...await collectJsonFiles(path));
    } else if (entry.isFile() && entry.name.endsWith('.json')) {
      files.push(path);
    }
  }
  return files;
}

const sourcePaths = await collectJsonFiles(rulesRoot);
const sources = [];
const files = {};
for (const sourcePath of sourcePaths) {
  const text = await readFile(sourcePath, 'utf8');
  const path = relative(rulesRoot, sourcePath).replaceAll('\\', '/');
  if (standaloneRulePaths.has(path)) {
    continue;
  }
  sources.push({ path, digest: sha256(text) });
  files[path] = JSON.parse(text);
}
const bundle = {
  schema: 'doppler.rule-bundle/v1',
  sources,
  files,
};
const output = `${JSON.stringify(bundle, null, 2)}\n`;

if (checkOnly) {
  let current = null;
  try {
    current = await readFile(outputPath, 'utf8');
  } catch {
    // The comparison below reports the required regeneration command.
  }
  if (current !== output) {
    throw new Error('Rule bundle is stale. Run: npm run rules:bundle:sync');
  }
  console.log(`rule bundle: current (${sources.length} source files)`);
} else {
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, output);
  console.log(`rule bundle: wrote ${relative(repoRoot, outputPath)} (${sources.length} source files)`);
}

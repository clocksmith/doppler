#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { materializeSafetensorsHeaderEvidence } from '../src/converter/safetensors-header-evidence.js';

function parseArgs(argv) {
  const options = { check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--recipe') options.recipe = argv[++index];
    else if (token === '--check') options.check = true;
    else throw new Error(`Unknown argument "${token}".`);
  }
  if (!options.recipe) throw new Error('Usage: --recipe <path> [--check]');
  return options;
}

function encodeRepository(repository) {
  return repository.split('/').map(encodeURIComponent).join('/');
}

async function readExactRange({ repository, revision, sourceFile, start, end }) {
  const url = `https://huggingface.co/${encodeRepository(repository)}/resolve/${encodeURIComponent(revision)}/${encodeURIComponent(sourceFile)}`;
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (response.status !== 206) {
    throw new Error(`Range request for "${sourceFile}" returned HTTP ${response.status}; refusing an unbounded body.`);
  }
  const expectedLength = end - start + 1;
  const contentLength = Number(response.headers.get('content-length'));
  if (contentLength !== expectedLength) {
    throw new Error(`Range request for "${sourceFile}" returned ${String(contentLength)} bytes; expected ${expectedLength}.`);
  }
  const contentRange = response.headers.get('content-range');
  if (!contentRange?.startsWith(`bytes ${start}-${end}/`)) {
    throw new Error(`Range request for "${sourceFile}" returned invalid Content-Range "${contentRange}".`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const recipePath = path.resolve(options.recipe);
const recipe = JSON.parse(await fs.readFile(recipePath, 'utf8'));
const evidence = await materializeSafetensorsHeaderEvidence(recipe, readExactRange);
const outputPath = path.resolve(repoRoot, recipe.output);
const outputText = `${JSON.stringify(evidence, null, 2)}\n`;
if (options.check) {
  const observed = await fs.readFile(outputPath, 'utf8');
  if (observed !== outputText) throw new Error(`SafeTensors header evidence drifted: ${recipe.output}.`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, outputText);
}
console.log(`${evidence.checkpointId}: ${evidence.tensorCount} tensors across ${recipe.shards.length} shards`);

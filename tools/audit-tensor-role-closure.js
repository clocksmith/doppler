#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createTensorRoleClosureReceipt } from '../src/converter/tensor-role-closure.js';

function parseArgs(argv) {
  const options = { check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') options.policy = argv[++index];
    else if (token === '--check') options.check = true;
    else throw new Error(`Unknown argument "${token}".`);
  }
  if (!options.policy) throw new Error('Usage: --policy <path> [--check]');
  return options;
}

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const policy = await readJson(path.resolve(options.policy));
const [modelIRReceipt, headers] = await Promise.all([
  readJson(path.resolve(repoRoot, policy.modelIRReceipt)),
  readJson(path.resolve(repoRoot, policy.headerEvidence)),
]);
const receipt = createTensorRoleClosureReceipt({ modelIR: modelIRReceipt.modelIR, headers, policy });
const outputPath = path.resolve(repoRoot, policy.output);
const outputText = `${JSON.stringify(receipt, null, 2)}\n`;
if (options.check) {
  const observed = await fs.readFile(outputPath, 'utf8');
  if (observed !== outputText) throw new Error(`Tensor-role closure evidence drifted: ${policy.output}.`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, outputText);
}
console.log(`${receipt.modelId}/${receipt.entryPointId}: ${receipt.observedTensorCount}/${receipt.expectedTensorCount} tensors`);

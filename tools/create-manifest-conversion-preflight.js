#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createManifestConversionPreflightReceipt } from '../src/converter/manifest-conversion-preflight.js';

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
const [
  rawConfig,
  conversionConfig,
  semanticReceipt,
  headers,
  weightIndex,
  tensorPolicy,
  tensorClosureReceipt,
  sourceAcquisitionReceipt,
] = await Promise.all([
  readJson(path.resolve(repoRoot, policy.rawConfig)),
  readJson(path.resolve(repoRoot, policy.conversionConfig)),
  readJson(path.resolve(repoRoot, policy.semanticReceipt)),
  readJson(path.resolve(repoRoot, policy.headerEvidence)),
  readJson(path.resolve(repoRoot, policy.weightIndex)),
  readJson(path.resolve(repoRoot, policy.tensorPolicy)),
  readJson(path.resolve(repoRoot, policy.tensorClosureReceipt)),
  readJson(path.resolve(repoRoot, policy.sourceAcquisitionReceipt)),
]);
const receipt = createManifestConversionPreflightReceipt({
  rawConfig,
  conversionConfig,
  semanticReceipt,
  headers,
  weightIndex,
  tensorPolicy,
  tensorClosureReceipt,
  sourceAcquisitionReceipt,
  policy,
});
const outputPath = path.resolve(repoRoot, policy.output);
const outputText = `${JSON.stringify(receipt, null, 2)}\n`;
if (options.check) {
  const observed = await fs.readFile(outputPath, 'utf8');
  if (observed !== outputText) throw new Error(`Manifest conversion preflight drifted: ${policy.output}.`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, outputText);
}
console.log(`${receipt.modelId}/${receipt.entryPointId}: ${receipt.sourceEvidence.scopedTensorCount} tensors preflighted`);

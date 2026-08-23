#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { materializeSemanticManifestCandidate } from '../src/converter/semantic-manifest-lowering.js';

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

async function readJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const recipePath = path.resolve(options.recipe);
const recipe = await readJson(recipePath);
const [modelIRReceipt, template] = await Promise.all([
  readJson(path.resolve(repoRoot, recipe.modelIRReceipt)),
  readJson(path.resolve(repoRoot, recipe.template)),
]);
const receipt = materializeSemanticManifestCandidate({
  modelIR: modelIRReceipt.modelIR,
  template,
  recipe,
});
const outputPath = path.resolve(repoRoot, recipe.output);
const receiptPath = path.resolve(repoRoot, recipe.receiptOutput);
const outputText = `${JSON.stringify(receipt.conversionConfig, null, 2)}\n`;
const receiptText = `${JSON.stringify(receipt, null, 2)}\n`;
if (options.check) {
  const [observedOutput, observedReceipt] = await Promise.all([
    fs.readFile(outputPath, 'utf8'),
    fs.readFile(receiptPath, 'utf8'),
  ]);
  if (observedOutput !== outputText || observedReceipt !== receiptText) {
    throw new Error('Semantic conversion outputs are stale. Run without --check.');
  }
} else {
  await Promise.all([
    fs.mkdir(path.dirname(outputPath), { recursive: true }),
    fs.mkdir(path.dirname(receiptPath), { recursive: true }),
  ]);
  await Promise.all([
    fs.writeFile(outputPath, outputText),
    fs.writeFile(receiptPath, receiptText),
  ]);
}
console.log(`${receipt.modelId}: ${receipt.conversionConfigDigest}`);

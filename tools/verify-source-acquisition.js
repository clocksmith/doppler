#!/usr/bin/env node

import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSourceAcquisitionReceipt } from '../src/converter/source-acquisition.js';

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

async function hashFile(file) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const policy = JSON.parse(await fs.readFile(path.resolve(options.policy), 'utf8'));
const sourceDir = path.resolve(repoRoot, policy.localDir);
const receipt = await createSourceAcquisitionReceipt(policy, {
  async listFiles() {
    const entries = await fs.readdir(sourceDir, { withFileTypes: true });
    return entries.filter((entry) => entry.isFile()).map((entry) => entry.name);
  },
  async statFile(file) {
    return Number((await fs.stat(path.join(sourceDir, file))).size);
  },
  async hashFile(file) {
    return hashFile(path.join(sourceDir, file));
  },
});
const outputPath = path.resolve(repoRoot, policy.output);
const outputText = `${JSON.stringify(receipt, null, 2)}\n`;
if (options.check) {
  const observed = await fs.readFile(outputPath, 'utf8');
  if (observed !== outputText) throw new Error(`Source acquisition receipt drifted: ${policy.output}.`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, outputText);
}
console.log(`${receipt.checkpointId}: ${receipt.fileCount} files, ${receipt.totalBytes} verified bytes`);

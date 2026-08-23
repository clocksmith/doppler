#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { createSourceTruthPacket, forgeModelIRV2 } from '../src/converter/source-truth-forge.js';

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--spec') options.spec = argv[++index];
    else if (token === '--out') options.out = argv[++index];
    else throw new Error(`Unknown argument "${token}".`);
  }
  if (!options.spec || !options.out) throw new Error('Usage: --spec <path> --out <path>');
  return options;
}

const options = parseArgs(process.argv.slice(2));
const specPath = path.resolve(options.spec);
const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
const sources = {};
for (const [artifactId, sourcePath] of Object.entries(spec.sources || {})) {
  sources[artifactId] = JSON.parse(await fs.readFile(path.resolve(repoRoot, sourcePath), 'utf8'));
}
delete spec.sources;
const packet = createSourceTruthPacket(spec, sources);
const receipt = forgeModelIRV2(packet, sources);
const outputPath = path.resolve(options.out);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(`${receipt.modelIR.modelId}: ${receipt.intakeDigest} -> ${outputPath}`);

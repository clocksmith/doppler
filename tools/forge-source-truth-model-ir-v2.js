#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { createSourceTruthPacket, forgeModelIRV2 } from '../src/converter/source-truth-forge.js';
import { parseSafetensorsHeaderEvidence, readSafetensorsHeaderLength } from '../src/converter/safetensors-header-evidence.js';
import { hashBytesSha256 } from '../src/formats/canonical-hash.js';

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
for (const [artifactId, source] of Object.entries(spec.sources || {})) {
  if (typeof source === 'string') {
    sources[artifactId] = JSON.parse(await fs.readFile(path.resolve(repoRoot, source), 'utf8'));
    continue;
  }
  if (!source || typeof source.path !== 'string' || !/^sha256:[0-9a-f]{64}$/u.test(source.hash)) {
    throw new Error(`Source "${artifactId}" requires path and pinned byte hash.`);
  }
  const sourcePath = path.resolve(repoRoot, source.path);
  let bytes;
  if (source.format === 'json') {
    bytes = await fs.readFile(sourcePath);
  } else if (source.format === 'safetensors-header') {
    const file = await fs.open(sourcePath, 'r');
    try {
      const prefix = Buffer.alloc(8);
      if ((await file.read(prefix, 0, 8, 0)).bytesRead !== 8) throw new Error('Truncated SafeTensors prefix.');
      bytes = Buffer.alloc(8 + readSafetensorsHeaderLength(prefix));
      let offset = 0;
      while (offset < bytes.length) {
        const { bytesRead } = await file.read(bytes, offset, bytes.length - offset, offset);
        if (bytesRead === 0) throw new Error('Truncated SafeTensors header.');
        offset += bytesRead;
      }
    } finally {
      await file.close();
    }
  } else {
    throw new Error(`Source "${artifactId}" has unsupported format "${source.format}".`);
  }
  const hash = hashBytesSha256(bytes);
  if (hash !== source.hash) throw new Error(`Source "${artifactId}" byte hash mismatch.`);
  const content = source.format === 'json' ? JSON.parse(bytes.toString('utf8'))
    : parseSafetensorsHeaderEvidence(bytes, { sourceFile: path.basename(sourcePath), expectedSha256: source.hash });
  sources[artifactId] = { content, hash };
}
delete spec.sources;
const packet = createSourceTruthPacket(spec, sources);
const receipt = forgeModelIRV2(packet, sources);
const outputPath = path.resolve(options.out);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
console.log(`${receipt.modelIR.modelId}: ${receipt.intakeDigest} -> ${outputPath}`);

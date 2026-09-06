import fs from 'node:fs/promises';
import path from 'node:path';
import { createSourceTruthPacket, forgeModelIRV2 } from '../converter/source-truth-forge.js';
import { parseSafetensorsHeaderEvidence, readSafetensorsHeaderLength } from '../converter/safetensors-header-evidence.js';
import { hashBytesSha256 } from '../formats/canonical-hash.js';

async function readHeader(filePath) {
  const file = await fs.open(filePath, 'r');
  try {
    const prefix = Buffer.alloc(8);
    if ((await file.read(prefix, 0, 8, 0)).bytesRead !== 8) throw new Error('Truncated SafeTensors prefix.');
    const bytes = Buffer.alloc(8 + readSafetensorsHeaderLength(prefix));
    let offset = 0;
    while (offset < bytes.length) {
      const { bytesRead } = await file.read(bytes, offset, bytes.length - offset, offset);
      if (bytesRead === 0) throw new Error('Truncated SafeTensors header.');
      offset += bytesRead;
    }
    return bytes;
  } finally {
    await file.close();
  }
}

export async function forgeSourceTruthFromFiles(spec, sourceRoot) {
  const sources = {};
  for (const [artifactId, source] of Object.entries(spec.sources || {})) {
    if (typeof source === 'string') {
      sources[artifactId] = JSON.parse(await fs.readFile(path.resolve(sourceRoot, source), 'utf8'));
      continue;
    }
    if (!source || typeof source.path !== 'string' || !/^sha256:[0-9a-f]{64}$/u.test(source.hash)) {
      throw new Error(`Source "${artifactId}" requires path and pinned byte hash.`);
    }
    const sourcePath = path.resolve(sourceRoot, source.path);
    let bytes;
    if (source.format === 'json') bytes = await fs.readFile(sourcePath);
    else if (source.format === 'safetensors-header') bytes = await readHeader(sourcePath);
    else throw new Error(`Source "${artifactId}" has unsupported format "${source.format}".`);
    const hash = hashBytesSha256(bytes);
    if (hash !== source.hash) throw new Error(`Source "${artifactId}" byte hash mismatch.`);
    const content = source.format === 'json' ? JSON.parse(bytes.toString('utf8'))
      : parseSafetensorsHeaderEvidence(bytes, { sourceFile: path.basename(sourcePath), expectedSha256: source.hash });
    sources[artifactId] = { content, hash };
  }
  const { sources: fileSources, ...packetSpec } = spec;
  return forgeModelIRV2(createSourceTruthPacket(packetSpec, sources), sources);
}

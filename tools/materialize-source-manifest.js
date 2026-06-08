#!/usr/bin/env node

import { once } from 'node:events';
import { createWriteStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';

import { materializeSourceRuntimeManifest } from '../src/tooling/source-runtime-materializer.js';
import { resolveNodeSourceRuntimeBundle } from '../src/tooling/node-source-runtime.js';

function fail(message) {
  console.error(`[materialize-source-manifest] ${message}`);
  process.exit(1);
}

function parseArgs(argv) {
  const args = {
    inputPath: null,
    modelId: null,
    manifestPath: null,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model-id') {
      args.modelId = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--manifest' || arg === '--manifest-path') {
      args.manifestPath = argv[i + 1] ?? null;
      i += 1;
      continue;
    }
    if (arg === '--dry-run') {
      args.dryRun = true;
      continue;
    }
    if (arg.startsWith('-')) {
      fail(`Unknown flag: ${arg}`);
    }
    if (!args.inputPath) {
      args.inputPath = arg;
      continue;
    }
    fail(`Unexpected positional argument: ${arg}`);
  }

  if (!args.inputPath) {
    fail(
      'Usage: node tools/materialize-source-manifest.js <source-path> ' +
      '[--model-id <id>] [--manifest <manifest.json>] [--dry-run]'
    );
  }

  return args;
}

function toBufferChunk(chunk) {
  if (Buffer.isBuffer(chunk)) return chunk;
  if (chunk instanceof Uint8Array) {
    return Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  }
  if (chunk instanceof ArrayBuffer) {
    return Buffer.from(chunk);
  }
  if (ArrayBuffer.isView(chunk)) {
    return Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  }
  throw new Error(`Unsupported source chunk type: ${typeof chunk}.`);
}

async function getFileSize(filePath) {
  const stats = await fs.stat(filePath).catch((error) => {
    if (error?.code === 'ENOENT') {
      return null;
    }
    throw error;
  });
  return stats?.isFile() ? Number(stats.size) : null;
}

async function writeSourceFileFromShard(storageContext, entry, artifactDir) {
  const targetPath = path.resolve(artifactDir, entry.path);
  const expectedSize = Number(entry.size);
  if (!Number.isSafeInteger(expectedSize) || expectedSize < 0) {
    throw new Error(`source file ${entry.index} has invalid size ${entry.size}.`);
  }

  const currentSize = await getFileSize(targetPath);
  if (currentSize === expectedSize) {
    return false;
  }

  await fs.mkdir(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.tmp-${process.pid}`;
  const stream = createWriteStream(tempPath, { flags: 'w' });
  let written = 0;
  try {
    for await (const chunk of storageContext.streamShardRange(entry.index, 0, expectedSize, {
      chunkBytes: 64 * 1024 * 1024,
    })) {
      const bytes = toBufferChunk(chunk);
      written += bytes.byteLength;
      if (!stream.write(bytes)) {
        await once(stream, 'drain');
      }
    }
    stream.end();
    await once(stream, 'finish');
    if (written !== expectedSize) {
      throw new Error(
        `source file ${entry.index} short write for "${entry.path}": wrote ${written}, expected ${expectedSize}.`
      );
    }
    await fs.rename(tempPath, targetPath);
    return true;
  } catch (error) {
    stream.destroy();
    await fs.rm(tempPath, { force: true }).catch(() => {});
    throw error;
  }
}

async function writeAuxiliaryFile(storageContext, entry, artifactDir) {
  const targetPath = path.resolve(artifactDir, entry.path);
  const expectedSize = Number(entry.size);
  if (!Number.isSafeInteger(expectedSize) || expectedSize < 0) {
    throw new Error(`auxiliary file "${entry.path}" has invalid size ${entry.size}.`);
  }

  const currentSize = await getFileSize(targetPath);
  if (currentSize === expectedSize) {
    return false;
  }
  if (typeof storageContext.loadAuxiliaryFile !== 'function') {
    throw new Error('materialized auxiliary files require bundle.storageContext.loadAuxiliaryFile().');
  }

  const payload = await storageContext.loadAuxiliaryFile(entry.path);
  const bytes = toBufferChunk(payload);
  if (bytes.byteLength !== expectedSize) {
    throw new Error(
      `auxiliary file "${entry.path}" size mismatch: read ${bytes.byteLength}, expected ${expectedSize}.`
    );
  }

  await fs.mkdir(path.dirname(targetPath), { recursive: true });
  const tempPath = `${targetPath}.tmp-${process.pid}`;
  await fs.writeFile(tempPath, bytes);
  await fs.rename(tempPath, targetPath);
  return true;
}

async function persistSourceFiles(bundle, materializedManifest, artifactDir) {
  const sourceFiles = materializedManifest?.metadata?.sourceRuntime?.sourceFiles;
  if (!Array.isArray(sourceFiles) || sourceFiles.length === 0) {
    return 0;
  }
  if (!bundle?.storageContext || typeof bundle.storageContext.streamShardRange !== 'function') {
    throw new Error('materialized source files require bundle.storageContext.streamShardRange().');
  }

  let written = 0;
  for (const entry of sourceFiles) {
    if (await writeSourceFileFromShard(bundle.storageContext, entry, artifactDir)) {
      written += 1;
    }
  }
  return written;
}

async function persistAuxiliaryFiles(bundle, materializedManifest, artifactDir) {
  const auxiliaryFiles = materializedManifest?.metadata?.sourceRuntime?.auxiliaryFiles;
  if (!Array.isArray(auxiliaryFiles) || auxiliaryFiles.length === 0) {
    return 0;
  }

  let written = 0;
  for (const entry of auxiliaryFiles) {
    if (await writeAuxiliaryFile(bundle.storageContext, entry, artifactDir)) {
      written += 1;
    }
  }
  return written;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(args.inputPath);
  const stats = await fs.stat(inputPath).catch((error) => {
    fail(`Failed to stat inputPath "${inputPath}": ${error.message}`);
  });
  const artifactDir = stats.isDirectory() ? inputPath : path.dirname(inputPath);
  const manifestPath = path.resolve(args.manifestPath || path.join(artifactDir, 'manifest.json'));

  const bundle = await resolveNodeSourceRuntimeBundle({
    inputPath,
    modelId: args.modelId,
    verifyHashes: true,
  });
  try {
    if (!bundle) {
      fail(
        `No direct-source model detected at "${inputPath}". ` +
        'Expected a Safetensors directory, existing direct-source artifact, or .gguf file.'
      );
    }

    const materializedManifest = materializeSourceRuntimeManifest(bundle.manifest, artifactDir);
    const manifestJson = `${JSON.stringify(materializedManifest, null, 2)}\n`;

    if (args.dryRun) {
      process.stdout.write(manifestJson);
      return;
    }

    const sourceFilesWritten = await persistSourceFiles(bundle, materializedManifest, artifactDir);
    const auxiliaryFilesWritten = await persistAuxiliaryFiles(bundle, materializedManifest, artifactDir);
    await fs.writeFile(manifestPath, manifestJson, 'utf8');
    console.log(
      `[materialize-source-manifest] wrote ${path.relative(process.cwd(), manifestPath)} ` +
      `for ${materializedManifest.modelId} ` +
      `(sourceFilesWritten=${sourceFilesWritten}, auxiliaryFilesWritten=${auxiliaryFilesWritten})`
    );
  } finally {
    await bundle?.storageContext?.close?.();
  }
}

await main();

import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const [root, modelDirectory] = process.argv.slice(2);
const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const inventory = JSON.parse(await readFile(resolve(root, 'inventory.json')));
for (const file of inventory) assert.equal(hash(await readFile(resolve(root, 'node_modules/doppler-gpu', file.path))), file.hash, file.path);
const manifestBytes = await readFile(resolve(root, 'manifest.json')), manifest = JSON.parse(manifestBytes);
const expected = JSON.parse(await readFile(resolve(root, 'inputs.json')));
for (const [path, digest] of Object.entries(expected)) assert.equal(hash(await readFile(resolve(root, path))), digest, path);
const { hash: blake3 } = await import(pathToFileURL(resolve(root, 'node_modules/doppler-gpu/src/storage/blake3.js')));
const shards = [];
for (const shard of manifest.shards) {
  const path = resolve(modelDirectory, shard.filename), bytes = await readFile(path);
  assert.equal(bytes.length, shard.size); assert.equal(Buffer.from(await blake3(bytes)).toString('hex'), shard.blake3);
  shards.push({ path, bytes: bytes.length, blake3: shard.blake3 });
}
await writeFile(resolve(root, 'verification.json'), JSON.stringify({ installedFiles: inventory.length,
  allInstalledBytesVerified: true, inputs: expected, modelDirectory, shards, node: process.version, gpuOpened: false }, null, 2));
console.log(JSON.stringify({ installedFiles: inventory.length, shards: shards.length, allVerified: true }));

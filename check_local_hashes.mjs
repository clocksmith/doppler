import fs from 'node:fs/promises';
import { computeHash } from './src/storage/shard-manager.js';

const localDir = '/home/x/deco/doppler/models/local/gemma-4-e2b-it-q4k-ehf16-af32';
const manifest = JSON.parse(await fs.readFile(`${localDir}/manifest.json`, 'utf8'));
const bad = [];
const missing = [];

for (const shard of manifest.shards) {
  const path = `${localDir}/${shard.filename}`;
  try {
    const buf = await fs.readFile(path);
    if (buf.byteLength !== shard.size) {
      bad.push({ filename: shard.filename, reason: 'size', local: buf.byteLength, expected: shard.size });
      continue;
    }
    const algorithm = manifest.hashAlgorithm || shard.hashAlgorithm || 'blake3';
    const hash = await computeHash(buf, algorithm);
    if (hash !== shard.hash) {
      bad.push({ filename: shard.filename, reason: 'hash', local: hash, expected: shard.hash });
    }
  } catch (err) {
    if (err.code === 'ENOENT') {
      missing.push(shard.filename);
    } else {
      throw err;
    }
  }
}

console.log(`missing=${missing.length}`);
for (const f of missing) console.log(`MISSING ${f}`);
console.log(`bad=${bad.length}`);
for (const b of bad) {
  if (b.reason === 'size') {
    console.log(`BAD ${b.filename} size=${b.local} expected=${b.expected}`);
  } else {
    console.log(`BAD ${b.filename} hash=${b.local} expected=${b.expected}`);
  }
}

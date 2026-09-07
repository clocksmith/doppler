import fs from 'node:fs/promises';
import { hash } from '../../src/storage/blake3.js';
const root = 'models/local/create-gemma-3-1b-qualification/';
const manifest = JSON.parse(await fs.readFile(root+'manifest.json','utf8'));
const rows=[];
for(const shard of manifest.shards) {
  const bytes = await fs.readFile(root+shard.filename);
  const actual = Buffer.from(await hash(bytes)).toString('hex');
  if(bytes.length !== shard.size || actual !== shard.blake3) throw new Error(`Shard integrity mismatch: ${shard.filename}`);
  rows.push({filename:shard.filename,bytes:bytes.length,blake3:actual});
  console.log(`verified ${shard.filename}`);
}
await fs.writeFile('artifacts/create-generation-qualification/shard-verification.json',JSON.stringify({capturedAt:new Date().toISOString(),algorithm:'blake3',rows},null,2)+'\n');

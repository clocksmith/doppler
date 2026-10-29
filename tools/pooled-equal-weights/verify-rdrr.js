import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
const [canonicalDir, modelDir, reportPath] = process.argv.slice(2);
const canonical = JSON.parse(await fs.readFile(path.join(canonicalDir, 'canonical.json'), 'utf8'));
if (!canonical.completed) throw new Error('Canonical export is incomplete');
const model = JSON.parse(await fs.readFile(path.join(modelDir, 'manifest.json'), 'utf8'));
const names = new Set(Object.keys(model.tensors));
const checks = [];
for (const expected of canonical.tensors) {
  const tensor = model.tensors[expected.name];
  if (!tensor || tensor.dtype !== 'F32' || JSON.stringify(tensor.shape) !== JSON.stringify(expected.shape)) {
    throw new Error(`Canonical tensor missing or changed: ${expected.name}`);
  }
  const hash = createHash('sha256'); let size = 0;
  for (const span of tensor.spans ?? [{ shardIndex: tensor.shard, offset: tensor.offset, size: tensor.size }]) {
    const shard = model.shards.find(x => x.index === span.shardIndex);
    const handle = await fs.open(path.join(modelDir, shard.filename));
    try {
      const buffer = Buffer.alloc(Math.min(span.size, 4 << 20));
      for (let off = 0; off < span.size;) {
        const { bytesRead } = await handle.read(buffer, 0, Math.min(buffer.length, span.size - off), span.offset + off);
        if (!bytesRead) throw new Error('Truncated converted tensor');
        hash.update(buffer.subarray(0, bytesRead)); off += bytesRead; size += bytesRead;
      }
    } finally { await handle.close(); }
  }
  const actual = hash.digest('hex');
  checks.push({ name: expected.name, bytes: size, expected: expected.canonicalBytesSha256, actual,
    pass: actual === expected.canonicalBytesSha256 && size === expected.bytes });
  names.delete(expected.name);
}
const report = { schema: 'doppler.canonical-rdrr-check.v1', scope: 'Exact F32 artifact bytes before engine materialization',
  checks, unexpectedTensors: [...names], pass: names.size === 0 && checks.every(x => x.pass) };
await fs.writeFile(reportPath, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ tensors: checks.length, pass: report.pass, unexpected: [...names] }));
if (!report.pass) process.exitCode = 1;

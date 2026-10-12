// Isolate one shader delta against the immutable state candidate, never promotion.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
const [archive, output] = process.argv.slice(2).map(value => resolve(value));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const parentArchiveSha256 = hash(await readFile(archive));
assert.equal(parentArchiveSha256, 'd0d53e8f6d72ef7de63433d5935b2e89a717d3b4d6e8d890bd19add499e5e85e');
const shader = await readFile(new URL('./q4-prefill-residual-candidate.wgsl', import.meta.url));
assert.equal(hash(shader), '5f54bbf28895cc3e489e3da43c7add98538fd6a6269dc2a5cdcde48843a569cb');
await mkdir(output);
execFileSync('tar', ['-xzf', archive, '-C', output]);
const root = join(output, 'package');
await writeFile(join(root, 'src/gpu/kernels/fused_matmul_q4_widetile.wgsl'), shader);
const packed = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output],
  { cwd: root, encoding: 'utf8' }))[0];
const record = { qualified: false, parentArchiveSha256,
  archiveSha256: hash(await readFile(join(output, packed.filename))), shaderSha256: hash(shader),
  onlyRuntimeFileChanged: 'src/gpu/kernels/fused_matmul_q4_widetile.wgsl',
  scope: 'Single prefill reduction experiment; unchanged two-F32 recurrent state, other operators and output storage' };
await writeFile(join(output, 'archive-binding.json'), JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify(record));

// Reuse the retained frequency experiment as one explicit diagnostic delta.
// Its known wider-range accuracy limitation remains; this is not promotion.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { resolve, join } from 'node:path';
const [archive, output] = process.argv.slice(2).map(value => resolve(value));
const hash = value => createHash('sha256').update(value).digest('hex');
assert.equal(hash(await readFile(archive)), '91b7c94d858f6fa44bb4b5208f5e48b0a95fb51f79806a802cf15f7b975e92bd');
const previous = new URL('../source-prefix-correction-20261012/', import.meta.url);
const oldInput = await readFile(new URL('frequency-precompute-input.wgsl', previous), 'utf8');
const oldCandidate = await readFile(new URL('candidate-frequency-residual-rope.wgsl', previous), 'utf8');
assert.equal(hash(oldCandidate), '217fcc8c7d5018d27f18ab529bf648ee68ed4d55857eb1bb8d57be802c948dc5');
const extentABI = source => source.replace('    _pad2: u32,', '    position_count: u32,')
  .replace('let count = u.max_seq_len * half_dim;', 'let count = u.position_count * half_dim;');
const source = extentABI(oldCandidate);
await mkdir(output); execFileSync('tar', ['-xzf', archive, '-C', output]);
const root = join(output, 'package'), target = join(root, 'src/gpu/kernels/rope_precompute.wgsl');
assert.equal(await readFile(target, 'utf8'), extentABI(oldInput), 'Parent arithmetic differs from retained frequency baseline');
await writeFile(target, source);
const packed = JSON.parse(execFileSync('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', output],
  { cwd: root, encoding: 'utf8' }))[0];
const binding = { qualified: false, parentArchiveSha256: hash(await readFile(archive)),
  archiveSha256: hash(await readFile(join(output, packed.filename))), shaderSha256: hash(source),
  reusedCandidateSha256: hash(oldCandidate), onlyRuntimeFileChanged: 'src/gpu/kernels/rope_precompute.wgsl',
  declaredIntervention: 'Retained compensated frequency calculation; only later position_count ABI propagated',
  scope: 'First eight fixed prefixes, theta10000000, unscaled, existing independently checked positions0–255. Wider-range caveat remains.' };
await writeFile(join(output, 'archive-binding.json'), JSON.stringify(binding, null, 2) + '\n');
console.log(JSON.stringify(binding));

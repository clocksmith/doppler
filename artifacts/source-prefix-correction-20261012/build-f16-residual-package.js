// One-change diagnostic archive, never an automatic promotion path.
import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, readdir, symlink } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, join } from 'node:path';
const [archive, model, output] = process.argv.slice(2).map(value => resolve(value));
const hash = value => createHash('sha256').update(value).digest('hex');
const parentHash = hash(await readFile(archive));
assert.equal(parentHash, 'c596ecab6d498b3bb6e4d070e65a640713e693020e77c897271481f197875aae');
await mkdir(output);
await promisify(execFile)('tar', ['-xzf', archive, '-C', output]);
const packageRoot = join(output, 'package');
const shader = await readFile(new URL('./candidate-f16-residual-tiled.wgsl', import.meta.url));
assert.equal(hash(shader), 'd99cbd20a752bae55318e271864ade0d9ed2178c90b9294f913154e2c2aba22f');
const shaderPath = 'src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl';
assert.equal(hash(await readFile(join(packageRoot, shaderPath))), '31b2f8d9623d77df223f12da61eb5a178209fa1ce43199fe94357d7791c2b3b5');
await writeFile(join(packageRoot, shaderPath), shader);
const modelRoot = join(output, 'model');
await mkdir(modelRoot);
for (const name of await readdir(model)) {
  if (name !== 'manifest.json') await symlink(join(model, name), join(modelRoot, name));
}
const manifest = JSON.parse(await readFile(join(model, 'manifest.json'), 'utf8'));
const updatedRefs = [];
for (const [id, kernel] of Object.entries(manifest.inference.execution.kernels)) {
  if (kernel.kernel !== 'matmul_f16w_f32a_tiled.wgsl') continue;
  kernel.digest = 'sha256:' + hash(shader.toString('utf8').replaceAll('\r\n', '\n') + '\n@@entry:' + kernel.entry);
  updatedRefs.push(id);
}
const manifestBytes = JSON.stringify(manifest, null, 2) + '\n';
await writeFile(join(modelRoot, 'manifest.json'), manifestBytes);
const pack = await promisify(execFile)('npm', ['pack', '--ignore-scripts', '--pack-destination', output], { cwd: packageRoot });
await writeFile(join(output, 'pack.log'), pack.stdout + pack.stderr);
const file = pack.stdout.trim().split('\n').at(-1);
const binding = { archiveSha256: hash(await readFile(join(output, file))),
  manifestSha256: hash(manifestBytes), sourceManifestSha256: '18fe30e2c65c47bfdaa1f6714ced9fa90ed8c482a8d57b95cf0cdd4490e842e9',
  parentArchiveSha256: parentHash, shaderSha256: hash(shader), updatedRefs, unchangedWeights: true,
  tolerance: 0.001, candidate: 'frequency-parent-plus-f16-tiled-residual', scope: 'Diagnostic only; no canonical arithmetic or reference change' };
await writeFile(join(output, 'binding.json'), JSON.stringify(binding, null, 2) + '\n');
console.log(JSON.stringify(binding));

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { KERNEL_REF_CONTENT_DIGESTS } from '../../src/config/kernels/kernel-ref-digests.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-candidate-digests-'));
const run = (...args) => spawnSync(process.execPath, ['tools/sync-conversion-kernel-digests.js', ...args], { encoding: 'utf8' });
try {
  const candidate = path.join(root, 'candidate.json');
  const previous = path.join(root, 'previous.json');
  const value = { execution: { kernels: { projection: {
    kernel: 'fused_matmul_q4.wgsl', entry: 'main_gemv', digest: `sha256:${'0'.repeat(64)}`,
  } } } };
  const original = JSON.stringify(value);
  await fs.writeFile(candidate, original);
  await fs.writeFile(previous, original);
  assert.notEqual(run('--file', candidate, '--check').status, 0);
  assert.equal(await fs.readFile(candidate, 'utf8'), original, 'checking must not mutate recipes');
  const sync = run('--file', candidate);
  assert.equal(sync.status, 0, sync.stderr);
  assert.equal(await fs.readFile(previous, 'utf8'), original, 'selected sync must not rewrite adjacent historical candidates');
  assert.equal(JSON.parse(await fs.readFile(candidate, 'utf8')).execution.kernels.projection.digest,
    `sha256:${KERNEL_REF_CONTENT_DIGESTS['fused_matmul_q4.wgsl#main_gemv']}`);
  assert.equal(run('--file', candidate, '--check').status, 0);
  assert.notEqual(run('--file').status, 0);
  assert.notEqual(run('--unknown').status, 0);
  await fs.writeFile(candidate, '{');
  assert.notEqual(run('--file', candidate, '--check').status, 0, 'selected malformed inputs must not report success');

  const packageRoot = path.join(root, 'installed');
  for (const directory of ['tools', 'src/config/kernels', 'src/config/conversion', 'src/config/source-packages', 'models/local']) {
    await fs.mkdir(path.join(packageRoot, directory), { recursive: true });
  }
  await fs.copyFile('tools/sync-conversion-kernel-digests.js', path.join(packageRoot, 'tools/sync-conversion-kernel-digests.js'));
  await fs.copyFile('src/config/kernels/kernel-ref-digests.js', path.join(packageRoot, 'src/config/kernels/kernel-ref-digests.js'));
  await fs.writeFile(path.join(packageRoot, 'src/config/kernels/registry.json'), JSON.stringify({
    operations: { matmul: { variants: { fixture: { wgsl: 'fused_matmul_q4.wgsl', entryPoint: 'main_gemv' } } } },
  }));
  await fs.mkdir(path.join(packageRoot, 'src/gpu/kernels'), { recursive: true });
  await fs.copyFile('src/gpu/kernels/fused_matmul_q4.wgsl', path.join(packageRoot, 'src/gpu/kernels/fused_matmul_q4.wgsl'));
  await fs.writeFile(path.join(packageRoot, 'package.json'), '{"type":"module"}');
  const recipe = path.join(packageRoot, 'src/config/conversion/recipe.json');
  const retainedManifest = path.join(packageRoot, 'models/local/manifest.json');
  await fs.writeFile(recipe, original);
  await fs.writeFile(retainedManifest, original);
  const rejectedPackage = run('--package-root', packageRoot, '--check');
  assert.notEqual(rejectedPackage.status, 0);
  assert.match(rejectedPackage.stderr, /recipe\.json.*fused_matmul_q4\.wgsl#main_gemv/);
  assert.equal(await fs.readFile(recipe, 'utf8'), original, 'package validation cannot repair installed bytes');
  const repair = spawnSync(process.execPath, ['tools/sync-conversion-kernel-digests.js', '--source-only'], {
    cwd: packageRoot, encoding: 'utf8',
  });
  assert.equal(repair.status, 0, repair.stderr);
  assert.equal(await fs.readFile(retainedManifest, 'utf8'), original, 'source sync must preserve retained model manifests');
  const acceptedPackage = run('--package-root', packageRoot, '--check');
  assert.equal(acceptedPackage.status, 0, acceptedPackage.stderr);
  const shader = path.join(packageRoot, 'src/gpu/kernels/fused_matmul_q4.wgsl');
  await fs.appendFile(shader, '\n// altered installed bytes\n');
  assert.notEqual(run('--package-root', packageRoot, '--check').status, 0,
    'installed shader bytes, not the repository digest mirror, control validation');
  await fs.copyFile('src/gpu/kernels/fused_matmul_q4.wgsl', shader);
  for (const args of [
    ['--package-root'], ['--package-root', packageRoot],
    ['--package-root', packageRoot, '--file', recipe, '--check'],
    ['--source-only', '--file', recipe],
  ]) assert.notEqual(run(...args).status, 0, JSON.stringify(args));
  await fs.writeFile(recipe, '{');
  assert.notEqual(run('--package-root', packageRoot, '--check').status, 0, 'malformed packaged config must fail');
  await fs.rm(recipe);
  await fs.rm(path.join(packageRoot, 'src/config/source-packages'), { recursive: true });
  assert.notEqual(run('--package-root', packageRoot, '--check').status, 0, 'missing package config directory must fail');
} finally {
  await fs.rm(root, { recursive: true, force: true });
}
console.log('conversion-digest-selection.test: ok');

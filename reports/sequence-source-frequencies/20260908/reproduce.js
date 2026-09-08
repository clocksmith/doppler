import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { spawnSync, execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../../..');
const args = process.argv.slice(2);
assert.equal(args.length, 6, 'Usage: node reproduce.js --source <checkpoint-directory> --out <new-directory> --browser <Chromium-executable>');
assert.deepEqual([args[0], args[2], args[4]], ['--source', '--out', '--browser']);
const source = path.resolve(args[1]);
const output = path.resolve(args[3]);
const browser = path.resolve(args[5]);
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const hashFile = async file => `sha256:${createHash('sha256').update(await fs.readFile(file)).digest('hex')}`;
assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: root, encoding: 'utf8' }).trim(), '', 'Reproduction requires a clean checkout.');
const evidence = await read(path.join(directory, 'evidence.json'));
for (const artifact of evidence.artifacts) {
  assert.equal(await hashFile(path.join(directory, artifact.path)), artifact.hash, artifact.path);
}
const reference = await read(path.join(directory, 'retained-control/full-reference.json'));
for (const [file, digest] of Object.entries(reference.source.files)) {
  assert.equal(await hashFile(path.join(source, file)), digest, file);
}
await fs.mkdir(output);
const run = (script, argv) => {
  const result = spawnSync(process.execPath, [path.join(root, script), ...argv], {
    cwd: root, env: process.env, stdio: 'inherit',
  });
  assert.equal(result.status, 0, `${script} failed: ${result.error?.message ?? result.signal ?? result.status}`);
};
const converted = path.join(output, 'converted');
run('tools/convert-safetensors-node.js', [source, '--config',
  'src/config/conversion/esm/esm2-t12-35m-ur50d-f32-af32.json', '--output-dir', converted]);
const policy = await read(path.join(directory, 'policy.json'));
for (const item of policy.cases) {
  const config = { modelDir: converted, referencePath: path.join(directory, item.id, 'full-reference.json'),
    outputPath: path.join(output, `${item.id}-qualification.json`), browserExecutablePath: browser,
    browserArgs: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
      '--disable-dawn-features=disallow_unsafe_apis', '--ignore-gpu-blocklist', '--use-angle=vulkan',
      '--enable-features=Vulkan', '--disable-vulkan-surface', '--enable-precise-memory-info'], timeoutMs: 300000 };
  const configPath = path.join(output, `${item.id}-config.json`);
  await fs.writeFile(configPath, JSON.stringify(config, null, 2) + '\n', { flag: 'wx' });
  run('tools/qualify-sequence-model-browser.js', ['--config', configPath]);
}

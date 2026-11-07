#!/usr/bin/env node
// Preparer only. Never include this file or the report directory in a blinded packet.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const [repository, revision, defectRevision, assetDirectory, destination, workflow] = process.argv.slice(2);
if (![repository, revision, defectRevision, assetDirectory, destination].every(Boolean)
    || !['standard', 'inspection'].includes(workflow)) {
  throw new Error('Usage: prepare-pilot.js <repo> <runtime-revision> <defect-revision> <model-dir> <new-packet-dir> <standard|inspection>');
}
const root = path.resolve(destination);
const assets = path.resolve(assetDirectory);
if (process.platform !== 'darwin' || !root.startsWith('/private/tmp/')) {
  throw new Error('This Metal pilot requires macOS and a new packet under /private/tmp/.');
}
await fs.mkdir(root); // Refuse to overwrite an existing packet or observations.
const command = (file, args, cwd = root) => execFileSync(file, args, { cwd, encoding: 'utf8' });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const archive = path.join(root, '.source.tar.gz');
execFileSync('git', ['archive', '--format=tar.gz', `--output=${archive}`, revision, 'package.json', 'package-lock.json',
  'AGENTS.md', 'CATSCAN.md', 'GOALS.md', 'INTENT.md', 'LICENSE', 'NOTICE', 'src', 'tests', 'tools', 'docs'],
{ cwd: repository });
command('tar', ['-xzf', archive]);
await fs.unlink(archive);
// Preserve normative/API docs; omit historical investigation narratives from both arms.
for (const directory of ['docs/status', 'docs/perf-investigations', 'docs/evidence']) {
  await fs.rm(path.join(root, directory), { recursive: true, force: true });
}
const shaderPath = 'src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl';
const defect = execFileSync('git', ['show', `${defectRevision}:${shaderPath}`], { cwd: repository });
await fs.writeFile(path.join(root, shaderPath), defect);
await fs.mkdir(path.join(root, 'bin'));
command('cp', ['-c', process.execPath, path.join(root, 'bin/node')]);
for (const dependency of ['webgpu', 'debug', 'ms', '@webgpu/types']) {
  const target = path.join(root, 'node_modules', dependency);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await fs.cp(path.join(repository, 'node_modules', dependency), target, { recursive: true, dereference: true });
}
await fs.mkdir(path.join(root, 'model'));
for (const name of await fs.readdir(assets)) {
  const source = path.join(assets, name);
  if (!(await fs.stat(source)).isFile()) throw new Error(`Unexpected model entry: ${name}`);
  command('cp', ['-c', source, path.join(root, 'model', name)]);
}
const node = path.join(root, 'bin/node');
command(node, ['tools/sync-kernel-ref-digests.js']);
command(node, ['tools/sync-conversion-kernel-digests.js', '--source-only']);
command(node, ['tools/refresh-converted-manifest.js', 'model', '--config',
  'src/config/conversion/gemma3/gemma-3-270m-it-f16-af32.json']);
await fs.mkdir(path.join(root, 'case'));
await fs.mkdir(path.join(root, 'observations'));
await fs.copyFile(new URL('./command.py', import.meta.url), path.join(root, 'case/command.py'));
await fs.copyFile(new URL('./application-control.json', import.meta.url), path.join(root, 'case/reference.json'));
const instructions = await fs.readFile(new URL('./packet-instructions.md', import.meta.url), 'utf8');
const workflows = JSON.parse(await fs.readFile(new URL('./workflows.json', import.meta.url)));
await fs.writeFile(path.join(root, 'case/START.md'), instructions
  .replaceAll('ABSOLUTE_PACKET', root).replace('WORKFLOW_TEXT', workflows[workflow]));
const deniedPaths = [path.resolve(repository), ...['.codex', '.agents', '.claude', '.ssh']
  .map(name => path.join(os.homedir(), name))];
await fs.writeFile(path.join(root, 'case/offline.sb'), `(version 1)
(allow default)
(deny network*)
(deny file-read* file-write* ${deniedPaths.map(value => `(subpath ${JSON.stringify(value)})`).join(' ')})
(deny file-read* file-write* (require-all (subpath "/private/tmp") (require-not (subpath ${JSON.stringify(root)}))))
(allow file-read-metadata (literal "/private/tmp"))
`);
const workload = {
  schema: 'doppler.debugging-pilot-workload/v1',
  prompt: 'The capital of France is',
  generation: { maxTokens: 12, temperature: 0, topK: 1, useChatTemplate: false },
};
await fs.writeFile(path.join(root, 'case/workload.json'), JSON.stringify(workload, null, 2) + '\n');
await fs.writeFile(path.join(root, 'case/run.js'), `import fs from 'node:fs/promises';
import { load } from '../src/index.js';
import { destroyDevice } from '../src/gpu/device.js';
import { releaseNodeWebGPU } from '../src/tooling/node-webgpu.js';
const workload=JSON.parse(await fs.readFile(new URL('./workload.json',import.meta.url)));
let model;
try {
  model=await load({url:new URL('../model/',import.meta.url).href},{isolatedLoader:true});
  const result=await model.generateWithEvidence(workload.prompt,workload.generation);
  await fs.writeFile(new URL('../observations/application.json',import.meta.url),JSON.stringify(result,null,2)+'\\n');
  console.log(JSON.stringify({outputText:result.outputText,tokenIds:result.tokenIds}));
} finally { await model?.unload(); destroyDevice(); releaseNodeWebGPU(); }
`);
const manifest = JSON.parse(await fs.readFile(path.join(root, 'model/manifest.json')));
const evidence = { schema: 'doppler.debugging-pilot-preparation/v1', runtimeRevision: revision,
  defectRevision, shaderPath, shaderSha256: hash(defect), preparationIsNotTrial: true,
  modifiedProgram: 'Current runtime with exactly the historical defective shader; generated kernel pins and diagnostic manifest refreshed honestly. No signed Capsule or historical whole-release claim.',
  originalModelManifestSha256: hash(await fs.readFile(path.join(assets, 'manifest.json'))),
  diagnosticManifestSha256: hash(await fs.readFile(path.join(root, 'model/manifest.json'))),
  modelId: manifest.modelId, nodeVersion: process.version, nodeSha256: hash(await fs.readFile(node)) };
console.log(JSON.stringify(evidence,null,2));

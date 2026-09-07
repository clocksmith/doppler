#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createReadStream, createWriteStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawn } from 'node:child_process';
import path from 'node:path';

// Reconstruct signed artifact bytes; never sign, promote, or change a model plan.
const [bundleArgument, outputArgument] = process.argv.slice(2);
if (!bundleArgument || !outputArgument || process.argv.length !== 4) {
  throw new Error('Usage: node tools/restore-capsule-baseline.js <extracted-bundle> <new-output-directory>');
}
const bundle = path.resolve(bundleArgument);
const output = path.resolve(outputArgument);
const read = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const recipe = await read(path.join(bundle, 'reproduction.json'));
assert.equal(recipe.schema, 'doppler.capsule-baseline-reproduction/v1');
function inside(root, relative) {
  assert.equal(typeof relative, 'string');
  const filename = path.resolve(root, relative);
  const resolved = path.relative(root, filename);
  assert(resolved && !resolved.startsWith('..') && !path.isAbsolute(resolved), 'Bundle paths must remain inside their root.');
  return filename;
}
async function hash(filename) {
  const digest = createHash('sha256');
  for await (const bytes of createReadStream(filename)) digest.update(bytes);
  return digest.digest('hex');
}
async function verify(filename, identity) {
  assert.equal((await fs.stat(filename)).size, identity.sizeBytes, `Size mismatch: ${filename}`);
  assert.equal(await hash(filename), identity.sha256, `SHA-256 mismatch: ${filename}`);
}
async function command(executable, args, name, cwd = output) {
  const log = await fs.open(path.join(output, `${name}.log`), 'wx');
  report.commands.push({ executable, args, cwd, log: `${name}.log` });
  try {
    await new Promise((resolve, reject) => {
      const child = spawn(executable, args, { cwd, stdio: ['ignore', log.fd, log.fd] });
      child.once('error', reject);
      child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${name} failed: ${code ?? signal}`)));
    });
  } finally { await log.close(); }
}
await fs.mkdir(output);
await fs.copyFile(path.join(bundle, 'reproduction.json'), path.join(output, 'reproduction-input.json'), 1);
const report = { schema: 'doppler.capsule-baseline-restoration/v1', passed: false,
  recipeSha256: await hash(path.join(bundle, 'reproduction.json')), nodeVersion: process.version,
  startedAtUtc: new Date().toISOString(), commands: [], downloads: [], artifacts: [], physicalExecution: false };
try {
  for (const file of recipe.files) await verify(inside(bundle, file.path), file);
  await fs.copyFile(inside(bundle, recipe.runtimeArchive), path.join(output, path.basename(recipe.runtimeArchive)), 1);
  await fs.copyFile(inside(bundle, recipe.runtimeReceipt), path.join(output, 'receipt.json'), 1);
  await fs.cp(path.join(bundle, 'retained'), path.join(output, 'retained'), { recursive: true, errorOnExist: true });
  const consumer = path.join(output, 'consumer');
  await fs.mkdir(consumer);
  await fs.writeFile(path.join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }), { flag: 'wx' });
  const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  await command(npm, ['install', inside(bundle, recipe.runtimeArchive), '--ignore-scripts', '--omit=optional',
    '--offline', '--no-audit', '--no-fund'], 'install-runtime', consumer);
  const installed = path.join(consumer, 'node_modules/doppler-gpu');
  for (const source of recipe.sources) {
    assert(/^[a-f0-9]{40}$/.test(source.revision), 'Immutable upstream revision required.');
    assert(/^[\w.-]+\/[\w.-]+$/.test(source.repository), 'Explicit upstream repository required.');
    const sourceDir = inside(output, `sources/${source.id}`);
    await fs.mkdir(sourceDir, { recursive: true });
    for (const file of source.files) {
      const filename = inside(sourceDir, file.path);
      await fs.mkdir(path.dirname(filename), { recursive: true });
      const url = `https://huggingface.co/${source.repository}/resolve/${source.revision}/${file.path.split('/').map(encodeURIComponent).join('/')}`;
      console.log(JSON.stringify({ stage: 'download', source: source.id, file: file.path }));
      const response = await fetch(url, { signal: AbortSignal.timeout(recipe.downloadTimeoutMs) });
      assert(response.ok && response.body, `Source download failed: ${response.status} ${url}`);
      await pipeline(Readable.fromWeb(response.body), createWriteStream(filename, { flags: 'wx' }));
      await verify(filename, file);
      report.downloads.push({ url, ...file });
    }
    const converted = inside(output, `converted/${source.id}`);
    console.log(JSON.stringify({ stage: 'conversion', source: source.id }));
    await command(process.execPath, [path.join(installed, 'tools/convert-safetensors-node.js'), sourceDir,
      '--config', inside(bundle, source.conversionConfig), '--output-dir', converted], `convert-${source.id}`);
    for (const destination of source.distributions) {
      const distribution = inside(output, `retained/${destination}`);
      const capsulePath = path.join(distribution, 'capsule-v3.json');
      const capsule = await read(capsulePath);
      for (const artifact of capsule.artifacts) {
        const target = inside(distribution, artifact.path);
        if (/^artifacts\/model\/shard_\d+\.bin$/.test(artifact.path)) {
          const generated = path.join(converted, path.basename(artifact.path));
          await verify(generated, { sizeBytes: artifact.sizeBytes, sha256: artifact.hash.slice(7) });
          await fs.mkdir(path.dirname(target), { recursive: true });
          await fs.copyFile(generated, target, 1);
        }
        await verify(target, { sizeBytes: artifact.sizeBytes, sha256: artifact.hash.slice(7) });
        report.artifacts.push({ distribution: destination, path: artifact.path, hash: artifact.hash, sizeBytes: artifact.sizeBytes });
      }
    }
  }
  report.passed = true;
} catch (error) {
  report.error = { message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  report.completedAtUtc = new Date().toISOString();
  await fs.writeFile(path.join(output, 'restoration.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}
console.log(JSON.stringify({ passed: report.passed, output, error: report.error?.message }));

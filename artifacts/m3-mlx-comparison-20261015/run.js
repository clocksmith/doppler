// From the repository root, after preparing the pinned source and converted artifacts.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { compare } from './compare.js';

const root = path.dirname(fileURLToPath(import.meta.url));
const read = async name => JSON.parse(await fs.readFile(path.join(root, name)));
const policy = await read('natural-eos-policy.json');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const resumePackedOnly = process.argv.includes('--packed-only');
const metadata = resumePackedOnly ? await read('environment.json') : {
  startedAt: new Date().toISOString(),
  commit: execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim(),
  diffSha256: hash(execFileSync('git', ['diff'])),
  hardware: execFileSync('sysctl', ['-n', 'machdep.cpu.brand_string', 'hw.memsize'], {encoding: 'utf8'}).trim(),
  os: execFileSync('sw_vers', [], {encoding: 'utf8'}).trim(),
  scripts: {}, runs: [],
  memoryCaveat: 'MLX allocator bytes and Doppler pool bytes have different coverage. Never subtract or compare them as total system residency.',
};
for (const name of ['doppler.js', 'mlx-reference.py', 'compare.js', 'run.js']) {
  metadata.scripts[name] = hash(await fs.readFile(path.join(root, name)));
}
await fs.writeFile(path.join(root, 'environment.json'), JSON.stringify(metadata, null, 2));
async function run(engine, policyName, resultName, modelDir) {
  const command = engine === 'mlx' ? '.cache/mlx-comparison/bin/python' : process.execPath;
  const script = engine === 'mlx' ? 'mlx-reference.py' : 'doppler.js';
  const args = [path.join(root, script), modelDir, path.join(root, 'inputs.json'),
    path.join(root, `${resultName}.json`), path.join(root, policyName)];
  const log = await fs.open(path.join(root, `${resultName}.log`), 'w');
  const startedAt = new Date().toISOString();
  let timedOut = false;
  try {
    const child = spawn(command, args, {stdio: ['ignore', log.fd, log.fd]});
    const timer = setTimeout(() => {timedOut = true; child.kill('SIGKILL');}, policy.shared.maxProcessSeconds * 1000);
    const exit = await new Promise((resolve, reject) => {
      child.once('error', reject);
      child.once('close', (code, signal) => resolve({code, signal}));
    }).finally(() => clearTimeout(timer));
    metadata.runs.push({engine, command, args, startedAt, endedAt: new Date().toISOString(), timedOut, ...exit});
    await fs.writeFile(path.join(root, 'environment.json'), JSON.stringify(metadata, null, 2));
    if (timedOut || exit.code !== 0) throw new Error(`${resultName} failed: ${JSON.stringify(exit)}; see retained log`);
    console.log(`${resultName}: complete`);
  } finally {await log.close();}
}
if (!resumePackedOnly) {
  await run('mlx', 'natural-eos-policy.json', 'mlx-natural', '.cache/mlx-gemma270m-source');
  await run('doppler', 'natural-eos-policy.json', 'doppler-natural', '.cache/mlx-gemma270m-rdrr');
  await fs.writeFile(path.join(root, 'comparison.json'), JSON.stringify(compare(await read('mlx-natural.json'), await read('doppler-natural.json')), null, 2));
  await run('doppler', 'batch4-policy.json', 'doppler-batch4', '.cache/mlx-gemma270m-rdrr');
  await fs.writeFile(path.join(root, 'batch4-comparison.json'), JSON.stringify(compare(await read('mlx-natural.json'), await read('doppler-batch4.json')), null, 2));
}
await run('doppler', 'packed-policy.json', 'doppler-packed', '.cache/mlx-gemma270m-q4k');
await run('doppler', 'packed-candidate-policy.json', 'doppler-packed-candidate', '.cache/mlx-gemma270m-q4k');
const dense = await read('doppler-packed.json'), packed = await read('doppler-packed-candidate.json');
await fs.writeFile(path.join(root, 'packed-comparison.json'), JSON.stringify({
  ...compare(dense, packed),
  baselineRetainedModelBytes: dense.loadedMemory.pool.resources.retainedModel.bytes,
  candidateRetainedModelBytes: packed.loadedMemory.pool.resources.retainedModel.bytes,
  caveat: 'Same stored Q4K bytes; different materialization and kernels. Exact tokens on two synthetic probes do not establish per-layer numerical parity or recommendation quality.',
}, null, 2));

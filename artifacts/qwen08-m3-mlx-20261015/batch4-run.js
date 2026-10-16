// From the repository root, after preparing the pinned source and converted artifacts.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';

const root = path.dirname(fileURLToPath(import.meta.url));
const read = async name => JSON.parse(await fs.readFile(path.join(root, name)));
const policy = await read('batch4-policy.json');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const metadata = {
  startedAt: new Date().toISOString(),
  runtimeCommit: (await fs.readFile(path.join(root, 'runtime-commit.txt'), 'utf8')).trim(),
  runtimeSource: 'Committed snapshot in .cache/qwen08-runtime, excludes concurrent working-tree changes',
  diffSha256: hash(execFileSync('git', ['diff'])),
  hardware: execFileSync('sysctl', ['-n', 'machdep.cpu.brand_string', 'hw.memsize'], {encoding: 'utf8'}).trim(),
  os: execFileSync('sw_vers', [], {encoding: 'utf8'}).trim(),
  scripts: {}, runs: [],
  memoryCaveat: 'MLX allocator bytes and Doppler pool bytes have different coverage. Never subtract or compare them as total system residency.',
};
for (const name of ['doppler.js', 'mlx-reference.py', 'batch4-run.js', 'batch4-policy.json', 'conversion.json', 'compare.js']) {
  metadata.scripts[name] = hash(await fs.readFile(path.join(root, name)));
}
await fs.writeFile(path.join(root, 'batch4-environment.json'), JSON.stringify(metadata, null, 2));
async function run(engine, policyName, resultName, modelDir) {
  const command = engine === 'mlx' ? '.cache/mlx-comparison/bin/python' : process.execPath;
  const script = engine === 'mlx' ? 'mlx-reference.py' : 'doppler.js';
  const scriptPath = engine === 'mlx' ? path.join(root, script) : path.resolve('.cache/qwen08-runtime/artifacts/qwen08-m3-mlx-20261015/doppler.js');
  if (engine === 'doppler' && hash(await fs.readFile(scriptPath)) !== metadata.scripts['doppler.js']) throw new Error('Frozen script differs');
  const args = [scriptPath, modelDir, path.join(root, 'inputs.json'),
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
    await fs.writeFile(path.join(root, 'batch4-environment.json'), JSON.stringify(metadata, null, 2));
    if (timedOut || exit.code !== 0) throw new Error(`${resultName} failed: ${JSON.stringify(exit)}; see retained log`);
    console.log(`${resultName}: complete`);
  } finally {await log.close();}
}
await run('doppler', 'batch4-policy.json', 'doppler-batch4', path.resolve('.cache/qwen08-f16'));

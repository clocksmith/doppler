import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const packageJson = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8'));

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    encoding: 'utf8',
  });
  assert.equal(
    result.status,
    0,
    `${command} ${args.join(' ')} failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`
  );
  return result;
}

assert.equal(
  packageJson.scripts['bench:rerank:compare:hf'],
  'node tools/compare-rerankers-hf.js',
  'package script must expose the HF/PyTorch rerank compare lane'
);

run(process.execPath, ['--check', 'tools/compare-rerankers-hf.js']);
run('python3', [
  '-c',
  [
    'from pathlib import Path',
    "path = Path('benchmarks/runners/hf-transformers-reranker-bench.py')",
    "compile(path.read_text(), str(path), 'exec')",
  ].join('; '),
]);

const help = run(process.execPath, ['tools/compare-rerankers-hf.js', '--help']).stdout;
assert.match(help, /--hf-device <device>/);
assert.match(help, /local_non_webgpu_baseline|HF Transformers model/);

const runnerText = fs.readFileSync(
  path.join(repoRoot, 'benchmarks/runners/hf-transformers-reranker-bench.py'),
  'utf8'
);
assert.match(runnerText, /local_files_only/);
assert.match(runnerText, /AutoModelForCausalLM/);
assert.match(runnerText, /trueTokenId/);
assert.match(runnerText, /falseTokenId/);

console.log('hf-reranker-bench-contract.test: ok');

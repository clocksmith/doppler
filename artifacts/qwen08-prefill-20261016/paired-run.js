import fs from 'node:fs/promises';
import { spawn, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const folder = path.dirname(fileURLToPath(import.meta.url));
const plan = JSON.parse(await fs.readFile(path.join(folder, 'paired-plan.json')));
const reference = JSON.parse(await fs.readFile(path.join(folder, 'baseline.json')));
const receipt = { plan, complete: false, runs: [], source: {} };
for (const file of [plan.runner, '.cache/qwen08-20261016/src/gpu/kernels/attention/executors.js',
  '.cache/qwen08-20261016/src/config/kernels/registry.json', path.join(folder, 'runtime.patch')]) {
  receipt.source[file] = createHash('sha256').update(await fs.readFile(file)).digest('hex');
}
const save = () => fs.writeFile(path.join(folder, 'paired-receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
try {
  for (let pair = 0; pair < plan.pairs.length; pair++) {
    for (const name of plan.pairs[pair]) {
      const output = path.join(folder, `paired-${pair}-${name}.json`);
      const policyPath = path.join(folder, `paired-${name}-policy.json`);
      const log = await fs.open(output.replace(/\.json$/, '.log'), 'w');
      const startedAt = new Date().toISOString();
      const host = execFileSync('ps', ['-axo', 'pid,pcpu,comm'], { encoding: 'utf8' });
      await fs.writeFile(output.replace(/\.json$/, '.host.txt'), host);
      const args = [plan.runner, plan.model, plan.inputs, output, policyPath];
      try {
        const child = spawn(process.execPath, args, { stdio: ['ignore', log.fd, log.fd] });
        const timer = setTimeout(() => child.kill('SIGKILL'), plan.maxChildSeconds * 1000);
        const exit = await new Promise((resolve, reject) => {
          child.once('error', reject);
          child.once('close', (code, signal) => resolve({ code, signal }));
        }).finally(() => clearTimeout(timer));
        receipt.runs.push({ pair, name, output, startedAt, endedAt: new Date().toISOString(), ...exit });
        await save();
        assert.equal(exit.code, 0, `${name}: child failed`);
        const result = JSON.parse(await fs.readFile(output));
        assert.equal(result.complete, true);
        assert.equal(result.runs.length, 4);
        for (const run of result.runs) {
          const expected = reference.runs.find(r => r.promptTokenIds.length === run.promptTokenIds.length);
          assert.deepEqual(run.promptTokenIds, expected.promptTokenIds);
          assert.deepEqual(run.evidence.tokenIds, expected.evidence.tokenIds);
          assert.equal(run.tokenTimes.length, 128);
        }
        console.log(JSON.stringify({ pair, name, timings: result.runs.filter(r => !r.warmup).map(r => ({ prompt: r.promptTokenIds.length, totalMs: r.totalRunMs, firstMs: r.firstTokenMs })) }));
      } finally { await log.close(); }
    }
  }
  receipt.complete = true;
} catch (error) {
  receipt.error = { message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally { await save(); }

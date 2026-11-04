import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { assessGpuBenchmarkMemory } from './gpu-memory-budget.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'gpu-memory-budget-'));
try {
  const evidence = Buffer.from('retained workload allocation observation\n');
  await fs.writeFile(path.join(root, 'evidence.json'), evidence);
  await fs.writeFile(path.join(root, 'mem_info_gtt_total'), '1000\n');
  await fs.writeFile(path.join(root, 'mem_info_gtt_used'), '100\n');
  const reference = { path: 'evidence.json', sha256: createHash('sha256').update(evidence).digest('hex') };
  const plan = { schema: 'doppler.gpu-benchmark-memory-plan/v1', sysfsDevicePath: root,
    headroomBytes: 50, reservations: [{ id: 'paired-engines', bytes: 650, evidence: reference }] };
  const admitted = await assessGpuBenchmarkMemory(plan, root);
  assert.equal(admitted.admitted, true);
  assert.equal(admitted.projectedBytes, 800);
  const exact = await assessGpuBenchmarkMemory({ ...plan,
    reservations: [{ ...plan.reservations[0], bytes: 850 }] }, root);
  assert.equal(exact.admitted, true);
  assert.equal(exact.remainingBytes, 0);
  const oversubscribed = { ...plan, reservations: [...plan.reservations,
    { id: 'additional-model', bytes: 400, evidence: reference }] };
  assert.equal((await assessGpuBenchmarkMemory(oversubscribed, root)).admitted, false);
  for (const invalid of [
    { ...plan, headroomBytes: undefined }, { ...plan, headroomBytes: -1 },
    { ...plan, headroomBytes: Number.MAX_SAFE_INTEGER },
    { ...plan, reservations: [...plan.reservations, plan.reservations[0]] },
    { ...plan, reservations: [{ ...plan.reservations[0], bytes: NaN }] },
    { ...plan, reservations: [{ ...plan.reservations[0], evidence: { ...reference, sha256: '0'.repeat(64) } }] },
  ]) await assert.rejects(assessGpuBenchmarkMemory(invalid, root));
  const input = path.join(root, 'plan.json'), output = path.join(root, 'admission.json');
  await fs.writeFile(input, JSON.stringify(oversubscribed));
  const cli = fileURLToPath(new URL('../probes/check-gpu-benchmark-budget.js', import.meta.url));
  const run = spawnSync(process.execPath, [cli, '--config', input, '--out', output], { encoding: 'utf8', cwd: os.tmpdir() });
  assert.equal(run.status, 2, run.stderr);
  assert.equal(JSON.parse(await fs.readFile(output, 'utf8')).admitted, false);
  const overwrite = spawnSync(process.execPath, [cli, '--config', input, '--out', output], { encoding: 'utf8' });
  assert.notEqual(overwrite.status, 0, 'Frozen admission receipts cannot be overwritten');
  await fs.writeFile(path.join(root, 'mem_info_gtt_used'), 'not a counter');
  await assert.rejects(assessGpuBenchmarkMemory(plan, root), /Invalid driver memory counter/);
  console.log('gpu-memory-budget.test: passed');
} finally { await fs.rm(root, { recursive: true, force: true }); }

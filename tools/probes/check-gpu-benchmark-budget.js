import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { assessGpuBenchmarkMemory } from '../pooled-equal-weights/gpu-memory-budget.js';

const options = {};
for (let index = 2; index < process.argv.length; index += 2) {
  const flag = process.argv[index], value = process.argv[index + 1];
  if (!['--config', '--out'].includes(flag) || !value || value.startsWith('--') || options[flag]) {
    throw new Error('Usage: node tools/probes/check-gpu-benchmark-budget.js --config <plan.json> --out <new-receipt.json>');
  }
  options[flag] = path.resolve(value);
}
if (!options['--config'] || !options['--out']) throw new Error('Both --config and --out are required.');
const raw = await fs.readFile(options['--config']);
const receipt = await assessGpuBenchmarkMemory(JSON.parse(raw.toString('utf8')), path.dirname(options['--config']));
receipt.planSha256 = createHash('sha256').update(raw).digest('hex');
await fs.mkdir(path.dirname(options['--out']), { recursive: true });
await fs.writeFile(options['--out'], JSON.stringify(receipt, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ admitted: receipt.admitted, projectedBytes: receipt.projectedBytes,
  capacityBytes: receipt.capacityBytes, receipt: options['--out'] }));
if (!receipt.admitted) process.exitCode = 2;

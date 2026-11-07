import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { installMoeRoutingObservation } from '../../tests/helpers/moe-routing-observation-loader.js';

// Observe real top-k buffers before importing the model pipeline. No replacement.
const path = process.env.DOPPLER_FLASH_NEXT_PROBE;
assert(path);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const runBytes = await fs.readFile(path), run = JSON.parse(runBytes);
assert.deepEqual(run.sequences, [{ id: 'long', maxTokens: 1 }]);
const url = new URL('../../src/inference/pipelines/text/moe-gpu.js', import.meta.url);
const rows = [];
const observation = installMoeRoutingObservation(url.href, hash(await fs.readFile(url)), async record => {
  assert.equal(record.numTokens, 2304);
  assert.equal(record.activationDtype, 'f32');
  const { readBuffer } = await import('../../src/memory/buffer-pool.js');
  const row = { layer: record.layer };
  for (const [name, buffer, suffix] of [['indices', record.indicesBuffer, 'u32'], ['weights', record.weightsBuffer, 'f32']]) {
    const bytes = Buffer.from(await readBuffer(buffer, record.numTokens * record.topK * 4));
    const file = `${run.outputPath}.layer${record.layer}.${name}.${suffix}`;
    await fs.writeFile(file, bytes);
    row[name] = { file, shape: [record.numTokens, record.topK], sha256: hash(bytes) };
  }
  rows.push(row);
});
let completed = false, failure;
try {
  await import('../../tests/integration/flash-next-learned-physical.test.js');
  assert.equal(rows.length, 48);
  completed = true;
} catch (error) {
  failure = String(error);
  throw error;
} finally {
  observation.restore();
  await fs.writeFile(`${run.outputPath}.routing.json`, JSON.stringify({ scope: 'Actual GPU routing observation only',
    runSha256: hash(runBytes), completed, failure, instrumentation: observation.receipt, rows }, null, 2)+'\n');
}

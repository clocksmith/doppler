import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

// Paired CPU microbenchmark of the fresh dispatch-geometry wrappers observed
// in matmul selection. This measures uniform preparation, not GPU throughput.
const config = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
assert.equal(config.schema, 'doppler.uniform-validation-profile/v1');
for (const field of ['warmupPairs', 'measuredPairs', 'iterations']) {
  assert(Number.isSafeInteger(config[field]) && config[field] > 0, field);
}
const report = { config, startedAt: new Date().toISOString(), sources: {}, pairs: [], passed: false };
const lanes = {};
for (const name of ['baseline', 'candidate']) {
  const root = path.resolve(config[name]);
  const source = path.join(root, 'src/gpu/kernels/uniform-utils.js');
  report.sources[name] = createHash('sha256').update(await fs.readFile(source)).digest('hex');
  const api = await import(pathToFileURL(source));
  const { getKernelConfig } = await import(pathToFileURL(path.join(root, 'src/config/kernel-registry-contract.js')));
  lanes[name] = config.kernels.map(([operation, variant]) => {
    const kernel = getKernelConfig(operation, variant);
    const values = Object.fromEntries(kernel.uniforms.fields.filter(field => !field.padding)
      .map(field => [field.name, field.type === 'f32' ? field.offset + 0.125 : field.offset + 1]));
    const view = new DataView(new ArrayBuffer(kernel.uniforms.size));
    return { api, kernel, values, view };
  });
}
function measure(name) {
  let checksum = 0;
  const start = performance.now();
  for (let i = 0; i < config.iterations; i++) {
    for (const { api, kernel, values, view } of lanes[name]) {
      const wrapper = Object.freeze({ ...kernel, workgroupSize: Object.freeze([128, 1, 1]) });
      checksum += api.getUniformByteLength(wrapper);
      api.writeUniformsFromObject(view, wrapper, values);
    }
  }
  return { ms: performance.now() - start, checksum };
}
for (let pair = -config.warmupPairs; pair < config.measuredPairs; pair++) {
  const order = pair % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'];
  const row = { pair, order };
  for (const name of order) row[name] = measure(name);
  assert.equal(row.baseline.checksum, row.candidate.checksum);
  for (let index = 0; index < config.kernels.length; index++) {
    assert.deepEqual(new Uint8Array(lanes.baseline[index].view.buffer),
      new Uint8Array(lanes.candidate[index].view.buffer), 'uniform bytes must match exactly');
  }
  if (pair >= 0) report.pairs.push(row);
}
report.passed = true;
await fs.writeFile(process.argv[3], `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ passed: true, pairs: report.pairs.length, report: process.argv[3] }));

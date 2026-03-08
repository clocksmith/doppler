import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const schemaPath = path.resolve('benchmarks/benchmark-schema.json');
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));
const required = new Set(schema.required || []);

for (const field of ['schemaVersion', 'timestamp', 'suite', 'runType', 'env', 'model', 'config', 'workload', 'metrics']) {
  assert.equal(required.has(field), true, `expected benchmark schema to require ${field}`);
}

assert.equal(schema.properties.env.minProperties, 1);
assert.equal(schema.properties.model.minProperties, 1);
assert.equal(schema.properties.config.minProperties, 1);
assert.equal(schema.properties.workload.minProperties, 1);
assert.equal(schema.properties.metrics.minProperties, 1);

console.log('benchmark-schema-contract.test: ok');

// The benchmark result schema must be internally consistent, and the
// standalone result validator (tests/validate-benchmark-results.js), which
// keeps a hand-written type map, must agree with the schema it enforces.
import assert from 'node:assert/strict';
import fs from 'node:fs';

const schema = JSON.parse(fs.readFileSync('benchmarks/benchmark-schema.json', 'utf8'));

assert.equal(schema.type, 'object');
assert.ok(Array.isArray(schema.required) && schema.required.length > 0, 'schema must declare required fields');
assert.equal(new Set(schema.required).size, schema.required.length, 'required fields must be unique');
for (const field of schema.required) {
  const property = schema.properties?.[field];
  assert.ok(property, `required field ${field} must be declared in properties`);
  if (property.type === 'object') {
    assert.ok(
      Number.isInteger(property.minProperties) && property.minProperties >= 1,
      `required object ${field} must reject an empty object`
    );
  }
}

// JSON Schema type -> JavaScript typeof, as the validator checks it.
const TYPEOF_FOR_SCHEMA_TYPE = { integer: 'number', number: 'number', string: 'string', object: 'object', boolean: 'boolean' };
const validatorSource = fs.readFileSync('tests/validate-benchmark-results.js', 'utf8');
const typeMapSource = /const REQUIRED_TYPE_MAP = (\{[\s\S]*?\});/.exec(validatorSource)?.[1];
assert.ok(typeMapSource, 'validator must declare REQUIRED_TYPE_MAP');
const validatorTypes = Object.fromEntries(
  [...typeMapSource.matchAll(/(\w+):\s*'(\w+)'/g)].map(([, key, type]) => [key, type])
);
assert.deepEqual(Object.keys(validatorTypes).sort(), [...schema.required].sort(), 'validator must type-check exactly the required fields');
for (const field of schema.required) {
  assert.equal(
    validatorTypes[field],
    TYPEOF_FOR_SCHEMA_TYPE[schema.properties[field].type],
    `validator type for ${field} must match schema type ${schema.properties[field].type}`
  );
}

console.log('benchmark-schema-contract.test: ok');

import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import * as canonical from '../../src/config/transforms/execution-graph-transforms.js';
import * as compatibility from '../../src/converter/execution-graph-transforms.js';
import * as rules from '../../src/rules/capability-transform-resolver.js';
import * as oldResolver from '../../src/converter/capability-transform-resolver.js';
import { dequantizeQ4KM } from '../../src/formats/q4k-dequant.js';
import { dequantizeQ4KM as converterDecoder } from '../../src/converter/quantizer.js';

// Compatibility paths forward to the same objects, including the registry;
// callers cannot accidentally use independent transform implementations.
assert.deepEqual(Object.keys(canonical), Object.keys(compatibility));
for (const name of Object.keys(canonical)) assert.equal(compatibility[name], canonical[name], name);
assert.equal(oldResolver.resolveCapabilityTransforms, rules.resolveCapabilityTransforms);
assert.equal(oldResolver.resolveFinitenessFallbackTransform, rules.resolveFinitenessFallbackTransform);
assert.equal(converterDecoder, dequantizeQ4KM);
const policy = JSON.parse(await fs.readFile(new URL('../../tools/policies/source-architecture-policy.json', import.meta.url), 'utf8'));
assert.equal(policy.allowedDependencies.inference.includes('converter'), false);
assert.equal(policy.allowedDependencies.config.includes('converter'), false);
console.log('runtime-transform-ownership.test: compatibility identity and runtime boundaries passed');

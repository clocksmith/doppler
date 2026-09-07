import assert from 'node:assert/strict';
import { globals } from 'webgpu';
import { bootstrapNodeWebGPUProvider } from '../../../../src/tooling/node-webgpu.js';
import { installNodeFileFetchShim } from '../../../../src/tooling/node-file-fetch.js';
import config from './hybrid-reference-provider.json' with { type: 'json' };

Object.assign(globalThis, globals);
installNodeFileFetchShim();
const provider = await bootstrapNodeWebGPUProvider(config.provider, { createArgs: config.createArgs });
assert(provider.ok);
const info = provider.session.adapter.info;
assert.equal(info.vendor, config.requiredVendor);
assert.equal(info.isFallbackAdapter, config.isFallbackAdapter);
console.log(JSON.stringify({ stage: 'physical-reference', hardware:
  Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter'].map(key => [key, info[key]])) }));

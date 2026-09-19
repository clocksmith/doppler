import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { KERNEL_CONFIGS } from '../../src/config/kernel-registry-contract.js';
import { getRequiredWgslFeatures } from '../../src/config/wgsl-language-contract.js';

const sources = new Map();
let variants = 0;
for (const [operation, configs] of Object.entries(KERNEL_CONFIGS)) {
  for (const [variant, config] of Object.entries(configs)) {
    if (!sources.has(config.shaderFile)) {
      const source = await readFile(new URL(`../../src/gpu/kernels/${config.shaderFile}`, import.meta.url), 'utf8');
      sources.set(config.shaderFile, getRequiredWgslFeatures(source));
    }
    assert.deepEqual([...config.requiredWgslFeatures].sort(), sources.get(config.shaderFile),
      `${operation}/${variant}: registry WGSL language requirements must match shader directives`);
    variants++;
  }
}
console.log(`kernel-language-contract.test: ${variants} variants, ${sources.size} modules agree`);

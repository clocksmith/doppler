import assert from 'node:assert/strict';
import { loadRuntimeModules, createRuntimeBridge } from '../../src/tooling/node-runtime-bridge.js';
import { loadRuntimeConfigFromRef } from '../../src/tooling/browser-harness/runtime-config.js';
import * as kernelPaths from '../../src/config/kernel-path-loader.js';

// Command execution and diagnosis share lazy imports, not duplicated runtime wiring.
const firstLoad = loadRuntimeModules();
assert.equal(loadRuntimeModules(), firstLoad);
const modules = await firstLoad;
assert.equal(await loadRuntimeModules(), modules);
const bridge = createRuntimeBridge(modules);
assert.notEqual(createRuntimeBridge(modules), bridge);
assert.equal(bridge.loadRuntimeConfigFromRef, loadRuntimeConfigFromRef);
for (const key of ['loadRuntimeProfile', 'loadRuntimeConfigFromUrl']) {
  assert.equal(bridge[key], modules.harness[key]);
}
for (const key of ['getRuntimeConfig', 'setRuntimeConfig', 'resetRuntimeConfig']) {
  assert.equal(bridge[key], modules.runtime[key]);
}
for (const key of ['getActiveKernelPath', 'getActiveKernelPathPolicy',
  'getActiveKernelPathSource', 'setActiveKernelPath']) {
  assert.equal(bridge[key], kernelPaths[key]);
}
console.log('node-runtime-bridge.test: ok');

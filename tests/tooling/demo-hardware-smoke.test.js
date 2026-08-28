import assert from 'node:assert/strict';

import { resolveBrowserArgs } from '../../tools/demo-hardware-smoke.js';

assert.deepEqual(
  resolveBrowserArgs('darwin'),
  ['--enable-unsafe-webgpu', '--use-angle=metal']
);
assert.deepEqual(
  resolveBrowserArgs('linux'),
  [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--use-angle=vulkan',
    '--disable-vulkan-surface',
  ]
);
assert.deepEqual(resolveBrowserArgs('win32'), ['--enable-unsafe-webgpu']);
assert.throws(
  () => resolveBrowserArgs('freebsd'),
  /Unsupported demo hardware smoke host platform "freebsd"/
);

console.log('demo-hardware-smoke.test: ok');

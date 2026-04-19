import assert from 'node:assert/strict';

import { disableRetainQ4KMaterialization } from '../../src/config/transforms/execution-graph-transforms.js';
import { resolveCapabilityTransforms } from '../../src/config/transforms/capability-transform-resolver.js';

{
  const r = resolveCapabilityTransforms(
    {
      hasSubgroups: true,
      hasF16: true,
      maxWorkgroupStorageSize: 32768,
      adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
    },
    { id: 'apple-m3', vendor: 'apple', architecture: 'metal-3' },
    {
      activationDtype: 'f32',
      kvDtype: 'f16',
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      retainQ4KMaterialization: true,
    }
  );
  assert.deepEqual(r.names, ['disableRetainQ4KMaterialization']);
  assert.equal(r.transforms.length, 1);
  assert.equal(r.transforms[0], disableRetainQ4KMaterialization);
}

{
  const r = resolveCapabilityTransforms(
    {
      hasSubgroups: true,
      hasF16: true,
      maxWorkgroupStorageSize: 32768,
      adapterInfo: { vendor: 'amd', architecture: 'rdna3' },
    },
    { id: 'amd-rdna3', vendor: 'amd', architecture: 'rdna3' },
    {
      activationDtype: 'f32',
      kvDtype: 'f16',
      modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
      retainQ4KMaterialization: true,
    }
  );
  assert.deepEqual(r.names, []);
  assert.equal(r.transforms.length, 0);
}

console.log('capability-retain-q4k-policy.test: ok');
